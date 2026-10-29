"""MiniMax H3 の Ulysses シーケンス並列を 2 台の DGX Spark にまたがらせる UNETLoader（2026-10-28）。

buqi-code/buqi-minimax-h3-multigpu（MIT）の minimax_sp を元に、
  - rank1 を同じ機械の子プロセスではなく「別ホストで手で起動した worker」にした
    （head は TCPStore を master_addr:master_port で開いて待つ。worker は sp_worker.py）
  - LoRA の patch 同期を「一時ファイルの道」から「gloo でバイト列を送る」に変えた
  - ComfyUI v0.42.3（＋5090 の H3 差分）の _forward に合わせた（rope_freqs(layout, device)、
    ModelAttentionBackend の optimized_attention_override を許す、all_gather_into_tensor）
UNETLoader をこのノードに置き換えるだけで、他のノード（RES4LYF の ClownsharKSampler_Beta、
LoraLoaderModelOnly、ModelAttentionBackend、H3RoutingMeter）はそのまま。
"""

import inspect
import logging
import os
import sys

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.ops
import comfy.sd
import folder_paths

from . import sp_group
from . import sp_vae
from .sp_forward import validate_transformer_options

sp_vae.install_head()   # MINIMAX_SP_VAE_SPLIT=1 のときだけ、2 台組で VAE のクリップを分担する

from comfy.ldm.minimax.model import Attention, DiTBlock, FinalLayer, MiniMaxH3Model, PackedLayout
from comfy.patcher_extension import CallbacksMP
import comfyui_version


def check_h3_api():
    problems = []
    forward_required = {"x", "timestep", "context", "transformer_options", "minimax_payload",
                        "denoise_mask", "audio_denoise_mask"}
    missing = forward_required.difference(inspect.signature(MiniMaxH3Model._forward).parameters)
    if missing:
        problems.append(f"MiniMaxH3Model._forward is missing {', '.join(sorted(missing))}")
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        problems.append("PackedLayout still requires frame_count")
    try:
        attention = Attention(1, 1, 1, 1e-6, operations=comfy.ops.disable_weight_init)
    except TypeError:
        attention = None
    if attention is None or not hasattr(attention, "comfy_attention"):
        problems.append("Attention does not expose comfy_attention")
    if "attention" not in inspect.signature(DiTBlock.forward).parameters:
        problems.append("DiTBlock.forward does not accept attention")
    final_required = {"sigma", "sample_sigmas", "shifts"}
    final_missing = final_required.difference(inspect.signature(FinalLayer.forward).parameters)
    if final_missing:
        problems.append(f"FinalLayer.forward is missing {', '.join(sorted(final_missing))}")
    if problems:
        raise RuntimeError("MiniMax SP requires the H3 API introduced by ComfyUI commit 8d534945: "
                           + "; ".join(problems) + ". Update ComfyUI and restart it.")
    logging.info("[minimax_sp] compatible ComfyUI %s MiniMaxH3Model%s",
                 comfyui_version.__version__, inspect.signature(MiniMaxH3Model))


class MiniMaxH3SPUNETLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPUNETLoader",
            display_name="MiniMax H3 Multi-Host Loader (Ulysses SP, 2 Sparks)",
            category="advanced/multigpu",
            description="Loads the MiniMax-H3 DiT and shards the packed sequence across "
                        "world_size hosts (Ulysses all-to-all over NCCL/RoCE). rank>=1 workers "
                        "must be started by hand with sp_worker.py on the other host(s).",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]),
                io.Int.Input("world_size", default=2, min=1, max=8,
                             tooltip="Ranks (hosts) to shard across. Must divide the 56 attention heads "
                                     "(valid: 1, 2, 4, 7, 8)."),
                io.String.Input("master_addr", default=os.environ.get("MINIMAX_SP_MASTER_ADDR", "127.0.0.1"),
                                tooltip="This host's fabric IPv4 address that the workers connect to (TCPStore)."),
                io.Int.Input("master_port", default=int(os.environ.get("MINIMAX_SP_MASTER_PORT", "29628")),
                             min=1024, max=65535, tooltip="TCPStore rendezvous port."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype, world_size, master_addr="127.0.0.1", master_port=29628) -> io.NodeOutput:
        if world_size > 1:
            check_h3_api()
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        if world_size <= 1:
            return io.NodeOutput(model)

        if sys.platform == "win32":
            raise RuntimeError("multi-GPU SP needs NCCL, which is not available on native Windows; "
                               "run ComfyUI inside WSL2 or on Linux")

        dit = model.get_model_object("diffusion_model")
        if not isinstance(dit, MiniMaxH3Model):
            raise RuntimeError("MiniMax SP loader requires a MiniMaxH3Model checkpoint")
        heads = dit.blocks[0].attn.heads
        if heads % world_size:
            raise ValueError(f"world_size {world_size} must divide the {heads} attention heads "
                             "(valid: 1, 2, 4, 7, 8)")

        key = (world_size, unet_name, weight_dtype, master_addr, int(master_port))
        sp_group.get_group(*key, unet_path=path)

        model = model.clone()
        dit = model.get_model_object("diffusion_model")

        def sp_entry(x, timestep, context, transformer_options=None, minimax_payload=None,
                     denoise_mask=None, audio_denoise_mask=None, **kwargs):
            transformer_options = transformer_options or {}
            validate_transformer_options(transformer_options)
            group = sp_group.get_group(*key, unet_path=path)
            return group.forward(dit, x, timestep, context, transformer_options, minimax_payload,
                                 denoise_mask, audio_denoise_mask)

        def sync_patches(patcher):
            group = sp_group.get_group(*key, unet_path=path)
            group.sync_patches(patcher)

        model.add_callback_with_key(CallbacksMP.ON_PRE_RUN, "minimax_sp_lora", sync_patches)
        # ModelPatcher installs/restores the inner call without bypassing outer wrappers.
        model.add_object_patch("diffusion_model._forward", sp_entry)
        logging.info(f"[minimax_sp] sequence parallel enabled, world_size={world_size} master={master_addr}:{master_port}")
        return io.NodeOutput(model)


class MiniMaxSPExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SPUNETLoader]


async def comfy_entrypoint() -> ComfyExtension:
    return MiniMaxSPExtension()
