"""H3 の Qwen（ComfyUI の CLIP.encode_from_tokens）の間だけ、fp32 の行列計算に TF32 を使う（DGX Spark 互換機での本番、2026-10-28）。
ComfyUI の H3 用 Qwen は内部を fp32 で計算し、GB10 では前処理（R2V 10 秒・参照 6 枚）が 25.8 秒かかる。TF32 では 10.0 秒。
TF32 は Tensor Core で掛け算を 10 ビットの仮数で行い、足し算は fp32 のまま。条件づけの数値がわずかに変わるので、
fp32 の版とは動画がビット一致しない（見比べて差はわずかと判断し、本番に採用）。
既定で有効。H3_QWEN_TF32=0 で ComfyUI を起動すると従来の fp32 のまま。終わったら元の設定に戻すので、DiT・VAE には効かない。
外すときはこのフォルダを消して ComfyUI を起動し直す。効果と画質の比べは README の「前処理」を参照。
"""
import functools, logging, os, time

NODE_CLASS_MAPPINGS = {}
ENABLED = os.environ.get("H3_QWEN_TF32", "1") != "0"

import torch
import comfy.sd

_orig = comfy.sd.CLIP.encode_from_tokens
if ENABLED and not getattr(_orig, "_h3_tf32", False):
    @functools.wraps(_orig)
    def encode_from_tokens(self, *a, **k):
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        t = time.perf_counter()
        try:
            out = _orig(self, *a, **k)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        logging.info("[h3-qwen-tf32] Qwen TF32 %.2f 秒", time.perf_counter() - t)
        return out

    encode_from_tokens._h3_tf32 = True
    comfy.sd.CLIP.encode_from_tokens = encode_from_tokens
    logging.info("[h3-qwen-tf32] 有効（Qwen の計算だけ TF32。H3_QWEN_TF32=0 で fp32）")
elif not ENABLED:
    logging.info("[h3-qwen-tf32] 無効（H3_QWEN_TF32=0、Qwen は fp32）")
