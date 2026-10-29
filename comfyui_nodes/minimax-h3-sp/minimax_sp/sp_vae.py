"""2 台組で H3 の動画 VAE のデコードを分担する（2026-10-28）。

H3 の動画 VAE は時間方向にクリップ（tokens_chunk_size 個の latent＋重なり）ごとに独立してデコードし、
その後に重なりをなじませて書き出す（MiniMaxH3VideoVAE.decode_temporal）。ここでは各クリップの
_adaptive_decode（区画分割を含む）だけを rank ごとに分け（クリップ i は rank i % world）、結果を head に
集めてから、元の decode_temporal をそのまま走らせる（つなぎ合わせ・書き出しは従来どおり）。
各クリップの計算は head と同じ関数・同じ入力・同じ種類の GPU なので、結果は単独と同じになる想定
（latent → 画素の一致で確かめる）。

有効化：head と worker の両方で環境変数 MINIMAX_SP_VAE_SPLIT=1。worker は MINIMAX_SP_VAE_PATH の VAE を
最初の命令で読み込む（head の VAELoader と同じファイルにすること）。
"""

import logging
import os
import time

import torch
import torch.distributed as dist

import comfy.ldm.minimax.vae as h3vae

ENABLED = os.environ.get("MINIMAX_SP_VAE_SPLIT") == "1"
OP = "vae_temporal"


def _clip_ranges(vae, z_len):
    pad_tokens, num_chunks = vae._decode_temporal_chunks(z_len)
    ranges = []
    for i in range(num_chunks):
        s = i * vae.tokens_chunk_size
        ranges.append((s, s + vae.tokens_chunk_size + vae.token_overlap))
    return pad_tokens, ranges


def _pad(z, pad_tokens):
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)
    return z


def _decode_local(vae, zz, ranges, mine):
    out = {}
    for i in mine:
        s, e = ranges[i]
        out[i] = vae._adaptive_decode(zz[:, :, s:e, :, :])
    return out


def install_head():
    """head（ComfyUI 本体のプロセス）で decode_temporal を包む。SP 組が無いときは従来どおり。"""
    if not ENABLED or getattr(h3vae.MiniMaxH3VideoVAE, "_sp_vae_wrapped", False):
        return
    from . import sp_group
    orig = h3vae.MiniMaxH3VideoVAE.decode_temporal

    def decode_temporal(self, z, output_buffer=None):
        grp = sp_group.active_group()
        # 2 台組と 4 台の輪（2026-10-28）。3 台は未対応
        if grp is None or grp.control_pg is None or grp.world not in (2, 4):
            return orig(self, z, output_buffer)
        t0 = time.perf_counter()
        pad_tokens, ranges = _clip_ranges(self, z.shape[2])
        if len(ranges) < 2:
            return orig(self, z, output_buffer)
        world = grp.world
        zz = _pad(z, pad_tokens).contiguous()
        owner = [i % world for i in range(len(ranges))]
        meta = {"op": OP, "shape": list(zz.shape), "dtype": str(zz.dtype).replace("torch.", ""),
                "ranges": ranges, "owner": owner}
        dist.broadcast_object_list([meta], src=0, group=grp.control_pg)
        dist.broadcast(zz, src=0, group=grp.tensor_pg)
        local = _decode_local(self, zz, ranges, [i for i in range(len(ranges)) if owner[i] == 0])
        shapes = [None] * world
        dist.all_gather_object(shapes, {"rank": 0}, group=grp.control_pg)
        results = dict(local)
        for r in range(1, world):   # 送受信の前に全部の worker の成否を見る（worker 側も同じ判断で送らない）
            if "error" in shapes[r]:
                raise RuntimeError(f"[minimax_sp] VAE worker rank {r} failed: {shapes[r]['error']}")
        # broadcast(src=r) は全 rank が参加する（4 台では他の worker も受け手として加わる。輪の Ring で隣どうしだけ）
        for r in range(1, world):
            info = shapes[r]
            for i, shp, dt in info["parts"]:
                buf = torch.empty(shp, dtype=getattr(torch, dt), device=zz.device)
                dist.broadcast(buf, src=r, group=grp.tensor_pg)
                results[i] = buf
        t_par = time.perf_counter()
        order = iter(results[i] for i in range(len(ranges)))

        def adaptive(clip_z, frames=slice(None)):
            part = next(order)
            return part if frames == slice(None) else part[:, :, frames]

        self._adaptive_decode = adaptive
        try:
            out = orig(self, z, output_buffer)
        finally:
            del self._adaptive_decode
        logging.info("[minimax_sp][vae] %d clips split over %d ranks: decode %.1fs, assemble %.1fs",
                     len(ranges), world, t_par - t0, time.perf_counter() - t_par)
        return out

    h3vae.MiniMaxH3VideoVAE.decode_temporal = decode_temporal
    h3vae.MiniMaxH3VideoVAE._sp_vae_wrapped = True
    logging.info("[minimax_sp] VAE temporal split enabled on head")


class WorkerVAE:
    """worker 側：最初の命令で VAE を読み込み、割り当てられたクリップをデコードして head へ送る。"""

    def __init__(self, rank):
        self.rank = rank
        self.vae = None

    def _load(self):
        if self.vae is not None:
            return self.vae
        import comfy.model_management
        import comfy.sd
        import comfy.utils
        path = os.environ.get("MINIMAX_SP_VAE_PATH")
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"MINIMAX_SP_VAE_PATH is not a file: {path}")
        t0 = time.perf_counter()
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        comfy.model_management.load_models_gpu([vae.patcher], force_full_load=True)
        self.vae = vae
        logging.info("rank %s: VAE loaded from %s in %.1fs", self.rank, path, time.perf_counter() - t0)
        return vae

    def handle(self, meta, control_pg, tensor_pg, world, device):
        zz = torch.empty(meta["shape"], dtype=getattr(torch, meta["dtype"]), device=device)
        dist.broadcast(zz, src=0, group=tensor_pg)
        reply = {"rank": self.rank, "parts": []}
        results = {}
        try:
            vae = self._load()
            mine = [i for i, o in enumerate(meta["owner"]) if o == self.rank]
            t0 = time.perf_counter()
            with torch.inference_mode():
                results = _decode_local(vae.first_stage_model, zz.to(vae.vae_dtype), meta["ranges"], mine)
            torch.cuda.synchronize()
            reply["parts"] = [(i, list(t.shape), str(t.dtype).replace("torch.", "")) for i, t in sorted(results.items())]
            logging.info("rank %s: VAE decoded %d clips in %.1fs", self.rank, len(mine), time.perf_counter() - t0)
        except Exception as exc:
            logging.exception("rank %s: VAE decode failed", self.rank)
            reply = {"rank": self.rank, "error": str(exc), "parts": []}
        replies = [None] * world
        dist.all_gather_object(replies, reply, group=control_pg)
        if any("error" in rep for rep in replies[1:]):
            return   # head が例外にする（どの rank も送受信しない）
        for r in range(1, world):
            if r == self.rank:
                for i, _, _ in reply["parts"]:
                    dist.broadcast(results[i].contiguous(), src=self.rank, group=tensor_pg)
            else:   # 他の worker の分：受け手として加わり、受けたものは捨てる
                for i, shp, dt in replies[r]["parts"]:
                    buf = torch.empty(shp, dtype=getattr(torch, dt), device=device)
                    dist.broadcast(buf, src=r, group=tensor_pg)
                    del buf
