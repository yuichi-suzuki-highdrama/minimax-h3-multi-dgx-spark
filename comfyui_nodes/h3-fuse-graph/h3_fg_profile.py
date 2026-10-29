"""H3 の forward を torch.profiler にかける計測用フック（2026-10-28）。
環境変数 H3_FG_PROFILE=3,7 のように、何回目の forward を測るかを渡したときだけ有効。
H3_FG_TIMING=1 なら、測らない回も forward ごとの秒数（前後で synchronize）をログに出す。
単独は MiniMaxH3Model._forward、2 台組は minimax_sp.sp_forward.sp_forward を包む。
出力: H3_FG_PROFILE_DIR（既定 ./h3_fg_prof）に chrome trace と表。
"""
import functools
import logging
import os
import time

import torch

log = logging.getLogger("h3-fuse-graph")
_state = {"calls": 0, "installed": set()}
OUTDIR = os.environ.get("H3_FG_PROFILE_DIR", "./h3_fg_prof")


def _targets():
    spec = os.environ.get("H3_FG_PROFILE", "")
    return {int(s) for s in spec.replace(" ", "").split(",") if s}


def _enabled():
    return bool(_targets()) or os.environ.get("H3_FG_TIMING", "0") == "1"


def _wrap(orig, label):
    targets = _targets()

    @functools.wraps(orig)
    def wrapped(*a, **k):
        _state["calls"] += 1
        n = _state["calls"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if n not in targets:
            out = orig(*a, **k)
            torch.cuda.synchronize()
            log.warning("[h3-fg] %s forward %d %.3f s", label, n, time.perf_counter() - t0)
            return out
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out = orig(*a, **k)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        os.makedirs(OUTDIR, exist_ok=True)
        tag = os.environ.get("H3_FG_PROFILE_TAG", "prof")
        rank = os.environ.get("H3_FG_RANK", "0")
        base = f"{OUTDIR}/{tag}-r{rank}-fwd{n}"
        prof.export_chrome_trace(base + ".json")
        ka = prof.key_averages()
        with open(base + ".txt", "w") as f:
            f.write(f"wall {wall:.3f} s\n")
            for key in ("self_device_time_total", "self_cuda_time_total"):
                try:
                    f.write(ka.table(sort_by=key, row_limit=120, max_name_column_width=90))
                    break
                except Exception:  # noqa: BLE001
                    continue
        log.warning("[h3-fg] %s forward %d を計測 %.3f s -> %s", label, n, wall, base)
        return out

    return wrapped


def install():
    if not _enabled() or "single" in _state["installed"]:
        return
    try:
        import comfy.ldm.minimax.model as mm
        mm.MiniMaxH3Model._forward = _wrap(mm.MiniMaxH3Model._forward, "single")
        _state["installed"].add("single")
    except Exception as e:  # noqa: BLE001
        log.warning("[h3-fg] 単独の forward を包めない: %s", e)


def install_sp(spf_module):
    if not _enabled() or "sp" in _state["installed"]:
        return
    spf_module.sp_forward = _wrap(spf_module.sp_forward, "sp")
    _state["installed"].add("sp")
    log.warning("[h3-fg] sp_forward を包んだ")
