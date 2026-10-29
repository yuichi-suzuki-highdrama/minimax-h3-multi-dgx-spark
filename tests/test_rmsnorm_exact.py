"""torch.nn.functional.rms_norm（CUDA、bf16）とビット一致する Triton の RMSNorm を探す（2026-10-28）。
PyTorch の vectorized_layer_norm_kernel（rms_norm 版）の足し算の順番をまねる:
  各スレッド t（0..NUMX-1）が VEC 個ずつのかたまり t, t+NUMX, ... を順に二乗和 → warp 内の shuffle-down の木
  → warp 間の木 → /N（IEEE の割り算）→ rsqrt(var+eps) → y = w * (rstd * x)
使い方: python test_rmsnorm_exact.py
"""
import os
import sys
import sysconfig

_inc = sysconfig.get_paths().get("include") or ""
if not os.path.exists(os.path.join(_inc, "Python.h")):
    p = os.environ.get("H3_LINEAR_FAST_PYINCLUDE", "../comfyui_nodes/h3-linear-fast/pyinclude")
    os.environ["CPATH"] = os.pathsep.join([os.path.join(p, "python3.12"), p] + ([os.environ["CPATH"]] if os.environ.get("CPATH") else []))

import torch
import triton
import triton.language as tl


@triton.jit
def _split_cols(t, VEC: tl.constexpr):
    # [NUMX, VEC] -> VEC 本の [NUMX]（VEC は 2/4/8）
    if VEC == 2:
        a, b = tl.split(t)
        return a, b, a, b, a, b, a, b
    elif VEC == 4:
        t = tl.reshape(t, [t.shape[0], 2, 2])
        e, o = tl.split(t)          # e[:, i] = t[:, 2i], o[:, i] = t[:, 2i+1]
        c0, c2 = tl.split(e)
        c1, c3 = tl.split(o)
        return c0, c1, c2, c3, c0, c1, c2, c3
    else:
        t = tl.reshape(t, [t.shape[0], 2, 2, 2])
        e, o = tl.split(t)          # 最後の軸 = 要素番号の 1 の位
        ee, eo = tl.split(e)
        oe, oo = tl.split(o)
        c0, c4 = tl.split(ee)
        c2, c6 = tl.split(eo)
        c1, c5 = tl.split(oe)
        c3, c7 = tl.split(oo)
        return c0, c1, c2, c3, c4, c5, c6, c7


@triton.jit
def _sumsq_tile(acc, t, VEC: tl.constexpr):
    c0, c1, c2, c3, c4, c5, c6, c7 = _split_cols(t, VEC)
    acc = acc + c0 * c0
    acc = acc + c1 * c1
    if VEC >= 4:
        acc = acc + c2 * c2
        acc = acc + c3 * c3
    if VEC >= 8:
        acc = acc + c4 * c4
        acc = acc + c5 * c5
        acc = acc + c6 * c6
        acc = acc + c7 * c7
    return acc


@triton.jit
def _tree(acc, NUMX: tl.constexpr):
    WARPS: tl.constexpr = NUMX // 32
    s = tl.reshape(acc, [WARPS, 32])
    s = tl.sum(tl.reshape(s, [WARPS, 2, 16]), axis=1)
    s = tl.sum(tl.reshape(s, [WARPS, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [WARPS, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [WARPS, 2, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [WARPS, 2, 1]), axis=1)
    s = tl.reshape(s, [WARPS])
    if WARPS >= 16:
        s = tl.sum(tl.reshape(s, [2, WARPS // 2]), axis=0)
    if WARPS >= 8:
        s = tl.sum(tl.reshape(s, [2, s.shape[0] // 2]), axis=0)
    if WARPS >= 4:
        s = tl.sum(tl.reshape(s, [2, s.shape[0] // 2]), axis=0)
    if WARPS >= 2:
        s = tl.sum(tl.reshape(s, [2, s.shape[0] // 2]), axis=0)
    return tl.sum(s, axis=0)


@triton.jit
def rms_kernel(X, W, Y, N, eps, NUMX: tl.constexpr, VEC: tl.constexpr, ITERS: tl.constexpr, RSQ: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    xb = X + row * N
    yb = Y + row * N
    t = tl.arange(0, NUMX)
    v = tl.arange(0, VEC)
    acc = tl.zeros([NUMX], tl.float32)
    for it in tl.static_range(ITERS):
        idx = (it * NUMX + t)[:, None] * VEC + v[None, :]
        x = tl.load(xb + idx, mask=idx < N, other=0.0).to(tl.float32)
        acc = _sumsq_tile(acc, x, VEC)
    tot = _tree(acc, NUMX)
    var = tl.math.div_rn(tot, N.to(tl.float32))
    if RSQ == 0:
        rstd = tl.math.rsqrt(var + eps)
    else:
        rstd = tl.math.div_rn(1.0, tl.math.sqrt_rn(var + eps))
    for it in tl.static_range(ITERS):
        idx = (it * NUMX + t)[:, None] * VEC + v[None, :]
        m = idx < N
        x = tl.load(xb + idx, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + idx, mask=m, other=0.0).to(tl.float32)
        tl.store(yb + idx, (w * (rstd * x)).to(tl.bfloat16), mask=m)


def run(x, w, eps, numx, vec, rsq=0):
    m, n = x.shape
    y = torch.empty_like(x)
    iters = triton.cdiv(triton.cdiv(n, vec), numx)
    rms_kernel[(m,)](x, w, y, n, eps, NUMX=numx, VEC=vec, ITERS=iters, RSQ=rsq, num_warps=4)
    return y


def main():
    torch.manual_seed(0)
    n = 5376
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    eps = 1e-5
    # 大きさの違う行を混ぜる（H3 の残差は行ごとに桁が違う）
    x = (torch.randn(m, n, device="cuda") * torch.logspace(-2, 3, m, device="cuda")[:, None]).to(torch.bfloat16)
    x[:, :7] *= 50
    w = (1 + 0.3 * torch.randn(n, device="cuda")).to(torch.bfloat16)
    ref = torch.nn.functional.rms_norm(x, (n,), w, eps)
    print("torch", torch.__version__, "triton", triton.__version__)
    for numx in (32, 64, 128, 256, 512):
        for vec in (2, 4, 8):
            for rsq in (0, 1):
                try:
                    y = run(x, w, eps, numx, vec, rsq)
                except Exception as e:  # noqa: BLE001
                    print(numx, vec, rsq, "失敗", str(e)[:200])
                    continue
                bad = (y.view(torch.int16) != ref.view(torch.int16))
                print(f"NUMX={numx:4d} VEC={vec} RSQ={rsq}  違う要素 {int(bad.sum()):9d}  違う行 {int(bad.any(1).sum()):6d}/{m}")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch.nn.functional.rms_norm(x, (n,), w, eps)
        torch.cuda.synchronize()
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or "kernel" in e.key.lower():
            print("kernel:", e.key[:200])


if __name__ == "__main__":
    main()
