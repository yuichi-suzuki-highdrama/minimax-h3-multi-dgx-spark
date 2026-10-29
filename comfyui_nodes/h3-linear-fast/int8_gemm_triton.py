"""INT8 GEMM（行ごと×列ごとの scale を掛けて bf16 で直接書く）の Triton 版（2026-10-28）。
out[m, n] = bf16( (float(sum_k xq[m,k]*w[n,k]) * xs[m]) * ws[n] )
掛け算の順番は comfy-kitchen の CUTLASS 経路と同じ（実測でビット一致を確認する）。
"""
import os
import torch
import triton
import triton.language as tl


def _configs():
    # 2026-10-28 に 1 台で 11 通りを自動調整し、H3 の 4 層×M=19,200／38,400 で選ばれたのはこの 3 つだけだった
    # （他の候補は H3_LINEAR_FAST_ALLCFG=1 で戻せる）
    cands = [(128, 128, 64, 4, 4), (128, 128, 64, 4, 8), (128, 256, 64, 4, 8)]
    if os.environ.get("H3_LINEAR_FAST_ALLCFG") == "1":
        cands += [(128, 256, 64, 3, 8), (256, 128, 64, 3, 8), (128, 128, 128, 3, 8), (128, 256, 128, 2, 8),
                  (256, 128, 128, 2, 8), (256, 128, 64, 4, 8), (64, 256, 128, 3, 4), (128, 128, 128, 4, 8)]
    cfgs = []
    for bm, bn, bk, st, wp in cands:
        cfgs.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_stages=st, num_warps=wp))
    return cfgs


_FIXED = os.environ.get("H3_LINEAR_FAST_CFG")  # 例 "128,256,64,3,8"（自動調整を省く）
if _FIXED:
    _bm, _bn, _bk, _st, _wp = [int(v) for v in _FIXED.split(",")]
    _CFGS = [triton.Config({"BM": _bm, "BN": _bn, "BK": _bk, "GROUP_M": 8}, num_stages=_st, num_warps=_wp)]
else:
    _CFGS = _configs()


@triton.autotune(configs=_CFGS, key=["N", "K", "M_BUCKET"])
@triton.jit
def _int8_mm_scaled_kernel(A, B, C, XS, WS, M, N, K, M_BUCKET,
                           stride_am, stride_bn, stride_cm,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_m = group_id * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rm_c = tl.where(rm < M, rm, 0)
    rn_c = tl.where(rn < N, rn, 0)
    a_ptr = A + rm_c[:, None] * stride_am + rk[None, :]
    b_ptr = B + rn_c[None, :] * stride_bn + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, K // BK):  # K は BK の倍数に限る（呼び出し側で確認）
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptr += BK
        b_ptr += BK
    xs = tl.load(XS + rm, mask=rm < M, other=0.0)
    ws = tl.load(WS + rn, mask=rn < N, other=0.0)
    c = acc.to(tl.float32) * xs[:, None]
    c = c * ws[None, :]
    c_ptr = C + rm[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptr, c.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def _bucket(m):
    # 自動調整のキー。M は 19,200 / 38,400 前後なので 4096 刻みでまとめる
    return max(1, (m + 4095) // 4096)


def int8_mm_scaled(xq, w, xs, ws, out=None):
    """xq [M,K] int8、w [N,K] int8、xs [M,1] fp32、ws [N] fp32 → [M,N] bf16"""
    M, K = xq.shape
    N = w.shape[0]
    assert w.shape[1] == K and xq.is_contiguous() and w.is_contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=xq.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)
    _int8_mm_scaled_kernel[grid](xq, w, out, xs, ws, M, N, K, _bucket(M), xq.stride(0), w.stride(0), out.stride(0))
    return out


def supported(K):
    return K % 128 == 0
