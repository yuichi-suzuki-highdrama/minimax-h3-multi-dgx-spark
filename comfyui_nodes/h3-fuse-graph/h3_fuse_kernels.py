"""H3 の DiT ブロックの細かい処理を融合する Triton カーネル（2026-10-28）。

元（comfy/ldm/minimax/model.py の DiTBlock.forward）:
  h = norm1(x)                                   # torch の rms_norm（vectorized_layer_norm_kernel<bf16, float, rms=true>）
  h[a:b].mul_(1 + scale[row]).add_(shift[row])   # 区間（mod_segments）ごとに 2 カーネル
  x[a:b].addcmul_(attn_out[a:b], gate[row])      # 区間ごとに 1 カーネル
  h = norm2(x); 同じ変調; x += mlp_out * gate
融合後:
  norm_mod        : h = ((rmsnorm(x) * s1[row]) + shift[row])            1 カーネル（x を読み h を書く）
  gate_norm_mod_  : x = x + o * g[row]; h = ((rmsnorm(x) * s1[row]) + shift[row])   1 カーネル
  gate_           : x = x + o * g[row]                                   1 カーネル
ビット一致のために:
  - bf16 に丸める位置は元と同じ（norm の出力、掛け算の後、足し算の後、残差の後）。
  - RMSNorm の二乗和は PyTorch の vectorized_layer_norm_kernel と同じ順番で足す（128 スレッド × 4 要素の
    かたまりを順に、warp 内は shuffle-down の木、warp 間は共有メモリの木）。/N は IEEE の割り算、rsqrt は近似命令、
    y = w * (rstd * x)。torch 2.11.0+cu130 で 0 ビット差を確認（test_fuse.py）。
"""
import os
import sysconfig

import torch


def _ensure_python_headers():
    inc = sysconfig.get_paths().get("include") or ""
    if os.path.exists(os.path.join(inc, "Python.h")):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for pyinc in (os.path.join(here, "pyinclude"), os.path.join(os.path.dirname(here), "h3-linear-fast", "pyinclude")):
        if os.path.exists(os.path.join(pyinc, "python3.12", "Python.h")):
            add = os.pathsep.join([os.path.join(pyinc, "python3.12"), pyinc])
            cur = os.environ.get("CPATH")
            os.environ["CPATH"] = add + (os.pathsep + cur if cur else "")
            return


_ensure_python_headers()

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

NUMX = 128   # PyTorch の num_threads()（warp 32 × 4）
VEC = 4      # PyTorch の vectorized_layer_norm_kernel の vec_size


@triton.jit
def _cols4(t):
    t = tl.reshape(t, [t.shape[0], 2, 2])
    e, o = tl.split(t)
    c0, c2 = tl.split(e)
    c1, c3 = tl.split(o)
    return c0, c1, c2, c3


@triton.jit
def _sumsq4(acc, t):
    c0, c1, c2, c3 = _cols4(t)
    acc = acc + c0 * c0
    acc = acc + c1 * c1
    acc = acc + c2 * c2
    acc = acc + c3 * c3
    return acc


@triton.jit
def _tree128(acc):
    s = tl.reshape(acc, [4, 32])
    s = tl.sum(tl.reshape(s, [4, 2, 16]), axis=1)
    s = tl.sum(tl.reshape(s, [4, 2, 8]), axis=1)
    s = tl.sum(tl.reshape(s, [4, 2, 4]), axis=1)
    s = tl.sum(tl.reshape(s, [4, 2, 2]), axis=1)
    s = tl.sum(tl.reshape(s, [4, 2, 1]), axis=1)
    s = tl.reshape(s, [4])
    s = tl.sum(tl.reshape(s, [2, 2]), axis=0)   # warp y と y+2
    return tl.sum(s, axis=0)                    # warp 0 と 1


@triton.jit
def _rbf16(v):
    # f32 -> bf16 の最近接偶数丸めを f32 のまま返す。.to(bf16).to(f32) と書くと Triton が往復を畳んで
    # 丸めが消え、次の掛け算・足し算と FMA にされる（実測）ので、整数演算で丸める。
    u = v.to(tl.uint32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    u = (u >> 16) << 16
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _norm_mod_kernel(X, W, RIDX, S1, SH, H, N, eps, ITERS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    r = tl.load(RIDX + row).to(tl.int64)
    xb = X + row * N
    hb = H + row * N
    s1b = S1 + r * N
    shb = SH + r * N
    t = tl.arange(0, 128)
    v = tl.arange(0, 4)
    acc = tl.zeros([128], tl.float32)
    for it in tl.static_range(ITERS):
        idx = (it * 128 + t)[:, None] * 4 + v[None, :]
        x = tl.load(xb + idx, mask=idx < N, other=0.0).to(tl.float32)
        acc = _sumsq4(acc, x)
    tot = _tree128(acc)
    rstd = tl.math.rsqrt(tl.math.div_rn(tot, N.to(tl.float32)) + eps)
    for it in tl.static_range(ITERS):
        idx = (it * 128 + t)[:, None] * 4 + v[None, :]
        m = idx < N
        x = tl.load(xb + idx, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + idx, mask=m, other=0.0).to(tl.float32)
        y = _rbf16(w * (rstd * x))
        s1 = tl.load(s1b + idx, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(shb + idx, mask=m, other=0.0).to(tl.float32)
        y = _rbf16(y * s1)
        tl.store(hb + idx, (y + sh).to(tl.bfloat16), mask=m)


@triton.jit
def _gate_norm_mod_kernel(X, O, G, W, RIDX, S1, SH, H, N, eps, ITERS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    r = tl.load(RIDX + row).to(tl.int64)
    xb = X + row * N
    ob = O + row * N
    hb = H + row * N
    gb = G + r * N
    s1b = S1 + r * N
    shb = SH + r * N
    t = tl.arange(0, 128)
    v = tl.arange(0, 4)
    acc = tl.zeros([128], tl.float32)
    # 1 周目: 残差を足した値の二乗和だけ（x はまだ書かない）
    for it in tl.static_range(ITERS):
        idx = (it * 128 + t)[:, None] * 4 + v[None, :]
        m = idx < N
        x = tl.load(xb + idx, mask=m, other=0.0).to(tl.float32)
        o = tl.load(ob + idx, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gb + idx, mask=m, other=0.0).to(tl.float32)
        xn = _rbf16(x + o * g)
        acc = _sumsq4(acc, xn)
    tot = _tree128(acc)
    rstd = tl.math.rsqrt(tl.math.div_rn(tot, N.to(tl.float32)) + eps)
    # 2 周目: 同じ値を作り直して x と h を書く
    for it in tl.static_range(ITERS):
        idx = (it * 128 + t)[:, None] * 4 + v[None, :]
        m = idx < N
        x = tl.load(xb + idx, mask=m, other=0.0).to(tl.float32)
        o = tl.load(ob + idx, mask=m, other=0.0).to(tl.float32)
        g = tl.load(gb + idx, mask=m, other=0.0).to(tl.float32)
        xn = _rbf16(x + o * g)
        tl.store(xb + idx, xn.to(tl.bfloat16), mask=m)
        w = tl.load(W + idx, mask=m, other=0.0).to(tl.float32)
        y = _rbf16(w * (rstd * xn))
        s1 = tl.load(s1b + idx, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(shb + idx, mask=m, other=0.0).to(tl.float32)
        y = _rbf16(y * s1)
        tl.store(hb + idx, (y + sh).to(tl.bfloat16), mask=m)


@triton.jit
def _gate_kernel(X, O, G, RIDX, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = col < N
    r = tl.load(RIDX + row).to(tl.int64)
    x = tl.load(X + row * N + col, mask=m, other=0.0).to(tl.float32)
    o = tl.load(O + row * N + col, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + r * N + col, mask=m, other=0.0).to(tl.float32)
    tl.store(X + row * N + col, (x + o * g).to(tl.bfloat16), mask=m)


def _iters(n):
    return triton.cdiv(triton.cdiv(n, VEC), NUMX)


def norm_mod(x, w, eps, ridx, s1, sh):
    s, n = x.shape
    h = torch.empty_like(x)
    _norm_mod_kernel[(s,)](x, w, ridx, s1, sh, h, n, eps, ITERS=_iters(n), num_warps=4)
    return h


def gate_norm_mod_(x, o, g, w, eps, ridx, s1, sh):
    s, n = x.shape
    h = torch.empty_like(x)
    _gate_norm_mod_kernel[(s,)](x, o, g, w, ridx, s1, sh, h, n, eps, ITERS=_iters(n), num_warps=4)
    return h


def gate_(x, o, g, ridx):
    s, n = x.shape
    block = 1024
    _gate_kernel[(s, triton.cdiv(n, block))](x, o, g, ridx, n, BLOCK=block, num_warps=4)
    return x
