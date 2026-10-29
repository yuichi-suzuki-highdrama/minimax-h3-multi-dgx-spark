"""H3 の DiTBlock.forward を、細かい処理を融合した版に差し替える（2026-10-28）。

有効にするのは環境変数 H3_FUSE=1 のときだけ。差し替えた forward は、元と同じ順番で同じ値を計算する
（norm・変調・ゲート・残差を Triton の 3 カーネルにまとめるだけ。attention と MLP は元のまま呼ぶ）。
条件に合わない呼び出し（bf16 以外、行列が 2 次元でない、区間が行を隙間なく覆っていない等）は元の forward へ。
単独（MiniMaxH3Model._forward）も 2 台組（minimax_sp の sp_block）も DiTBlock.forward を通るので両方に効く。
"""
import importlib.abc
import importlib.machinery
import functools
import logging
import os
import sys

import torch

log = logging.getLogger("h3-fuse-graph")
_HERE = os.path.dirname(os.path.abspath(__file__))
_state = {"installed": False, "fused": 0, "fallback": 0, "ridx_key": None, "ridx": None}


def enabled():
    return os.environ.get("H3_FUSE", "0") == "1"


def _fallback(reason):
    _state["fallback"] += 1
    if _state["fallback"] <= 5:
        log.warning("[h3-fuse] 元の forward へ: %s", reason)


def _row_index(mod_segments, s, device):
    """mod_segments（(start, stop, row) の並び。row は整数か行ごとの LongTensor）→ 行ごとの int32 添字 [s]。
    同じ forward の 50 ブロックは同じリストを使うので、最後のものを覚えておく（リストは参照を持って id の再利用を防ぐ）。"""
    key = _state["ridx_key"]
    if key is not None and key[0] is mod_segments and key[1] == s and key[2] == device:
        return _state["ridx"]
    segs = sorted(((a, b, row) for a, b, row in mod_segments), key=lambda t: t[0])
    pos = 0
    for a, b, row in segs:
        if a != pos or b < a:
            return None
        if torch.is_tensor(row) and row.numel() != b - a:
            return None
        pos = b
    if pos != s:
        return None
    r = torch.empty(s, dtype=torch.int32, device=device)
    for a, b, row in segs:
        if torch.is_tensor(row):
            r[a:b] = row.to(device=device, dtype=torch.int32)
        else:
            r[a:b] = int(row)
    _state["ridx_key"] = (mod_segments, s, device)
    _state["ridx"] = r
    return r


def _norm_weight(norm, x):
    """comfy.ops の RMSNorm.forward と同じ取り方で重みを得る（コンテキストを返す）。"""
    import comfy.ops
    if norm.comfy_cast_weights or len(norm.weight_function) > 0 or len(norm.bias_function) > 0:
        return norm.cast_weight(x)
    return _Plain(norm.weight)


class _Plain:
    def __init__(self, w):
        self.w = w

    def __enter__(self):
        return self.w, None

    def __exit__(self, *a):
        return False


def install():
    if _state["installed"] or not enabled():
        return _state["installed"]
    try:
        import comfy.ldm.minimax.model as mm
        import comfy.ops
    except Exception as e:  # noqa: BLE001
        log.warning("[h3-fuse] comfy.ldm.minimax.model を読めないので無効: %s", e)
        return False
    if _HERE not in sys.path:
        sys.path.append(_HERE)
    import h3_fuse_kernels as fk

    orig = mm.DiTBlock.forward
    run_every_op = getattr(comfy.ops, "run_every_op", None)
    bf16 = torch.bfloat16

    @functools.wraps(orig)
    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        n1, n2 = self.norm1, self.norm2
        s = x.shape[0] if x.dim() == 2 else 0
        ok = (x.dim() == 2 and x.is_cuda and x.dtype == bf16 and x.is_contiguous() and x.data_ptr() % 16 == 0
              and x.shape[1] % 4 == 0 and n1.eps is not None and n2.eps is not None
              and tuple(n1.normalized_shape) == (x.shape[1],) and tuple(n2.normalized_shape) == (x.shape[1],)
              and not torch.is_grad_enabled())
        if not ok:
            _fallback(f"x={tuple(x.shape)} {x.dtype} contiguous={x.is_contiguous()} grad={torch.is_grad_enabled()}")
            return orig(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, attention=attention)
        ridx = _row_index(mod_segments, s, x.device)
        if ridx is None:
            _fallback("mod_segments が行を隙間なく覆っていない")
            return orig(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, attention=attention)
        attention = self.attn if attention is None else attention
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        # 元の _mod_row(vecs, row, dtype) = vecs[row].to(dtype)、変調は 1.0 + その値（bf16）。表ごと同じ演算で作る
        s1a = (1.0 + scale_msa.to(bf16)).contiguous()
        sha = shift_msa.to(bf16).contiguous()
        ga = gate_msa.to(bf16).contiguous()
        s1m = (1.0 + scale_mlp.to(bf16)).contiguous()
        shm = shift_mlp.to(bf16).contiguous()
        gm = gate_mlp.to(bf16).contiguous()
        if run_every_op is not None:
            run_every_op()
        with _norm_weight(n1, x) as (w1, _):
            if w1 is None or w1.dtype != bf16 or not w1.is_contiguous():
                _fallback(f"norm1 の重み {None if w1 is None else w1.dtype}")
                return orig(self, x, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, attention=attention)
            h = fk.norm_mod(x, w1, float(n1.eps), ridx, s1a, sha)
        o = attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        del h
        if not o.is_contiguous():
            o = o.contiguous()
        if run_every_op is not None:
            run_every_op()
        with _norm_weight(n2, x) as (w2, _):
            h = fk.gate_norm_mod_(x, o, ga, w2, float(n2.eps), ridx, s1m, shm)
        del o
        o = self.mlp(h)
        del h
        if not o.is_contiguous():
            o = o.contiguous()
        fk.gate_(x, o, gm, ridx)
        _state["fused"] += 1
        if _state["fused"] in (1, 1000, 10000):
            log.warning("[h3-fuse] 融合経路 %d 回目（元の経路 %d 回）", _state["fused"], _state["fallback"])
        return x

    forward._h3_fuse_orig = orig
    mm.DiTBlock.forward = forward
    _state["installed"] = True
    log.warning("[h3-fuse] DiTBlock.forward を融合版に差し替えた")
    return True


class _AfterImport(importlib.abc.MetaPathFinder):
    """名前が合うモジュールの読み込み直後に関数を呼ぶ（worker の起動用）。"""

    def __init__(self, match, fn):
        self.match = match
        self.fn = fn

    def find_spec(self, name, path=None, target=None):
        if not self.match(name):
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module
        fn = self.fn
        me = self

        def exec_module(module):
            orig_exec(module)
            if me in sys.meta_path:
                sys.meta_path.remove(me)
            fn(module)

        spec.loader.exec_module = exec_module
        return spec


def after_import(match, fn):
    sys.meta_path.insert(0, _AfterImport(match, fn))
