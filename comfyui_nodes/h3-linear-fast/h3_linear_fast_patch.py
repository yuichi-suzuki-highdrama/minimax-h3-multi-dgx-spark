"""H3 の INT8 ConvRot Linear を速くする差し替え（2026-10-28）。

元の経路（comfy-kitchen 0.2.39 の CUDA backend `int8_linear`）:
  1. `_C.quantize_int8_rowwise_convrot64`（活性の 256 群 Hadamard 回転＋行ごと INT8 量子化。fc2 は SwiGLU も同じカーネル）
  2. `_C.cutlass_int8_dequant`（CUTLASS の INT8 GEMM＋行×列 scale を掛けて bf16 で書く）
GB10（sm121）では 2 の CUTLASS カーネルが 75〜100 TOPS しか出ず、時間の 8〜9 割を占める。

差し替え: 1 はそのまま使い、2 だけを Triton の INT8 GEMM（行×列 scale、bf16 直書き、180〜190 TOPS）に替える。
逆量子化の式 bf16((float(acc) * xs) * ws) は CUTLASS 経路と同じで、出力はビット一致（実測）。
条件に合わない呼び出し（M が小さい、bias・residual あり、bf16 以外、回転なし等）は元の関数に回す。
有効にするのは環境変数 H3_LINEAR_FAST=1 のときだけ。
"""
import logging
import os
import sys
import sysconfig

import torch

log = logging.getLogger("h3-linear-fast")
_HERE = os.path.dirname(os.path.abspath(__file__))
_state = {"installed": False, "orig": None, "fast_calls": 0, "fallback_calls": 0, "disabled": False}
MIN_M = int(os.environ.get("H3_LINEAR_FAST_MIN_M", "256"))


def _ensure_python_headers():
    """Triton は起動時に小さな C の部品を gcc で作るので Python.h が要る。01〜08 には python3.12-dev が無いので、
    同梱の pyinclude（minimax-h3-kitchen イメージの /usr/include から写した 3.12 の見出し）を CPATH に足す。"""
    inc = sysconfig.get_paths().get("include") or ""
    if os.path.exists(os.path.join(inc, "Python.h")):
        return
    pyinc = os.path.join(_HERE, "pyinclude")
    if not os.path.exists(os.path.join(pyinc, "python3.12", "Python.h")):
        log.warning("[h3-linear-fast] Python.h が見つかりません（Triton の部品を作れない可能性）")
        return
    add = os.pathsep.join([os.path.join(pyinc, "python3.12"), pyinc])
    cur = os.environ.get("CPATH")
    os.environ["CPATH"] = add + (os.pathsep + cur if cur else "")


def install():
    if _state["installed"]:
        return True
    if os.environ.get("H3_LINEAR_FAST", "0") != "1":
        return False
    _ensure_python_headers()
    try:
        from comfy_kitchen.backends import cuda as ckc
    except Exception as e:  # noqa: BLE001
        log.warning("[h3-linear-fast] comfy_kitchen の CUDA backend を読めないので無効: %s", e)
        return False
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import int8_gemm_triton as tg

    orig = ckc.int8_linear
    _C = ckc._C
    W = ckc._wrap_for_dlpack

    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False, convrot_groupsize=256,
                    input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None):
        out_dtype_eff = out_dtype or x.dtype
        act_w = 2 if input_act == "swiglu" else 1
        ok = (not _state["disabled"] and convrot and convrot_groupsize == 256 and bias is None and residual is None
              and input_act in (None, "none", "swiglu") and x.is_cuda and x.dtype == torch.bfloat16
              and out_dtype_eff == torch.bfloat16 and weight.dtype == torch.int8 and weight.dim() == 2)
        if ok:
            kin = x.shape[-1]
            k = kin // act_w
            m = x.numel() // kin if kin else 0
            n = weight.shape[0]
            ok = (m >= MIN_M and kin % act_w == 0 and k % 256 == 0 and 256 <= k <= ckc._CONVROT_FUSED_MAX_K
                  and weight.shape[1] == k and tg.supported(k) and n % 8 == 0)
        if not ok:
            _state["fallback_calls"] += 1
            if _state["fallback_calls"] <= 5:
                log.warning("[h3-linear-fast] 元の経路へ: x=%s w=%s convrot=%s bias=%s act=%s", tuple(x.shape),
                            tuple(weight.shape), convrot, bias is not None, input_act)
            return orig(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act,
                        input_act_weight, input_act_eps, residual, residual_scale)
        x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, kin).contiguous()
        if not ckc._convrot_fused_shared_memory_fits(x_2d, k, convrot_groupsize):
            _state["fallback_calls"] += 1
            return orig(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act,
                        input_act_weight, input_act_eps, residual, residual_scale)
        if not weight.is_contiguous():
            weight = weight.contiguous()
        stream_ptr = torch.cuda.current_stream(x.device).cuda_stream
        # 1. 回転＋量子化（元と同じカーネル・同じ引数）
        x_q = torch.empty((m, k), dtype=torch.int8, device=x.device)
        x_s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        act_weight_arg = ckc._act_weight_arg(input_act, input_act_weight, x.device, x_2d.dtype)
        _C.quantize_int8_rowwise_convrot64(W(x_2d), W(x_q), W(x_s), convrot_groupsize, False,
                                           ckc._input_act_code(input_act), 0,
                                           W(act_weight_arg), float(input_act_eps), stream_ptr)
        # 2. INT8 GEMM＋逆量子化（Triton）
        ws = ckc._int8_weight_scale_arg(weight_scale, x.device)
        if ws.numel() != n:
            ws = ws.expand(n).contiguous()
        try:
            out = tg.int8_mm_scaled(x_q, weight, x_s, ws)
        except Exception as e:  # noqa: BLE001
            _state["disabled"] = True
            log.warning("[h3-linear-fast] Triton の GEMM が失敗したので以後は元の経路: %s", e)
            return orig(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize, input_act,
                        input_act_weight, input_act_eps, residual, residual_scale)
        _state["fast_calls"] += 1
        if _state["fast_calls"] in (1, 1000, 10000):
            log.warning("[h3-linear-fast] 差し替え経路 %d 回目（元の経路 %d 回）", _state["fast_calls"], _state["fallback_calls"])
        return out if x.dim() == 2 else out.reshape(*x.shape[:-1], n)

    ckc.int8_linear = int8_linear
    _state["orig"] = orig
    _state["installed"] = True
    log.warning("[h3-linear-fast] 有効: comfy_kitchen CUDA int8_linear の ConvRot 経路の GEMM を Triton に差し替え（MIN_M=%d）", MIN_M)
    return True


def uninstall():
    if not _state["installed"]:
        return
    from comfy_kitchen.backends import cuda as ckc
    ckc.int8_linear = _state["orig"]
    _state["installed"] = False


def stats():
    return {k: _state[k] for k in ("installed", "fast_calls", "fallback_calls", "disabled")}
