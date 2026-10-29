"""H3 の INT8 ConvRot Linear 1 つを本物の重み（blocks.0）で組み立て、元の経路の内訳を測る（2026-10-28）。
使い方: venv/bin/python bench_linear.py [M ...]   （既定 19200 38400）
出力: 層ごとに 全体の時間、回転＋量子化だけ、CUTLASS GEMM だけ、cuBLAS _int_mm だけ、profiler のカーネル表。
"""
import os, sys, time, torch
from safetensors import safe_open
import comfy_kitchen as ck
from comfy_kitchen.registry import registry
from comfy_kitchen.backends import cuda as ckc

MODEL = os.environ.get("H3_DIT", "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
LAYERS = {"qkv": "attn.qkv_proj", "out": "attn.out_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
Ms = [int(a) for a in sys.argv[1:] if a.isdigit()] or [19200, 38400]
dev = torch.device("cuda", 0)
_C = ckc._C


def bench(fn, iters=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


def load(name):
    with safe_open(MODEL, "pt", device="cpu") as f:
        w = f.get_tensor(f"blocks.0.{LAYERS[name]}.weight").to(dev)
        s = f.get_tensor(f"blocks.0.{LAYERS[name]}.weight_scale").to(dev)
    return w, s


def quant(x, act):
    m, kk = x.shape
    k = kk // (2 if act == "swiglu" else 1)
    xq = torch.empty((m, k), dtype=torch.int8, device=dev); xs = torch.empty((m, 1), dtype=torch.float32, device=dev)
    sp = torch.cuda.current_stream().cuda_stream
    _C.quantize_int8_rowwise_convrot64(ckc._wrap_for_dlpack(x), ckc._wrap_for_dlpack(xq), ckc._wrap_for_dlpack(xs), 256, False,
                                       ckc._input_act_code(act), 0, ckc._wrap_for_dlpack(ckc._act_weight_arg(act, None, x.device, x.dtype)),
                                       0.0, sp)
    return xq, xs


def cutlass(xq, w, xs, ws, n):
    out = torch.empty((xq.shape[0], n), dtype=torch.bfloat16, device=dev)
    ok = _C.cutlass_int8_dequant(ckc._wrap_for_dlpack(xq), ckc._wrap_for_dlpack(w), ckc._wrap_for_dlpack(xs), ckc._wrap_for_dlpack(ws),
                                 ckc._wrap_for_dlpack(ckc._empty_cuda_tensor(dev, torch.bfloat16)), ckc._wrap_for_dlpack(out), 2,
                                 torch.cuda.current_stream().cuda_stream)
    assert ok
    return out


def main():
    print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "ck", ck.__version__ if hasattr(ck, "__version__") else "")
    for name in ["fc1", "qkv", "out", "fc2"]:
        w, s = load(name)
        n, k = w.shape
        act = "swiglu" if name == "fc2" else None
        kin = k * 2 if act else k
        ws = s.reshape(-1).float().contiguous()
        for M in Ms:
            torch.manual_seed(0)
            x = torch.randn(M, kin, device=dev, dtype=torch.bfloat16)
            kw = dict(x=x, weight=w, weight_scale=s, bias=None, out_dtype=torch.bfloat16, convrot=True, convrot_groupsize=256,
                      input_act=act, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None)
            be = registry.get_capable_backend("int8_linear", kw)
            full = lambda: ck.int8_linear(x, w, s, None, torch.bfloat16, convrot=True, convrot_groupsize=256, input_act=act)
            t_full = bench(full)
            t_q = bench(lambda: quant(x, act))
            xq, xs = quant(x, act)
            t_g = bench(lambda: cutlass(xq, w, xs, ws, n))
            wt = w.t()
            t_im = bench(lambda: torch._int_mm(xq, wt))
            fl = 2 * M * k * n
            print(f"{name} M={M} 経路={be}  全体 {t_full*1e3:.2f}ms ({fl/t_full/1e12:.0f}T)  回転+量子化 {t_q*1e3:.2f}ms  "
                  f"CUTLASS {t_g*1e3:.2f}ms ({fl/t_g/1e12:.0f}T)  _int_mm {t_im*1e3:.2f}ms ({fl/t_im/1e12:.0f}T)", flush=True)
            if M == Ms[0]:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    for _ in range(3): full()
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=90))
            del x, xq, xs
            torch.cuda.empty_cache()
        del w, s


if __name__ == "__main__":
    main()
