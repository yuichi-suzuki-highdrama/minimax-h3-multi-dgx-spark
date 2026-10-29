"""Triton の INT8 GEMM を本物の重みで、CUTLASS 経路とのビット一致と速さで比べる（2026-10-28）。"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comfyui_nodes", "h3-linear-fast"))
import bench_linear as bl
import int8_gemm_triton as tg

dev = torch.device("cuda", 0)
Ms = [int(a) for a in sys.argv[1:] if a.isdigit()] or [19200, 38400]
for name in ["fc1", "qkv", "out", "fc2"]:
    w, s = bl.load(name)
    n, k = w.shape
    act = "swiglu" if name == "fc2" else None
    ws = s.reshape(-1).float().contiguous()
    for M in Ms:
        torch.manual_seed(1)
        x = torch.randn(M, k * (2 if act else 1), device=dev, dtype=torch.bfloat16)
        xq, xs = bl.quant(x, act)
        ref = bl.cutlass(xq, w, xs, ws, n)
        o = tg.int8_mm_scaled(xq, w, xs, ws)
        eq = torch.equal(o, ref)
        t_c = bl.bench(lambda: bl.cutlass(xq, w, xs, ws, n))
        t_t = bl.bench(lambda: tg.int8_mm_scaled(xq, w, xs, ws))
        fl = 2 * M * k * n
        best = tg._int8_mm_scaled_kernel.best_config
        print(f"{name} M={M} ビット一致={eq} 不一致={int((o != ref).sum())}  CUTLASS {t_c*1e3:.2f}ms ({fl/t_c/1e12:.0f}T)  "
              f"Triton {t_t*1e3:.2f}ms ({fl/t_t/1e12:.0f}T)  設定 {best}", flush=True)
        del x, xq, xs, ref, o
        torch.cuda.empty_cache()
    del w, s
