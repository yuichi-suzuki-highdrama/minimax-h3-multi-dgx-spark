"""GB10 の行列計算の素の速さを、H3 の DiT と同じ形で測る（2026-10-28）。
形は 1 台あたり 19,200 トークン × H3 の 4 つの線形層（qkv 5376→21504、out 7168→5376、fc1 5376→28672、fc2 14336→5376）。
BF16（torch.matmul）、FP8 e4m3（torch._scaled_mm、行・列ごとの scale）、INT8（torch._int_mm）を比べる。
"""
import time, torch

M = 19200
SHAPES = {"qkv": (5376, 21504), "out": (7168, 5376), "fc1": (5376, 28672), "fc2": (14336, 5376)}
dev = "cuda"
print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())


def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters


tot = {}
for name, (k, n) in SHAPES.items():
    flops = 2 * M * k * n
    a = torch.randn(M, k, device=dev, dtype=torch.bfloat16); b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    res = {"bf16": bench(lambda: a @ b)}
    try:
        a8 = a.to(torch.float8_e4m3fn); b8 = b.t().contiguous().to(torch.float8_e4m3fn).t()
        sa = torch.ones(M, 1, device=dev, dtype=torch.float32); sb = torch.ones(1, n, device=dev, dtype=torch.float32)
        res["fp8"] = bench(lambda: torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
    except Exception as e:
        print(name, "fp8 失敗", str(e)[:120])
    try:
        ai = torch.randint(-127, 127, (M, k), device=dev, dtype=torch.int8)
        bi = torch.randint(-127, 127, (n, k), device=dev, dtype=torch.int8).t()
        res["int8"] = bench(lambda: torch._int_mm(ai, bi))
    except Exception as e:
        print(name, "int8 失敗", str(e)[:120])
    line = "  ".join(f"{kk} {v * 1e3:.2f}ms {flops / v / 1e12:.0f}T" for kk, v in res.items())
    print(f"{name:4s} {line}")
    for kk, v in res.items(): tot[kk] = tot.get(kk, 0) + v
print("4 層の合計（1 ブロック）:", "  ".join(f"{kk} {v * 1e3:.1f}ms ×50 段={v * 50:.2f}s" for kk, v in tot.items()))
