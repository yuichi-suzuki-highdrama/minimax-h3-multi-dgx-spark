"""2 台の間の all_to_all_single の速さを、H3 の Ulysses2 と同じ大きさで測る（2026-10-28）。
1 回あたり [19200 トークン, 56 頭, 128] の bf16（275 MB、相手へは半分の 137 MB）を送る。qkv の 3 回分を 1 回にまとめた大きさも測る。
使い方（両台で同時に）：python a2a_bench.py --rank 0|1 --master <head の直結 IP> --port 29700
環境変数 NCCL_IB_HCA・NCCL_SOCKET_IFNAME などは呼び出し側で設定する。
"""
import argparse, os, time, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--master", required=True)
ap.add_argument("--port", type=int, default=29700)
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
dist.init_process_group("nccl", init_method=f"tcp://{a.master}:{a.port}", rank=a.rank, world_size=2)
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)
S, H, D = 19200, 56, 128


def bench(n_elem, iters):
    send = torch.randn(n_elem, device=dev, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    for _ in range(3):
        dist.all_to_all_single(recv, send)
    torch.cuda.synchronize(); dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_to_all_single(recv, send)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    cross = n_elem * 2 / 2  # 半分を相手へ送り、半分を受け取る
    return dt, cross / dt / 1e9


rows = [("q だけ（275 MB）", S * H * D), ("qkv をまとめて（826 MB）", 3 * S * H * D), ("小さめ（34 MB）", S * H * D // 8)]
for name, n in rows:
    dt, gbs = bench(n, a.iters)
    if a.rank == 0:
        print(f"{name:24s} 1 回 {dt * 1e3:7.2f} ms  片方向 {gbs:5.1f} GB/s", flush=True)
if a.rank == 0:
    hca = os.environ.get("NCCL_IB_HCA")
    print(f"NCCL_IB_HCA={hca}")
dist.destroy_process_group()
