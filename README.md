# Speeding up MiniMax H3 on DGX Spark-class machines (2-node and 4-node)

English | [日本語](README.ja.md)

Code and notes for generating MiniMax H3 videos (ComfyUI) faster by linking two or four DGX Spark-class machines (GB10) and splitting each video across them.

![Speedup steps and results](docs/h3_speedup_x2_x4.png)

*(The figure is in Japanese. The tables below carry the same numbers.)*

## Results

| Video | 1 node | 2 nodes | 4 nodes | RTX 5090 |
|---|---:|---:|---:|---:|
| I2V 5 s (124 frames) | 431.6* | 237.7 | 137.9 | — |
| FL2V 8 s (192 frames, first and last frame)† | — | 496.4* | 279.7 | 257.9 |
| R2V 10 s (243 frames, 6 reference images) | — | 654.7 | **363.2** | 391.1 |
| R2V 10 s, same prompt, different seed | — | 641.7 | **345.7** | 383.9 |

- Seconds, from ComfyUI execution start to finish, after the weights are loaded. * = first run right after startup.
- The two R2V rows use the production settings, which include the TF32 text encoder and background video writing. † FL2V was measured before those two were added.
- Bit-exact output (same hash for the latent and the mp4) was verified for I2V on 1, 2 and 4 nodes, and for R2V and FL2V on 2 and 4 nodes (they were not run on a single node). The RTX 5090 is a different machine, so its output is not bit-identical, though the motion and composition are the same.
- On R2V 10 s, four nodes are 7–10% faster than an RTX 5090. On FL2V 8 s attention is a smaller share of the work, and the RTX 5090 is still faster. RTX 5090 times also vary by about 5% between runs.

## Setup

- Hardware: DGX Spark-class machines (GB10, 128 GB unified memory, ConnectX-7 with two ports). Machines are cabled directly with QSFP; no switch.
- Software: ComfyUI v0.42.3 (with H3 support), torch 2.11 (CUDA 13.0), Triton 3.6, comfy-kitchen 0.2.39.
- Weights: the H3 INT8 ConvRot release (DiT about 21 GB, Qwen3-VL text encoder about 27 GB, video and audio VAEs, the DMAD 4-step LoRA).
- Workflow: N8D3 (8 steps, then 3 DMAD steps), 1376×768, 24 fps.
- H3 uses 60–70 GB on one node, so it does not fit next to a large LLM on the same machine.

## Contents

| Path | Purpose |
|---|---|
| `comfyui_nodes/minimax-h3-sp/` | Ulysses (sequence-parallel) group loader and worker, for 2 and 4 nodes. The 4-node ring uses a relay all-to-all (`ring_a2a.py`). The video VAE is also split across nodes |
| `comfyui_nodes/h3-linear-fast/` | Replaces the INT8 linear-layer GEMM with a Triton kernel (`int8_gemm_triton.py`). Workers load it through `boot/sitecustomize.py` |
| `comfyui_nodes/h3-fuse-graph/` | Fuses the DiT block's norm, adaLN modulation, gate and residual into three Triton kernels. Workers start through `boot/h3_fuse_launch.py` |
| `comfyui_nodes/h3-qwen-tf32/` | Allows TF32 only while the text encoder (Qwen3-VL) runs. **This is the one change that is not bit-exact** |
| `scripts/` | Start scripts for the head and workers, and a template NCCL config for the 4-node ring (`nccl_ring4_env.sh`) |
| `bench/` | All-to-all bandwidth (`a2a_bench.py`) and raw GB10 matmul throughput (`gemm_peak.py`) |
| `tests/` | Bit-exactness and speed of the Triton GEMM against the original CUTLASS path, and bit-exactness of the fused RMSNorm |

## Configurations

| Configuration | Wiring | Good for |
|---|---|---|
| 1 node | none | Many seeds in parallel (one per node) |
| 2 nodes | two nodes cabled directly | One video faster; run two or four pairs for volume |
| 4 nodes | a ring (A→B→D→C→A); opposite nodes (A↔D, B↔C) are not connected | One video as fast as possible |

- The split is Ulysses: attention heads are divided across nodes, with one all-to-all before and one after attention.
- ComfyUI runs on one node (the head). The other nodes (workers) load the same DiT and compute their share of each forward when the head asks. They do not hold the text encoder or VAE weights, except when sharing the video VAE.
- The only workflow JSON change is replacing `UNETLoader` with the group loader (which takes world_size and the head address). The two samplers and the LoRA work unchanged.

## ×2: speeding up two nodes (I2V 5 s, warm)

| Step | Time per video | What changed |
|---|---:|---|
| 1 node | 576.6 s | — |
| Split across 2 nodes | 342.9 s | Ulysses. One forward is 35% attention, 32% MLP and similar, 15% qkv, 13% network |
| Use both ports | — | Each cable shows two RDMA devices, so pass both to NCCL (list both in `NCCL_IB_HCA`). All-to-all goes from 12.9 to 20.6 GB/s |
| Overlap network and attention | — | Split heads into four groups and compute while sending. Without GPUDirect RDMA this hid only about 0.2 s |
| Split the video VAE across 2 nodes | 319.3 s | Split the temporal clips across nodes and join them on the head. About 30 s → 13 s |
| Rewrite the INT8 linear layers in Triton | 240.9 s | The stock CUTLASS INT8 GEMM reached only 75–103 TOPS on sm121. The Triton version reaches 175–193 TOPS. The dequantization formula and rounding points match the original |
| Fuse small ops | 237.7 s* | Norm, adaLN modulation, gate and residual in three Triton kernels. Summation order and bf16 rounding points match PyTorch |

\* Fusion alone was measured on a different pair (246.4 → 237.7 s, −3.5%).

- The linear layers helped most. On this machine there is almost no CPU-side launch overhead, so making the math itself faster is what pays off. CUDA graphs did not help (only about 0.05 s per forward to remove).
- One node with the same improvements takes 431.6 s (first run after startup). Two nodes are 1.8× faster than that.

## ×4: building the four-node ring (R2V 10 s, 6 reference images)

- **Relay all-to-all.** In a ring, opposite nodes cannot talk directly. The all-to-all is built only from sends and receives between neighbors; data for the opposite node is received by a neighbor and passed on. Gathering K and V on every node (all_gather) was also tried, but this was faster. One q tensor (148 MB) takes 7.5 ms.
- **Modified NCCL.** Each node's ports sit on different subnets, so a modified NCCL 2.30.7 (with switchless-ring support) that picks the port matching the peer's subnet is loaded with `LD_PRELOAD`. It is not the author's work, so neither the binary nor the source is included here. See `scripts/nccl_ring4_env.sh` for the settings. Startup traffic (TCPStore, gloo, NCCL bootstrap) goes over a management LAN that reaches all four nodes.
- **Bit-exact across 1, 2 and 4 nodes.** cuBLAS picks a different fp32 GEMM algorithm depending on the row count, so the final output layer is multiplied with the same row count as on one node. Start the head with `--disable-cuda-malloc` (freeing tensors used for send/receive failed under cudaMallocAsync).
- **Video VAE split across 4 nodes.** The 14 clips are split 4/4/3/3. This is the floor; going further would require splitting inside a clip.
- **Preprocessing: Qwen3-VL in TF32.** The H3 text encoder computes in fp32 and took 25.8 s to read the prompt and six reference images. Allowing TF32 during encoding cuts that to 10.0 s. Fine details change slightly, so it was adopted after a visual comparison.
- **Postprocessing: background writing.** The save node returns immediately and the mp4 is finished in the background (overlapped conversion plus x264 with 16 threads). 17.5 s → 12.8 s. This is a change to a local save node and is not included here. Note that the mp4 is still being written when ComfyUI reports the job as done.

## Usage (outline)

1. Put the four folders under `comfyui_nodes/` into each node's ComfyUI `custom_nodes/`.
2. Set environment variables on each node:
   - `H3_BASE`: the folder holding the venv, ComfyUI and models.
   - `MINIMAX_SP_MASTER_ADDR`: the head's address.
   - `FABRIC_IF` and `FABRIC_HCA`: the names of the direct-link port.
   - For four nodes, also `RING_NCCL_LIB`, `RING_IB_ADDR_RANGE` and `RING_MGMT_IF`.
3. Start the workers first:
   - 2 nodes: `scripts/start_sp_worker.sh start`
   - 4 nodes: add `MINIMAX_SP_FABRIC=ring4 MINIMAX_SP_WORLD=4 MINIMAX_SP_RANK=1..3`.
4. Start the head with `scripts/start_h3_sp.sh start` (add `MINIMAX_SP_FABRIC=ring4` for four nodes).
5. Replace `UNETLoader` in your workflow with the group loader and run it.

The scripts were written for the author's setup and are not guaranteed to work as-is. Adjust port names and folder layout to your environment. Comments in the code are in Japanese.

## What was not adopted

| Approach | Result |
|---|---|
| Sol-Attn | About 5× faster, but visibly worse quality |
| SageAttention2 | −3 to −5% (−1 to −2% on pairs). Changes the output (PSNR about 30 dB) |
| SageAttention3 | Slower, with larger error |
| NVFP4 weights (FP4 for the MLP only) | Slower (240.9 → 254.3 s), and the motion changes |
| FP8 | Slower than BF16 in cuBLAS on this machine (54–65 TOPS). INT8 is the right choice |
| CUDA graphs | H3's DiT is outside ComfyUI's CUDA graph support. No effect |

comfy-kitchen's INT8 attention already reaches about 155 TOPS here, close to the INT8 GEMM ceiling (170–190 TOPS).

## Pitfalls

- If ComfyUI runs with `--base-directory`, put weights and input images in that folder's `models/` and `input/`.
- To switch weights in a group (for example I2V to R2V), stop and restart the whole group. Workers keep the weights they started with.
- If a worker dies, the head does not notice until the NCCL timeout (up to 15 minutes). After a failure, restart the whole group.
- Right after copying many files, the page cache can fill unified memory and CUDA allocations can fail.
- Resolution must be a multiple of 32; frame counts are 5 + 17k (124, 141, …, 192, 243).
- Not supported in a group: Fun ControlNet, Sparse Attention, dynamic hooks or patches.

## License and credits

- `comfyui_nodes/minimax-h3-sp/` is based on `minimax_sp` from [buqi-code/buqi-minimax-h3-multigpu](https://github.com/buqi-code/buqi-minimax-h3-multigpu) (MIT), reworked for 2 nodes and a 4-node ring across separate hosts. The original license is in `comfyui_nodes/minimax-h3-sp/LICENSE-buqi-code-MIT.txt`.
- Everything else is MIT (`LICENSE`).
- The MiniMax H3 weights and model are subject to MiniMax's official license.
