#!/bin/bash
# head の台で、H3 用 ComfyUI を「2 台 Ulysses（NCCL/RoCE）」用の環境変数付きで起動・停止・確認する（2026-10-28）。
# 使い方: start_h3_sp.sh start|stop|status   （待ち受けは通常版と同じ 127.0.0.1:8188。通常版 start_h3.sh と同時に起動しない）
# worker 側は start_sp_worker.sh start で先に起動しておく（head の TCPStore に繋がるまで待つ）。
B=${H3_BASE:?H3_BASE に ComfyUI 一式（venv・ComfyUI・models）を置いたフォルダを指定}
LOG=$B/logs/ulysses/comfy_sp.log
PIDF=$B/logs/ulysses/comfy_sp.pid
FABRIC_IF=${FABRIC_IF:-enp1s0f1np1}    # 相手の台とつながっている直結 NIC（例）
FABRIC_HCA=${FABRIC_HCA:-rocep1s0f1}   # その RoCE 装置（例）
if [ "${MINIMAX_SP_FABRIC:-pair}" = ring4 ]; then
  # 2026-10-28：4 台の輪（Ulysses4）。改変版 NCCL を差し込み、制御は管理用 LAN（nccl_ring4_env.sh）
  source "$(dirname "$0")/nccl_ring4_env.sh" || exit 1
  # 送受信（P2P）に使った tensor の解放が cudaMallocAsync＋DynamicVRAM の head で CUDA error: invalid argument に
  # なったため、worker と同じ PyTorch 既定の allocator にする（計算は変わらない）
  COMFY_EXTRA="--disable-cuda-malloc"
else
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA=$FABRIC_HCA NCCL_IB_GID_INDEX=3 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_SOCKET_IFNAME=$FABRIC_IF GLOO_SOCKET_IFNAME=$FABRIC_IF NCCL_CROSS_NIC=0 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
fi
export MINIMAX_SP_MASTER_ADDR=${MINIMAX_SP_MASTER_ADDR:?head の直結 IP を MINIMAX_SP_MASTER_ADDR に指定} MINIMAX_SP_MASTER_PORT=${MINIMAX_SP_MASTER_PORT:-29628}
export MINIMAX_SP_STARTUP_TIMEOUT=${MINIMAX_SP_STARTUP_TIMEOUT:-3600} MINIMAX_SP_COLLECTIVE_TIMEOUT=${MINIMAX_SP_COLLECTIVE_TIMEOUT:-900}
export MINIMAX_SP_PROFILE=${MINIMAX_SP_PROFILE:-1}
mkdir -p "$B/logs/ulysses"
case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "起動済み pid=$(cat $PIDF)"; exit 0; fi
    if [ -f "$B/logs/comfy.pid" ] && kill -0 "$(cat $B/logs/comfy.pid)" 2>/dev/null; then echo "通常版の ComfyUI が動いています。先に start_h3.sh stop"; exit 1; fi
    cd "$B"
    export H3_FFMPEG=$(cat "$B/logs/ffmpeg.path")
    export H3_LINEAR_FAST=${H3_LINEAR_FAST:-1}   # 2026-10-28：INT8 GEMM を Triton 版に（出力はビット一致、h3-linear-fast ノード）
    export H3_FUSE=${H3_FUSE:-1}   # 2026-10-28：DiT ブロックの norm・変調・ゲート・残差を融合（出力はビット一致、h3-fuse-graph ノード）
    nohup ./venv/bin/python -s ComfyUI/main.py --base-directory "$B" --listen 127.0.0.1 --port 8188 \
      --use-ck-attention --disable-cuda-graphs --preview-method none $COMFY_EXTRA > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    for i in $(seq 1 100); do grep -q "To see the GUI" "$LOG" && break; sleep 3; done
    grep -q "To see the GUI" "$LOG" && echo "起動 pid=$(cat $PIDF)" || { echo "起動を確認できません"; tail -5 "$LOG"; exit 1; }
    grep -n "minimax_sp\|MiniMaxH3SPUNETLoader\|IMPORT FAILED" "$LOG" | head -5
    ;;
  stop)
    if [ -f "$PIDF" ]; then kill "$(cat $PIDF)" 2>/dev/null && echo "停止"; rm -f "$PIDF"; else echo "動いていません"; fi
    ;;
  status)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then
      echo "pid=$(cat $PIDF) $(curl -s -m 5 http://127.0.0.1:8188/queue | head -c 200)"
    else echo "停止中"; fi
    free -g | awk '/Mem:/{print "空きメモリ " $7 " GB"}'
    ;;
  *) echo "start|stop|status"; exit 1;;
esac
