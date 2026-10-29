#!/bin/bash
# worker の台で、2 台 Ulysses の rank1 worker（DiT だけを持つ）を起動・停止・確認する（2026-10-28）。
# 使い方: start_sp_worker.sh start|stop|status
# head（start_h3_sp.sh）の TCPStore <head の直結 IP>:29628 に繋がるまで待ち（最長 MINIMAX_SP_STARTUP_TIMEOUT 秒）、
# 起動情報を受け取ってから重みを読む。head の ComfyUI が止まる（shutdown を送る）と自分も終わる。
B=${H3_BASE:?H3_BASE に ComfyUI 一式（venv・ComfyUI・models）を置いたフォルダを指定}
# 2026-10-28：4 台の輪（Ulysses4）にも対応。MINIMAX_SP_RANK（1..3）・MINIMAX_SP_WORLD（4）・MINIMAX_SP_FABRIC=ring4 を渡す
SP_RANK=${MINIMAX_SP_RANK:-1}
SP_WORLD=${MINIMAX_SP_WORLD:-2}
LOG=$B/logs/ulysses/sp_worker${SP_RANK}.log
PIDF=$B/logs/ulysses/sp_worker${SP_RANK}.pid
FABRIC_IF=${FABRIC_IF:-enp1s0f0np0}    # head とつながっている直結 NIC（例）
FABRIC_HCA=${FABRIC_HCA:-rocep1s0f0}   # その RoCE 装置（例）
MASTER=${MINIMAX_SP_MASTER_ADDR:?head の直結 IP を MINIMAX_SP_MASTER_ADDR に指定}
PORT=${MINIMAX_SP_MASTER_PORT:-29628}
if [ "${MINIMAX_SP_FABRIC:-pair}" = ring4 ]; then
  # 4 台の輪：改変版 NCCL（相手のサブネットに合う口を選ぶ）を差し込み、制御は管理用 LAN（nccl_ring4_env.sh）
  source "$(dirname "$0")/nccl_ring4_env.sh" || exit 1
else
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA=$FABRIC_HCA NCCL_IB_GID_INDEX=3 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_SOCKET_IFNAME=$FABRIC_IF GLOO_SOCKET_IFNAME=$FABRIC_IF NCCL_CROSS_NIC=0 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
fi
export MINIMAX_SP_STARTUP_TIMEOUT=${MINIMAX_SP_STARTUP_TIMEOUT:-3600} MINIMAX_SP_COLLECTIVE_TIMEOUT=${MINIMAX_SP_COLLECTIVE_TIMEOUT:-900}
export MINIMAX_SP_PROFILE=${MINIMAX_SP_PROFILE:-1}
mkdir -p "$B/logs/ulysses"
case "${1:-status}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "起動済み pid=$(cat $PIDF)"; exit 0; fi
    if [ -f "$B/logs/comfy.pid" ] && kill -0 "$(cat $B/logs/comfy.pid)" 2>/dev/null; then echo "この台の ComfyUI が動いています。先に start_h3.sh stop"; exit 1; fi
    cd "$B"
    # 2026-10-28：INT8 GEMM を Triton 版に（出力はビット一致）。worker はカスタムノードを読まないので sitecustomize で差し替える
    export H3_LINEAR_FAST=${H3_LINEAR_FAST:-1}
    if [ "$H3_LINEAR_FAST" = 1 ]; then export PYTHONPATH="$B/ComfyUI/custom_nodes/h3-linear-fast/boot${PYTHONPATH:+:$PYTHONPATH}"; fi
    # 2026-10-28：DiT ブロックの norm・変調・ゲート・残差を融合（出力はビット一致）。worker には起動口 h3_fuse_launch.py から効かせる
    export H3_FUSE=${H3_FUSE:-1}
    LAUNCH=""
    if [ "$H3_FUSE" = 1 ] && [ -f "$B/ComfyUI/custom_nodes/h3-fuse-graph/boot/h3_fuse_launch.py" ]; then LAUNCH="$B/ComfyUI/custom_nodes/h3-fuse-graph/boot/h3_fuse_launch.py"; fi
    nohup ./venv/bin/python -s $LAUNCH ComfyUI/custom_nodes/minimax-h3-sp/minimax_sp/sp_worker.py \
      --rank "$SP_RANK" --world "$SP_WORLD" --master-addr "$MASTER" --port "$PORT" --comfy-root "$B/ComfyUI" \
      --startup-timeout "$MINIMAX_SP_STARTUP_TIMEOUT" --collective-timeout "$MINIMAX_SP_COLLECTIVE_TIMEOUT" > "$LOG" 2>&1 &
    echo $! > "$PIDF"
    sleep 3
    if kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "起動 pid=$(cat $PIDF)（head に繋がるまで待機）"; tail -2 "$LOG"; else echo "すぐ終了しました"; tail -20 "$LOG"; exit 1; fi
    ;;
  stop)
    n=0
    for f in "$B"/logs/ulysses/sp_worker*.pid; do   # rank の番号によらず、この台の worker を止める
      [ -f "$f" ] || continue
      kill "$(cat $f)" 2>/dev/null && { echo "停止 $(basename $f .pid)"; n=1; }; rm -f "$f"
    done
    [ $n = 1 ] || echo "動いていません"
    ;;
  status)
    n=0
    for f in "$B"/logs/ulysses/sp_worker*.pid; do
      [ -f "$f" ] && kill -0 "$(cat $f)" 2>/dev/null && { echo "$(basename $f .pid) pid=$(cat $f)"; tail -3 "${f%.pid}.log"; n=1; }
    done
    [ $n = 1 ] || echo "停止中"
    free -g | awk '/Mem:/{print "空きメモリ " $7 " GB"}'
    ;;
  *) echo "start|stop|status"; exit 1;;
esac
