#!/bin/bash
# 4 台の輪（A→B→D→C→A、スイッチ無し。向かいどうし A↔D・B↔C はつながっていない）で NCCL を使うための環境変数（Ulysses4 用）。source で読み込む。
# スイッチレスの輪に対応した改変版 NCCL 2.30.7（相手のサブネットに合う自分の口を選ぶ）を LD_PRELOAD で差し込む。
# venv の nvidia/nccl/lib/libnccl.so.2 は書き換えない（外せば元どおり）。
# 口の名前・サブネット・管理用 LAN の口は環境に合わせて書き換えること（下の値は例）。
RING_NCCL_LIB=${RING_NCCL_LIB:?改変版 libnccl.so.2.30.7 の場所を RING_NCCL_LIB に指定}
if [ ! -f "$RING_NCCL_LIB" ]; then echo "改変版 NCCL が見つかりません: $RING_NCCL_LIB" >&2; return 1 2>/dev/null || exit 1; fi
export LD_PRELOAD="$RING_NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
# 口：4 つ全部を並べ、つなぐ時に相手のサブネット（/24）に合う口を NCCL が選ぶ
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA="${RING_IB_HCA:-=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1}"
export NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24 NCCL_IB_MERGE_NICS=0
export NCCL_IB_EXTENDED_IPV4_GIDS=1 NCCL_IB_PRESERVE_PCI_DOMAIN=1 NCCL_IB_ADDR_RANGE=${RING_IB_ADDR_RANGE:?直結の口のアドレス範囲（例 10.0.0.0/22）を RING_IB_ADDR_RANGE に指定}
export NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ROCE_VERSION_NUM=2 NCCL_CROSS_NIC=1
# 輪の隣だけ：Ring に固定し、Tree などの接続を作らない（向かいの台へ繋ぎに行かない）
export NCCL_ALGO=Ring NCCL_SWITCHLESS_RING_ONLY=1 NCCL_SKIP_TREE_CONNECT=1
export NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1
# 起動時のやりとり（NCCL の bootstrap・gloo・TCPStore）は 4 台全部が届く管理用 LAN で
export NCCL_SOCKET_IFNAME=${RING_MGMT_IF:?管理用 LAN の口の名前を RING_MGMT_IF に指定} GLOO_SOCKET_IFNAME=$RING_MGMT_IF
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
