"""4 台の輪（スイッチ無し）の上の中継つき all-to-all（2026-10-28、Ulysses4 用）。

rank は輪の順に並べる（rank r の隣は r-1 と r+1、向かいは r+2 で、向かいとは直接つながっていない）。
各 rank は隣 2 台へは直接送り、向かいの台宛ての分は半分ずつ左右の隣に送って中継してもらう。
NCCL の送受信（batch_isend_irecv）は隣どうしだけで行うので、向かいへの接続は作らない。

段 1：右へ [右宛て, 向かい宛ての前半]、左へ [左宛て, 向かい宛ての後半] を送り、
      左から [左→自分, 左→右（＝左の向かい）の前半]、右から [右→自分, 右→左の後半] を受ける。
段 2：預かった「左→右の前半」を右へ、「右→左の後半」を左へ送り、
      向かい→自分の前半を左から、後半を右から受ける。
段 1 と段 2 は同じ NCCL の通信路（同じ stream）に積むので、段 2 は段 1 が終わってから走る（CPU は待たない）。

並べ替えだけなので、結果は普通の all_to_all_single とビット単位で同じになる。
"""

import torch
import torch.distributed as dist


def ring_peers(rank, world):
    if world != 4:
        raise ValueError(f"ring relay all-to-all supports world=4 only, got {world}")
    return (rank - 1) % world, (rank + 1) % world, (rank + 2) % world


def _prefix(sizes):
    out, s = [], 0
    for n in sizes:
        out.append(s)
        s += n
    return out


def relay_all_to_all(out, inp, sizes, rank, group, async_op=False):
    """中継つき all-to-all。

    sizes[src][dst]：src から dst へ送る要素数（全 rank で同じ表を渡す）。
    inp：1 次元、宛先 rank の順に塊を並べたもの（塊 j の長さは sizes[rank][j]）。
    out：1 次元、送り主 rank の順に塊を並べる（塊 j の長さは sizes[j][rank]）。
    async_op=True のときは待つべき work の一覧を返す（呼び側で全部 wait する）。
    """
    world = len(sizes)
    left, right, opp = ring_peers(rank, world)
    gl = dist.get_global_rank(group, left) if group is not None else left
    gr = dist.get_global_rank(group, right) if group is not None else right
    in_off = _prefix([sizes[rank][j] for j in range(world)])
    out_off = _prefix([sizes[j][rank] for j in range(world)])

    def iblk(j):
        return inp[in_off[j]:in_off[j] + sizes[rank][j]]

    def oblk(j):
        return out[out_off[j]:out_off[j] + sizes[j][rank]]

    oblk(rank).copy_(iblk(rank))

    h_mine = sizes[rank][opp] // 2           # 自分→向かい：前半は右経由、後半は左経由
    h_lr = sizes[left][right] // 2           # 左→右の前半が自分に来る（右へ中継）
    n_rl = sizes[right][left]
    h_rl = n_rl // 2                         # 右→左の後半が自分に来る（左へ中継）
    h_or = sizes[opp][rank] // 2             # 向かい→自分：前半は左から、後半は右から
    relay_a = torch.empty(h_lr, dtype=inp.dtype, device=inp.device)
    relay_b = torch.empty(n_rl - h_rl, dtype=inp.dtype, device=inp.device)
    mine_opp = iblk(opp)
    from_opp = oblk(opp)

    def ops(pairs):
        return [dist.P2POp(fn, t, peer, group) for fn, t, peer in pairs if t.numel() > 0]

    # 同じ相手への送りと受けは、両側で同じ順に並べる（直接分 → 中継分）
    stage1 = ops([
        (dist.isend, iblk(right), gr), (dist.isend, mine_opp[:h_mine], gr),
        (dist.isend, iblk(left), gl), (dist.isend, mine_opp[h_mine:], gl),
        (dist.irecv, oblk(left), gl), (dist.irecv, relay_a, gl),
        (dist.irecv, oblk(right), gr), (dist.irecv, relay_b, gr),
    ])
    stage2 = ops([
        (dist.isend, relay_a, gr), (dist.isend, relay_b, gl),
        (dist.irecv, from_opp[:h_or], gl), (dist.irecv, from_opp[h_or:], gr),
    ])
    works = []
    if stage1:
        works += dist.batch_isend_irecv(stage1)
    if stage2:
        works += dist.batch_isend_irecv(stage2)
    keep = (relay_a, relay_b, inp, out)
    if async_op:
        return RelayWork(works, keep)
    for w in works:
        w.wait()
    return None


class RelayWork:
    """relay_all_to_all(async_op=True) の戻り。wait() で全部の送受信を待つ。"""

    def __init__(self, works, keep):
        self.works = works
        self.keep = keep

    def wait(self):
        for w in self.works:
            w.wait()
        self.works = []
        self.keep = None
        return True


def a2a_sizes_uniform_send(splits, chunk):
    """各 rank が自分の行（splits[src] 行）を全員へ同じ幅 chunk で送る形（seq を集めて頭を配る側）。"""
    w = len(splits)
    return [[splits[src] * chunk for _ in range(w)] for src in range(w)]


def a2a_sizes_uniform_recv(splits, chunk):
    """各 rank が宛先の行（splits[dst] 行）を送る形（seq を配って頭を集める側）。"""
    w = len(splits)
    return [[splits[dst] * chunk for dst in range(w)] for _ in range(w)]


def ring_gather_rows(buf, rank, group):
    """隣どうしの送受信だけで all_gather（全 rank が同じ長さの buf を出し、rank 順に並べた 1 本を返す）。

    all_gather_into_tensor が輪の上で通らない場合の予備。左右から 1 つずつ受け取り、
    2 段目で向かいの分を（前半を左から、後半を右から）中継する。
    """
    world = dist.get_world_size(group)
    n = buf.numel()
    flat = buf.contiguous().view(-1)
    sizes = [[n for _ in range(world)] for _ in range(world)]
    inp = flat.repeat(world)
    out = torch.empty(world * n, dtype=buf.dtype, device=buf.device)
    relay_all_to_all(out, inp, sizes, rank, group)
    return out
