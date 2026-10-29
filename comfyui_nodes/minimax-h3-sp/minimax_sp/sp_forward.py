"""Ulysses sequence-parallel forward for MiniMax-H3（buqi 版を ComfyUI v0.42.3＋H3 差分に合わせた）。

The packed sequence is split by rows across ranks. Every per-token op (patch
proj, AdaLN, RoPE, MLP, out_proj) is row-local and needs no communication; only
attention does, via two all-to-all exchanges that trade the head dim for the
sequence dim. Attention math is therefore unchanged: each rank runs complete
attention over the full sequence for 56/P of the heads.

Row splits are deliberately uneven (base + remainder) instead of padding the
sequence, so no attention mask is needed and results stay numerically equivalent
to the single-GPU path.

sp_forward mirrors MiniMaxH3Model._forward of ComfyUI v0.42.3 (+ the 5090 H3 overlay:
rope_freqs(layout, device), h3_timing in PackedLayout) and must be re-checked when it changes.

2 台版での変更点:
  - dit.rope_freqs の引数が layout か position_ids かを署名で見分ける
  - ModelAttentionBackend ノードが置く transformer_options["optimized_attention_override"] を許す
    （worker には関数名だけ送り、同じ backend を worker 側で組み立てる）
  - torch 2.11 に無い dist.all_gather_single を all_gather_into_tensor に置き換えた
"""

import inspect
import logging
import os
import time

import torch
import torch.distributed as dist

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.quant_ops
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    mask_row_values,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from . import ring_a2a
from comfy.ldm.modules.attention import (
    AttentionTensorContainer,
    REGISTERED_ATTENTION_FUNCTIONS,
    get_attention_function,
    optimized_attention,
)

PROFILE_OPS = bool(os.environ.get("MINIMAX_SP_PROFILE_OPS"))
# 2026-10-28：頭を N 組に分け、回線（all-to-all）と attention を重ねる。0/1 で従来どおり。
# 両方の rank で同じ値にすること（違うと集合通信が噛み合わず止まる）。
OVERLAP_CHUNKS = int(os.environ.get("MINIMAX_SP_OVERLAP_CHUNKS", "0") or 0)
_prof_acc = {}
# 2026-10-28：world=4 で FinalLayer の出力 Linear を単独と同じ行数で掛ける（latent を単独とビット一致させる）。0 で外す。
FINAL_FIXED_M = os.environ.get("MINIMAX_SP_FINAL_FIXED_M", "1") == "1"

ALLOWED_ATTENTION_KEYS = ("optimized_attention_override",)


class region:
    """Accumulate wall time per named region; only active under MINIMAX_SP_PROFILE_OPS."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            _prof_acc[self.name] = _prof_acc.get(self.name, 0.0) + time.perf_counter() - self.t0
        return False


def prof_report(total):
    rows = sorted(_prof_acc.items(), key=lambda kv: -kv[1])
    body = "  ".join(f"{k}={v * 1000:.0f}ms" for k, v in rows)
    logging.info(f"[minimax_sp][ops] step={total * 1000:.0f}ms  {body}")
    _prof_acc.clear()


def validate_transformer_options(transformer_options):
    unsupported = [key for key in ("patches_replace", "patches") if transformer_options.get(key)]
    unsupported.extend(key for key in transformer_options
                       if "attention" in key.lower() and key not in unsupported
                       and key not in ALLOWED_ATTENTION_KEYS)
    if unsupported:
        raise RuntimeError(f"MiniMax SP does not support transformer {', '.join(unsupported)}; remove those patches or use the single-GPU model")
    if transformer_options.get("multigpu_thread_device") is not None or transformer_options.get("multigpu_clones"):
        raise RuntimeError("MiniMax SP cannot be combined with ComfyUI threaded MultiGPU clones; disable one of the two MultiGPU modes")


def attention_override_name(transformer_options):
    """ModelAttentionBackend が置いた override から、登録名（例 comfy_kitchen_int8）を割り出す。"""
    override = transformer_options.get("optimized_attention_override")
    if override is None:
        return None
    candidates = []
    for cell in getattr(override, "__closure__", None) or ():
        try:
            candidates.append(cell.cell_contents)
        except ValueError:
            pass
    cf = getattr(override, "container_function", None)
    for name, fn in REGISTERED_ATTENTION_FUNCTIONS.items():
        if any(c is fn for c in candidates):
            return name
        if cf is not None and getattr(fn, "container_function", None) is cf:
            return name
    return "unknown"


def make_attention_override(name):
    """worker 側で ModelPatcher.set_model_optimized_attention と同じ override を組み立てる。"""
    if name is None:
        return None
    fn = get_attention_function(name, None)
    if fn is None:
        logging.warning("[minimax_sp] attention backend %r is unavailable on this rank; using the default", name)
        return None

    def optimized_attention_override(_, *args, **kwargs):
        return fn(*args, **kwargs)

    if getattr(fn, "container_function", None) is not None:
        optimized_attention_override.container_function = fn.container_function
    return optimized_attention_override


class SPContext:
    """Row-sharding descriptor for one forward pass."""

    def __init__(self, rank, world, group, seq_len):
        self.rank = rank
        self.world = world
        self.group = group
        self.seq_len = seq_len
        base, rem = divmod(seq_len, world)
        self.splits = [base + (1 if r < rem else 0) for r in range(world)]
        self.start = sum(self.splits[:rank])
        self.stop = self.start + self.splits[rank]

    @property
    def local(self):
        return self.splits[self.rank]


def _seq_gather(out, send, ctx, width, async_op=False):
    """各 rank が自分の行を幅 width で全員に送り、全行を受け取る all-to-all。
    world=4（スイッチ無しの輪）は中継つき（ring_a2a）、それ以外は従来の all_to_all_single。"""
    if ctx.world == 4:
        return ring_a2a.relay_all_to_all(out, send, ring_a2a.a2a_sizes_uniform_send(ctx.splits, width),
                                         ctx.rank, ctx.group, async_op=async_op)
    return dist.all_to_all_single(out, send, output_split_sizes=[n * width for n in ctx.splits],
                                  input_split_sizes=[ctx.local * width] * ctx.world,
                                  group=ctx.group, async_op=async_op)


def _seq_scatter(out, send, ctx, width, async_op=False):
    """_seq_gather の逆（全行を持つ側から、宛先の行を配る）。"""
    if ctx.world == 4:
        return ring_a2a.relay_all_to_all(out, send, ring_a2a.a2a_sizes_uniform_recv(ctx.splits, width),
                                         ctx.rank, ctx.group, async_op=async_op)
    return dist.all_to_all_single(out, send, output_split_sizes=[ctx.local * width] * ctx.world,
                                  input_split_sizes=[n * width for n in ctx.splits],
                                  group=ctx.group, async_op=async_op)


def a2a_gather_seq_scatter_heads(t, ctx):
    """[s_local, H, D] -> [S_total, H/P, D]."""
    s_local, heads, dim = t.shape
    p = ctx.world
    hp = heads // p
    chunk = hp * dim
    send = t.reshape(s_local, p, chunk).transpose(0, 1).contiguous().view(-1)
    out = torch.empty(ctx.seq_len * chunk, dtype=t.dtype, device=t.device)
    _seq_gather(out, send, ctx, chunk)
    return out.view(ctx.seq_len, hp, dim)


def a2a_scatter_seq_gather_heads(t, ctx):
    """[S_total, (H/P)*D] -> [s_local, H*D]"""
    chunk = t.shape[1]
    p = ctx.world
    s_local = ctx.local
    out = torch.empty(p * s_local * chunk, dtype=t.dtype, device=t.device)
    _seq_scatter(out, t.contiguous().view(-1), ctx, chunk)
    return out.view(p, s_local, chunk).transpose(0, 1).reshape(s_local, p * chunk)


def run_optimized_attention(attn, q, k, v, heads, transformer_options):
    out = optimized_attention(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
        heads, preferred_attention=attn.comfy_attention, mask=None, skip_reshape=True,
        transformer_options=transformer_options)
    if not isinstance(out, torch.Tensor):
        raise TypeError("ComfyUI optimized_attention must return a tensor")
    return out


def sp_attention_overlap(attn, q, k, v, ctx, transformer_options, nchunks):
    """a2a と attention を頭の組ごとに重ねる版。q, k, v: [s_local, H, D] -> [s_local, H*D]。

    宛先 rank j に送る頭は j*hp .. (j+1)*hp（従来と同じ割り当て）。それを nchunks 組（hc 頭ずつ）に分け、
    組 i の q/k/v の all-to-all を全部先に非同期で投げ、組 i が届いたら attention、その出力の all-to-all を
    非同期で投げて次の組へ進む。各頭の attention の計算は従来と同じ（頭は互いに独立）。
    """
    s_local, heads, dim = q.shape
    p = ctx.world
    hp = heads // p
    hc = hp // nchunks
    width = hc * dim
    keep = []   # 非同期の集合通信が終わるまで送受信のテンソルを生かしておく

    def prep(t, i):
        x = t.reshape(s_local, p, nchunks, width)[:, :, i]   # [s_local, p, hc*D]
        return x.transpose(0, 1).contiguous().view(-1)        # [p, s_local, hc*D]

    pending = []
    for i in range(nchunks):
        trio = []
        for t in (q, k, v):
            send = prep(t, i)
            recv = torch.empty(ctx.seq_len * width, dtype=t.dtype, device=t.device)
            work = _seq_gather(recv, send, ctx, width, async_op=True)
            keep.append(send)
            trio.append((recv, work))
        pending.append(trio)

    final = torch.empty(s_local, p, nchunks, width, dtype=q.dtype, device=q.device)
    outs = []
    for i in range(nchunks):
        parts = []
        for recv, work in pending[i]:
            work.wait()
            parts.append(recv.view(ctx.seq_len, hc, dim))
        qi, ki, vi = parts
        o = run_optimized_attention(
            attn, qi.transpose(0, 1).unsqueeze(0), ki.transpose(0, 1).unsqueeze(0),
            vi.transpose(0, 1).unsqueeze(0), hc, transformer_options)
        o = o.squeeze(0).contiguous().view(-1)                # [S_total * hc*D]
        ob = torch.empty(p * s_local * width, dtype=o.dtype, device=o.device)
        work = _seq_scatter(ob, o, ctx, width, async_op=True)
        keep.append(o)
        outs.append((i, ob, work))
    for i, ob, work in outs:
        work.wait()
        final[:, :, i] = ob.view(p, s_local, width).transpose(0, 1)
    out = final.view(s_local, heads * dim)
    del keep
    return out


def overlap_chunks(hp, world):
    """重ね合わせの組の数。world=2 は従来どおり MINIMAX_SP_OVERLAP_CHUNKS をそのまま使う。
    world=4 は 1 台の頭が 14 なので、4 が割り切れないときは 14 を割り切る N 以下の最大の数（4 → 2）にする。
    MINIMAX_SP_OVERLAP_CHUNKS_W4 があればそれを優先（全 rank で同じ値にすること）。"""
    if world != 4:
        return OVERLAP_CHUNKS
    n = int(os.environ.get("MINIMAX_SP_OVERLAP_CHUNKS_W4", "0") or 0) or OVERLAP_CHUNKS
    while n > 1 and hp % n:
        n -= 1
    return n


def sp_attention(attn, x, rope_freqs, ctx, transformer_options):
    s = x.shape[0]
    heads, dim = attn.heads, attn.head_dim
    with region("qkv_proj+qknorm+rope"):
        q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
        v = v.view(s, heads, dim)
        if rope_freqs is not None:
            q = q.view(1, s, heads, dim)
            k = k.view(1, s, heads, dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, heads, dim))
            k = attn.k_norm(k.view(s, heads, dim))

    hp = heads // ctx.world
    nchunks = overlap_chunks(hp, ctx.world)
    if nchunks > 1 and hp % nchunks == 0:
        with region("attn+a2a_overlap"):
            out = sp_attention_overlap(attn, q, k, v, ctx, transformer_options, nchunks)
        with region("out_proj"):
            return attn.out_proj(out)
    with region("a2a_qkv"):
        q = a2a_gather_seq_scatter_heads(q, ctx)
        k = a2a_gather_seq_scatter_heads(k, ctx)
        v = a2a_gather_seq_scatter_heads(v, ctx)
    with region("attn"):
        out = run_optimized_attention(
            attn, q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), hp, transformer_options)
    with region("a2a_out"):
        out = a2a_scatter_seq_gather_heads(out.squeeze(0), ctx)
    with region("out_proj"):
        return attn.out_proj(out)


def sp_block(block, x, t_emb, mod_segments, rope_freqs, ctx, transformer_options):
    def attention(h, rope_freqs=None, transformer_options=None):
        return sp_attention(block.attn, h, rope_freqs, ctx, transformer_options)

    with region("block"):
        return block(x, t_emb, mod_segments, rope_freqs,
                     transformer_options=transformer_options, attention=attention)


def shard_segments(segments, start, stop):
    """Clip global (start, stop, row) triples to a rank's row window, in local coords."""
    out = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, stop)
        if hi > lo:
            out.append((lo - start, hi - start, row[lo-a:hi-a] if torch.is_tensor(row) else row))
    return out


def final_layer_fixed_m(fl, x, t_emb, video_seg, audio_seg, totals, sigma, sample_sigmas, shifts):
    """FinalLayer.forward（ComfyUI v0.42.3）の写し。出力の fp32 Linear（video_out・audio_out）だけ、
    自分の行を 0 で埋めて「単独で計算するときと同じ行数」（その流れの全行数）にしてから掛ける。

    cuBLAS の fp32 GEMM は行数 M で計算の仕方を変え、M が小さいと（実測：video_out は約 1 万行未満、
    audio_out はもっと広い範囲で）同じ行でも結果のビットが変わる（quad/m_dep.py）。4 台組では 1 台の
    video の行が約 1 万を切るため、単独・2 台組と latent が一致しなかった。行数を単独と同じにすれば
    同じ計算の仕方になり、各行の結果も単独と同じになる。0 で埋めた行の結果は捨てる。
    """
    from comfy.ldm.minimax.model import _mod_row, _pdd_head
    shift, scale = fl.adaln_proj(t_emb)

    def mod(seg):
        a, b, row = seg
        return (fl.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype)) + _mod_row(shift, row, shift.dtype)).to(torch.float32)

    def padded(h, total):
        m = h.shape[0]
        if m == 0 or m >= total:
            return h, m
        hp = h.new_zeros(total, h.shape[1])
        hp[:m] = h
        return hp, m

    n = fl.video_out.weight.shape[0] // fl.video_dim
    if n != 1:
        if sample_sigmas is None:
            raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")
        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        start, stop = (round(float(1.0 - time_shift_sigma(s, shifts[0], 1.0)) * n) for s in (sigma, sigma_next))
        start = min(start, n - 1)
        stop = max(stop, start + 1)
    outs = []
    for head, seg, total, flow_shift in ((fl.video_out, video_seg, totals[0], shifts[0]),
                                         (fl.audio_out, audio_seg, totals[1], shifts[1])):
        hin, m = padded(mod(seg), total)
        y = head(hin) if n == 1 else _pdd_head(head, hin, n, start, stop, flow_shift)
        outs.append(y[:m])
    return outs[0], outs[1]


def _gather_rows(local, counts, ctx, out_dim, device):
    """Padded all_gather of variable-length row blocks, concatenated in rank order."""
    maxn = max(counts)
    buf = torch.zeros(maxn, out_dim, dtype=torch.float32, device=device)
    if local is not None:
        buf[:local.shape[0]] = local
    gathered = torch.empty(ctx.world * maxn, out_dim, dtype=torch.float32, device=buf.device)
    dist.all_gather_into_tensor(gathered, buf, group=ctx.group)
    gathered = gathered.view(ctx.world, maxn, out_dim)
    return torch.cat([gathered[r, :counts[r]] for r in range(ctx.world) if counts[r] > 0], dim=0)


def _rope_freqs(dit, layout, device):
    params = list(inspect.signature(dit.rope_freqs).parameters)
    if params and params[0] == "layout":
        return dit.rope_freqs(layout, device)
    return dit.rope_freqs(layout.position_ids, device)


def sp_forward(dit, x, timestep, context, transformer_options, minimax_payload, rank, world, group,
               denoise_mask=None, audio_denoise_mask=None):
    """Sequence-parallel mirror of MiniMaxH3Model._forward.

    Returns [video_velocity, audio_velocity] on rank 0, None elsewhere.
    """
    validate_transformer_options(transformer_options)
    t_fwd = time.perf_counter()
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, dit.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    transformer_options["minimax_h3_layout"] = layout

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dit.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dit.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    # Current ComfyUI labels masked rows at their own stream sigma.
    t_pin_v = max(t_v, VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, AUDIO_COND_TIMESTEP)
    video_rows_t = audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                      | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                      | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

    def rows_to_mod_index(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()], dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    if video_rows_t is not None:
        video_seg = (*video_seg[:2], rows_to_mod_index(video_rows_t, 0) // 3)
    if audio_rows_t is not None:
        audio_seg = (*audio_seg[:2], rows_to_mod_index(audio_rows_t, 0) // 3)

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), dit.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = dit._cond_video_rows(payload, device)
    cond_audio_rows = dit._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = dit.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = dit.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != dit.hidden_size:
        text_states = dit.token_refiner(dit.condition_proj(text_states),
                                        transformer_options=transformer_options)

    ctx = SPContext(rank, world, group, layout.seq_len)
    # embedding assembly is row-local and cheap; every rank builds only its own window
    h = torch.empty(ctx.local, dit.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        lo, hi = max(a, ctx.start), min(b, ctx.stop)
        if hi > lo:
            if kind == "text":
                h[lo - ctx.start:hi - ctx.start] = text_states[lo - a:hi - a]
            elif kind in ("cond", "ref_img", "video"):
                h[lo - ctx.start:hi - ctx.start] = video_embed[voff + (lo - a):voff + (hi - a)]
            else:
                h[lo - ctx.start:hi - ctx.start] = audio_embed[aoff + (lo - a):aoff + (hi - a)]
        if kind in ("cond", "ref_img", "video"):
            voff += n
        elif kind != "text":
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if dit.use_adaln_curves:
        table = comfy.model_management.cast_to(dit.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = dit.time_embedder(t_vals).to(dtype)

    rope_freqs = rope_rotation_table(_rope_freqs(dit, layout, device), dtype)
    local_segments = shard_segments(mod_segments, ctx.start, ctx.stop)
    rope_for_blocks = rope_freqs[:, ctx.start:ctx.stop].contiguous()

    if PROFILE_OPS:
        torch.cuda.synchronize()
        _prof_acc["pre"] = time.perf_counter() - t_fwd

    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dit.blocks), device, transformer_options)
    for i, block in enumerate(dit.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, malloc_scope="block")
        transformer_options["block_index"] = i
        h = sp_block(block, h, t_emb, local_segments, rope_for_blocks, ctx, transformer_options)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")

    def counts_for(seg):
        a, b, _ = seg
        out = []
        s = 0
        for n in ctx.splits:
            lo, hi = max(a, s), min(b, s + n)
            out.append(max(0, hi - lo))
            s += n
        return out

    with region("final+gather"):
        def local_seg(seg):
            a, b, row = seg
            lo, hi = max(a, ctx.start), min(b, ctx.stop)
            if hi <= lo:
                return (0, 0, row[:0] if torch.is_tensor(row) else row)
            return (lo-ctx.start, hi-ctx.start, row[lo-a:hi-a] if torch.is_tensor(row) else row)

        if world == 4 and FINAL_FIXED_M:
            v_local, a_local = final_layer_fixed_m(
                dit.final_layer, h, t_emb, local_seg(video_seg), local_seg(audio_seg),
                (video_seg[1] - video_seg[0], audio_seg[1] - audio_seg[0]),
                sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a))
        else:
            v_local, a_local = dit.final_layer(h, t_emb, local_seg(video_seg), local_seg(audio_seg),
                                              sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a))

        v = _gather_rows(v_local, counts_for(video_seg), ctx, dit.final_layer.video_out.out_features, device)
        a = _gather_rows(a_local, counts_for(audio_seg), ctx, dit.final_layer.audio_out.out_features, device)
    if PROFILE_OPS and rank == 0:
        prof_report(time.perf_counter() - t_fwd)
    if rank != 0:
        return None

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, dit.latents_dim, dit.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
