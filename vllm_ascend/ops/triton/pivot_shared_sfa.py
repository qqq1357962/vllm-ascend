# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consume a Pivot R4 plan without sharing any query's attention weights.

The first COMMON_TOPK indices must agree in every valid group of four rows.
The final three indices remain private to each row. The caller provides global
indices before DCP remapping and excludes the indexer's dense prefix. No index
intersection or approximate union is computed here.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element

TOPK = 2048
COMMON_TOPK = 2045
GROUP_SIZE = 4
LATENT_DIM = 512
ROPE_DIM = 64
MAX_CACHED_ROWS = 64


@triton.jit(do_not_specialize=["N"])
def _plan(
    RAW,
    BT,
    QLEN,
    COMMON,
    PRIVATE,
    VALID,
    N,
    HCHUNKS: tl.constexpr,
    DCP: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    PAGE: tl.constexpr,
    TOPK: tl.constexpr = TOPK,
    COMMON_TOPK: tl.constexpr = COMMON_TOPK,
):
    g = tl.program_id(0)
    active_n = tl.minimum(N, tl.load(QLEN))
    j = tl.arange(0, TOPK)
    base = tl.load(RAW + (g * 4) * TOPK + j, (j < COMMON_TOPK) & (g * 4 < active_n), other=-1)
    if DCP == 1:
        for hc in tl.static_range(HCHUNKS):
            tl.store(COMMON + (g * HCHUNKS + hc) * COMMON_TOPK + j, base, j < COMMON_TOPK)
    # Contiguous row loads avoid Ascend's strided-gather runtime host copy.
    col = tl.arange(0, 4)
    for row in tl.static_range(4):
        active = tl.load(RAW + (g * 4 + row) * TOPK, g * 4 + row < active_n, other=-1) >= 0
        tl.store(VALID + g * 4 + row, active.to(tl.int32), g * 4 + row < N)
        private = tl.load(
            RAW + (g * 4 + row) * TOPK + COMMON_TOPK + col, (col < 3) & (g * 4 + row < active_n), other=-1
        )
        valid = (private >= 0) & ((private // INTERLEAVE) % DCP == RANK) & (col < 3)
        local = (private // (DCP * INTERLEAVE)) * INTERLEAVE + private % INTERLEAVE
        page = tl.load(BT + local // PAGE, valid, other=0)
        tl.store(
            PRIVATE + g * 12 + row * 3 + col,
            tl.where(valid, page * PAGE + local % PAGE, -1),
            (col < 3) & (g * 4 + row < N),
        )


@triton.jit
def _compact(
    RAW,
    QLEN,
    CHUNKS,
    COUNTS,
    COMMON,
    DCP: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    HCHUNKS: tl.constexpr,
    TOPK: tl.constexpr = TOPK,
    COMMON_TOPK: tl.constexpr = COMMON_TOPK,
):
    chunk, g = tl.program_id(0), tl.program_id(1)
    j = chunk * 128 + tl.arange(0, 128)
    ids = tl.load(RAW + g * 4 * TOPK + j, (j < COMMON_TOPK) & (g * 4 < tl.load(QLEN)), other=-1)
    keep = ((ids >= 0) & ((ids // INTERLEAVE) % DCP == RANK)).to(tl.int32)
    local = (ids // (DCP * INTERLEAVE)) * INTERLEAVE + ids % INTERLEAVE
    for hc in tl.static_range(HCHUNKS):
        tl.store(COMMON + (g * HCHUNKS + hc) * COMMON_TOPK + j, -1, j < COMMON_TOPK)
    count = 0
    for i in range(128):
        if get_element(keep, (i,)) == 1:
            tl.store(CHUNKS + (g * 16 + chunk) * 128 + count, get_element(local, (i,)))
            count += 1
    tl.store(COUNTS + g * 16 + chunk, count)


@triton.jit
def _gather(CHUNKS, COUNTS, COMMON, HCHUNKS: tl.constexpr, COMMON_TOPK: tl.constexpr = COMMON_TOPK):
    g, hc = tl.program_id(0), tl.program_id(1)
    start = 0
    j = tl.arange(0, 128)
    for chunk in range(16):
        count = tl.load(COUNTS + g * 16 + chunk)
        x = tl.load(CHUNKS + (g * 16 + chunk) * 128 + j, j < count, other=0)
        tl.store(COMMON + (g * HCHUNKS + hc) * COMMON_TOPK + start + j, x, j < count)
        start += count


@triton.jit(do_not_specialize=["N", "QS0", "QS1", "RS0", "RS1"])
def _pack(
    Q,
    QR,
    QLEN,
    PQ,
    PQR,
    N,
    H: tl.constexpr,
    PH: tl.constexpr,
    QS0,
    QS1,
    RS0,
    RS1,
    LATENT_DIM: tl.constexpr = LATENT_DIM,
):
    packed = tl.program_id(0)
    g, hc = packed // (H // PH), packed % (H // PH)
    x = tl.arange(0, LATENT_DIM)
    row = tl.program_id(1)
    query = g * 4 + row // PH
    active_n = tl.minimum(N, tl.load(QLEN))
    head = hc * PH + row % PH
    q = tl.load(Q + query * QS0 + head * QS1 + x, query < active_n, other=0)
    tl.store(PQ + (packed * 4 * PH + row) * LATENT_DIM + x, q)
    dr = tl.arange(0, 64)
    qr = tl.load(QR + query * RS0 + head * RS1 + dr, query < active_n, other=0)
    tl.store(PQR + (packed * 4 * PH + row) * 64 + dr, qr)


@triton.jit(do_not_specialize=["N", "QS0", "QS1", "RS0", "RS1"])
def _merge(
    Q,
    QR,
    KV,
    KR,
    PRIVATE,
    VALID,
    CORE_OUTPUT,
    M,
    S,
    OUT,
    MAX,
    SUM,
    N,
    H: tl.constexpr,
    PH: tl.constexpr,
    QS0,
    QS1,
    RS0,
    RS1,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KRS0: tl.constexpr,
    KRS1: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    LATENT_DIM: tl.constexpr = LATENT_DIM,
):
    query = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d, dr = tl.arange(0, LATENT_DIM), tl.arange(0, 64)
    q = tl.load(Q + query * QS0 + h[:, None] * QS1 + d[None, :]).to(tl.float32)
    qr = tl.load(QR + query * RS0 + h[:, None] * RS1 + dr[None, :]).to(tl.float32)
    packed = ((query // 4) * (H // PH) + h // PH) * (4 * PH) + (query % 4) * PH + h % PH
    m = tl.load(M + packed)
    s = tl.load(S + packed)
    m = tl.where(s > 0, m, -float("inf"))
    core = tl.load(CORE_OUTPUT + packed[:, None] * LATENT_DIM + d[None, :]).to(tl.float32)
    # CANN returns the normalized common output plus FP32 max/sum. Rebuild
    # its numerator in that scale, then merge each row's three private keys.
    # QK, softmax state and PV remain distinct for every (query, head).
    numerator = tl.where(s[:, None] > 0, core * s[:, None], 0.0)
    for p in tl.static_range(3):
        slot = tl.load(PRIVATE + query * 3 + p)
        k = tl.load(KV + (slot // PAGE) * KS0 + (slot % PAGE) * KS1 + d, slot >= 0, other=0).to(tl.float32)
        kr = tl.load(KR + (slot // PAGE) * KRS0 + (slot % PAGE) * KRS1 + dr, slot >= 0, other=0).to(tl.float32)
        score = tl.where(slot >= 0, (tl.sum(q * k[None, :], 1) + tl.sum(qr * kr[None, :], 1)) * SCALE, -float("inf"))
        nm = tl.maximum(m, score)
        safe = tl.where(nm > -float("inf"), nm, 0.0)
        alpha = tl.exp(m - safe)
        beta = tl.exp(score - safe)
        numerator = numerator * alpha[:, None] + beta[:, None] * k[None, :]
        s = s * alpha + beta
        m = nm
    active = tl.load(VALID + query) > 0
    tl.store(
        OUT + (query * H + h[:, None]) * LATENT_DIM + d[None, :],
        tl.where(active, numerator / tl.where(s[:, None] > 0, s[:, None], 1.0), 0.0),
    )
    tl.store(MAX + query * H + h, tl.where(active & (s > 0), m, 0.0))
    tl.store(SUM + query * H + h, tl.where(active, s, 0.0))


class PivotSharedSFA:
    """Own graph-stable scratch; outputs never alias reusable workspace.

    A caller may share this object across layers on the same stream. Different
    streams have separate scratch. Only small shapes are retained; eager prefill
    scratch is released after use. Every invocation reads current Q/KV/indices,
    page tables and lengths, including during graph replay.
    """

    def __init__(self, operator=None):
        self.operator = operator if operator is not None else torch.ops._C_ascend.npu_sparse_flash_attention
        self.query_lengths = {}
        self.workspaces = {}

    def __call__(
        self, q, qr, kv, kr, raw, bt, klen, scale, *, dcp=1, rank=0, interleave=128, actual_query_lengths=None
    ):
        n, h = q.shape[:2]
        if (
            n < 1
            or raw.shape != (n, 1, TOPK)
            or h not in (8, 64)
            or q.shape[-1] != LATENT_DIM
            or qr.shape != (n, h, ROPE_DIM)
            or q.stride(-1) != 1
            or qr.stride(-1) != 1
            or not raw.is_contiguous()
            or raw.dtype != torch.int32
            or bt.ndim != 2
            or bt.shape[0] != 1
            or bt.stride(-1) != 1
            or kv.ndim != 4
            or kv.shape[2:] != (1, LATENT_DIM)
            or kr.shape != (*kv.shape[:3], ROPE_DIM)
            or kv.stride(-1) != 1
            or kr.stride(-1) != 1
            or any(t.dtype != torch.bfloat16 for t in (q, qr, kv, kr))
            or dcp < 1
            or not 0 <= rank < dcp
            or interleave < 1
        ):
            raise ValueError("Unsupported Pivot shared SFA shape, dtype, stride or DCP geometry")
        # Pack four query rows as heads of one pseudo-query. The CANN tile
        # can reuse common KV loads; no query vector or attention weight is shared.
        ph = min(h, 32)
        groups = triton.cdiv(n, 4)
        pc = groups * (h // ph)
        if actual_query_lengths is None:
            key = (q.device, n)
            if key not in self.query_lengths:
                self.query_lengths[key] = torch.tensor([n], dtype=torch.int32, device=q.device)
            actual_query_lengths = self.query_lengths[key]
        can_view = h == ph and n % 4 == 0 and q.stride(0) == h * q.stride(1) and qr.stride(0) == h * qr.stride(1)

        def empty(shape, dtype=torch.int32):
            return torch.empty(shape, dtype=dtype, device=q.device)

        # ACLNN graph inputs need stable addresses, not a separate allocation per
        # layer. Scratch is fully consumed before the next call on the same stream.
        # Keep different streams isolated; final outputs never alias this pool.
        stream_id = torch.npu.current_stream(q.device).npu_stream
        cache_key = (stream_id, n, h, dcp, rank, q.device, q.dtype, qr.dtype, can_view)
        ws = self.workspaces.get(cache_key) if n <= MAX_CACHED_ROWS else None
        if ws is None:
            ws = dict(common=empty((pc, 1, 2045)), private=empty((n, 3)), valid=empty((n,)))
            if not can_view:
                ws.update(pq=empty((pc, 4 * ph, 512), q.dtype), pqr=empty((pc, 4 * ph, 64), qr.dtype))
            if dcp > 1:
                ws.update(chunks=empty((groups * 16, 128)), counts=empty((groups, 16)))
            if n <= MAX_CACHED_ROWS:
                self.workspaces[cache_key] = ws
        common, private = (ws[k] for k in ("common", "private"))
        _plan[(groups,)](
            raw,
            bt,
            actual_query_lengths,
            common,
            private,
            ws["valid"],
            N=n,
            HCHUNKS=h // ph,
            DCP=dcp,
            RANK=rank,
            INTERLEAVE=interleave,
            PAGE=kv.shape[1],
            multibuffer=False,
        )
        if dcp > 1:
            _compact[(16, groups)](
                raw,
                actual_query_lengths,
                ws["chunks"],
                ws["counts"],
                common,
                DCP=dcp,
                RANK=rank,
                INTERLEAVE=interleave,
                HCHUNKS=h // ph,
                multibuffer=False,
            )
            _gather[(groups, h // ph)](ws["chunks"], ws["counts"], common, HCHUNKS=h // ph, multibuffer=False)
        if can_view:
            pq = q.view(groups, 4 * h, 512)
            pqr = qr.view(groups, 4 * h, 64)
        else:
            pq, pqr = ws["pq"], ws["pqr"]
            _pack[(pc, 4 * ph)](
                q,
                qr,
                actual_query_lengths,
                pq,
                pqr,
                N=n,
                H=h,
                PH=ph,
                QS0=q.stride(0),
                QS1=q.stride(1),
                RS0=qr.stride(0),
                RS1=qr.stride(1),
            )
        key = (q.device, pc)
        if key not in self.query_lengths:
            self.query_lengths[key] = torch.tensor([pc], dtype=torch.int32, device=q.device)
        qlen = self.query_lengths[key]
        o, m, s = self.operator(
            query=pq,
            key=kv,
            value=kv,
            sparse_indices=common,
            scale_value=scale,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            block_table=bt[:1],
            actual_seq_lengths_query=qlen,
            actual_seq_lengths_kv=klen[:1],
            query_rope=pqr,
            key_rope=kr,
            return_softmax_lse=True,
        )
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        maximum = empty((1, n, h), torch.float32)
        total = empty((1, n, h), torch.float32)
        _merge[(n, h // 8)](
            q,
            qr,
            kv,
            kr,
            private,
            ws["valid"],
            o,
            m,
            s,
            out,
            maximum,
            total,
            N=n,
            H=h,
            PH=ph,
            QS0=q.stride(0),
            QS1=q.stride(1),
            RS0=qr.stride(0),
            RS1=qr.stride(1),
            KS0=kv.stride(0),
            KS1=kv.stride(1),
            KRS0=kr.stride(0),
            KRS1=kr.stride(1),
            PAGE=kv.shape[1],
            SCALE=scale,
            BH=8,
        )
        return out, maximum, total
