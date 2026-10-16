# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless Q/RoPE materialization for the ordinary DCP decode SFA path."""

import torch
from vllm.triton_utils import tl, triton

MAX_DECODE_ROWS = 64
DECODE_HEADS = 64
LATENT_DIM = 512
ROPE_DIM = 64
ROWS_PER_PROGRAM = 64


@triton.jit
def _restore_query(
    Q,
    R,
    OUT_Q,
    OUT_R,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    LATENT_DIM: tl.constexpr = LATENT_DIM,
    ROPE_DIM: tl.constexpr = ROPE_DIM,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    query, head = row // HEADS, row % HEADS
    d = tl.arange(0, LATENT_DIM)
    dr = tl.arange(0, ROPE_DIM)
    q = tl.load(Q + query[:, None] * QS0 + head[:, None] * QS1 + d[None, :], row[:, None] < ROWS * HEADS, other=0)
    r = tl.load(R + query[:, None] * RS0 + head[:, None] * RS1 + dr[None, :], row[:, None] < ROWS * HEADS, other=0)
    tl.store(OUT_Q + row[:, None] * LATENT_DIM + d[None, :], q, row[:, None] < ROWS * HEADS)
    tl.store(OUT_R + row[:, None] * ROPE_DIM + dr[None, :], r, row[:, None] < ROWS * HEADS)


def materialize_sfa_queries(
    query: torch.Tensor, query_rope: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy the two strided DCP query fragments into contiguous SFA inputs.

    Allocate outputs per call so independent graph captures and streams do not
    share mutable scratch. During capture, the graph memory pool owns their
    storage. SFA consumes both outputs on the caller's stream.
    """
    if (
        query_rope is None
        or query.ndim != 3
        or not 1 <= query.shape[0] <= MAX_DECODE_ROWS
        or query.shape[1:] != (DECODE_HEADS, LATENT_DIM)
        or query_rope.shape != (*query.shape[:2], ROPE_DIM)
        or query.stride(-1) != 1
        or query_rope.stride(-1) != 1
        or query.dtype != torch.bfloat16
        or query_rope.dtype != query.dtype
        or query.device.type != "npu"
        or query_rope.device != query.device
        or query.is_contiguous()
        or query_rope.is_contiguous()
    ):
        return query, query_rope
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    rope = torch.empty(query_rope.shape, dtype=query_rope.dtype, device=query_rope.device)
    _restore_query[(triton.cdiv(query.shape[0] * DECODE_HEADS, ROWS_PER_PROGRAM),)](
        query,
        query_rope,
        out,
        rope,
        ROWS=query.shape[0],
        HEADS=DECODE_HEADS,
        QS0=query.stride(0),
        QS1=query.stride(1),
        RS0=query_rope.stride(0),
        RS1=query_rope.stride(1),
        BLOCK_ROWS=ROWS_PER_PROGRAM,
        multibuffer=False,
    )
    return out, rope
