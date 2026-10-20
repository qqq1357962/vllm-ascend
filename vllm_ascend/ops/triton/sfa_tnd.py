# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged logical indices -> physical slots -> zero-copy TND sparse attention.

Each query is mapped independently. No common-prefix or Pivot contract is used.
"""

from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton

_TOPK = 2048
_INDEX_TILE = 1024
_PAGE_SIZE = 128
_LATENT_DIM = 512
_ROPE_DIM = 64


@triton.jit
def _map_physical_slots(
    indices,
    block_table,
    query_length,
    kv_length,
    mapped,
    physical_length,
    PAGE: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    CAUSAL: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(indices + row * TOPK + column)
    qlen = tl.load(query_length)
    klen = tl.load(kv_length)
    valid = (row < qlen) & (logical >= 0) & (logical < klen)
    if CAUSAL:
        # Right-down causal visibility is defined in LOGICAL token coordinates.
        valid = valid & (logical <= klen - qlen + row)
    page = logical // PAGE
    valid = valid & (page >= 0) & (page < TABLE_SIZE)
    physical_page = tl.load(block_table + page, mask=valid, other=-1)
    slot = physical_page * PAGE + logical % PAGE
    valid = valid & (physical_page >= 0) & (physical_page < CAPACITY // PAGE)
    tl.store(mapped + row * TOPK + column, tl.where(valid, slot, -1))
    if (row == 0) & (tile == 0):
        # The flattened pool can be larger than the logical request context.
        tl.store(physical_length, CAPACITY)


def try_physical_tnd_sfa(
    operator: Callable,
    query: torch.Tensor,
    query_rope: torch.Tensor,
    kv: torch.Tensor,
    key_rope: torch.Tensor,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    query_length: torch.Tensor,
    kv_length: torch.Tensor,
    scale: float,
    *,
    sparse_mode: int = 3,
):
    """Return the SFA result, or None when the original PA path is required.

    Initial scope: A5 caller, BF16, one full-KV request, page128, latent512 /
    RoPE64 and Top2048. The caller must guarantee a full, causally valid
    selected row for each active query; short-context chunks use PA instead.
    Quantized KV, DCP-local/LSE and mixed batches stay on their existing paths.
    Metadata checks do not read device scalar values.

    Logical validity and mode3 causal filtering happen before page lookup.
    TND then uses mode0: physical page placement has no causal meaning.
    Output rows beyond query_length retain the operator's padding semantics.
    """
    if (
        sparse_mode not in (0, 3)
        or query.ndim != 3
        or query.shape[0] == 0
        or query.shape[-1] != _LATENT_DIM
        or kv.ndim != 4
        or kv.shape[1:] != (_PAGE_SIZE, 1, _LATENT_DIM)
        or key_rope.shape != (*kv.shape[:3], _ROPE_DIM)
        or query_rope.shape != (*query.shape[:2], _ROPE_DIM)
        or indices.shape != (query.shape[0], 1, _TOPK)
        or block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] == 0
        or query_length.shape != (1,)
        or kv_length.shape != (1,)
        or kv.numel() == 0
    ):
        return None
    tensors = (query, query_rope, kv, key_rope, indices, block_table, query_length, kv_length)
    if any(t.device != query.device or not t.is_contiguous() for t in tensors):
        return None
    if any(t.dtype != torch.bfloat16 for t in tensors[:4]):
        return None
    if any(t.dtype != torch.int32 for t in tensors[4:]):
        return None

    # Allocation is graph-pool managed; no persistent cache or mutable global.
    mapped = torch.empty_like(indices)
    physical_length = torch.empty_like(kv_length)
    capacity = kv.shape[0] * kv.shape[1]
    _map_physical_slots[(query.shape[0], _TOPK // _INDEX_TILE)](
        indices,
        block_table,
        query_length,
        kv_length,
        mapped,
        physical_length,
        PAGE=kv.shape[1],
        TABLE_SIZE=block_table.shape[1],
        CAPACITY=capacity,
        CAUSAL=sparse_mode == 3,
        TOPK=_TOPK,
        BLOCK=_INDEX_TILE,
        multibuffer=False,
    )
    flat_kv = kv.view(capacity, 1, _LATENT_DIM)
    return operator(
        query=query,
        key=flat_kv,
        value=flat_kv,
        sparse_indices=mapped,
        scale_value=scale,
        actual_seq_lengths_query=query_length,
        actual_seq_lengths_kv=physical_length,
        query_rope=query_rope,
        key_rope=key_rope.view(capacity, 1, _ROPE_DIM),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=False,
    )
