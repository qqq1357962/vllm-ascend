# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Side-stream preparation of a compact sparse MLA KV cache for decode."""

import re
from collections.abc import Mapping
from typing import NamedTuple

import torch
from vllm.logger import logger
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_ascend.attention.attention_v1 import AscendAttentionState

_QUERY_ROWS = 4  # One anchor and three speculative tokens.
_POOL_SLOTS = 2048  # Per-query capacity of the tagged SFA layout.
_PAGE_ROWS = 128
_KV_WIDTH = 512
_ROPE_WIDTH = 64
_COMPACT_TABLE_TAG = -77701313  # Shared with the arch22 SFA consumer.
# Leave the verification rows and another verification-width margin for recent KV writes.
_RECENT_WINDOW = 2 * _QUERY_ROWS


class CompactInputs(NamedTuple):
    cache: tuple[torch.Tensor, torch.Tensor]
    indices: torch.Tensor
    query_lengths: torch.Tensor
    kv_lengths: torch.Tensor
    block_table: torch.Tensor


def _supports_cache(cache) -> bool:
    if not isinstance(cache, (tuple, list)) or len(cache) < 2:
        return False
    key, rope = cache[:2]
    return (
        key.dtype == rope.dtype == torch.bfloat16
        and key.shape[1:] == (_PAGE_ROWS, 1, _KV_WIDTH)
        and rope.shape[1:] == (_PAGE_ROWS, 1, _ROPE_WIDTH)
        and key.shape[0] == rope.shape[0]
        and key.stride(1) == _KV_WIDTH
        and rope.stride(1) == _ROPE_WIDTH
        and key.stride(-1) == rope.stride(-1) == 1
    )


def _decode_context(metadata, dcp: bool):
    if (
        metadata is None
        or metadata.attn_state not in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
        or metadata.num_actual_tokens != _QUERY_ROWS
        or metadata.num_input_tokens != _QUERY_ROWS
        or metadata.num_prefills
        or getattr(metadata, "dsa_cp_context", None) is not None
        or getattr(metadata, "pcp_has_global_prefill", False)
    ):
        return None
    context = getattr(metadata, "dcp_context", None) if dcp else metadata
    if (
        context is None
        or context.seq_lens.dtype != torch.int32
        or context.seq_lens.numel() != 1
        or context.block_table.dtype != torch.int32
        or context.block_table.ndim != 2
        or context.block_table.shape[0] != 1
        or context.block_table.shape[1] == 0
    ):
        return None
    return context


def _pack_visible_indices(
    indices: torch.Tensor,
    length: torch.Tensor,
    query_lengths: torch.Tensor,
    slot_order: torch.Tensor,
) -> torch.Tensor:
    # Native no-DCP SFA masks future positions. Stable packing retains the
    # original order even when those positions appear inside the top-k list.
    visible_end = (length - _QUERY_ROWS + query_lengths).view(-1, 1, 1)
    valid = (indices >= 0) & (indices < visible_end)
    sort_keys = torch.where(valid, slot_order, slot_order + indices.size(-1))
    order = torch.argsort(sort_keys, dim=-1)
    return torch.gather(indices.masked_fill(~valid, -1), dim=-1, index=order).contiguous()


class _PrefetchPool:
    def __init__(self, device: torch.device, dcp: bool):
        self.dcp = dcp
        self.stream = torch.npu.Stream(device=device)
        pages = _QUERY_ROWS * (_POOL_SLOTS // _PAGE_ROWS)
        page_table = torch.arange(pages, device=device, dtype=torch.int32).view(_QUERY_ROWS, -1)
        tag = torch.full((_QUERY_ROWS, 1), _COMPACT_TABLE_TAG, device=device, dtype=torch.int32)
        self.inputs = CompactInputs(
            cache=(
                torch.empty((pages, _PAGE_ROWS, 1, _KV_WIDTH), device=device, dtype=torch.bfloat16),
                torch.empty((pages, _PAGE_ROWS, 1, _ROPE_WIDTH), device=device, dtype=torch.bfloat16),
            ),
            indices=torch.empty((_QUERY_ROWS, 1, _POOL_SLOTS), device=device, dtype=torch.int32),
            query_lengths=torch.arange(1, _QUERY_ROWS + 1, device=device, dtype=torch.int32),
            kv_lengths=torch.empty(_QUERY_ROWS, device=device, dtype=torch.int32),
            block_table=torch.cat((page_table, tag), dim=1),
        )
        self.slot_order = torch.arange(_POOL_SLOTS, device=device, dtype=torch.float32)
        self.pending = None

    def _gather(self, cache, indices, table, lengths, *, refresh: bool) -> None:
        torch.ops._C_ascend.sfa_kv_pool_gather(
            cache[0],
            cache[1],
            indices,
            table,
            lengths,
            *self.inputs.cache,
            self.inputs.indices,
            self.inputs.kv_lengths,
            _RECENT_WINDOW,
            refresh,
            self.dcp,
        )

    def prepare(self, layer_name: str) -> None:
        if self.pending is not None:
            torch.npu.current_stream().wait_stream(self.stream)
            self.pending = None
        try:
            metadata, layer, cache, _ = get_attention_context(layer_name)
        except KeyError:
            return
        context = _decode_context(metadata, self.dcp)
        if context is None or not _supports_cache(cache):
            return
        indices = layer.impl._get_indexcache_topk_indices(_QUERY_ROWS)
        if indices.shape != self.inputs.indices.shape or indices.dtype != torch.int32:
            return

        # The prior SFA consumer has finished using the common scratch on the main stream.
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            if self.dcp:
                indices = layer.impl._remap_sparse_indices(indices).contiguous()
            else:
                indices = _pack_visible_indices(indices, context.seq_lens, self.inputs.query_lengths, self.slot_order)
            table = context.block_table.contiguous()
            lengths = context.seq_lens.contiguous()
            self._gather(cache, indices, table, lengths, refresh=False)
        self.pending = (layer.impl, metadata, cache[0], cache[1], indices, table, lengths)

    def take(self, impl, metadata, cache, num_queries: int) -> CompactInputs | None:
        pending = self.pending
        if pending is None or pending[0] is not impl:
            return None
        self.pending = None
        _, prepared_metadata, key, rope, indices, table, lengths = pending
        main = torch.npu.current_stream()
        main.wait_stream(self.stream)
        if (
            metadata is not prepared_metadata
            or cache[0] is not key
            or cache[1] is not rope
            or num_queries != _QUERY_ROWS
        ):
            return None

        # The target KV writes have been enqueued on the main stream. Refresh its
        # recent slots there after the historical gather, then consume the pool.
        self._gather(cache, indices, table, lengths, refresh=True)
        for tensor in (indices, table, lengths):
            tensor.record_stream(main)
        return self.inputs


def initialize_sfa_kv_prefetch(layers: Mapping, config) -> None:
    if not hasattr(torch.ops._C_ascend, "sfa_kv_pool_gather"):
        raise RuntimeError("SFA KV prefetch requires vLLM-Ascend built with custom operators for Ascend 910B.")
    if (
        config.scheduler_config.max_num_seqs != 1
        or config.parallel_config.prefill_context_parallel_size != 1
        or config.kv_transfer_config is not None
    ):
        raise ValueError("SFA KV prefetch requires max_num_seqs=1, PCP=1 and KV transfer disabled.")
    pool = None
    for name, layer in layers.items():
        impl = getattr(layer, "impl", None)
        cache = getattr(layer, "kv_cache", None)
        if not getattr(impl, "skip_topk", False) or getattr(impl, "_is_mtp_layer", False) or not _supports_cache(cache):
            continue
        match = re.fullmatch(r"(.*layers\.)(\d+)(\..*)", name)
        if match is None or int(match[2]) == 0:
            continue
        previous = layers.get(f"{match[1]}{int(match[2]) - 1}{match[3]}")
        predecessor = getattr(previous, "impl", None)
        if predecessor is None:
            continue
        dcp = getattr(impl, "dcp_size", 1) > 1
        if pool is None:
            pool = _PrefetchPool(cache[0].device, dcp)
        impl._sfa_kv_prefetch_pool = pool
        predecessor._sfa_kv_prefetch_next = (pool, name)
    if pool is None:
        raise RuntimeError("SFA KV prefetch found no eligible BF16 Pivot reuse layers.")
    logger.info("Enabled SFA KV prefetch for BF16 sparse MLA decode.")


def prepare_next_sfa_kv(impl) -> None:
    target = getattr(impl, "_sfa_kv_prefetch_next", None)
    if target is not None:
        pool, layer_name = target
        pool.prepare(layer_name)


def take_prepared_sfa_kv(impl, metadata, cache, num_queries: int) -> CompactInputs | None:
    pool = getattr(impl, "_sfa_kv_prefetch_pool", None)
    if isinstance(pool, _PrefetchPool):
        return pool.take(impl, metadata, cache, num_queries)
    return None
