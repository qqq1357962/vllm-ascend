# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative serving dispatch for the experimental Pivot shared consumer."""

import torch

PIVOT_DENSE_BOUNDARY = 4095
PIVOT_GROUP_SIZE = 4
PIVOT_TOPK = 2048


def shared_prefill_prefix(q, qr, kv_cache, indices, metadata, qlen, klen, block_table, return_lse):
    """Return the dense-prefix row count, or None to use ordinary SFA.

    This host decision is restricted to eager prefill with one unpadded request.
    Decode graphs must not capture a CPU sequence-length decision: sequence
    lengths and batch membership can change on replay. DCP prefill is eligible
    after the existing all-KV gather; its indices are still global.
    """
    # attention_v1 imports DeviceOperator, so resolve its enum after worker
    # initialization instead of introducing a module-import cycle.
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    rows = q.shape[0]
    if (
        return_lse
        or metadata.attn_state
        not in {
            AscendAttentionState.PrefillNoCache,
            AscendAttentionState.PrefillCacheHit,
            AscendAttentionState.ChunkedPrefill,
        }
        or metadata.num_prefills not in (0, 1)
        or metadata.num_decodes != 0
        or metadata.num_actual_tokens != rows
        or metadata.max_query_len != rows
        or qlen.numel() != 1
        or klen.numel() != 1
        or block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.stride(-1) != 1
        or rows < PIVOT_GROUP_SIZE
        or q.ndim != 3
        or q.shape[1] not in (8, 64)
        or q.shape[2] != 512
        or qr is None
        or qr.shape != (*q.shape[:2], 64)
        or q.stride(-1) != 1
        or qr.stride(-1) != 1
        or indices.shape != (rows, 1, PIVOT_TOPK)
        or not indices.is_contiguous()
        or indices.dtype != torch.int32
        or len(kv_cache) < 2
    ):
        return None
    kv, kr = kv_cache[:2]
    if (
        kv.ndim != 4
        or kv.shape[2:] != (1, 512)
        or kr.shape != (*kv.shape[:3], 64)
        or kv.stride(-1) != 1
        or kr.stride(-1) != 1
        or any(t.dtype != torch.bfloat16 for t in (q, qr, kv, kr))
    ):
        return None
    lengths = metadata.seq_lens_cpu
    if lengths is None or lengths.device.type != "cpu" or lengths.numel() != 1:
        return None
    context = int(lengths[0]) - rows
    if context < 0:
        return None
    dense = min(
        rows, max(0, (PIVOT_DENSE_BOUNDARY - context + PIVOT_GROUP_SIZE - 1) // PIVOT_GROUP_SIZE * PIVOT_GROUP_SIZE)
    )
    return dense if rows - dense >= PIVOT_GROUP_SIZE else None


def try_shared_prefill(impl, q, qr, kv_cache, indices, metadata, qlen, klen, block_table, return_lse):
    prefix = shared_prefill_prefix(q, qr, kv_cache, indices, metadata, qlen, klen, block_table, return_lse)
    if prefix is None:
        return None
    if q.device.type == "npu" and torch.npu.is_current_stream_capturing():
        return None
    # Load Triton only for workers which explicitly enabled this experiment.
    from vllm_ascend.ops.triton.pivot_shared_sfa import PivotSharedSFA

    if impl.pivot_shared_sfa is None:
        impl.pivot_shared_sfa = PivotSharedSFA()
    output, _, _ = impl.pivot_shared_sfa(
        q[prefix:],
        qr[prefix:],
        *kv_cache[:2],
        indices[prefix:],
        block_table,
        klen,
        impl.scale,
        actual_query_lengths=qlen - prefix if prefix else qlen,
    )
    if prefix:
        kv, kr = kv_cache[:2]
        # Keep the original right-down causal coordinates for the dense prefix.
        # The indexer's indices already enforce visibility, so mode 0 avoids
        # shifting its mask when the prefix is submitted as a separate call.
        dense, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q[:prefix],
            key=kv,
            value=kv,
            sparse_indices=indices[:prefix],
            scale_value=impl.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=qlen.clamp_max(prefix),
            actual_seq_lengths_kv=klen,
            query_rope=qr[:prefix],
            key_rope=kr,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )
        output = torch.cat((dense, output), dim=0)
    return output
