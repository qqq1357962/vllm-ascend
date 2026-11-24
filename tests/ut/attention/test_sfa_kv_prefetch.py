# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_kv_prefetch import _decode_context, _pack_visible_indices


def test_pack_visible_indices_preserves_order_and_duplicates():
    # In a 10-token context, the four query positions can see 7, 8, 9, 10 tokens.
    indices = torch.tensor([[[9, 3, 7, 3, -1, 1]]] * 4, dtype=torch.int32)
    lengths = torch.tensor([10], dtype=torch.int32)
    query_lengths = torch.arange(1, 5, dtype=torch.int32)
    slot_order = torch.arange(indices.shape[-1], dtype=torch.float32)

    packed = _pack_visible_indices(indices, lengths, query_lengths, slot_order)

    assert packed.tolist() == [
        [[3, 3, 1, -1, -1, -1]],
        [[3, 7, 3, 1, -1, -1]],
        [[3, 7, 3, 1, -1, -1]],
        [[9, 3, 7, 3, 1, -1]],
    ]


def test_decode_context_rejects_prefill_and_token_shards():
    context = SimpleNamespace(
        seq_lens=torch.tensor([128], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        num_actual_tokens=4,
        num_input_tokens=4,
        num_prefills=0,
        dcp_context=context,
        dsa_cp_context=None,
    )
    assert _decode_context(metadata, dcp=True) is context

    metadata.dsa_cp_context = object()
    assert _decode_context(metadata, dcp=True) is None

    metadata.dsa_cp_context = None
    metadata.attn_state = AscendAttentionState.ChunkedPrefill
    assert _decode_context(metadata, dcp=True) is None
