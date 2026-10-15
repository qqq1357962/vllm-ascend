# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.pivot_shared_sfa import shared_prefill_prefix, try_shared_prefill


def inputs(rows=8, context=8192):
    q = torch.empty(rows, 8, 512, dtype=torch.bfloat16)
    qr = torch.empty(rows, 8, 64, dtype=torch.bfloat16)
    cache = (torch.empty(64, 128, 1, 512, dtype=torch.bfloat16), torch.empty(64, 128, 1, 64, dtype=torch.bfloat16))
    indices = torch.empty(rows, 1, 2048, dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=1,
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_decodes=0,
        num_actual_tokens=rows,
        max_query_len=rows,
        seq_lens_cpu=torch.tensor([context + rows]),
    )
    return [
        q,
        qr,
        cache,
        indices,
        metadata,
        torch.tensor([rows]),
        torch.tensor([context + rows]),
        torch.zeros(1, 64, dtype=torch.int32),
        False,
    ]


@pytest.mark.parametrize(
    "context,rows,expected", [(0, 8, None), (4090, 16, 8), (4091, 8, 4), (4095, 8, 0), (8192, 5, 0), (8192, 3, None)]
)
def test_dense_prefix_matches_pivot_geometry(context, rows, expected):
    assert shared_prefill_prefix(*inputs(rows, context)) == expected


@pytest.mark.parametrize(
    "change",
    [
        "decode",
        "mixed",
        "batch",
        "padded",
        "length_missing",
        "fp16",
        "quantized",
        "lse",
        "index_width",
        "strided_indices",
        "heads",
        "query_rope_missing",
    ],
)
def test_unsupported_routes_fall_back(change):
    args = inputs()
    metadata = args[4]
    if change == "decode":
        metadata.num_prefills, metadata.num_decodes = 0, 1
    elif change == "mixed":
        metadata.num_decodes = 1
    elif change == "batch":
        args[5] = torch.tensor([4, 8])
    elif change == "padded":
        metadata.num_actual_tokens = 5
    elif change == "length_missing":
        metadata.seq_lens_cpu = None
    elif change == "fp16":
        args[0] = args[0].half()
    elif change == "quantized":
        args[2] = (args[2][0].to(torch.int8), args[2][1])
    elif change == "lse":
        args[-1] = True
    elif change == "index_width":
        args[3] = args[3][:, :, :2045]
    elif change == "strided_indices":
        args[3] = torch.empty(8, 1, 4096, dtype=torch.int32)[:, :, ::2]
    elif change == "heads":
        args[0] = args[0][:, :4]
    elif change == "query_rope_missing":
        args[1] = None
    assert shared_prefill_prefix(*args) is None


def test_dense_prefix_and_private_rows_use_exact_original_indices():
    args = inputs(rows=16, context=4090)
    args[3].copy_(torch.arange(16 * 2048).reshape(16, 1, 2048))
    grouped = Mock(return_value=(torch.full_like(args[0][8:], 2), None, None))
    impl = SimpleNamespace(scale=1 / 24, pivot_shared_sfa=grouped)
    with patch.object(
        torch.ops._C_ascend,
        "npu_sparse_flash_attention",
        create=True,
        return_value=(torch.ones_like(args[0][:8]), None, None),
    ) as ordinary:
        result = try_shared_prefill(impl, *args)
    torch.testing.assert_close(grouped.call_args.args[4], args[3][8:])
    assert grouped.call_args.args[0].data_ptr() == args[0][8:].data_ptr()
    torch.testing.assert_close(ordinary.call_args.kwargs["sparse_indices"], args[3][:8])
    assert ordinary.call_args.kwargs["sparse_mode"] == 0
    assert torch.all(result[:8] == 1) and torch.all(result[8:] == 2)


def test_non_dcp_metadata_has_no_prefill_count():
    args = inputs()
    args[4].num_prefills = 0
    assert shared_prefill_prefix(*args) == 0
    args[4].attn_state = AscendAttentionState.SpecDecoding
    assert shared_prefill_prefix(*args) is None
