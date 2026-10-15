# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-prefix boundaries and changing-input replay for arch22 paged SFA."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

pytestmark = pytest.mark.skipif("910" not in torch.npu.get_device_name(0), reason="Requires an arch22 SFA kernel")


def make_inputs(dtype, rows=8, capacity=2048):
    generator = torch.Generator().manual_seed(20261015)
    query = torch.randn(rows, 64, 512, generator=generator, dtype=dtype)
    rope = torch.randn(rows, 64, 64, generator=generator, dtype=dtype)
    key = torch.randn(64, 128, 1, 512, generator=generator, dtype=dtype)
    key_rope = torch.randn(64, 128, 1, 64, generator=generator, dtype=dtype)
    table = torch.stack([torch.randperm(64, generator=generator, dtype=torch.int32) for _ in range(2)])
    pages = key.npu()
    return {
        "query": query.npu(),
        "query_rope": rope.npu(),
        "key": pages,
        "value": pages,
        "key_rope": key_rope.npu(),
        "sparse_indices": torch.full((rows, 1, capacity), -1, dtype=torch.int32, device="npu"),
        "block_table": table.npu(),
        "actual_seq_lengths_query": torch.tensor([3, rows], dtype=torch.int32).npu(),
        "actual_seq_lengths_kv": torch.tensor([8192, 8192], dtype=torch.int32).npu(),
    }


def attention(inputs, return_lse=True):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        **inputs,
        scale_value=1 / 24,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=return_lse,
    )


def fill_indices(inputs, counts, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.full(inputs["sparse_indices"].shape, -1, dtype=torch.int32)
    for row, count in enumerate(counts):
        # Valid indices deliberately remain unordered and differ by query.
        indices[row, 0, :count] = torch.randperm(8192, generator=generator)[:count].int()
    inputs["sparse_indices"].copy_(indices)


def assert_reference(inputs, result, return_lse=True):
    output, maximum, total = [tensor.cpu() for tensor in result]
    q, qr = inputs["query"].cpu().double(), inputs["query_rope"].cpu().double()
    key = inputs["key"].cpu().reshape(-1, 512).double()
    kr = inputs["key_rope"].cpu().reshape(-1, 64).double()
    values = inputs["value"].cpu().reshape(-1, 512).double()
    indices = inputs["sparse_indices"].cpu()
    table = inputs["block_table"].cpu().long()
    lengths = inputs["actual_seq_lengths_query"].cpu().tolist()
    kv_lengths = inputs["actual_seq_lengths_kv"].cpu().tolist()
    start = 0
    for batch, end in enumerate(lengths):
        for row in range(start, end):
            selected = indices[row, 0]
            selected = selected[: min(selected.numel(), kv_lengths[batch])]
            selected = selected[selected >= 0].long()
            if selected.numel() == 0:
                assert torch.count_nonzero(output[row]) == 0
                if return_lse:
                    assert torch.count_nonzero(total[0, row]) == 0
                    assert torch.isneginf(maximum[0, row] + total[0, row].log()).all()
                continue
            physical = table[batch, selected // 128] * 128 + selected % 128
            logits = (q[row] @ key[physical].T + qr[row] @ kr[physical].T) / 24
            expected = logits.softmax(-1) @ values[physical]
            torch.testing.assert_close(output[row].double(), expected, atol=0.03, rtol=0.01)
            if return_lse:
                lse = maximum[0, row].double() + total[0, row].double().log()
                torch.testing.assert_close(lse, logits.logsumexp(-1), atol=0.005, rtol=0.001)
        start = end


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("return_lse", [True, False])
def test_sparse_prefix_boundaries_and_graph_replay(dtype, return_lse):
    enable_custom_op()
    inputs = make_inputs(dtype)
    counts = [0, 1, 127, 128, 129, 255, 256, 257]
    fill_indices(inputs, counts, 1)
    for _ in range(3):
        attention(inputs, return_lse)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = attention(inputs, return_lse)
    for step, current_counts in enumerate([counts, [2047, 2048, 257, 256, 255, 129, 128, 127], [1] * 8, [0] * 8]):
        fill_indices(inputs, current_counts, step + 2)
        inputs["query"].mul_(0.9)
        inputs["query_rope"].mul_(0.8)
        inputs["key"].mul_(0.7)
        inputs["key_rope"].mul_(0.6)
        inputs["block_table"].copy_(inputs["block_table"].flip(-1))
        inputs["actual_seq_lengths_query"].copy_(torch.tensor([3 + step % 2, 8], dtype=torch.int32).npu())
        graph.replay()
        torch.npu.synchronize()
        assert_reference(inputs, result, return_lse)


def test_zero_kv_and_padded_query_rows():
    enable_custom_op()
    inputs = make_inputs(torch.bfloat16)
    fill_indices(inputs, [0, 0, 0, 1, 127, 128, 129, 257], 10)
    inputs["actual_seq_lengths_kv"].copy_(torch.tensor([0, 8192], dtype=torch.int32).npu())
    inputs["actual_seq_lengths_query"].copy_(torch.tensor([3, 6], dtype=torch.int32).npu())
    result = attention(inputs)
    torch.npu.synchronize()
    assert_reference(inputs, result)
