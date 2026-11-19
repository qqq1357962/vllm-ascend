# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

GROUP_SIZE = 4
HEADS = 32
HEAD_DIM = 128
PAGE = 128
CONTEXT = 8192


@pytest.fixture
def inputs():
    ops = torch.ops._C_ascend
    if not hasattr(ops, "npu_indexer_coarse_screen") or not 220 <= torch_npu.npu.get_soc_version() <= 225:
        pytest.skip("requires the arch22 PivotCoarseSelect operator")
    torch.manual_seed(20261119)
    batch = 3
    q = torch.randn(batch * GROUP_SIZE, HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    w = torch.rand(batch * GROUP_SIZE, HEADS, device="npu", dtype=q.dtype)
    k = torch.randn(batch * CONTEXT // PAGE, PAGE, 1, HEAD_DIM, device="npu", dtype=q.dtype)
    table = torch.arange(batch * CONTEXT // PAGE, device="npu", dtype=torch.int32).reshape(batch, -1)
    ends = torch.arange(1, batch + 1, device="npu", dtype=torch.int32) * GROUP_SIZE
    lengths = torch.full_like(ends, CONTEXT)
    return (
        q,
        w,
        k,
        dict(
            actual_seq_lengths_query=ends,
            actual_seq_lengths_key=lengths,
            block_table=table,
        ),
    )


@torch.inference_mode()
@pytest.mark.parametrize("width", [2048, 4096])
def test_grouped_proxies_agree_and_respect_window(inputs, width):
    q, w, k, metadata = inputs
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    actual, lengths = op(q, w, k, **metadata, coarse_count=width, proxy_mode=1)
    q_bar = q.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(q.dtype)
    w_bar = w.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(w.dtype)
    expected, expected_lengths = op(q_bar, w_bar, k, **metadata, coarse_count=width)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)

    history = CONTEXT - GROUP_SIZE
    budget = 2 * GROUP_SIZE - 1
    for row in range(len(q_bar)):
        length = int(lengths[row])
        row_ids = actual[row].cpu()
        retained = row_ids[: width - budget].long()
        visible = row_ids[:length].tolist()
        assert len(set(visible)) == length
        assert set(range(history - GROUP_SIZE + 1, CONTEXT)).issubset(visible)
        assert (row_ids[length:] == -1).all()
        assert ((retained >= 0) & (retained < history)).all()
        request_key = k[row * (CONTEXT // PAGE) : (row + 1) * (CONTEXT // PAGE)].reshape(-1, HEAD_DIM)
        scores = ((request_key[:history].float() @ q_bar[row].float().T).relu() @ w_bar[row].float()).cpu()
        cutoff = scores.topk(len(retained)).values[-1]
        torch.testing.assert_close(scores[retained].min(), cutoff, rtol=3e-5, atol=1e-4)


@torch.inference_mode()
@pytest.mark.parametrize("width", [2048, 4096])
def test_local_window_updates_only_the_group_tail(inputs, width):
    q, w, k, metadata = inputs
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    candidates, lengths = op(q, w, k, **metadata, coarse_count=width, proxy_mode=1, is_modify_localwindow=True)
    by_group = candidates.unflatten(0, (-1, GROUP_SIZE)).cpu()
    first_visible = CONTEXT - GROUP_SIZE + 1
    for offset in range(1, GROUP_SIZE):
        torch.testing.assert_close(by_group[:, offset, :-offset], by_group[:, 0, :-offset], rtol=0, atol=0)
        expected = torch.arange(first_visible, first_visible + offset, dtype=torch.int32)
        torch.testing.assert_close(by_group[:, offset, -offset:], expected.expand(len(by_group), -1), rtol=0, atol=0)
    assert (lengths.cpu() == width).all()

    if width == 2048:
        reused, _ = torch.ops._C_ascend.npu_pivot_lightning_indexer(
            q,
            k,
            w,
            **metadata,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=width,
            sparse_mode=3,
        )
        torch.testing.assert_close(candidates, reused.squeeze(1), rtol=0, atol=0)


@torch.inference_mode()
def test_short_grouped_window_deduplicates_and_empty_rows_are_padded(inputs):
    q, w, k, metadata = inputs
    metadata["actual_seq_lengths_key"] = torch.tensor([0, 2, 10], dtype=torch.int32, device="npu")
    candidates, lengths = torch.ops._C_ascend.npu_indexer_coarse_screen(
        q,
        w,
        k,
        **metadata,
        coarse_count=4096,
        proxy_mode=1,
    )
    assert lengths.tolist() == [0, 2, 10]
    for row, count in enumerate(lengths.tolist()):
        indices = candidates[row].tolist()
        assert sorted(indices[:count]) == list(range(count))
        assert set(indices[count:]) == {-1}


@torch.inference_mode()
def test_paired_groups_use_their_own_key_tables(inputs):
    q, w, k, metadata = inputs
    props = torch_npu.npu.get_device_properties(torch.npu.current_device())
    core_count = getattr(props, "cube_core_num", None)
    if not core_count:
        pytest.skip("requires device Cube core count to select the paired schedule")
    rows = 2 * core_count + 1
    # Alternate different KV tables, including a final unpaired row.
    pattern = torch.arange(rows, device="npu") % len(metadata["block_table"])
    table = metadata["block_table"].index_select(0, pattern)
    base_q = q.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(q.dtype)
    base_w = w.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(w.dtype)
    lengths = metadata["actual_seq_lengths_key"]
    # Use different tail lengths to exercise partially filled score buffers.
    lengths[1] -= PAGE + 1
    ends = metadata["actual_seq_lengths_query"]
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    expected, sizes = op(
        base_q,
        base_w,
        k,
        actual_seq_lengths_query=ends,
        actual_seq_lengths_key=lengths,
        block_table=metadata["block_table"],
        coarse_count=4096,
    )
    actual, actual_sizes = op(
        base_q.index_select(0, pattern),
        base_w.index_select(0, pattern),
        k,
        actual_seq_lengths_query=torch.arange(1, rows + 1, device="npu", dtype=torch.int32) * GROUP_SIZE,
        actual_seq_lengths_key=lengths.index_select(0, pattern),
        block_table=table,
        coarse_count=4096,
    )
    torch.testing.assert_close(actual, expected.index_select(0, pattern), rtol=0, atol=0)
    torch.testing.assert_close(actual_sizes, sizes.index_select(0, pattern), rtol=0, atol=0)
