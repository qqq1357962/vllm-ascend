# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

PAGE_SIZE = 128
HEADS = 32
HEAD_DIM = 128
GROUP_SIZE = 4
KV_LENGTH = 32768


@pytest.fixture
def inputs():
    ops = torch.ops._C_ascend
    if not hasattr(ops, "npu_indexer_coarse_screen") or not 220 <= torch_npu.npu.get_soc_version() <= 225:
        pytest.skip("requires the arch22 coarse screen operator")
    torch.manual_seed(20261118)
    ends = torch.tensor([GROUP_SIZE, 2 * GROUP_SIZE], device="npu", dtype=torch.int32)
    lengths = torch.full_like(ends, KV_LENGTH)
    n_pages = KV_LENGTH // PAGE_SIZE
    table = torch.arange(2 * n_pages, device="npu", dtype=torch.int32).reshape(2, n_pages)
    q = torch.randn(2 * GROUP_SIZE, HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    w = torch.rand(2 * GROUP_SIZE, HEADS, device="npu", dtype=q.dtype)
    k = torch.randn(2 * n_pages, PAGE_SIZE, 1, HEAD_DIM, device="npu", dtype=q.dtype)
    return q, w, k, ends, lengths, table


def _call(q, w, k, ends, lengths, table, *, coarse_count, raw=True, modify=False):
    return torch.ops._C_ascend.npu_indexer_coarse_screen(
        q,
        w,
        k,
        actual_seq_lengths_query=ends,
        actual_seq_lengths_key=lengths,
        block_table=table,
        coarse_count=coarse_count,
        proxy_mode=int(raw),
        is_modify_localwindow=modify,
    )


@pytest.mark.parametrize("width", [2048, 4096])
@torch.inference_mode()
def test_coarse_screen_raw_and_precomputed_proxies_match(inputs, width):
    q, w, k, ends, lengths, table = inputs
    candidates, aslk = _call(*inputs, coarse_count=width)
    proxy_q = q.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(q.dtype)
    proxy_w = w.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(w.dtype)
    proxy, proxy_aslk = _call(proxy_q, proxy_w, k, ends, lengths, table, coarse_count=width, raw=False)
    torch.testing.assert_close(candidates, proxy, rtol=0, atol=0)
    torch.testing.assert_close(aslk.cpu(), torch.full((2,), width, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(aslk, proxy_aslk, rtol=0, atol=0)
    window = 2 * GROUP_SIZE - 1
    torch.testing.assert_close(
        candidates[:, -window:].cpu(),
        torch.arange(KV_LENGTH - window, KV_LENGTH, dtype=torch.int32).expand(2, -1),
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_coarse_screen_2048_with_window_matches_pivot_reuse(inputs):
    q, w, k, ends, lengths, table = inputs
    reuse = torch.ops._C_ascend.npu_pivot_lightning_indexer
    expected, _ = reuse(
        q,
        k,
        w,
        actual_seq_lengths_query=ends,
        actual_seq_lengths_key=lengths,
        block_table=table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
    )
    candidates, aslk = _call(*inputs, coarse_count=2048, modify=True)
    torch.testing.assert_close(candidates, expected.squeeze(1), rtol=0, atol=0)
    torch.testing.assert_close(aslk.cpu(), torch.full((len(q),), 2048, dtype=torch.int32), rtol=0, atol=0)


@torch.inference_mode()
def test_coarse_screen_4096_replaces_the_reused_tail(inputs):
    candidates, aslk = _call(*inputs, coarse_count=4096, modify=True)
    candidates = candidates.cpu().reshape(-1, GROUP_SIZE, 4096)
    first_visible = KV_LENGTH - GROUP_SIZE + 1
    for row in range(1, GROUP_SIZE):
        torch.testing.assert_close(candidates[:, row, :-row], candidates[:, 0, :-row], rtol=0, atol=0)
        expected = torch.arange(first_visible, first_visible + row, dtype=torch.int32)
        torch.testing.assert_close(candidates[:, row, -row:], expected.expand(len(candidates), -1), rtol=0, atol=0)
    torch.testing.assert_close(aslk.cpu(), torch.full((2 * GROUP_SIZE,), 4096, dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("width", [2048, 4096])
@torch.inference_mode()
def test_coarse_screen_empty_kv_returns_empty_token_pools(inputs, width):
    q, w, k, ends, lengths, table = inputs
    candidates, aslk = _call(q, w, k, ends, torch.zeros_like(lengths), table, coarse_count=width, modify=True)
    torch.testing.assert_close(candidates.cpu(), torch.full((len(q), width), -1, dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(aslk.cpu(), torch.zeros(len(q), dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("modify", [False, True])
@torch.inference_mode()
def test_li_v2_prefill_matches_coarse_selection(width, modify):
    if not 220 <= torch_npu.npu.get_soc_version() <= 225:
        pytest.skip("requires the arch22 LIv2 backend")
    torch.manual_seed(20261119)
    batch = 2
    # Include a partial second chunk and a different key length per request.
    v2_topk_alignment = 1024
    chunk_groups = v2_topk_alignment // GROUP_SIZE
    groups_per_request = chunk_groups + 5
    query_length = groups_per_request * GROUP_SIZE
    q = torch.randn(batch * query_length, HEADS, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    w = torch.rand(batch * query_length, HEADS, device="npu", dtype=q.dtype)
    pages_per_request = KV_LENGTH // PAGE_SIZE
    k = torch.randn(batch * pages_per_request, PAGE_SIZE, 1, HEAD_DIM, device="npu", dtype=q.dtype)
    table = torch.arange(batch * pages_per_request, device="npu", dtype=torch.int32).reshape(batch, -1)
    ends = torch.arange(1, batch + 1, device="npu", dtype=torch.int32) * query_length
    lengths = torch.tensor([KV_LENGTH, KV_LENGTH - PAGE_SIZE], device="npu", dtype=torch.int32)
    arguments = dict(
        actual_seq_lengths_query=ends,
        actual_seq_lengths_key=lengths,
        block_table=table,
        coarse_count=width,
        is_modify_localwindow=modify,
        prefill_min_prefix_length=KV_LENGTH - PAGE_SIZE - query_length,
    )
    op = torch.ops._C_ascend.npu_indexer_coarse_screen
    candidates, aslk = op(q, w, k, proxy_mode=1, **arguments)
    count = batch * (query_length if modify else groups_per_request)
    assert candidates.shape == (count, width)
    torch.testing.assert_close(aslk.cpu(), torch.full((count,), width, dtype=torch.int32), rtol=0, atol=0)

    windows = 0 if modify else 2 * GROUP_SIZE - 1
    proxy_q = q.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(q.dtype)
    proxy_w = w.unflatten(0, (-1, GROUP_SIZE)).float().mean(1).to(w.dtype)
    if not modify:
        precomputed, _ = op(proxy_q, proxy_w, k, proxy_mode=0, **arguments)
        torch.testing.assert_close(candidates, precomputed, rtol=0, atol=0)

    # Probe both sides of the LIv2 chunk boundary in each request.
    for request, key_length in enumerate(lengths.tolist()):
        request_key = k[request * pages_per_request : (request + 1) * pages_per_request].flatten(0, 1).squeeze(1)
        for group in (0, chunk_groups - 1, chunk_groups, groups_per_request - 1):
            proxy_row = request * groups_per_request + group
            first_visible = key_length - query_length + group * GROUP_SIZE + 1
            history = first_visible if modify else first_visible - GROUP_SIZE
            src_row = proxy_row * GROUP_SIZE if modify else proxy_row
            row = candidates[src_row].cpu()
            picked = row[: width - windows].long()
            assert torch.unique(picked).numel() == len(picked)
            assert ((picked >= 0) & (picked < history)).all()
            scores = (request_key[:history].float() @ proxy_q[proxy_row].float().T).relu() @ proxy_w[proxy_row].float()
            scores = scores.cpu()
            cutoff = scores.topk(len(picked)).values[-1]
            torch.testing.assert_close(scores[picked].min(), cutoff, rtol=2e-5, atol=1e-4)
            if modify:
                for offset in range(1, GROUP_SIZE):
                    reused = candidates[src_row + offset].cpu()
                    torch.testing.assert_close(reused[:-offset], row[:-offset], rtol=0, atol=0)
                    torch.testing.assert_close(
                        reused[-offset:],
                        torch.arange(first_visible, first_visible + offset, dtype=torch.int32),
                        rtol=0,
                        atol=0,
                    )
            else:
                torch.testing.assert_close(
                    row[-windows:], torch.arange(history, history + windows, dtype=torch.int32), rtol=0, atol=0
                )
