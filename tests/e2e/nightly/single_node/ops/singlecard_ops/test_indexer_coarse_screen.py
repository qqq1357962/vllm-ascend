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
