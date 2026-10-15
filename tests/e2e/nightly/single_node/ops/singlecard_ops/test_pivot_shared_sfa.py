# SPDX-License-Identifier: Apache-2.0
"""Exact-support and replay regressions; requires PA SFA FP32 max/sum support."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.pivot_shared_sfa import try_shared_prefill
from vllm_ascend.ops.triton.pivot_shared_sfa import PivotSharedSFA
from vllm_ascend.utils import enable_custom_op


@pytest.fixture
def consumer():
    enable_custom_op()
    return PivotSharedSFA()


def fixture(rows, heads):
    torch.manual_seed(13579)
    context, page = 16384, 128
    q = torch.randn(rows, heads, 512, dtype=torch.bfloat16).npu()
    qr = torch.randn(rows, heads, 64, dtype=torch.bfloat16).npu()
    kv = torch.randn(context // page, page, 1, 512, dtype=torch.bfloat16).npu()
    kr = torch.randn(context // page, page, 1, 64, dtype=torch.bfloat16).npu()
    indices = torch.empty(rows, 1, 2048, dtype=torch.int32)
    for start in range(0, rows, 4):
        common = torch.randperm(context - 16)[:2045].int()
        for row in range(start, min(start + 4, rows)):
            indices[row, 0, :2045] = common
            indices[row, 0, -3:] = torch.tensor([context - 3 - row, context - 2 - row, context - 1 - row])
    bt = torch.randperm(context // page, dtype=torch.int32).reshape(1, -1).npu()
    return q, qr, kv, kr, indices.npu(), bt


def reference(q, qr, kv, kr, raw, bt, active, dcp, rank):
    q, qr, kv, kr = (t.cpu().double() for t in (q, qr, kv, kr))
    raw, bt = raw.cpu(), bt.cpu()[0].long()
    out = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], -torch.inf, dtype=torch.float64)
    for row in range(active):
        ids = raw[row, 0].long()
        if ids[0] < 0:
            continue
        ids = ids[(ids >= 0) & (ids // 128 % dcp == rank)]
        if ids.numel() == 0:
            continue
        local = ids // (dcp * 128) * 128 + ids % 128
        k = kv[bt[local // 128], local % 128, 0]
        r = kr[bt[local // 128], local % 128, 0]
        logits = (q[row] @ k.T + qr[row] @ r.T) / 24
        out[row] = logits.softmax(-1) @ k
        lse[row] = logits.logsumexp(-1)
    return out, lse


def check(result, args, active, dcp, rank):
    expected, expected_lse = reference(*args, active, dcp, rank)
    actual = result[0].cpu().double()
    observed_lse = (result[1] + result[2].log())[0].cpu().double()
    assert torch.isfinite(actual).all()
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
    assert error < 0.01
    valid = torch.isfinite(expected_lse)
    torch.testing.assert_close(observed_lse[valid], expected_lse[valid], atol=0.004, rtol=0)
    assert torch.isneginf(observed_lse[~valid]).all()
    assert torch.count_nonzero(actual[~valid]) == 0


@pytest.mark.parametrize(
    "rows,heads,dcp,rank", [(8, 8, 1, 0), (8, 64, 1, 0), (4, 64, 8, 0), (8, 64, 8, 3), (3, 64, 8, 7), (1, 64, 8, 0)]
)
def test_replay_reads_current_inputs(consumer, rows, heads, dcp, rank):
    args = fixture(rows, heads)
    q, qr, kv, kr, raw, bt = args
    if heads == 64:
        # Match the head-major DCP gather views consumed by serving.
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        qr = qr.transpose(0, 1).contiguous().transpose(0, 1)
        args = (q, qr, kv, kr, raw, bt)
    qlen = torch.tensor([rows], dtype=torch.int32).npu()
    klen = torch.tensor([16384 // dcp], dtype=torch.int32).npu()

    def run():
        return consumer(*args, klen, 1 / 24, dcp=dcp, rank=rank, actual_query_lengths=qlen)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    graph.replay()
    torch.npu.synchronize()
    check(result, args, rows, dcp, rank)
    # Change values at the same addresses, including physical page ownership.
    q.mul_(0.9)
    qr.mul_(0.8)
    kv.mul_(0.7)
    kr.mul_(0.6)
    raw[:, :, :2045].copy_(raw[:, :, :2045].flip(-1))
    bt.copy_(bt.flip(-1))
    # Stale positive indices in inactive rows must not survive graph replay.
    active = max(1, rows - 1)
    qlen.fill_(active)
    graph.replay()
    torch.npu.synchronize()
    check(result, args, active, dcp, rank)


def test_empty_rank_private_only_and_workspace_lifetime(consumer):
    args = fixture(8, 64)
    raw = args[4]
    qlen = torch.tensor([8], dtype=torch.int32).npu()
    klen = torch.tensor([2048], dtype=torch.int32).npu()
    owner_one = torch.tensor([i for i in range(16384) if i // 128 % 8 == 1], dtype=torch.int32).npu()
    raw.copy_(owner_one.reshape(1, 1, -1).expand_as(raw))
    previous = []
    for private in (False, True, False):
        if private:
            raw[:, :, -3:] = torch.tensor([0, 1, 2], dtype=torch.int32).npu()
        else:
            raw.copy_(owner_one.reshape(1, 1, -1).expand_as(raw))
        result = consumer(*args, klen, 1 / 24, dcp=8, rank=0, actual_query_lengths=qlen)
        torch.npu.synchronize()
        check(result, args, 8, 8, 0)
        for old, snapshot in previous:
            torch.testing.assert_close(old, snapshot, atol=0, rtol=0)
        previous.append((result[0], result[0].clone()))


@pytest.mark.parametrize("rows,context", [(16, 4090), (128, 8192)])
def test_serving_prefill_preserves_dense_prefix_and_private_tail(consumer, rows, context):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    args = fixture(rows, 8)
    q, qr, kv, kr, raw, bt = args
    # Make all selected indices causally visible without relying on a mask.
    raw.remainder_(context)
    qlen = torch.tensor([rows], dtype=torch.int32).npu()
    klen = torch.tensor([context + rows], dtype=torch.int32).npu()
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_actual_tokens=rows,
        max_query_len=rows,
        seq_lens_cpu=torch.tensor([context + rows]),
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    impl = SimpleNamespace(scale=1 / 24, pivot_shared_sfa=consumer)
    out = try_shared_prefill(impl, q, qr, (kv, kr), raw, metadata, qlen, klen, bt, False)
    assert out is not None
    expected, _ = reference(*args, rows, 1, 0)
    actual = out.cpu().double()
    assert (actual - expected).norm() / expected.norm() < 0.01
    # The serving gate must never capture a CPU context-length decision.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert try_shared_prefill(impl, q, qr, (kv, kr), raw, metadata, qlen, klen, bt, False) is None
