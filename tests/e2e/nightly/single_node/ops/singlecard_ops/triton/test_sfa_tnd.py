# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.sfa_tnd import try_physical_tnd_sfa


def _inputs():
    torch.manual_seed(20261020)
    n, h, page, pages, logical_length = 16, 8, 128, 32, 3072
    q = torch.randn(n, h, 512, device="npu", dtype=torch.bfloat16)
    qr = torch.randn(n, h, 64, device="npu", dtype=torch.bfloat16)
    kv = torch.randn(pages, page, 1, 512, device="npu", dtype=torch.bfloat16)
    kr = torch.randn(pages, page, 1, 64, device="npu", dtype=torch.bfloat16)
    # A larger physical pool and descending physical pages expose accidental
    # use of logical context length / physical order for TND visibility.
    bt = torch.arange(pages - 1, 7, -1, device="npu", dtype=torch.int32)[None].contiguous()
    ids = torch.stack([torch.randperm(2800, device="npu", dtype=torch.int32)[:2048] for _ in range(n)])[:, None, :]
    qlen = torch.tensor([n], device="npu", dtype=torch.int32)
    klen = torch.tensor([logical_length], device="npu", dtype=torch.int32)
    return [q, qr, kv, kr, ids, bt, qlen, klen]


def _reference(args, mode):
    q, qr, kv, kr, ids, bt, qlen, klen = [t.cpu() for t in args]
    n = int(qlen[0])
    rows = []
    for row in range(n):
        logical = ids[row, 0].long()
        valid = (logical >= 0) & (logical < klen[0])
        if mode == 3:
            valid &= logical <= klen[0] - n + row
        logical = logical[valid]
        slots = bt[0, logical // 128].long() * 128 + logical % 128
        k = kv.view(-1, 512)[slots].double()
        r = kr.view(-1, 64)[slots].double()
        score = (q[row].double() @ k.T + qr[row].double() @ r.T) / 24
        rows.append(torch.softmax(score, dim=-1) @ k)
    return torch.stack(rows)


@pytest.mark.parametrize("mode", [0, 3])
@torch.inference_mode()
def test_independent_indices_with_larger_nonmonotonic_pool(mode):
    args = _inputs()
    # Independent, unsorted query rows; no shared2045 premise.
    got = try_physical_tnd_sfa(torch_npu.npu_sparse_flash_attention, *args, 1 / 24, sparse_mode=mode)
    assert got is not None
    expected = _reference(args, mode)
    error = (got[0].cpu().double() - expected).norm() / expected.norm()
    assert torch.isfinite(got[0]).all()
    assert error < 0.006


@torch.inference_mode()
def test_graph_updates_indices_lengths_pages_and_values():
    args = _inputs()

    def call():
        return try_physical_tnd_sfa(torch_npu.npu_sparse_flash_attention, *args, 1 / 24, sparse_mode=3)

    for _ in range(3):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = call()
    for _ in range(2):
        args[0].normal_()
        args[1].normal_()
        args[2].normal_()
        args[3].normal_()
        args[4].copy_(
            torch.stack([torch.randperm(2800, device="npu", dtype=torch.int32)[:2048] for _ in range(16)])[:, None, :]
        )
        args[5].copy_(args[5].flip(-1))
        args[6].fill_(12)
        args[7].fill_(2944)
        graph.replay()
        torch.npu.synchronize()
        expected = _reference(args, 3)
        actual = result[0][:12].cpu().double()
        assert (actual - expected).norm() / expected.norm() < 0.006


@torch.inference_mode()
def test_invalid_and_future_slots_are_masked_before_page_lookup():
    args = _inputs()
    args[4][:, :, 0] = -1
    args[4][:, :, 1] = 3071
    args[4][:, :, 2] = 3200
    # Inspect actual NPU map output. Attention with incomplete rows is outside
    # this fast path's contract and stays on PA in the production dispatcher.
    mapped, length = try_physical_tnd_sfa(
        lambda **kw: (kw["sparse_indices"], kw["actual_seq_lengths_kv"]),
        *args,
        1 / 24,
        sparse_mode=3,
    )
    ids, bt = args[4].cpu().long(), args[5].cpu().long()
    valid = (ids >= 0) & (ids < 3072) & (ids <= 3072 - 16 + torch.arange(16)[:, None, None])
    safe = ids.clamp(0, 3071)
    expected = torch.where(valid, bt[0, safe // 128] * 128 + safe % 128, -1).int()
    torch.testing.assert_close(mapped.cpu(), expected, rtol=0, atol=0)
    assert length.cpu().tolist() == [4096]


@pytest.mark.parametrize("change", ["batch", "strides", "dtype", "mode", "topk"])
@torch.inference_mode()
def test_unsupported_layout_returns_none(change):
    args = _inputs()
    mode = 3
    if change == "batch":
        args[5] = args[5].expand(2, -1).contiguous()
    elif change == "strides":
        args[2] = args[2].transpose(0, 1)
    elif change == "dtype":
        args[2] = args[2].to(torch.int8)
    elif change == "mode":
        mode = 1
    else:
        args[4] = args[4][:, :, :1024].contiguous()
    assert try_physical_tnd_sfa(None, *args, 1 / 24, sparse_mode=mode) is None
