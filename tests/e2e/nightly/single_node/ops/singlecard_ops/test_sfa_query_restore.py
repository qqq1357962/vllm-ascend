# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP query materialization, fallback and graph storage lifetimes."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sfa_query_restore import materialize_sfa_queries


def make_queries(rows=8, heads=64, latent=512, rotary=64, dtype=torch.bfloat16, device="npu"):
    fused = torch.randn(heads, rows, latent + rotary, dtype=dtype, device=device)
    return fused.transpose(0, 1).split((latent, rotary), dim=-1)


def assert_bits_equal(actual, expected):
    # Compare raw BF16 bits on CPU, including signed zero and NaN payloads.
    assert torch.equal(actual.cpu().contiguous().view(torch.int16), expected.cpu().contiguous().view(torch.int16))


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 16, 64])
@pytest.mark.parametrize("head_major", [True, False])
def test_materialize_queries(rows, head_major):
    q, qr = make_queries(rows)
    if not head_major:
        packed = torch.cat((q, qr), dim=-1)
        q, qr = packed.split((512, 64), dim=-1)
    rq, rr = materialize_sfa_queries(q, qr)
    assert rq.is_contiguous() and rr.is_contiguous()
    assert_bits_equal(rq, q)
    assert_bits_equal(rr, qr)


def test_materialize_special_values():
    q, qr = make_queries()
    # +0, -0, infinities, NaNs with distinct payloads, and subnormal values.
    values = torch.tensor([0, -32768, 32640, -128, 32705, -63, 1, -32767], dtype=torch.int16).view(torch.bfloat16).npu()
    q[..., :8].copy_(values)
    qr[..., :8].copy_(values)
    rq, rr = materialize_sfa_queries(q, qr)
    assert_bits_equal(rq, q)
    assert_bits_equal(rr, qr)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "too_many_rows",
        "heads",
        "latent",
        "rotary",
        "rank",
        "rope_shape",
        "no_rope",
        "dtype",
        "rope_dtype",
        "cpu",
        "rope_device",
        "q_inner_stride",
        "rope_inner_stride",
        "contiguous",
        "q_contiguous",
        "rope_contiguous",
    ],
)
def test_unsupported_inputs_return_original_objects(case):
    options = {
        "empty": {"rows": 0},
        "too_many_rows": {"rows": 65},
        "heads": {"heads": 32},
        "latent": {"latent": 256},
        "rotary": {"rotary": 32},
        "dtype": {"dtype": torch.float16},
        "cpu": {"device": "cpu"},
    }
    q, qr = make_queries(**options.get(case, {}))
    if case == "rank":
        q = q[0]
    elif case == "rope_shape":
        qr = qr[:1]
    elif case == "no_rope":
        qr = None
    elif case == "rope_dtype":
        qr = qr.float()
    elif case == "rope_device":
        qr = qr.cpu()
    elif case == "q_inner_stride":
        q = torch.empty(8, 64, 1024, dtype=q.dtype, device=q.device)[..., ::2]
    elif case == "rope_inner_stride":
        qr = torch.empty(8, 64, 128, dtype=qr.dtype, device=qr.device)[..., ::2]
    elif case == "contiguous":
        q, qr = q.contiguous(), qr.contiguous()
    elif case == "q_contiguous":
        q = q.contiguous()
    elif case == "rope_contiguous":
        qr = qr.contiguous()
    with patch("vllm_ascend.ops.triton.sfa_query_restore._restore_query") as kernel:
        rq, rr = materialize_sfa_queries(q, qr)
    assert rq is q and rr is qr
    kernel.__getitem__.assert_not_called()


def test_independent_graphs_shapes_and_current_inputs():
    inputs = [make_queries(rows) for rows in (3, 8, 64, 8)]
    graphs, outputs = [], []
    for q, qr in inputs:
        materialize_sfa_queries(q, qr)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = materialize_sfa_queries(q, qr)
        graphs.append(graph)
        outputs.append(result)
    assert len({tensor.data_ptr() for pair in outputs for tensor in pair}) == 2 * len(outputs)
    for magnitude in (0.5, -1.0, 0.0):
        for graph, (q, qr), result in zip(graphs, inputs, outputs):
            q.mul_(magnitude)
            qr.add_(magnitude)
            graph.replay()
            torch.npu.synchronize()
            for actual, reference in zip(result, (q, qr)):
                assert_bits_equal(actual, reference)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    torch.npu.synchronize()
    concurrent = []
    for stream, args in zip(streams, inputs):
        with torch.npu.stream(stream):
            concurrent.append(materialize_sfa_queries(*args))
    for stream in streams:
        stream.synchronize()
    for result, expected in zip(concurrent, inputs):
        for actual, reference in zip(result, expected):
            assert_bits_equal(actual, reference)


def make_attention_inputs(requests):
    q, qr = make_queries()
    kv = torch.randn(64, 128, 1, 512, dtype=torch.bfloat16, device="npu")
    kr = torch.randn(64, 128, 1, 64, dtype=torch.bfloat16, device="npu")
    indices = torch.full((8, 1, 2048), -1, dtype=torch.int32)
    for row in range(8):
        indices[row, 0, :256] = torch.randperm(8192)[:256].int()
    bt = torch.stack([torch.randperm(64, dtype=torch.int32) for _ in range(requests)]).npu()
    lengths = torch.tensor([8 // requests * (i + 1) for i in range(requests)], dtype=torch.int32).npu()
    klen = torch.full((requests,), 8192, dtype=torch.int32, device="npu")
    return q, qr, kv, kr, indices.npu(), bt, lengths, klen


def run_attention(args, enabled, dcp=True, remap_rank=None, interleave=128):
    # Exercise the real DCP decode entry and device adapter. Output reduction
    # is bypassed; callers can exercise the existing Triton index remap too.
    from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl, DCPGatherContext

    q, qr, kv, kr, indices, bt, qlen, klen = args
    if not dcp:
        from vllm_ascend.device.device_op import BaseDeviceAdaptor

        if enabled:
            q, qr = materialize_sfa_queries(q, qr)
        return BaseDeviceAdaptor.execute_sparse_flash_attention_process(
            SimpleNamespace(scale=1 / 24),
            q,
            qr,
            (kv, kr),
            indices,
            None,
            qlen,
            klen,
            block_table=bt,
            sparse_mode=0,
            return_lse=False,
        )
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.enable_sparse_sfa_turboquant = False
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sfa_dcp_query_restore = enabled
    impl.scale = 1 / 24
    impl.dcp_group = object()
    impl._has_prefill = lambda _: False
    if remap_rank is None:
        impl._remap_sparse_indices = lambda value: value
    else:
        impl.dcp_size = 8
        impl.dcp_rank = remap_rank
        impl._dcp_interleave_size = interleave
        impl._dcp_index_topk = indices.shape[-1]
    impl._merge_dcp_outputs = lambda output, *_: output
    gathered = q.as_strided((64, 8, 576), (8 * 576, 576, 1))
    context = SimpleNamespace(
        gather_context=DCPGatherContext(gathered, None, (1, 0, 2), (512, 64)),
        block_table=bt,
        seq_lens=klen,
    )
    metadata = SimpleNamespace(dcp_context=context)
    with patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_force_tmajor_restore", return_value=False):
        return impl._execute_sparse_flash_attention_process(q, qr, (kv, kr), indices, metadata, qlen, klen)


@pytest.mark.parametrize("requests", [1, 2])
@pytest.mark.parametrize("dcp", [False, True])
def test_complete_sfa_replay(requests, dcp):
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    args = make_attention_inputs(requests)
    # Some public CANN packages cannot execute the existing PA + LSE path.
    # Probe the unmodified control before testing the optional materialization.
    try:
        run_attention(args, False, dcp)
    except RuntimeError as error:
        if dcp and "return_softmax_lse" in str(error) and "PA_BSND" in str(error):
            pytest.skip("Installed CANN SFA does not support the existing paged DCP LSE path")
        raise
    for _ in range(3):
        run_attention(args, True, dcp)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run_attention(args, True, dcp)
    q, qr, kv, kr, indices, bt, qlen, klen = args
    for step, active in enumerate((8, 1, 4, 0)):
        q.mul_(0.9)
        qr.mul_(0.8)
        kv.mul_(0.7)
        kr.mul_(0.6)
        indices[:, :, :256].remainder_(4096)
        indices[:, :, :256].copy_(indices[:, :, :256].flip(-1))
        bt.copy_(bt.flip(-1))
        klen.fill_(4096)
        if requests == 1:
            qlen.fill_(active)
        else:
            active = 8
            qlen.copy_(torch.tensor([3 + step % 2, 8], dtype=torch.int32).npu())
        if step == 2:
            indices[0].fill_(-1)
        expected = run_attention(args, False, dcp)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result[:active], expected[:active], atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("rank,interleave", [(0, 128), (7, 1)])
def test_query_restore_with_existing_remap(rank, interleave):
    from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton
    from vllm_ascend.utils import enable_custom_op

    enable_custom_op()
    args = make_attention_inputs(requests=2)
    q, qr, kv, kr, indices, bt, qlen, klen = args
    # Probe the existing paged LSE boundary before introducing either kernel.
    try:
        run_attention(args, False)
    except RuntimeError as error:
        if "return_softmax_lse" in str(error) and "PA_BSND" in str(error):
            pytest.skip("Installed CANN SFA does not support the existing paged DCP LSE path")
        raise
    for _ in range(3):
        run_attention(args, True, remap_rank=rank, interleave=interleave)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run_attention(args, True, remap_rank=rank, interleave=interleave)
    generator = torch.Generator().manual_seed(1234 + rank)
    for step in range(3):
        # Mixed owners, interior invalid entries, and an entirely empty row.
        global_indices = torch.full(indices.shape, -1, dtype=indices.dtype)
        global_indices[..., :512] = torch.randint(65536, (8, 1, 512), generator=generator, dtype=indices.dtype)
        global_indices[..., ::13] = -1
        global_indices[step].fill_(-1)
        local_indices = torch.full_like(global_indices, -1)
        for source, target in zip(global_indices.view(8, -1), local_indices.view(8, -1)):
            owned = source[(source >= 0) & ((source // interleave) % 8 == rank)]
            target[: owned.numel()] = (owned // (8 * interleave)) * interleave + owned % interleave
        indices.copy_(global_indices)
        actual_indices = remap_sparse_indices_triton(indices, 8, rank, interleave)
        assert torch.equal(actual_indices.cpu(), local_indices)
        q.mul_(0.9)
        qr.mul_(0.8)
        kv.mul_(0.7)
        kr.mul_(0.6)
        bt.copy_(bt.flip(-1))
        qlen.copy_(torch.tensor([3 + step % 2, 8], dtype=torch.int32).npu())
        # Compare the composed kernels to SFA with independently remapped
        # indices and its original query materialization path.
        control_args = (q, qr, kv, kr, local_indices.npu(), bt, qlen, klen)
        expected = run_attention(control_args, False)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(indices.cpu(), global_indices)
        torch.testing.assert_close(result, expected, atol=0, rtol=0, equal_nan=True)
