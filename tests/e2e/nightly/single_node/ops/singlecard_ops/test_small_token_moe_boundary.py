# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.fused_moe.small_token_boundary import (
    _small_token_shared_add,
    add_shared_expert_output,
    mask_local_expert_weights,
)


def assert_exact(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    finite = torch.isfinite(expected)
    assert torch.equal(actual[finite], expected[finite])
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    valid = ~torch.isnan(expected)
    assert torch.equal(torch.signbit(actual[valid]), torch.signbit(expected[valid]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,top_k", [(0, 8), (1, 1), (3, 7), (4, 8), (8, 8), (32, 8), (33, 8)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_mask_strides_ownership_and_negative_ids(dtype, tokens, top_k, index_dtype):
    weights = torch.randn(top_k, tokens, device="npu", dtype=dtype).T
    ids = torch.randint(-256, 256, (top_k, tokens), device="npu", dtype=index_dtype).T
    mapping = torch.arange(512, device="npu", dtype=index_dtype)[::2]
    for ownership in ("mixed", "all", "none"):
        mapping.copy_(torch.arange(256, device="npu", dtype=index_dtype))
        if ownership == "mixed":
            mapping[::3] = -1
            mapping[1::7] = -2  # Only -1 means nonlocal.
        elif ownership == "none":
            mapping.fill_(-1)
        before = weights.clone(), ids.clone(), mapping.clone()
        output = mask_local_expert_weights(weights, ids, mapping, enabled=True)
        assert_exact(output, weights * (mapping[ids] != -1))
        for current, original in zip((weights, ids, mapping), before):
            assert torch.equal(current, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mask_special_values_and_graph_updates(dtype):
    values = [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"), 1.0e-8]
    weights = torch.tensor([values, values], dtype=dtype, device="npu")
    ids = torch.tensor([[64] * 8, [0] * 8], dtype=torch.int32, device="npu")
    mapping = torch.full((256,), -1, dtype=torch.int32, device="npu")
    mapping[64:96] = torch.arange(32, dtype=torch.int32, device="npu")
    for _ in range(3):
        mask_local_expert_weights(weights, ids, mapping, enabled=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = mask_local_expert_weights(weights, ids, mapping, enabled=True)
    for iteration in range(4):
        if iteration:
            weights.copy_(torch.randn_like(weights))
            ids.copy_(torch.randint(-256, 256, ids.shape, dtype=ids.dtype, device="npu"))
            mapping.copy_(torch.randint(-1, 32, mapping.shape, dtype=mapping.dtype, device="npu"))
        graph.replay()
        assert_exact(output, weights * (mapping[ids] != -1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 33])
@pytest.mark.parametrize("compiled_boundary", [False, True])
def test_shared_add_graph_replay_after_producer_stream(dtype, tokens, compiled_boundary):
    def combine(shared, routed):
        if compiled_boundary:
            return _small_token_shared_add(shared, routed)
        return add_shared_expert_output(shared, routed, enabled=True)

    shared = torch.randn(6144, tokens, device="npu", dtype=dtype).T
    routed = torch.randn_like(shared)
    for _ in range(3):
        combine(shared, routed)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = combine(shared, routed)
    producer = torch.npu.Stream()
    for _ in range(3):
        with torch.npu.stream(producer):
            shared.copy_(torch.randn_like(shared))
        torch.npu.current_stream().wait_stream(producer)
        routed.copy_(torch.randn_like(routed))
        saved_shared, saved_routed = shared.clone(), routed.clone()
        graph.replay()
        assert_exact(output, torch._foreach_add([shared], [routed])[0])
        assert torch.equal(shared, saved_shared)
        assert torch.equal(routed, saved_routed)
        assert output.data_ptr() not in (shared.data_ptr(), routed.data_ptr())
