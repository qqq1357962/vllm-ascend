# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.ops.fused_moe.small_token_boundary import (
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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("bad_id", [-5, 4])
def test_cpu_fallback_retains_invalid_index_errors(enabled, bad_id):
    with pytest.raises(IndexError):
        mask_local_expert_weights(torch.ones(1, 1), torch.tensor([[bad_id]]), torch.arange(4), enabled=enabled)


@pytest.mark.parametrize("tokens", [0, 1, 4, 32, 33])
@pytest.mark.parametrize("enabled", [False, True])
def test_add_preserves_inputs_and_output_storage(tokens, enabled):
    shared = torch.randn(8, tokens).T
    routed = torch.randn_like(shared)
    before_shared, before_routed = shared.clone(), routed.clone()
    output = add_shared_expert_output(shared, routed, enabled=enabled)
    assert_exact(output, torch._foreach_add([shared], [routed])[0])
    assert torch.equal(shared, before_shared)
    assert torch.equal(routed, before_routed)
    if tokens:
        assert output.data_ptr() not in (shared.data_ptr(), routed.data_ptr())


def test_config_switches_default_off():
    from vllm_ascend.ascend_config import AscendConfig

    config = AscendConfig(sparse_kv_offload_config=None)
    assert config.enable_moe_expert_map_fusion is False
    assert config.enable_moe_shared_output_add is False
    enabled = AscendConfig(
        sparse_kv_offload_config=None, enable_moe_expert_map_fusion=True, enable_moe_shared_output_add=True
    )
    assert enabled.enable_moe_expert_map_fusion is True
    assert enabled.enable_moe_shared_output_add is True
