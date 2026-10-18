# SPDX-License-Identifier: Apache-2.0
"""Opt-in execution changes for small MoE token capacities."""

import torch

MAX_SMALL_MOE_TOKENS = 32
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_INDEX_DTYPES = (torch.int32, torch.int64)


def mask_local_expert_weights(weights, expert_ids, expert_map, *, enabled=False):
    """Mask router-produced IDs without changing their routing decisions.

    The enabled NPU path requires valid IDs in [-E, E), where E is the map
    length. Routers and EPLB must satisfy this existing dispatch contract;
    invalid-ID error equivalence is not promised by the opt-in path. Arbitrary
    ownership, valid negative IDs and strided inputs remain supported.
    """
    if not (
        enabled
        and weights.device.type == "npu"
        and weights.device == expert_ids.device == expert_map.device
        and weights.ndim == expert_ids.ndim == 2
        and weights.shape == expert_ids.shape
        and weights.shape[0] <= MAX_SMALL_MOE_TOKENS
        and expert_map.ndim == 1
        and expert_map.numel() > 0
        and weights.dtype in _FLOAT_DTYPES
        and expert_ids.dtype in _INDEX_DTYPES
        and expert_map.dtype in _INDEX_DTYPES
    ):
        return weights * (expert_map[expert_ids] != -1)

    # Import only on the supported path, preserving non-Triton fallbacks.
    from vllm_ascend.ops.triton.moe_expert_map import fused_expert_map_weights

    return fused_expert_map_weights(weights, expert_ids, expert_map)


def add_shared_expert_output(shared_output, routed_output, *, enabled=False):
    """Join outputs after their existing stream waits, without input mutation."""
    if (
        enabled
        and shared_output.ndim == routed_output.ndim == 2
        and shared_output.shape == routed_output.shape
        and shared_output.shape[0] <= MAX_SMALL_MOE_TOKENS
        and shared_output.dtype == routed_output.dtype
        and shared_output.dtype in _FLOAT_DTYPES
    ):
        return torch.add(shared_output, routed_output)
    return torch._foreach_add([shared_output], [routed_output])[0]
