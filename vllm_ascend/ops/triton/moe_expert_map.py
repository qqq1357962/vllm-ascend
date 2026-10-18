# SPDX-License-Identifier: Apache-2.0
"""Expert-map gather, ownership comparison and weight multiplication."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _expert_map_weights_kernel(
    weights,
    expert_ids,
    expert_map,
    output,
    NUMEL: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    WEIGHT_STRIDE_ROW: tl.constexpr,
    WEIGHT_STRIDE_COL: tl.constexpr,
    ID_STRIDE_ROW: tl.constexpr,
    ID_STRIDE_COL: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows, columns = offsets // TOP_K, offsets % TOP_K
    ids = tl.load(expert_ids + rows * ID_STRIDE_ROW + columns * ID_STRIDE_COL, offsets < NUMEL, other=0)
    ids = tl.where(ids < 0, ids + NUM_EXPERTS, ids)
    # Bound every load even if a caller violates the valid-ID contract.
    mapped = tl.load(expert_map + ids * MAP_STRIDE, (offsets < NUMEL) & (ids >= 0) & (ids < NUM_EXPERTS), other=-1)
    values = tl.load(weights + rows * WEIGHT_STRIDE_ROW + columns * WEIGHT_STRIDE_COL, offsets < NUMEL, other=0)
    # Do not use where(mask, value, 0): multiplication preserves signed zero
    # and the reference's NaN/Inf behavior, including inf * 0.
    result = values.to(tl.float32) * (mapped != -1).to(tl.float32)
    tl.store(output + offsets, result, offsets < NUMEL)


def fused_expert_map_weights(weights, expert_ids, expert_map):
    output = torch.empty(weights.shape, dtype=weights.dtype, device=weights.device)
    numel = weights.numel()
    if numel:
        block_size = min(256, max(32, triton.next_power_of_2(numel)))
        _expert_map_weights_kernel[(triton.cdiv(numel, block_size),)](
            weights,
            expert_ids,
            expert_map,
            output,
            numel,
            weights.shape[1],
            expert_map.numel(),
            *weights.stride(),
            *expert_ids.stride(),
            expert_map.stride(0),
            block_size,
        )
    return output
