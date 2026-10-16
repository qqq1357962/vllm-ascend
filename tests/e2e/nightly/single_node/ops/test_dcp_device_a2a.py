# SPDX-License-Identifier: Apache-2.0
"""Eight-rank hardware regression, run explicitly with torchrun.

See the DCP device exchange configuration guide for the build and run commands.
The optional benchmark prints paired current-main HCCL / device graph timings.
"""

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.distributed.device_communicators.dcp_device_a2a import EPOCH_OFFSET, initialize_device_a2a
from vllm_ascend.ops.triton.dcp import dcp_a2a
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def _reference(parts, lse, rank, group):
    gathered = [None] * 8
    dist.all_gather_object(gathered, (parts.cpu(), lse.cpu()), group=group)
    outputs = torch.stack([o[:, rank * 8 : (rank + 1) * 8].float() for o, _ in gathered])
    lses = torch.stack([lse_part[:, rank * 8 : (rank + 1) * 8] for _, lse_part in gathered])
    valid = torch.isfinite(lses)
    weights = torch.softmax(lses.masked_fill(~valid, float("-inf")), dim=0).nan_to_num(nan=0.0)
    return (torch.where(valid, outputs, 0.0) * weights).sum(0)


def _check(actual, reference):
    actual = actual.cpu().float()
    assert torch.isfinite(actual).all()
    error = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
    assert error < 0.004, error
    return float(error)


def _seed_wrap(workspace):
    # Seed all worker counters together before the first invocation. Both flag
    # slots are still zero; four calls exercise fffffffd, fffffffe, ffffffff, 0.
    workspace.buffer[EPOCH_OFFSET : EPOCH_OFFSET + 32 * 64].view(torch.int32)[::16].fill_(-4)
    torch.npu.synchronize()
    dist.barrier(group=workspace.control_group)


def _assert_wrapped(workspace):
    counters = workspace.buffer[EPOCH_OFFSET : EPOCH_OFFSET + 32 * 64].view(torch.int32)[::16].cpu()
    assert (counters == 0).all()


def _capture(call, calls):
    for _ in range(3):
        call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            result = call()
    graph.replay()
    torch.npu.synchronize()
    return graph, result


def _benchmark(graphs, group, calls):
    rounds = []
    for round_idx in range(3):
        measurements = {}
        order = ("hccl", "device") if round_idx % 2 == 0 else ("device", "hccl")
        for name in order:
            times = []
            for _ in range(21):
                dist.barrier(group=group)
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graphs[name][0].replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000 / calls)
            ranks = [None] * 8
            dist.all_gather_object(ranks, statistics.median(times), group=group)
            measurements[name] = max(ranks)
        rounds.append(measurements)
    return {name: statistics.median(r[name] for r in rounds) for name in ("hccl", "device")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_device_properties_triton()
    init_distributed_environment(
        world_size=8, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group([list(range(8))], local_rank, "hccl", group_name="dcp_device_test")
    parallel_config = SimpleNamespace(
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        prefill_context_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
    )
    workspace = initialize_device_a2a(group, parallel_config, True)
    _seed_wrap(workspace)
    results = []
    benchmark_tokens = (1, 4, 8, 16, 32, 64, 128)
    small_graph = small_result = small_parts = small_lse = None
    for tokens in (1, 4, 8, 16, 32, 33, 64, 65, 127, 128):
        generator = torch.Generator().manual_seed(348 + rank + tokens)
        parts_cpu = torch.randn(tokens, 64, 512, generator=generator).to(torch.bfloat16)
        lse_cpu = torch.randn(tokens, 64, 1, generator=generator) * 4
        parts, lse = parts_cpu.npu(), lse_cpu.npu()

        def candidate(parts=parts, lse=lse):
            return torch.ops.vllm.dcp_a2a_fused(parts, lse, 8, 1, group.unique_name)

        def baseline(parts=parts, lse=lse):
            return dcp_a2a.dcp_a2a_fused_combine(parts, lse, 8, 1, group.device_group)

        reference = _reference(parts, lse, rank, group.cpu_group)
        for _ in range(4):
            _check(candidate(), reference)
        if tokens == 1:
            _assert_wrapped(workspace)
        calls = 32
        graphs = {"hccl": _capture(baseline, calls), "device": _capture(candidate, calls)}
        if tokens == 1:
            small_graph, small_result = graphs["device"]
            small_parts, small_lse = parts, lse
        for label in ("dynamic", "one_empty", "seven_empty", "all_empty", "wide_lse", "nonfinite"):
            parts.copy_(parts_cpu + rank + 1)
            lse.copy_(lse_cpu - rank)
            empty = (label == "one_empty" and rank == 3) or (label == "seven_empty" and rank != 3)
            if empty or label == "all_empty":
                parts.fill_(float("nan"))
                lse.fill_(float("-inf"))
            if label == "wide_lse":
                lse.add_(10000)
            if label == "nonfinite" and rank < 2:
                parts.fill_(float("nan"))
                lse.fill_(float("nan") if rank == 0 else float("inf"))
            reference = _reference(parts, lse, rank, group.cpu_group)
            for name in ("hccl", "device"):
                graphs[name][0].replay()
                _check(graphs[name][1], reference)
            workspace.check_status()
        parts.copy_(parts_cpu)
        lse.copy_(lse_cpu)
        reference = _reference(parts, lse, rank, group.cpu_group)
        # Padded noncontiguous views must include the adapter's copy and match.
        parts_view = parts.transpose(0, 1).contiguous().transpose(0, 1)
        lse_view = lse.transpose(0, 1).contiguous().transpose(0, 1)
        _check(workspace.run(parts_view, lse_view), reference)
        dist.barrier(group=group.cpu_group)
        if rank == 3:
            time.sleep(0.02)
        _check(candidate(), reference)
        if args.benchmark and tokens in benchmark_tokens:
            results.append({"tokens": tokens, **_benchmark(graphs, group.cpu_group, calls)})
        if rank == 0:
            print(f"DCP_DEVICE_A2A_SHAPE_OK T={tokens}", flush=True)
        del graphs
    # Reuse an older small-shape graph after the maximum-size graph has changed
    # the shared epoch/slot history. Its addresses remain fixed throughout.
    small_parts.fill_(rank + 1)
    small_lse.fill_(-rank)
    reference = _reference(small_parts, small_lse, rank, group.cpu_group)
    small_graph.replay()
    _check(small_result, reference)
    workspace.check_status()
    workspace.close()
    group.dcp_device_a2a = None
    if rank == 0:
        print("DCP_DEVICE_A2A_RESULT " + json.dumps(results), flush=True)
    group.destroy()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
