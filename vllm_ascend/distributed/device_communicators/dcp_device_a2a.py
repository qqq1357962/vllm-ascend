# SPDX-License-Identifier: Apache-2.0
"""Optional single-node 910B2 DCP8 push/merge backend.

The DCP coordinator owns the workspace. Initialize before graph capture, use
one ordered execution stream on every rank, and close after the last replay.
All ranks must choose the same backend and issue calls in the same order.
"""

import socket

import torch
import torch.distributed as dist

RANKS = 8
MAX_TOKENS = 32
HEADS = 64
HEAD_DIM = 512
LOCAL_HEADS = HEADS // RANKS
WORKSPACE_BYTES = 10 << 20
EPOCH_OFFSET = 8 << 20
ERROR_OFFSET = (8 << 20) + 4096
WORKERS = 32


def supports_device_a2a(partial_output, softmax_lse, scatter_dim, return_lse, defer_combine):
    return (
        scatter_dim == 1
        and not return_lse
        and not defer_combine
        and partial_output.ndim == 3
        and 1 <= partial_output.shape[0] <= MAX_TOKENS
        and partial_output.shape[1:] == (HEADS, HEAD_DIM)
        and softmax_lse.shape == (*partial_output.shape[:2], 1)
        and partial_output.dtype == torch.bfloat16
        and softmax_lse.dtype == torch.float32
        and partial_output.device.type == "npu"
        and partial_output.device == softmax_lse.device
    )


def initialize_device_a2a(group, parallel_config, enabled):
    """Collectively initialize the explicitly selected backend before warmup."""
    if not enabled:
        return None
    if not (
        parallel_config.tensor_parallel_size == RANKS
        and parallel_config.decode_context_parallel_size == RANKS
        and parallel_config.prefill_context_parallel_size == 1
        and parallel_config.data_parallel_size == 1
        and parallel_config.pipeline_parallel_size == 1
        and dist.get_world_size() == RANKS
        and group.world_size == RANKS
    ):
        raise ValueError("enable_dcp_device_a2a requires TP8/DCP8/PCP1/DP1/PP1 on one 910B2 node.")
    if getattr(group, "dcp_device_a2a", None) is not None:
        raise RuntimeError("DCP device workspace is already initialized.")
    workspace = DcpDeviceA2A(group.cpu_group)
    group.dcp_device_a2a = workspace
    return workspace


def _load_device_a2a_ops():
    # Load the package's existing native extension after selecting the NPU.
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "dcp_device_a2a_status"):
        raise RuntimeError(
            "DCP device A2A is unavailable. Install the Ascend SHMEM SDK and rebuild vllm-ascend "
            "with VLLM_ASCEND_SHMEM_ROOT pointing to its installed prefix."
        )
    return torch.ops._C_ascend


class DcpDeviceA2A:
    """One SHMEM instance, symmetric heap and epoch history per DCP group.

    Initialization and close are collective. There is deliberately no destructor
    that could enter a collective during Python garbage collection.
    """

    def __init__(self, control_group):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Initialize DCP device A2A before graph capture.")
        self.control_group = control_group
        self.rank = dist.get_rank(control_group)
        self.device = torch.device("npu", torch.npu.current_device())
        self.closed = False
        self.buffer = None
        group_size = dist.get_world_size(control_group)
        peers = [None] * group_size
        # Preflight errors must reach every rank before entering SDK bootstrap.
        error = None
        try:
            if group_size != RANKS:
                raise ValueError("DCP device A2A requires eight ranks.")
            if "910B2" not in torch.npu.get_device_name(self.device):
                raise ValueError("DCP device A2A currently supports Ascend 910B2 only.")
            self.ops = _load_device_a2a_ops()
            if self.ops.dcp_device_a2a_status() != 0:
                raise RuntimeError("DCP device A2A requires exclusive ownership of the SHMEM SDK instance.")
        except (ImportError, OSError, AttributeError, ValueError, RuntimeError) as exc:
            error = str(exc)
        dist.all_gather_object(peers, (socket.gethostname(), error), group=control_group)
        errors = [message for _, message in peers if message]
        if errors or len({host for host, _ in peers}) != 1:
            raise RuntimeError(f"DCP device A2A preflight failed: {errors or 'multiple hosts'}")
        uid = [None]
        if self.rank == 0:
            try:
                uid[0] = (None, self.ops.dcp_device_a2a_unique_id())
            except RuntimeError as exc:
                uid[0] = (str(exc), None)
        # The unique ID contains the SDK's allocated bootstrap endpoint; no
        # fixed TCP port or user-managed port reservation is needed.
        dist.broadcast_object_list(uid, src=dist.get_global_rank(control_group, 0), group=control_group)
        error, raw = uid[0]
        if error:
            raise RuntimeError(f"DCP device A2A unique ID failed: {error}")
        self.buffer, addresses = self.ops.dcp_device_a2a_init(self.rank, raw)
        self.peers = torch.tensor(addresses, dtype=torch.int64, device=self.device)
        torch.npu.synchronize(self.device)
        dist.barrier(group=control_group)

    def run(self, partial_output, softmax_lse):
        if self.closed:
            raise RuntimeError("DCP device workspace is closed.")
        if not supports_device_a2a(partial_output, softmax_lse, 1, False, False):
            raise ValueError("DCP device A2A expects BF16 [T,64,512], FP32 [T,64,1], 1 <= T <= 32.")
        if partial_output.device != self.device:
            raise ValueError("DCP input device differs from the workspace device.")
        partial_output = partial_output.contiguous()
        softmax_lse = softmax_lse.contiguous()
        return self.ops.dcp_device_a2a(partial_output, softmax_lse, self.peers, self.buffer, self.rank)

    def check_status(self):
        """Explicit diagnostic synchronization; never called in the hot path."""
        if self.closed:
            raise RuntimeError("DCP device workspace is closed.")
        torch.npu.synchronize(self.device)
        errors = self.buffer[ERROR_OFFSET : ERROR_OFFSET + WORKERS * 64].view(torch.int32)[::16].cpu().tolist()
        failed = [(worker, source - 1) for worker, source in enumerate(errors) if source]
        if failed:
            raise RuntimeError(f"DCP device A2A flag timeout (worker, source): {failed}")

    def close(self):
        if self.closed:
            return
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.control_group)
        self.ops.dcp_device_a2a_close(self.buffer)
        self.buffer = None
        self.closed = True
