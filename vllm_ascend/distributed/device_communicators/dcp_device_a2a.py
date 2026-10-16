# SPDX-License-Identifier: Apache-2.0
"""Optional single-node 910B2 DCP8 push/merge backend.

The DCP coordinator owns the workspace. Initialize before graph capture, use
one ordered execution stream on every rank, and close after the last replay.
All ranks must choose the same backend and issue calls in the same order.
"""

import ctypes
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist

RANKS = 8
MAX_TOKENS = 32
HEADS = 64
HEAD_DIM = 512
LOCAL_HEADS = HEADS // RANKS
WORKSPACE_BYTES = 10 << 20
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


def initialize_device_a2a(group, parallel_config, library):
    """Collectively initialize the explicitly selected backend before warmup."""
    if library is None:
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
        raise ValueError("dcp_device_a2a_library requires TP8/DCP8/PCP1/DP1/PP1 on one 910B2 node.")
    if getattr(group, "dcp_device_a2a", None) is not None:
        raise RuntimeError("DCP device workspace is already initialized.")
    workspace = DcpDeviceA2A(library, group.cpu_group)
    group.dcp_device_a2a = workspace
    return workspace


class DcpDeviceA2A:
    """One SHMEM instance, symmetric heap and epoch history per DCP group.

    Initialization and close are collective. There is deliberately no destructor
    that could enter a collective during Python garbage collection.
    """

    def __init__(self, library: str, control_group):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Initialize DCP device A2A before graph capture.")
        self.control_group = control_group
        self.rank = dist.get_rank(control_group)
        self.device = torch.device("npu", torch.npu.current_device())
        self.closed = False
        self.local = None
        group_size = dist.get_world_size(control_group)
        peers = [None] * group_size
        # Preflight errors must reach every rank before entering SDK bootstrap.
        error = None
        try:
            if group_size != RANKS:
                raise ValueError("DCP device A2A requires eight ranks.")
            if "910B2" not in torch.npu.get_device_name(self.device):
                raise ValueError("DCP device A2A currently supports Ascend 910B2 only.")
            self.acl = ctypes.CDLL("libascendcl.so")
            # MemFabric exports overlapping Hybm symbols. Bind SHMEM internally
            # even when vLLM imported MemFabric first.
            self.lib = ctypes.CDLL(str(Path(library).resolve()), mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
            self._configure_abi()
            if self.lib.dcp_shmem_status() != 0:
                raise RuntimeError("DCP device A2A requires exclusive ownership of the SHMEM SDK instance.")
        except (OSError, AttributeError, ValueError, RuntimeError) as exc:
            error = str(exc)
        dist.all_gather_object(peers, (socket.gethostname(), error), group=control_group)
        errors = [message for _, message in peers if message]
        if errors or len({host for host, _ in peers}) != 1:
            raise RuntimeError(f"DCP device A2A preflight failed: {errors or 'multiple hosts'}")
        uid = [None]
        if self.rank == 0:
            buffer = ctypes.create_string_buffer(self.lib.dcp_unique_id_size())
            code = self.lib.dcp_get_unique_id(buffer)
            uid[0] = (code, buffer.raw)
        # The unique ID contains the SDK's allocated bootstrap endpoint; no
        # fixed TCP port or user-managed port reservation is needed.
        dist.broadcast_object_list(uid, src=dist.get_global_rank(control_group, 0), group=control_group)
        code, raw = uid[0]
        self._check(code, "get SHMEM unique ID")
        addresses = (ctypes.c_uint64 * RANKS)()
        self._check(self.lib.init_dcp_shmem(self.rank, raw, addresses), "initialize SHMEM")
        self.local = ctypes.c_void_p(addresses[self.rank])
        self._check(self.acl.aclrtMemset(self.local, WORKSPACE_BYTES, 0, WORKSPACE_BYTES), "clear workspace")
        self.peers = torch.tensor(list(addresses), dtype=torch.int64, device=self.device)
        torch.npu.synchronize(self.device)
        dist.barrier(group=control_group)

    def _configure_abi(self):
        self.lib.launch_dcp.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_uint32] * 2
        self.lib.launch_dcp.restype = None
        self.lib.dcp_unique_id_size.argtypes = []
        self.lib.dcp_unique_id_size.restype = ctypes.c_size_t
        self.lib.dcp_get_unique_id.argtypes = [ctypes.c_void_p]
        self.lib.dcp_get_unique_id.restype = ctypes.c_int
        self.lib.dcp_shmem_status.argtypes = []
        self.lib.dcp_shmem_status.restype = ctypes.c_int
        self.lib.init_dcp_shmem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        self.lib.init_dcp_shmem.restype = ctypes.c_int
        self.lib.close_dcp_shmem.argtypes = [ctypes.c_void_p]
        self.lib.close_dcp_shmem.restype = ctypes.c_int
        self.acl.aclrtMemset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t]
        self.acl.aclrtMemset.restype = ctypes.c_int
        self.acl.aclrtMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.acl.aclrtMemcpy.restype = ctypes.c_int

    @staticmethod
    def _check(code, operation):
        if code:
            raise RuntimeError(f"DCP device A2A: {operation} failed with code {code}.")

    def run(self, partial_output, softmax_lse):
        if self.closed:
            raise RuntimeError("DCP device workspace is closed.")
        if not supports_device_a2a(partial_output, softmax_lse, 1, False, False):
            raise ValueError("DCP device A2A expects BF16 [T,64,512], FP32 [T,64,1], 1 <= T <= 32.")
        if partial_output.device != self.device:
            raise ValueError("DCP input device differs from the workspace device.")
        partial_output = partial_output.contiguous()
        softmax_lse = softmax_lse.contiguous()
        output = torch.empty(
            partial_output.shape[0], LOCAL_HEADS, HEAD_DIM, dtype=partial_output.dtype, device=self.device
        )
        self.lib.launch_dcp(
            torch.npu.current_stream(self.device).npu_stream,
            partial_output.data_ptr(),
            softmax_lse.data_ptr(),
            output.data_ptr(),
            self.peers.data_ptr(),
            self.rank,
            partial_output.shape[0],
        )
        return output

    def check_status(self):
        """Explicit diagnostic synchronization; never called in the hot path."""
        if self.closed:
            raise RuntimeError("DCP device workspace is closed.")
        torch.npu.synchronize(self.device)
        errors = (ctypes.c_int32 * (WORKERS * 16))()
        size = ctypes.sizeof(errors)
        self._check(
            self.acl.aclrtMemcpy(errors, size, ctypes.c_void_p(self.local.value + ERROR_OFFSET), size, 2),
            "read device status",
        )
        failed = [(worker, errors[worker * 16] - 1) for worker in range(WORKERS) if errors[worker * 16]]
        if failed:
            raise RuntimeError(f"DCP device A2A flag timeout (worker, source): {failed}")

    def close(self):
        if self.closed:
            return
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.control_group)
        self._check(self.lib.close_dcp_shmem(self.local), "finalize SHMEM")
        self.closed = True
