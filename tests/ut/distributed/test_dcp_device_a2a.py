# SPDX-License-Identifier: Apache-2.0

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.device_communicators import dcp_device_a2a as device_a2a


def _inputs(tokens=8, heads=64, dim=512, dtype=torch.bfloat16, lse_dtype=torch.float32):
    device = SimpleNamespace(type="npu")
    output = SimpleNamespace(ndim=3, shape=(tokens, heads, dim), dtype=dtype, device=device)
    lse = SimpleNamespace(shape=(tokens, heads, 1), dtype=lse_dtype, device=device)
    return output, lse


@pytest.mark.parametrize("tokens,expected", [(0, False), (1, True), (32, True), (33, False)])
def test_token_boundaries(tokens, expected):
    assert device_a2a.supports_device_a2a(*_inputs(tokens), 1, False, False) is expected


@pytest.mark.parametrize(
    "kwargs",
    [{"heads": 96}, {"dim": 256}, {"dtype": torch.float32}, {"lse_dtype": torch.bfloat16}],
)
def test_unsupported_layout_falls_back(kwargs):
    assert not device_a2a.supports_device_a2a(*_inputs(**kwargs), 1, False, False)


@pytest.mark.parametrize("scatter,return_lse,defer", [(0, False, False), (1, True, False), (1, False, True)])
def test_unsupported_modes_fall_back(scatter, return_lse, defer):
    assert not device_a2a.supports_device_a2a(*_inputs(), scatter, return_lse, defer)


def test_default_does_not_load_sdk(monkeypatch):
    load = Mock(side_effect=AssertionError("default must not load SHMEM"))
    monkeypatch.setattr(device_a2a.ctypes, "CDLL", load)
    assert device_a2a.initialize_device_a2a(None, None, None) is None
    load.assert_not_called()


def test_rejects_unsupported_topology():
    config = SimpleNamespace(tensor_parallel_size=4)
    with pytest.raises(ValueError, match="TP8/DCP8"):
        device_a2a.initialize_device_a2a(None, config, "library.so")


def test_graph_capture_cannot_initialize(monkeypatch):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        device_a2a.DcpDeviceA2A("library.so", None)


def test_closed_workspace_rejects_launch():
    workspace = object.__new__(device_a2a.DcpDeviceA2A)
    workspace.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        workspace.run(None, None)
    # Closing twice must not re-enter the SDK's collective finalization.
    workspace.close()


def test_status_reports_worker_and_source(monkeypatch):
    workspace = object.__new__(device_a2a.DcpDeviceA2A)
    workspace.closed = False
    workspace.device = "npu:0"
    workspace.local = ctypes.c_void_p(4096)
    monkeypatch.setattr(torch.npu, "synchronize", Mock())

    def copy_status(dst, *_):
        dst[7 * 16] = 4
        return 0

    workspace.acl = SimpleNamespace(aclrtMemcpy=copy_status)
    with pytest.raises(RuntimeError, match=r"\(7, 3\)"):
        workspace.check_status()
