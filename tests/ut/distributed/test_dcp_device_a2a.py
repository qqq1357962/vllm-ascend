# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.parametrize(
    "tokens,expected", [(0, False), (1, True), (32, True), (33, True), (64, True), (128, True), (129, False)]
)
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
    monkeypatch.setattr(device_a2a, "_load_device_a2a_ops", load)
    assert device_a2a.initialize_device_a2a(None, None, False) is None
    load.assert_not_called()


def test_rejects_unsupported_topology():
    config = SimpleNamespace(tensor_parallel_size=4)
    with pytest.raises(ValueError, match="TP8/DCP8"):
        device_a2a.initialize_device_a2a(None, config, True)


def test_graph_capture_cannot_initialize(monkeypatch):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        device_a2a.DcpDeviceA2A(None)


def test_missing_native_backend_fails_before_collective_bootstrap(monkeypatch):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    monkeypatch.setattr(torch.npu, "get_device_name", lambda _: "Ascend 910B2")
    monkeypatch.setattr(device_a2a.dist, "get_rank", lambda _: 0)
    monkeypatch.setattr(device_a2a.dist, "get_world_size", lambda _: 8)
    monkeypatch.setattr(
        device_a2a, "_load_device_a2a_ops", Mock(side_effect=RuntimeError("SHMEM backend was not built"))
    )

    def gather(peers, local, **_):
        peers[:] = [local] * 8

    monkeypatch.setattr(device_a2a.dist, "all_gather_object", gather)
    bootstrap = Mock(side_effect=AssertionError("failed preflight must not enter bootstrap"))
    monkeypatch.setattr(device_a2a.dist, "broadcast_object_list", bootstrap)
    with pytest.raises(RuntimeError, match="preflight failed.*SHMEM backend was not built"):
        device_a2a.DcpDeviceA2A(None)
    bootstrap.assert_not_called()


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
    workspace.buffer = torch.zeros(device_a2a.WORKSPACE_BYTES, dtype=torch.uint8)
    workspace.buffer[device_a2a.ERROR_OFFSET :].view(torch.int32)[7 * 16] = 4
    monkeypatch.setattr(torch.npu, "synchronize", Mock())

    with pytest.raises(RuntimeError, match=r"\(7, 3\)"):
        workspace.check_status()
