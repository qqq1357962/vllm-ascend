# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton.dcp import dcp_a2a


@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("pcp", [None, "pcp"])
def test_device_backend_dispatch_and_fallback(monkeypatch, supported, pcp):
    workspace = Mock()
    group = SimpleNamespace(world_size=8, device_group=object(), dcp_device_a2a=workspace)
    monkeypatch.setattr(dcp_a2a, "_groups", {"dcp": lambda: group, "pcp": lambda: group})
    monkeypatch.setattr(dcp_a2a, "supports_device_a2a", Mock(return_value=supported))
    fallback = Mock()
    monkeypatch.setattr(dcp_a2a, "dcp_a2a_fused_combine", fallback)
    output = torch.empty(8, 64, 512, dtype=torch.bfloat16)
    lse = torch.empty(8, 64, 1)
    actual = dcp_a2a.dcp_a2a_fused(output, lse, 8, 1, "dcp", pcp)
    if supported and pcp is None:
        assert actual is workspace.run.return_value
        workspace.run.assert_called_once_with(output, lse)
        fallback.assert_not_called()
    else:
        assert actual is fallback.return_value
        workspace.run.assert_not_called()


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("dtype,packed_extra", [(torch.float32, 1), (torch.bfloat16, 4)])
def test_deferred_fake_layout(scatter_dim, dtype, packed_extra):
    output = torch.empty(8, 12, 96, dtype=dtype, device="meta")
    lse = torch.empty(8, 12, 1, device="meta")
    actual = dcp_a2a.dcp_a2a_fused_fake(output, lse, 2, scatter_dim, "dcp", defer_combine=True)
    assert actual.shape == (2, output.shape[scatter_dim] // 2, output.shape[1 - scatter_dim], 96 + packed_extra)
    assert actual.dtype == dtype


@pytest.mark.parametrize("scatter_size", [1, 2])
def test_deferred_exchange_skips_combine(monkeypatch, scatter_size):
    send = torch.empty(scatter_size, 2, 3, 5)
    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", Mock(return_value=send))
    collective = Mock()
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", collective)
    combine = Mock(side_effect=AssertionError("deferred exchange must not combine"))
    monkeypatch.setattr(dcp_a2a, "fused_dcp_lse_combine", combine)
    result = dcp_a2a.dcp_a2a_fused_combine(
        torch.empty(2, 3, 4), torch.empty(2, 3, 1), scatter_size, 1, object(), defer_combine=True
    )
    assert result.shape == send.shape
    assert collective.call_count == int(scatter_size > 1)
    combine.assert_not_called()
    if scatter_size == 1:
        assert result is send
