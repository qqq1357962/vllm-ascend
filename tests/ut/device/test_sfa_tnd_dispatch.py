# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


@pytest.mark.parametrize(
    "enabled,prefills,lse,seq_len,num_decodes",
    [
        (False, 1, False, 4096, 0),
        (True, 0, False, 4096, 1),
        (True, 1, True, 4096, 0),
        (True, 1, False, 1024, 0),
        (True, 1, False, 4096, 1),
        (True, 2, False, 4096, 0),
    ],
)
def test_opt_out_decode_short_context_mixed_and_lse_keep_pa_call(enabled, prefills, lse, seq_len, num_decodes):
    impl = SimpleNamespace(scale=1 / 24, sfa_tnd_physical_indices=enabled)
    metadata = SimpleNamespace(
        num_prefills=prefills,
        num_decodes=num_decodes,
        max_seq_len=seq_len,
        max_query_len=16,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    q = torch.empty(16, 8, 512, dtype=torch.bfloat16)
    qr = torch.empty(16, 8, 64, dtype=torch.bfloat16)
    cache = (torch.empty(1, 128, 1, 512, dtype=torch.bfloat16), torch.empty(1, 128, 1, 64, dtype=torch.bfloat16))
    ids = torch.zeros(16, 1, 2048, dtype=torch.int32)
    qlen, klen = torch.tensor([16], dtype=torch.int32), torch.tensor([128], dtype=torch.int32)
    output = (q, torch.empty(0), torch.empty(0))
    with (
        mock.patch("vllm_ascend.device.device_op.get_current_hardware_profile") as profile,
        mock.patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", create=True, return_value=output) as op,
    ):
        profile.return_value.device_adaptor_family = DeviceAdaptorFamily.FP8_OPTIMIZED
        got = BaseDeviceAdaptor.execute_sparse_flash_attention_process(
            impl,
            q,
            qr,
            cache,
            ids,
            metadata,
            qlen,
            klen,
            return_lse=lse,
        )
    assert got is (output if lse else q)
    assert op.call_args.kwargs["layout_kv"] == "PA_BSND"
    assert op.call_args.kwargs["sparse_indices"] is ids


@pytest.mark.parametrize("supported", [False, True])
def test_enabled_prefill_dispatch_and_layout_fallback(supported):
    impl = SimpleNamespace(scale=1 / 24, sfa_tnd_physical_indices=True)
    metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        max_seq_len=4096,
        max_query_len=16,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    q = torch.empty(16, 8, 512, dtype=torch.bfloat16)
    qr = torch.empty(16, 8, 64, dtype=torch.bfloat16)
    cache = (torch.empty(1, 128, 1, 512, dtype=torch.bfloat16), torch.empty(1, 128, 1, 64, dtype=torch.bfloat16))
    ids = torch.zeros(16, 1, 2048, dtype=torch.int32)
    length = torch.tensor([16], dtype=torch.int32)
    with (
        mock.patch("vllm_ascend.device.device_op.get_current_hardware_profile") as profile,
        mock.patch(
            "vllm_ascend.ops.triton.sfa_tnd.try_physical_tnd_sfa", return_value=(q,) if supported else None
        ) as tnd,
        mock.patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", create=True, return_value=(q,)) as op,
    ):
        profile.return_value.device_adaptor_family = DeviceAdaptorFamily.FP8_OPTIMIZED
        got = BaseDeviceAdaptor.execute_sparse_flash_attention_process(
            impl,
            q,
            qr,
            cache,
            ids,
            metadata,
            length,
            length,
            sparse_mode=3,
        )
    assert got is q
    assert tnd.call_args.kwargs["sparse_mode"] == 3
    assert op.call_count == (0 if supported else 1)
