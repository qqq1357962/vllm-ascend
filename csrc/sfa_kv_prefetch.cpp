// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#ifdef VLLM_ASCEND_ENABLE_SFA_KV_PREFETCH

  #include <ATen/ATen.h>
  #include <acl/acl.h>
  #include <torch/library.h>

  #include "torch_npu/csrc/core/npu/NPUStream.h"

namespace vllm_ascend {
void launch_sfa_kv_pool_gather(bool dcp, uint32_t cores, void* stream, uint8_t* src, uint8_t* rope, uint8_t* ids,
                               uint8_t* table, uint8_t* lens, uint8_t* pool, uint8_t* poolR, uint8_t* dense,
                               uint8_t* count, uint32_t queries, uint32_t capacity, uint32_t tableCols, uint32_t page,
                               uint32_t srcPages, uint32_t refresh, uint32_t tailGuard, uint64_t keyPageStride,
                               uint64_t ropePageStride);

namespace {
void sfa_kv_pool_gather(const at::Tensor& key, const at::Tensor& rope, const at::Tensor& ids, const at::Tensor& table,
                        const at::Tensor& lengths, at::Tensor pool, at::Tensor ropePool, at::Tensor dense,
                        at::Tensor count, int64_t tailGuard, bool refresh, bool dcp) {
  // The compact SFA reader uses four query rows and a 2048-slot BF16 MLA pool per row.
  constexpr int64_t queries = 4;
  constexpr int64_t slots = 2048;
  constexpr int64_t pageRows = 128;
  constexpr int64_t keyWidth = 512;
  constexpr int64_t ropeWidth = 64;
  constexpr int64_t poolPages = queries * slots / pageRows;
  auto hasCacheLayout = [](const at::Tensor& tensor, int64_t width) {
    return tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 4 && tensor.size(1) == pageRows &&
           tensor.size(2) == 1 && tensor.size(3) == width && tensor.stride(1) == width && tensor.stride(3) == 1;
  };
  for (const auto& tensor : {key, rope, ids, table, lengths, pool, ropePool, dense, count}) {
    TORCH_CHECK(tensor.device() == key.device() && tensor.is_privateuseone(),
                "SFA KV pool inputs must be on the same NPU.");
  }
  TORCH_CHECK(hasCacheLayout(key, keyWidth) && hasCacheLayout(rope, ropeWidth) && key.size(0) == rope.size(0),
              "SFA KV pool requires BF16 MLA KV in 128-token pages.");
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() &&
                  ids.sizes() == at::IntArrayRef({queries, 1, slots}) && table.scalar_type() == at::kInt &&
                  table.is_contiguous() && table.dim() == 2 && table.size(0) == 1 && table.size(1) > 0 &&
                  lengths.scalar_type() == at::kInt && lengths.is_contiguous() && lengths.numel() == 1,
              "SFA KV pool requires B1/Q4 and 2048 indices per query.");
  TORCH_CHECK(hasCacheLayout(pool, keyWidth) && hasCacheLayout(ropePool, ropeWidth) && pool.size(0) == poolPages &&
                  ropePool.size(0) == poolPages && pool.is_contiguous() && ropePool.is_contiguous() &&
                  dense.scalar_type() == at::kInt && dense.sizes() == ids.sizes() && dense.is_contiguous() &&
                  count.scalar_type() == at::kInt && count.numel() == queries && count.is_contiguous(),
              "Invalid SFA KV pool output layout.");
  TORCH_CHECK(tailGuard >= 0 && tailGuard <= 2 * queries, "Invalid SFA KV pool recent-token window.");

  int32_t device = -1;
  int64_t vectorCores = 0;
  TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS &&
                  aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores) == ACL_SUCCESS &&
                  vectorCores > 0,
              "Could not query the NPU vector core count for SFA KV prefetch.");
  auto ptr = [](const at::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); };
  auto stream = c10_npu::getCurrentNPUStream(key.device().index()).stream(true);
  // The batched kernel stages a page table no larger than its index scratch.
  const bool useScalarGather = dcp || table.size(1) > ids.numel();
  launch_sfa_kv_pool_gather(useScalarGather, vectorCores, stream, ptr(key), ptr(rope), ptr(ids), ptr(table),
                            ptr(lengths), ptr(pool), ptr(ropePool), ptr(dense), ptr(count), queries, slots,
                            table.size(1), pageRows, key.size(0), refresh, tailGuard, key.stride(0), rope.stride(0));
}

void sfa_kv_pool_gather_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, bool, bool) {}
}  // namespace
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def(
      "sfa_kv_pool_gather(Tensor key, Tensor rope, Tensor ids, Tensor table, Tensor lengths, "
      "Tensor(a!) pool, Tensor(b!) rope_pool, Tensor(c!) dense, Tensor(d!) count, "
      "int tail_guard, bool refresh, bool dcp) -> ()");
  ops.impl("sfa_kv_pool_gather", c10::DispatchKey::PrivateUse1, &vllm_ascend::sfa_kv_pool_gather);
  ops.impl("sfa_kv_pool_gather", c10::DispatchKey::Meta, &vllm_ascend::sfa_kv_pool_gather_meta);
}
#endif
