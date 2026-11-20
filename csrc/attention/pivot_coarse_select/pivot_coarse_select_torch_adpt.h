// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef PIVOT_COARSE_SELECT_TORCH_ADPT_H
#define PIVOT_COARSE_SELECT_TORCH_ADPT_H

#include "op_kernel/pivot_coarse_select_pivot_geometry.h"

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor> npu_indexer_coarse_screen(
    const at::Tensor& q_bar, const at::Tensor& w_bar, const at::Tensor& key,
    const c10::optional<at::Tensor>& actual_seq_lengths_query, const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const at::Tensor& block_table, int64_t coarse_count, int64_t has_window, int64_t group_size, int64_t proxy_mode,
    bool is_modify_localwindow) {
  using namespace PivotCoarseSelectCommon;
  TORCH_CHECK(coarse_count == PIVOT_TOPK || coarse_count == COARSE_TOPK, "coarse_count must be 2048 or 4096.");
  TORCH_CHECK(has_window == 1, "has_window must be 1.");
  TORCH_CHECK(group_size >= 1 && group_size <= MAX_COARSE_GROUP_SIZE, "group_size must be between 1 and 16.");
  TORCH_CHECK(proxy_mode == 0 || proxy_mode == 1, "proxy_mode must be 0 or 1.");
  TORCH_CHECK(q_bar.dim() == 3 && q_bar.numel() > 0 && w_bar.dim() == 2 && key.dim() == 4 && block_table.dim() == 2,
              "Expected nonempty TND query, rank-two weights, PA_BSND key and paged block table.");
  TORCH_CHECK(!is_modify_localwindow ||
                  (proxy_mode == 1 && group_size == PIVOT_GROUP_SIZE && q_bar.size(1) == PIVOT_WEIGHT_WIDTH),
              "Local-window updates require raw queries, groups of four, and 32 query heads.");
  TORCH_CHECK(actual_seq_lengths_query.has_value() && actual_seq_lengths_query->dim() == 1 &&
                  actual_seq_lengths_key.has_value() && actual_seq_lengths_key->dim() == 1,
              "Cumulative query lengths and visible key lengths are required.");
  const int64_t rows = is_modify_localwindow ? q_bar.size(0) : actual_seq_lengths_query->size(0);
  const auto options = q_bar.options().dtype(at::kInt);
  auto indices = at::empty({rows, 1, coarse_count}, options);
  auto values = at::empty({0}, q_bar.options());
  auto aslk = at::empty({rows}, options);

  std::string query_layout = "TND";
  std::string key_layout = "PA_BSND";
  auto* query_layout_ptr = query_layout.data();
  auto* key_layout_ptr = key_layout.data();
  const int64_t sparse_mode = is_modify_localwindow ? 3 : 0;
  const int64_t full_range = std::numeric_limits<int64_t>::max();
  constexpr bool return_values = false;
  EXEC_NPU_CMD(aclnnPivotCoarseSelect, q_bar, key, w_bar, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
               query_layout_ptr, key_layout_ptr, coarse_count, sparse_mode, full_range, full_range, return_values,
               group_size, proxy_mode, is_modify_localwindow, indices, values, aslk);
  return {indices.squeeze(1), aslk};
}

}  // namespace vllm_ascend
#endif
