/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INDEXER_COARSE_SCREEN_TORCH_ADPT_H
#define INDEXER_COARSE_SCREEN_TORCH_ADPT_H

#include "indexer_coarse_screen_v2.h"

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor> construct_indexer_coarse_screen_output_tensor(
    const at::Tensor& actual_seq_lengths_query, const at::Tensor& q_bar, int64_t coarse_count, int64_t has_window,
    int64_t proxy_mode, bool is_modify_localwindow) {
  TORCH_CHECK(has_window == 1, "has_window must be 1; other modes are unsupported.");
  constexpr int64_t SIZE = 8;
  constexpr int64_t DIM_0 = 0;

  at::SmallVector<int64_t, SIZE> output_size;
  for (size_t i = 0; i < q_bar.sizes().size(); i++) {
    TORCH_CHECK(q_bar.size(i) > 0,
                "All values within q_bar's shape should be greater "
                "than 0, but shape[",
                i, "] is ", q_bar.size(i));
  }
  TORCH_CHECK(coarse_count == 2048 || coarse_count == 4096,
              "coarse_count must be 2048 or 4096, including the local window.");

  TORCH_CHECK(!is_modify_localwindow || proxy_mode == 1, "is_modify_localwindow requires raw queries (proxy_mode=1).");
  const int64_t req_num = is_modify_localwindow ? q_bar.size(DIM_0) : actual_seq_lengths_query.size(DIM_0);
  const int64_t out_w = coarse_count;
  output_size = {req_num, out_w};

  at::Tensor candidates_out = at::empty(output_size, q_bar.options().dtype(at::kInt));

  at::Tensor aslk_out = at::empty({req_num}, q_bar.options().dtype(at::kInt));
  return std::tuple<at::Tensor, at::Tensor>(candidates_out, aslk_out);
}

std::tuple<at::Tensor, at::Tensor> npu_indexer_coarse_screen(
    const at::Tensor& q_bar, const at::Tensor& w_bar, const at::Tensor& key,
    const c10::optional<at::Tensor>& actual_seq_lengths_query, const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const at::Tensor& block_table, int64_t coarse_count, int64_t has_window, int64_t group_size, int64_t proxy_mode,
    bool is_modify_localwindow, c10::optional<int64_t> prefill_min_prefix_length) {
  TORCH_CHECK(q_bar.numel() > 0, "Tensor q_bar is empty.");
  TORCH_CHECK(w_bar.numel() > 0, "Tensor w_bar is empty.");
  TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
  TORCH_CHECK(block_table.numel() > 0, "Tensor block_table is empty.");

  TORCH_CHECK(actual_seq_lengths_query.has_value() && actual_seq_lengths_query->dim() == 1,
              "actual_seq_lengths_query must be a vector of cumulative raw query lengths.");
  TORCH_CHECK(actual_seq_lengths_key.has_value() && actual_seq_lengths_key->dim() == 1,
              "actual_seq_lengths_key must be a vector of visible key lengths.");
  if (prefill_min_prefix_length.has_value()) {
    TORCH_CHECK(has_window == 1, "has_window must be 1.");
    return indexer_coarse_screen_v2(q_bar, w_bar, key, *actual_seq_lengths_query, *actual_seq_lengths_key, block_table,
                                    coarse_count, group_size, proxy_mode, is_modify_localwindow,
                                    *prefill_min_prefix_length);
  }

  auto coarse_screen_output = construct_indexer_coarse_screen_output_tensor(
      actual_seq_lengths_query.value(), q_bar, coarse_count, has_window, proxy_mode, is_modify_localwindow);
  at::Tensor candidates_out = std::get<0>(coarse_screen_output);
  at::Tensor aslk_out = std::get<1>(coarse_screen_output);

  EXEC_NPU_CMD(aclnnIndexerCoarseScreen, q_bar, w_bar, key, actual_seq_lengths_query, actual_seq_lengths_key,
               block_table, coarse_count, has_window, group_size, proxy_mode, is_modify_localwindow, candidates_out,
               aslk_out);

  return std::tuple<at::Tensor, at::Tensor>(candidates_out, aslk_out);
}
}  // namespace vllm_ascend

#endif
