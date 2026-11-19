/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef INDEXER_COARSE_SCREEN_V2_H
#define INDEXER_COARSE_SCREEN_V2_H

#include "op_kernel/indexer_coarse_screen_workspace.h"

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer_v2_npu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights, int64_t topk,
    const c10::optional<at::Tensor>& cu_seqlens_q, const c10::optional<at::Tensor>& cu_seqlens_k,
    const c10::optional<at::Tensor>& seqused_q, const c10::optional<at::Tensor>& seqused_k,
    const c10::optional<at::Tensor>& cmp_residual_k, const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& output_idx_offset, const c10::optional<at::Tensor>& metadata, int64_t max_seqlen_q,
    c10::string_view layout_q, c10::string_view layout_k, int64_t mask_mode, int64_t cmp_ratio, int64_t return_value);

inline std::tuple<at::Tensor, at::Tensor> indexer_coarse_screen_v2(const at::Tensor& q, const at::Tensor& w,
                                                                   const at::Tensor& key, const at::Tensor& query_ends,
                                                                   const at::Tensor& key_lengths,
                                                                   const at::Tensor& block_table, int64_t coarse_count,
                                                                   int64_t group_size, int64_t proxy_mode,
                                                                   bool modify_localwindow, int64_t min_prefix_length) {
  constexpr int64_t V2_TOPK_ALIGNMENT = 1024;  // Arch22 LIv2 requires multiples of 1024 above 2048.
  constexpr int64_t V2_MIN_FAST_Q_ROWS = 3;    // Fast tiling rejects batches consisting entirely of Q1/Q2.
  constexpr int64_t CAUSAL_MASK = 3;           // Right-aligned causal mask on the chunk's key prefix.
  constexpr int64_t UNCOMPRESSED_KEYS = 1;
  constexpr int64_t INDICES_ONLY = 0;
  constexpr int64_t V2_HEAD_ALIGNMENT = 8;  // Arch22 fast tiling accepts 8 to 64 query heads in steps of 8.
  constexpr int64_t V2_MAX_HEADS = 64;
  constexpr int64_t LARGE_COARSE_COUNT = 2 * LICommon::PIVOT_TOPK;
  constexpr int64_t INVALID_INDEX = -1;
  TORCH_CHECK(coarse_count == LICommon::PIVOT_TOPK || coarse_count == LARGE_COARSE_COUNT,
              "coarse_count must be 2048 or 4096.");
  TORCH_CHECK(group_size > 0 && group_size <= CoarseWorkspace::MAX_GROUP_SIZE, "group_size must be in [1, 16].");
  TORCH_CHECK(proxy_mode == 0 || proxy_mode == 1, "proxy_mode must be 0 or 1.");
  TORCH_CHECK(!modify_localwindow || proxy_mode == 1, "is_modify_localwindow requires raw queries (proxy_mode=1).");
  TORCH_CHECK(q.dim() == 3 && w.dim() == 2 && key.dim() == 4 && block_table.dim() == 2,
              "LIv2 coarse screen expects TND queries, TN weights, paged keys and a rank-two block table.");
  TORCH_CHECK(w.size(0) == q.size(0) && w.size(1) == q.size(1) && q.size(2) == key.size(3) && key.size(2) == 1,
              "Query, weight and key dimensions do not match.");
  TORCH_CHECK(q.size(2) == CoarseWorkspace::HEAD_DIM && q.size(1) >= V2_HEAD_ALIGNMENT && q.size(1) <= V2_MAX_HEADS &&
                  q.size(1) % V2_HEAD_ALIGNMENT == 0,
              "LIv2 coarse screen requires head dimension 128 and 8 to 64 query heads in steps of 8.");
  TORCH_CHECK(query_ends.scalar_type() == at::kInt && key_lengths.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt,
              "LIv2 coarse-screen metadata must be int32.");
  TORCH_CHECK((q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf) &&
                  key.scalar_type() == q.scalar_type() && w.scalar_type() == q.scalar_type(),
              "LIv2 coarse screen requires matching BF16 or FP16 query, weight and key dtypes.");
  const int64_t batch = key_lengths.numel();
  const int64_t rows_per_group = proxy_mode == 1 ? group_size : 1;
  TORCH_CHECK(batch > 0 && query_ends.numel() == batch && block_table.size(0) == batch &&
                  q.size(0) % (batch * rows_per_group) == 0,
              "LIv2 coarse screen requires uniform, group-aligned requests and one metadata row per request.");
  const int64_t groups_per_request = q.size(0) / (batch * rows_per_group);
  TORCH_CHECK(groups_per_request >= V2_MIN_FAST_Q_ROWS,
              "LIv2 coarse screen requires at least three groups per request.");

  // LIv2 advances the causal prefix by one per row. The real prefix advances by group_size.
  // A chunk spans at most one LIv2 top-k step in raw queries. At most
  // (chunk_size - 1) * (group_size - 1) extra keys can enter a proxy's ranked list.
  const int64_t chunk_size =
      group_size == 1 ? groups_per_request : std::min(groups_per_request, V2_TOPK_ALIGNMENT / group_size);
  const int64_t backend_topk = coarse_count + (group_size == 1 ? 0 : V2_TOPK_ALIGNMENT);
  const auto int_options = key_lengths.options();
  const int64_t key_capacity = block_table.size(1) * key.size(1);
  const int64_t raw_rows = groups_per_request * group_size;
  TORCH_CHECK(min_prefix_length >= 0 && min_prefix_length <= key_capacity - raw_rows,
              "Prefill cached prefix must be nonnegative and fit in the block table.");
  const int64_t window_size = modify_localwindow ? 0 : 2 * group_size - 1;
  const int64_t selected = coarse_count - window_size;
  const int64_t min_history = min_prefix_length + 1 - (modify_localwindow ? 0 : group_size);
  TORCH_CHECK(min_history >= selected && min_history + (chunk_size - 1) * group_size >= backend_topk,
              "Prefill prefix is too short for LIv2 coarse screen at this coarse_count.");
  TORCH_CHECK(!modify_localwindow || coarse_count != LICommon::PIVOT_TOPK || group_size != LICommon::PIVOT_GROUP_SIZE ||
                  min_prefix_length + 1 >= LICommon::FIRST_GROUP_CONTEXT,
              "Use the existing reuse route for the dense prefix below the first Pivot group.");

  // The caller supplies uniform request geometry and a host lower bound on the cached prefix.
  // Derive offsets from the tensor shapes to avoid synchronizing with the device in prefill.
  const auto history_base = key_lengths.view({batch, 1}) - raw_rows + 1 - (modify_localwindow ? 0 : group_size);
  const auto group_ids = at::arange(groups_per_request, int_options);
  const auto history = (history_base + group_ids * group_size).reshape({-1, 1});
  const auto chunk_ends =
      at::arange(chunk_size, groups_per_request + chunk_size, chunk_size, int_options).clamp_max(groups_per_request);
  const int64_t chunks_per_request = chunk_ends.numel();
  const int64_t chunks = batch * chunks_per_request;
  const auto request_offsets = at::arange(batch, int_options).view({batch, 1}) * groups_per_request;
  const auto cu_seqlens_q = at::cat({at::zeros({1}, int_options), (request_offsets + chunk_ends).reshape({-1})});
  const auto chunk_lengths = (history_base + (chunk_ends - 1) * group_size).reshape({-1});
  const auto chunk_table = at::repeat_interleave(block_table, chunks_per_request, 0, chunks);

  const auto proxy_q = proxy_mode == 0
                           ? q.contiguous()
                           : q.reshape({-1, group_size, q.size(1), q.size(2)}).mean(1, false, at::kFloat).to(q.dtype());
  const auto proxy_w =
      proxy_mode == 0 ? w.to(at::kFloat).contiguous()
                      : w.reshape({-1, group_size, w.size(1)}).mean(1, false, at::kFloat).to(w.dtype()).to(at::kFloat);
  // The arch22 metadata ABI reserves 2048 int32 entries, including fast-kernel tiling.
  constexpr int64_t ARCH22_METADATA_SIZE = 2048;
  auto metadata = at::empty({ARCH22_METADATA_SIZE}, int_options);
  const auto unused_lengths = at::empty({0}, int_options);
  const auto query_heads = q.size(1);
  const auto key_heads = key.size(2);
  const auto head_dim = q.size(2);
  std::string query_layout = "TND";
  std::string key_layout = "PA_BBND";
  auto* query_layout_ptr = query_layout.data();
  auto* key_layout_ptr = key_layout.data();
  EXEC_NPU_CMD(aclnnLightningIndexerV2Metadata, cu_seqlens_q, unused_lengths, unused_lengths, chunk_lengths,
               unused_lengths, query_heads, key_heads, head_dim, backend_topk, chunks, chunk_size, key_capacity,
               query_layout_ptr, key_layout_ptr, CAUSAL_MASK, UNCOMPRESSED_KEYS, metadata);
  const auto ids = std::get<0>(npu_lightning_indexer_v2_npu(
                                   proxy_q, key, proxy_w, backend_topk, cu_seqlens_q, c10::nullopt, c10::nullopt,
                                   chunk_lengths, c10::nullopt, chunk_table, c10::nullopt, metadata, chunk_size,
                                   query_layout, key_layout, CAUSAL_MASK, UNCOMPRESSED_KEYS, INDICES_ONLY))
                       .squeeze(1);

  const auto valid = (ids >= 0) & (ids < history);
  const auto rank = valid.cumsum(1, at::kInt) - 1;
  // Invalid and surplus entries each get their own scratch column, outside the retained prefix.
  const auto scratch_columns = at::arange(selected, selected + backend_topk, int_options);
  const auto destination = at::where(valid & (rank < selected), rank, scratch_columns).to(at::kLong);
  auto compacted = at::full({ids.size(0), selected + backend_topk}, INVALID_INDEX, int_options);
  compacted.scatter_(1, destination, ids);
  const auto top = compacted.slice(1, 0, selected);
  if (!modify_localwindow) {
    const auto window = history + at::arange(window_size, int_options);
    return {at::cat({top, window}, 1), at::full({ids.size(0)}, coarse_count, int_options)};
  }

  auto expanded = top.unsqueeze(1).expand({-1, group_size, -1}).contiguous();
  if (group_size > 1) {
    const auto tail = expanded.slice(2, coarse_count - (group_size - 1), coarse_count);
    const auto offsets = at::arange(group_size, int_options).view({1, group_size, 1});
    const auto slots = at::arange(1 - group_size, 0, int_options).view({1, 1, -1}) + offsets;
    tail.copy_(at::where(slots >= 0, history.unsqueeze(-1) + slots, tail));
  }
  return {expanded.reshape({-1, coarse_count}), at::full({q.size(0)}, coarse_count, int_options)};
}

}  // namespace vllm_ascend

#endif
