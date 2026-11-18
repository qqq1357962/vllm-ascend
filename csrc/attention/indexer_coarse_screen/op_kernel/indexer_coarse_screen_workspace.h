/*
 * Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0.
 */
#ifndef INDEXER_COARSE_SCREEN_WORKSPACE_H
#define INDEXER_COARSE_SCREEN_WORKSPACE_H

#include "indexer_coarse_screen_reuse_geometry.h"

namespace CoarseWorkspace {
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t MAX_GROUP_SIZE = 16;
constexpr uint32_t DMA_BLOCK_BYTES = 32;
constexpr uint32_t KEY_TILE = 512;
constexpr uint32_t CUBE_ALIGN = 16;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t MERGE_PARAMS = 16;
constexpr uint32_t COARSE_ROWS = 2;
constexpr uint32_t NATIVE_ROWS = 8;

struct Group {
  uint32_t outputRow;
  uint32_t copies;
  uint32_t history;
  uint32_t windowEnd;
};

struct Layout {
  uint64_t weights;
  uint64_t history;
  uint64_t groups;
  uint64_t scan;

  LI_PIVOT_INLINE Layout(uint64_t count, uint32_t heads, bool raw)
      : weights(raw ? LICommon::PivotAlign(count * heads * HEAD_DIM * sizeof(uint16_t)) : 0),
        history(weights + (raw ? LICommon::PivotAlign(count * heads * sizeof(uint16_t)) : 0)),
        groups(history + LICommon::PivotAlign(count * sizeof(uint32_t))),
        scan(groups + LICommon::PivotAlign(count * sizeof(Group))) {}
};

LI_PIVOT_INLINE bool UseReuse(uint32_t count, bool modify, uint32_t heads, uint32_t groupSize) {
  return count == LICommon::PIVOT_TOPK && modify && heads == LICommon::PIVOT_WEIGHT_WIDTH &&
         groupSize == LICommon::PIVOT_GROUP_SIZE;
}

LI_PIVOT_INLINE uint64_t ScanWorkspace(uint32_t cores, uint32_t heads, uint32_t rows, uint32_t topk) {
  const uint64_t cubeRows = (rows * heads + CUBE_ALIGN - 1) / CUBE_ALIGN * CUBE_ALIGN;
  // Each core has two QK tiles and two split-K boundaries, each with values, IDs, and metadata.
  return cores *
         (DOUBLE_BUFFER * cubeRows * KEY_TILE * sizeof(float) + rows * DOUBLE_BUFFER * 2 * topk * sizeof(float) +
          rows * DOUBLE_BUFFER * MERGE_PARAMS * sizeof(int64_t));
}
}  // namespace CoarseWorkspace
#undef LI_PIVOT_INLINE

#endif
