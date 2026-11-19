/* Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0. */
#ifndef LIGHTNING_INDEXER_PIVOT_GEOMETRY_H
#define LIGHTNING_INDEXER_PIVOT_GEOMETRY_H

#include <cstdint>

#if defined(__CCE_AICORE__)
  #define LI_PIVOT_INLINE __aicore__ inline
#else
  #define LI_PIVOT_INLINE inline
#endif

namespace LICommon {
constexpr uint32_t PIVOT_GROUP_SIZE = 4;
constexpr uint32_t PIVOT_TOPK = 2048;
constexpr uint32_t PIVOT_WEIGHT_WIDTH = 32;
constexpr uint32_t PIVOT_QUERY_WIDTH = PIVOT_WEIGHT_WIDTH * 128;
// Dense rows precede the first group that can see a 4K context.
constexpr uint32_t FIRST_GROUP_CONTEXT = 2 * PIVOT_TOPK;
constexpr uint32_t GM_ALIGNMENT = 512;

struct PivotGroup {
  uint32_t sourceRow;
  uint32_t copies;
  uint32_t visibleKeys;
};

struct PivotGeometry {
  uint32_t rows;
  uint32_t dense;

  LI_PIVOT_INLINE PivotGeometry(uint32_t rowCount, uint32_t keyCount) : rows(rowCount), dense(0) {
    if (keyCount >= rows && keyCount - rows + 1 < FIRST_GROUP_CONTEXT) {
      uint32_t boundary = (4095 - (keyCount - rows) + 3) / 4 * 4;
      dense = rows < boundary ? rows : boundary;
    }
  }

  LI_PIVOT_INLINE uint32_t Groups() const { return dense + (rows - dense + PIVOT_GROUP_SIZE - 1) / PIVOT_GROUP_SIZE; }

  LI_PIVOT_INLINE PivotGroup Group(uint32_t start, uint32_t keyCount, uint32_t group) const {
    uint32_t row = group < dense ? group : dense + PIVOT_GROUP_SIZE * (group - dense);
    uint32_t count = group < dense ? 1 : (rows - row < PIVOT_GROUP_SIZE ? rows - row : PIVOT_GROUP_SIZE);
    return {start + row, count, keyCount == 0 ? 0 : keyCount - rows + row + 1};
  }
};

LI_PIVOT_INLINE uint64_t PivotAlign(uint64_t bytes) { return (bytes + GM_ALIGNMENT - 1) / GM_ALIGNMENT * GM_ALIGNMENT; }

// Host reserves for every original row; device uses only the actual group count.
struct PivotWorkspace {
  uint64_t weights;
  uint64_t indices;
  uint64_t groups;
  uint64_t lengths;
  uint64_t native;

  LI_PIVOT_INLINE PivotWorkspace(uint64_t rows, uint64_t batches)
      : weights(PivotAlign(rows * PIVOT_QUERY_WIDTH * 2)),
        indices(weights + PivotAlign(rows * PIVOT_WEIGHT_WIDTH * sizeof(float))),
        groups(indices + PivotAlign(rows * PIVOT_TOPK * sizeof(int32_t))),
        lengths(groups + PivotAlign(rows * sizeof(PivotGroup))),
        native(lengths + PivotAlign(batches * sizeof(uint32_t))) {}
};
}  // namespace LICommon

#endif
