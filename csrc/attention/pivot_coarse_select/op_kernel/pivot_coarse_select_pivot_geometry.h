/* Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0. */
#ifndef PIVOT_COARSE_SELECT_PIVOT_GEOMETRY_H
#define PIVOT_COARSE_SELECT_PIVOT_GEOMETRY_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define LI_PIVOT_INLINE __aicore__ inline
#else
#define LI_PIVOT_INLINE inline
#endif

namespace PivotCoarseSelectCommon {
constexpr uint32_t PIVOT_GROUP_SIZE = 4;
constexpr uint32_t PIVOT_TOPK = 2048;
constexpr uint32_t COARSE_TOPK = 4096;
constexpr uint32_t MAX_COARSE_GROUP_SIZE = 16; // Match the reference grouped API's limit.
constexpr uint32_t INDEXER_HEAD_DIM = 128;
constexpr uint32_t PIVOT_QUERY_WIDTH = 4096;
constexpr uint32_t PIVOT_WEIGHT_WIDTH = 32;

struct PivotGroup {
    uint32_t sourceRow;
    uint32_t copies;
    uint32_t visibleKeys;
};

struct PivotGeometry {
    uint32_t rows;
    uint32_t dense;

    LI_PIVOT_INLINE PivotGeometry(uint32_t rowCount, uint32_t keyCount) : rows(rowCount), dense(0)
    {
        if (keyCount >= rows && keyCount - rows < 4095) {
            uint32_t boundary = (4095 - (keyCount - rows) + 3) / 4 * 4;
            dense = rows < boundary ? rows : boundary;
        }
    }

    LI_PIVOT_INLINE uint32_t Groups() const
    {
        return dense + (rows - dense + PIVOT_GROUP_SIZE - 1) / PIVOT_GROUP_SIZE;
    }

    LI_PIVOT_INLINE PivotGroup Group(uint32_t start, uint32_t keyCount, uint32_t group) const
    {
        uint32_t row = group < dense ? group : dense + PIVOT_GROUP_SIZE * (group - dense);
        uint32_t count = group < dense ? 1 : (rows - row < PIVOT_GROUP_SIZE ? rows - row : PIVOT_GROUP_SIZE);
        return {start + row, count, keyCount == 0 ? 0 : keyCount - rows + row + 1};
    }
};

LI_PIVOT_INLINE uint64_t PivotAlign(uint64_t bytes)
{
    return (bytes + 511) / 512 * 512;
}

// Host reserves for every original row; device uses only the actual group count.
struct PivotWorkspace {
    uint64_t weights;
    uint64_t indices;
    uint64_t groups;
    uint64_t lengths;
    uint64_t native;

    LI_PIVOT_INLINE PivotWorkspace(uint64_t rows, uint64_t batches, uint32_t topk = PIVOT_TOPK)
        : weights(PivotAlign(rows * PIVOT_QUERY_WIDTH * 2)),
          indices(weights + PivotAlign(rows * PIVOT_WEIGHT_WIDTH * sizeof(float))),
          groups(indices + PivotAlign(rows * topk * sizeof(int32_t))),
          lengths(groups + PivotAlign(rows * sizeof(PivotGroup))),
          native(lengths + PivotAlign(batches * sizeof(uint32_t)))
    {}
};

struct GroupedWorkspace {
    uint64_t weights;
    uint64_t picks;
    uint64_t queryEnds;
    uint64_t history;
    uint64_t scan;

    LI_PIVOT_INLINE GroupedWorkspace(uint64_t rows, uint64_t heads, uint32_t topk, bool raw)
        : weights(raw ? PivotAlign(rows * heads * INDEXER_HEAD_DIM * sizeof(uint16_t)) : 0),
          picks(weights + (raw ? PivotAlign(rows * heads * sizeof(float)) : 0)),
          queryEnds(picks + PivotAlign(rows * topk * sizeof(int32_t))),
          history(queryEnds + PivotAlign(rows * sizeof(uint32_t))),
          scan(history + PivotAlign(rows * sizeof(uint32_t)))
    {}
};
} // namespace PivotCoarseSelectCommon

#undef LI_PIVOT_INLINE
#endif
