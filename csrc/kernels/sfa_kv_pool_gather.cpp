// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t KEY_WIDTH = 512;
constexpr uint32_t ROPE_WIDTH = 64;
constexpr uint32_t ROW_ELEMENTS = KEY_WIDTH + ROPE_WIDTH;
constexpr uint32_t DMA_ALIGNMENT = 32;  // Ascend C local copy alignment in bytes.
constexpr uint32_t BATCH_SLOTS = DMA_ALIGNMENT / sizeof(int32_t);
}  // namespace

// DCP decode contract: one request, packed valid local-index prefix per query.
// The two launches copy historical positions and refresh recent positions, respectively.
extern "C" __global__ __aicore__ void sfa_kv_pool_gather_dcp(GM_ADDR src, GM_ADDR rope, GM_ADDR ids, GM_ADDR table,
                                                             GM_ADDR lens, GM_ADDR pool, GM_ADDR poolR, GM_ADDR dense,
                                                             GM_ADDR count, uint32_t queries, uint32_t capacity,
                                                             uint32_t tableCols, uint32_t page, uint32_t srcPages,
                                                             uint32_t refresh, uint32_t tailGuard,
                                                             uint64_t keyPageStride, uint64_t ropePageStride) {
  TPipe pipe;
  TQue<TPosition::VECIN, 1> q;
  // Retain the historical scalar gather queue depth and barriers.
  constexpr uint32_t queueDepth = 2;
  pipe.InitBuffer(q, queueDepth, ROW_ELEMENTS * sizeof(uint16_t));
  GlobalTensor<uint16_t> x, r, y, z;
  GlobalTensor<int32_t> ix, bt, kl, di, n;
  x.SetGlobalBuffer((__gm__ uint16_t*)src);
  r.SetGlobalBuffer((__gm__ uint16_t*)rope);
  y.SetGlobalBuffer((__gm__ uint16_t*)pool);
  z.SetGlobalBuffer((__gm__ uint16_t*)poolR);
  ix.SetGlobalBuffer((__gm__ int32_t*)ids);
  bt.SetGlobalBuffer((__gm__ int32_t*)table);
  kl.SetGlobalBuffer((__gm__ int32_t*)lens);
  di.SetGlobalBuffer((__gm__ int32_t*)dense);
  n.SetGlobalBuffer((__gm__ int32_t*)count);
  TBuf<TPosition::VECCALC> meta;
  pipe.InitBuffer(meta, DMA_ALIGNMENT);
  auto metadata = meta.Get<int32_t>();
  if (GetBlockIdx() == 0 && !refresh) {
    for (uint32_t row = 0; row < queries; row++) {
      uint32_t lo = 0, hi = capacity;
      while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (ix.GetValue(row * capacity + mid) >= 0)
          lo = mid + 1;
        else
          hi = mid;
      }
      metadata.SetValue(row, lo);
    }
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyExtParams cp = {1, static_cast<uint32_t>(queries * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(n, metadata, cp);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
  }
  int32_t length = kl.GetValue(0);
  int32_t historical = length > tailGuard ? length - tailGuard : 0;
  for (uint32_t base = GetBlockIdx() * BATCH_SLOTS; base < queries * capacity; base += GetBlockNum() * BATCH_SLOTS) {
    for (uint32_t j = 0; j < BATCH_SLOTS; j++) {
      uint32_t i = base + j;
      int32_t id = ix.GetValue(i);
      bool valid = id >= 0 && id < length;
      if (!refresh) metadata.SetValue(j, valid ? i % capacity : -1);
      if (!valid || (!refresh && id >= historical) || (refresh && id < historical)) continue;
      uint32_t logicalPage = id / page;
      if (logicalPage >= tableCols) continue;
      int32_t physicalPage = bt.GetValue(logicalPage);
      if (physicalPage < 0 || physicalPage >= srcPages) continue;
      auto a = q.AllocTensor<uint16_t>();
      DataCopy(a, x[(uint64_t)physicalPage * keyPageStride + (id % page) * KEY_WIDTH], KEY_WIDTH);
      DataCopy(a[KEY_WIDTH], r[(uint64_t)physicalPage * ropePageStride + (id % page) * ROPE_WIDTH], ROPE_WIDTH);
      PipeBarrier<PIPE_ALL>();
      q.EnQue(a);
      a = q.DeQue<uint16_t>();
      DataCopy(y[(uint64_t)i * KEY_WIDTH], a, KEY_WIDTH);
      DataCopy(z[(uint64_t)i * ROPE_WIDTH], a[KEY_WIDTH], ROPE_WIDTH);
      PipeBarrier<PIPE_ALL>();
      q.FreeTensor(a);
    }
    if (!refresh) {
      SetFlag<HardEvent::S_MTE3>(0);
      WaitFlag<HardEvent::S_MTE3>(0);
      DataCopy(di[base], metadata, BATCH_SLOTS);
      SetFlag<HardEvent::MTE3_S>(0);
      WaitFlag<HardEvent::MTE3_S>(0);
    }
  }
}

extern "C" __global__ __aicore__ void sfa_kv_pool_gather_batched(
    GM_ADDR src, GM_ADDR rope, GM_ADDR ids, GM_ADDR table, GM_ADDR lens, GM_ADDR pool, GM_ADDR poolR, GM_ADDR dense,
    GM_ADDR count, uint32_t queries, uint32_t capacity, uint32_t tableCols, uint32_t page, uint32_t srcPages,
    uint32_t refresh, uint32_t tailGuard, uint64_t keyPageStride, uint64_t ropePageStride) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> data, indexBuf, tableBuf;
  pipe.InitBuffer(data, BATCH_SLOTS * ROW_ELEMENTS * sizeof(uint16_t));
  pipe.InitBuffer(indexBuf, queries * capacity * sizeof(int32_t));
  pipe.InitBuffer(tableBuf, (tableCols * sizeof(int32_t) + DMA_ALIGNMENT - 1) / DMA_ALIGNMENT * DMA_ALIGNMENT);
  auto a = data.Get<uint16_t>();
  auto localIds = indexBuf.Get<int32_t>();
  auto localTable = tableBuf.Get<int32_t>();
  GlobalTensor<uint16_t> x, r, y, z;
  GlobalTensor<int32_t> ix, bt, kl, di, n;
  x.SetGlobalBuffer((__gm__ uint16_t*)src);
  r.SetGlobalBuffer((__gm__ uint16_t*)rope);
  y.SetGlobalBuffer((__gm__ uint16_t*)pool);
  z.SetGlobalBuffer((__gm__ uint16_t*)poolR);
  ix.SetGlobalBuffer((__gm__ int32_t*)ids);
  bt.SetGlobalBuffer((__gm__ int32_t*)table);
  kl.SetGlobalBuffer((__gm__ int32_t*)lens);
  di.SetGlobalBuffer((__gm__ int32_t*)dense);
  n.SetGlobalBuffer((__gm__ int32_t*)count);
  TBuf<TPosition::VECCALC> meta;
  pipe.InitBuffer(meta, DMA_ALIGNMENT);
  auto metadata = meta.Get<int32_t>();
  DataCopy(localIds, ix, queries * capacity);
  DataCopyExtParams tableCp = {1, static_cast<uint32_t>(tableCols * sizeof(int32_t)), 0, 0, 0};
  DataCopyPadExtParams<int32_t> padding = {false, 0, 0, 0};
  DataCopyPad(localTable, bt, tableCp, padding);
  if (!refresh) Duplicate(a, (uint16_t)0, BATCH_SLOTS * ROW_ELEMENTS);
  PipeBarrier<PIPE_ALL>();
  if (GetBlockIdx() == 0 && !refresh) {
    for (uint32_t row = 0; row < queries; row++) {
      uint32_t lo = 0, hi = capacity;
      while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (localIds.GetValue(row * capacity + mid) >= 0)
          lo = mid + 1;
        else
          hi = mid;
      }
      metadata.SetValue(row, lo);
    }
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyExtParams cp = {1, static_cast<uint32_t>(queries * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(n, metadata, cp);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
  }
  int32_t length = kl.GetValue(0);
  int32_t historical = length > tailGuard ? length - tailGuard : 0;
  for (uint32_t base = GetBlockIdx() * BATCH_SLOTS; base < queries * capacity; base += GetBlockNum() * BATCH_SLOTS) {
    bool anyCopy = false;
    for (uint32_t j = 0; j < BATCH_SLOTS; j++) {
      uint32_t i = base + j;
      int32_t id = localIds.GetValue(i);
      bool valid = id >= 0 && id < length;
      if (!refresh) metadata.SetValue(j, valid ? i % capacity : -1);
      if (!valid || (!refresh && id >= historical) || (refresh && id < historical)) continue;
      uint32_t logicalPage = id / page;
      if (logicalPage >= tableCols) continue;
      int32_t physicalPage = localTable.GetValue(logicalPage);
      if (physicalPage < 0 || physicalPage >= srcPages) continue;
      anyCopy = true;
      DataCopy(a[j * KEY_WIDTH], x[(uint64_t)physicalPage * keyPageStride + (id % page) * KEY_WIDTH], KEY_WIDTH);
      DataCopy(a[BATCH_SLOTS * KEY_WIDTH + j * ROPE_WIDTH],
               r[(uint64_t)physicalPage * ropePageStride + (id % page) * ROPE_WIDTH], ROPE_WIDTH);
    }
    if (refresh && !anyCopy) continue;
    // Complete a batch of indexed reads before contiguous output writes.
    PipeBarrier<PIPE_ALL>();
    if (!refresh) {
      DataCopy(y[(uint64_t)base * KEY_WIDTH], a, BATCH_SLOTS * KEY_WIDTH);
      DataCopy(z[(uint64_t)base * ROPE_WIDTH], a[BATCH_SLOTS * KEY_WIDTH], BATCH_SLOTS * ROPE_WIDTH);
    } else {
      for (uint32_t j = 0; j < BATCH_SLOTS; j++) {
        int32_t id = localIds.GetValue(base + j);
        if (id < 0 || id >= length || id < historical) continue;
        uint32_t logicalPage = id / page;
        if (logicalPage >= tableCols) continue;
        int32_t physicalPage = localTable.GetValue(logicalPage);
        if (physicalPage < 0 || physicalPage >= srcPages) continue;
        DataCopy(y[(uint64_t)(base + j) * KEY_WIDTH], a[j * KEY_WIDTH], KEY_WIDTH);
        DataCopy(z[(uint64_t)(base + j) * ROPE_WIDTH], a[BATCH_SLOTS * KEY_WIDTH + j * ROPE_WIDTH], ROPE_WIDTH);
      }
    }
    if (!refresh) {
      SetFlag<HardEvent::S_MTE3>(0);
      WaitFlag<HardEvent::S_MTE3>(0);
      DataCopy(di[base], metadata, BATCH_SLOTS);
      SetFlag<HardEvent::MTE3_S>(0);
      WaitFlag<HardEvent::MTE3_S>(0);
    }
    // Finish MTE3 before the next batch overwrites the local data buffer.
    PipeBarrier<PIPE_ALL>();
  }
}

namespace vllm_ascend {
void launch_sfa_kv_pool_gather(bool dcp, uint32_t cores, void* stream, uint8_t* src, uint8_t* rope, uint8_t* ids,
                               uint8_t* table, uint8_t* lens, uint8_t* pool, uint8_t* poolR, uint8_t* dense,
                               uint8_t* count, uint32_t queries, uint32_t capacity, uint32_t tableCols, uint32_t page,
                               uint32_t srcPages, uint32_t refresh, uint32_t tailGuard, uint64_t keyPageStride,
                               uint64_t ropePageStride) {
  if (dcp) {
    sfa_kv_pool_gather_dcp<<<cores, nullptr, stream>>>(src, rope, ids, table, lens, pool, poolR, dense, count, queries,
                                                       capacity, tableCols, page, srcPages, refresh, tailGuard,
                                                       keyPageStride, ropePageStride);
  } else {
    sfa_kv_pool_gather_batched<<<cores, nullptr, stream>>>(src, rope, ids, table, lens, pool, poolR, dense, count,
                                                           queries, capacity, tableCols, page, srcPages, refresh,
                                                           tailGuard, keyPageStride, ropePageStride);
  }
}
}  // namespace vllm_ascend
