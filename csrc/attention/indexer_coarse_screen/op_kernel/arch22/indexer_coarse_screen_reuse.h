/* Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0. */
#ifndef LIGHTNING_INDEXER_PIVOT_REUSE_H
#define LIGHTNING_INDEXER_PIVOT_REUSE_H

// PIVOT_REUSE_DEBUG is defined by lightning_indexer_common.h, which
// arch22/lightning_indexer_kernel_arch22.h pulls in before this header; when it is 0 the
// macro expands to nothing, so not even the argument expressions survive. Every
// record carries its own flag literal inside its format string, so two sites can
// never be folded into one printf. The AIV / first-block guard lives in the macro
// rather than at each site because the body of TryPivotReuse is compiled for the AIC
// and for every AIV: an unguarded print would repeat once per core.
#if PIVOT_REUSE_DEBUG
  #define LI_PIVOT_PROBE(...)           \
    do {                                \
      if ASCEND_IS_AIV {                \
        if (GetBlockIdx() == 0) {       \
          AscendC::printf(__VA_ARGS__); \
        }                               \
      }                                 \
    } while (0)
#else
  #define LI_PIVOT_PROBE(...)
#endif

namespace LIKernel {
// Event 7 publishes grouped inputs independently of the score pipeline events.
constexpr uint32_t PIVOT_READY_EVENT = 7;

template <typename LIT>
struct PivotProxyType : LIType<typename LIT::queryType, typename LIT::keyType, typename LIT::outputType, true,
                               LI_LAYOUT::TND, LI_LAYOUT::PA_BSND, LIT::weightsTypeFlag> {
  static constexpr bool pivotReuse = true;
};

template <typename T>
__aicore__ inline void PivotMeanRows(__gm__ uint8_t* input, __gm__ uint8_t* output, uint32_t width, uint32_t sourceRow,
                                     uint32_t outputRow, uint32_t count, LocalTensor<T> raw, LocalTensor<float> accum,
                                     LocalTensor<float> temp) {
  GlobalTensor<T> src, dst;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
  DataCopy(raw, src[sourceRow * width], count * width);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  if constexpr (std::is_same<T, float>::value) {
    Adds(accum, raw, 0.0f, width);
  } else {
    Cast(accum, raw, RoundMode::CAST_NONE, width);
  }
  for (uint32_t row = 1; row < count; ++row) {
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<T, float>::value) {
      Adds(temp, raw[row * width], 0.0f, width);
    } else {
      Cast(temp, raw[row * width], RoundMode::CAST_NONE, width);
    }
    PipeBarrier<PIPE_V>();
    Add(accum, accum, temp, width);
  }
  PipeBarrier<PIPE_V>();
  Muls(accum, accum, 1.0f / count, width);
  PipeBarrier<PIPE_V>();
  if constexpr (std::is_same<T, float>::value) {
    Adds(raw, accum, 0.0f, width);
  } else {
    Cast(raw, accum, RoundMode::CAST_RINT, width);
  }
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopy(dst[outputRow * width], raw, width);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  // The next input DMA reuses raw: ordering only the vector pipe is not enough.
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename LIT>
__aicore__ inline bool TryPivotReuse(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
                                     __gm__ uint8_t* queryLengths, __gm__ uint8_t* keyLengths, __gm__ uint8_t* table,
                                     __gm__ uint8_t* indices, __gm__ uint8_t* values, __gm__ uint8_t* workspace,
                                     const LITilingData& original, TPipe& pipe, __gm__ uint8_t* aslk) {
  using Q = typename LIT::queryType;
  using W = typename std::conditional<LIT::weightsTypeFlag, float, Q>::type;
  // flag1 is emitted before every gate, so a refused call still shows the tiling and
  // the two length pointers that refused it.
  LI_PIVOT_PROBE(
      "[LI_PIVOT] flag1 entry bSize=%u gSize=%u sparseCount=%u sparseMode=%u s1Size=%u s2Size=%u qNull=%u kNull=%u "
      "bsnd=%u tnd=%u\n",
      original.bSize, original.gSize, original.sparseCount, original.sparseMode, original.s1Size, original.s2Size,
      static_cast<uint32_t>(queryLengths == nullptr), static_cast<uint32_t>(keyLengths == nullptr),
      static_cast<uint32_t>(LIT::layout == LI_LAYOUT::BSND), static_cast<uint32_t>(LIT::layout == LI_LAYOUT::TND));
  if constexpr (!LIT::pageAttention || LIT::keyLayout != LI_LAYOUT::PA_BSND) {
    LI_PIVOT_PROBE("[LI_PIVOT] flag2 layout pageAttention=%u keyPA=%u\n", static_cast<uint32_t>(LIT::pageAttention),
                   static_cast<uint32_t>(LIT::keyLayout == LI_LAYOUT::PA_BSND));
    return false;
  }
  if (original.bSize == 0 || original.gSize != PIVOT_WEIGHT_WIDTH || original.sparseCount != PIVOT_TOPK ||
      original.sparseMode != 3 || original.returnValue || original.usedCoreNum == 0 || queryLengths == nullptr ||
      keyLengths == nullptr || (LIT::layout != LI_LAYOUT::TND && original.bSize != 1)) {
    // The flag1 field set, plus the two terms of this gate that flag1 does not
    // carry; without them the refusal cannot be attributed to a specific term.
    LI_PIVOT_PROBE(
        "[LI_PIVOT] flag3 tiling bSize=%u gSize=%u sparseCount=%u sparseMode=%u s1Size=%u s2Size=%u qNull=%u kNull=%u "
        "bsnd=%u tnd=%u returnValue=%u usedCoreNum=%u\n",
        original.bSize, original.gSize, original.sparseCount, original.sparseMode, original.s1Size, original.s2Size,
        static_cast<uint32_t>(queryLengths == nullptr), static_cast<uint32_t>(keyLengths == nullptr),
        static_cast<uint32_t>(LIT::layout == LI_LAYOUT::BSND), static_cast<uint32_t>(LIT::layout == LI_LAYOUT::TND),
        original.returnValue, original.usedCoreNum);
    return false;
  }

  GlobalTensor<uint32_t> qlen, klen;
  qlen.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(queryLengths));
  klen.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(keyLengths));
  uint32_t rowStart = 0;
  uint32_t totalGroups = 0;
  bool pooled = false;
  for (uint32_t b = 0; b < original.bSize; ++b) {
    uint32_t end = qlen.GetValue(b);
    uint32_t keys = klen.GetValue(b);
#if PIVOT_REUSE_DEBUG
    // flag4 is the per-batch census and covers every batch the loop walks, the one
    // that trips the refusal below included: a call refused here must still show
    // what the two length arrays held. rows, dense and groups are 0xFFFFFFFF when
    // end ran behind start -- that underflow IS the refusal, and the geometry must
    // not be built on it.
    {
      uint32_t rows = end >= rowStart ? end - rowStart : 0xFFFFFFFFu;
      uint32_t dense = 0xFFFFFFFFu;
      uint32_t groups = 0xFFFFFFFFu;
      bool pooledThisBatch = false;
      if (end >= rowStart) {
        PivotGeometry geometry(end - rowStart, keys);
        dense = geometry.dense;
        groups = geometry.Groups();
        pooledThisBatch = keys != 0 && geometry.Groups() < geometry.rows;
      }
      LI_PIVOT_PROBE(
          "[LI_PIVOT] flag4 batch b=%u end=%u keys=%u start=%u rows=%u dense=%u groups=%u pooledThisBatch=%u\n", b, end,
          keys, rowStart, rows, dense, groups, static_cast<uint32_t>(pooledThisBatch));
    }
#endif
    if (end < rowStart || (keys != 0 && keys < end - rowStart)) {
      LI_PIVOT_PROBE("[LI_PIVOT] flag5 refuse b=%u end=%u start=%u keys=%u\n", b, end, rowStart, keys);
      return false;
    }
    PivotGeometry geometry(end - rowStart, keys);
    totalGroups += geometry.Groups();
    pooled |= keys != 0 && geometry.Groups() < geometry.rows;
    rowStart = end;
  }
  if (!pooled || (LIT::layout == LI_LAYOUT::BSND && rowStart != original.s1Size)) {
    LI_PIVOT_PROBE("[LI_PIVOT] flag6 pooled pooled=%u rowStart=%u s1Size=%u\n", static_cast<uint32_t>(pooled), rowStart,
                   original.s1Size);
    return false;
  }

  // flag7: TAKEN. The group table is not written yet, so the census above is replayed
  // to report how many groups the device is about to store with visibleKeys == 0;
  // flag8 reads the stored table back.
#if PIVOT_REUSE_DEBUG
  {
    uint32_t zeroVisible = 0;
    uint32_t scanStart = 0;
    for (uint32_t b = 0; b < original.bSize; ++b) {
      uint32_t end = qlen.GetValue(b);
      uint32_t keys = klen.GetValue(b);
      PivotGeometry geometry(end - scanStart, keys);
      for (uint32_t g = 0; g < geometry.Groups(); ++g) {
        zeroVisible += geometry.Group(scanStart, keys, g).visibleKeys == 0 ? 1 : 0;
      }
      scanStart = end;
    }
    LI_PIVOT_PROBE("[LI_PIVOT] flag7 taken totalGroups=%u bSize=%u zeroVisibleGroups=%u\n", totalGroups, original.bSize,
                   zeroVisible);
  }
#endif

  PivotWorkspace offsets(totalGroups, original.bSize);
  auto groups = reinterpret_cast<__gm__ PivotGroup*>(workspace + offsets.groups);
  GlobalTensor<uint32_t> groupWords, groupEnds;
  groupWords.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(groups));
  groupEnds.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(workspace + offsets.lengths));

  if ASCEND_IS_AIV {
    TBuf<TPosition::VECCALC> rawBuf, accumBuf, tempBuf, metaBuf;
    pipe.InitBuffer(rawBuf, PIVOT_GROUP_SIZE * PIVOT_QUERY_WIDTH * sizeof(Q));
    pipe.InitBuffer(accumBuf, PIVOT_QUERY_WIDTH * sizeof(float));
    pipe.InitBuffer(tempBuf, PIVOT_QUERY_WIDTH * sizeof(float));
    pipe.InitBuffer(metaBuf, CoarseWorkspace::DMA_BLOCK_BYTES);
    auto meta = metaBuf.Get<uint32_t>();
    uint32_t id = GetBlockIdx();
    uint32_t cores = GetBlockNum() * 2;
    uint32_t base = 0;
    rowStart = 0;
    for (uint32_t b = 0; b < original.bSize; ++b) {
      uint32_t end = qlen.GetValue(b);
      uint32_t keys = klen.GetValue(b);
      PivotGeometry geometry(end - rowStart, keys);
      uint32_t count = geometry.Groups();
      if (id == 0) {
        meta.SetValue(0, base + count);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(groupEnds[b], meta, {1, sizeof(uint32_t), 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
      }
      // Distribute the flattened group space, including across short batches.
      for (uint32_t g = (id + cores - base % cores) % cores; g < count; g += cores) {
        PivotGroup group = geometry.Group(rowStart, keys, g);
        if (keys != 0) {
          PivotMeanRows<Q>(query, workspace, PIVOT_QUERY_WIDTH, group.sourceRow, base + g, group.copies,
                           rawBuf.Get<Q>(), accumBuf.Get<float>(), tempBuf.Get<float>());
          PivotMeanRows<W>(weights, workspace + offsets.weights, PIVOT_WEIGHT_WIDTH, group.sourceRow, base + g,
                           group.copies, rawBuf.Get<W>(), accumBuf.Get<float>(), tempBuf.Get<float>());
        }
        meta.SetValue(0, group.sourceRow);
        meta.SetValue(1, group.copies);
        meta.SetValue(2, group.visibleKeys);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(groupWords[(base + g) * 3], meta, {1, sizeof(PivotGroup), 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
      }
      base += count;
      rowStart = end;
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    CrossCoreSetFlag<2, PIPE_MTE3>(PIVOT_READY_EVENT);
    pipe.Reset();
  } else {
    CrossCoreWaitFlag(PIVOT_READY_EVENT);
  }

  LITilingData proxy = original;
  proxy.s1Size = totalGroups;
  PivotCoarseIndexerKernel<PivotProxyType<LIT>> op;
  op.Init(workspace, key, workspace + offsets.weights, workspace + offsets.lengths, keyLengths, table,
          workspace + offsets.indices, values, workspace + offsets.native, &proxy, &pipe, groups);
  op.Process();

  if ASCEND_IS_AIV {
    // LD may finish on another core. Publish every complete top-k before expansion.
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    pipe.Reset();
    TBuf<TPosition::VECCALC> resultBuf, lengthBuf;
    pipe.InitBuffer(resultBuf, PIVOT_TOPK * sizeof(int32_t));
    pipe.InitBuffer(lengthBuf, CoarseWorkspace::DMA_BLOCK_BYTES);
    auto result = resultBuf.Get<int32_t>();
    auto lengths = lengthBuf.Get<int32_t>();
    GlobalTensor<int32_t> outputLengths;
    outputLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(aslk));
    GlobalTensor<int32_t> packed, output;
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace + offsets.indices));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
#if PIVOT_REUSE_DEBUG
    // flag8: the visibleKeys == 0 case is the one that decides whether a zero key
    // length is a safe empty list or silent corruption, so it is counted out of the
    // group table the device just wrote. The expansion loop below is partitioned
    // across the AIVs, so the count is taken by scanning the whole table on the
    // recording core rather than by summing per-core counters; the scan is
    // debug-only, and the first three offenders are printed with their geometry.
    {
      uint32_t zeroVisible = 0;
      uint32_t shown = 0;
      for (uint32_t g = 0; g < totalGroups; ++g) {
        PivotGroup group{groups[g].sourceRow, groups[g].copies, groups[g].visibleKeys};
        if (group.visibleKeys != 0) {
          continue;
        }
        ++zeroVisible;
        if (shown < 3) {
          LI_PIVOT_PROBE("[LI_PIVOT] flag8 zeroGroup g=%u sourceRow=%u copies=%u visibleKeys=%u\n", g, group.sourceRow,
                         group.copies, group.visibleKeys);
          ++shown;
        }
      }
      LI_PIVOT_PROBE("[LI_PIVOT] flag8 zeroSummary zeroVisibleGroups=%u totalGroups=%u\n", zeroVisible, totalGroups);
    }
#endif
    for (uint32_t g = GetBlockIdx(); g < totalGroups; g += GetBlockNum() * 2) {
      PivotGroup group{groups[g].sourceRow, groups[g].copies, groups[g].visibleKeys};
      if (group.visibleKeys == 0) {
        Duplicate(result, int32_t(-1), PIVOT_TOPK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
      } else {
        DataCopy(result, packed[static_cast<uint64_t>(g) * PIVOT_TOPK], PIVOT_TOPK);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      }
      for (uint32_t r = 0; r < group.copies; ++r) {
        lengths.SetValue(r, group.visibleKeys == 0 ? 0 : Min(PIVOT_TOPK, group.visibleKeys + r));
        if (group.visibleKeys != 0) {
          for (uint32_t j = 0; j < r; ++j) {
            result.SetValue(PIVOT_TOPK - r + j, group.visibleKeys + j);
          }
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(output[static_cast<uint64_t>(group.sourceRow + r) * PIVOT_TOPK], result, PIVOT_TOPK);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
      }
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      DataCopyPad(outputLengths[group.sourceRow], lengths,
                  {1, static_cast<uint16_t>(group.copies * sizeof(int32_t)), 0, 0});
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
  }
  return true;
}
}  // namespace LIKernel
#endif
