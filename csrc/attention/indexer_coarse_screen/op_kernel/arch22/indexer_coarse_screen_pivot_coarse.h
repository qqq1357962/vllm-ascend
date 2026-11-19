/*
 * Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0.
 */
#ifndef PIVOT_COARSE_INDEXER_PROXY_H
#define PIVOT_COARSE_INDEXER_PROXY_H

namespace LIKernel {
constexpr uint32_t COARSE_READY_EVENT = 7;

template <typename LIT>
struct PivotCoarseType : LIType<typename LIT::queryType, typename LIT::keyType, typename LIT::outputType, true,
                                LI_LAYOUT::BSND, LI_LAYOUT::PA_BSND, LIT::weightsTypeFlag> {
  static constexpr bool pivotPool4K = true;
};

template <typename T>
__aicore__ inline void CoarseMeanRows(__gm__ uint8_t* input, __gm__ uint8_t* output, uint32_t width, uint32_t sourceRow,
                                      uint32_t outputRow, uint32_t count, LocalTensor<T> raw, LocalTensor<float> accum,
                                      LocalTensor<float> temp) {
  GlobalTensor<T> src, dst;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
  for (uint32_t row = 0; row < count; ++row) {
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    DataCopyPad(raw, src[(static_cast<uint64_t>(sourceRow) + row) * width],
                {1, static_cast<uint16_t>(width * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    auto target = row == 0 ? accum : temp;
    Cast(target, raw, RoundMode::CAST_NONE, width);
    if (row != 0) {
      PipeBarrier<PIPE_V>();
      Add(accum, accum, temp, width);
    }
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
  DataCopyPad(dst[static_cast<uint64_t>(outputRow) * width], raw, {1, static_cast<uint16_t>(width * sizeof(T)), 0, 0});
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  // The next input DMA reuses raw: ordering only the vector pipe is not enough.
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename LIT>
__aicore__ inline void RunPivotCoarse(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
                                      __gm__ uint8_t* groupEnds, __gm__ uint8_t* keyLengths, __gm__ uint8_t* table,
                                      __gm__ uint8_t* indices, __gm__ uint8_t* aslk, __gm__ uint8_t* workspace,
                                      const IndexerCoarseScreenTilingData& tiling, TPipe& pipe) {
  using Q = typename LIT::queryType;
  using W = typename LIT::queryType;
  const uint32_t groups = tiling.bSize;
  const uint32_t heads = tiling.gSize;
  const uint32_t queryWidth = heads * CoarseWorkspace::HEAD_DIM;
  const auto layout = CoarseWorkspace::Layout(groups, heads, tiling.proxyMode != 0);
  const uint32_t budget = tiling.modifyLocalwindow ? tiling.groupSize : tiling.windowG;
  auto geometry = reinterpret_cast<__gm__ CoarseWorkspace::Group*>(workspace + layout.groups);
  auto historyGm = reinterpret_cast<__gm__ uint8_t*>(workspace + layout.history);

  if ASCEND_IS_AIV {
    GlobalTensor<uint32_t> ends, keys, history, metadata;
    GlobalTensor<int32_t> output, outputLengths;
    ends.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(groupEnds));
    keys.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(keyLengths));
    history.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(historyGm));
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(geometry));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    outputLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(aslk));

    TBuf<TPosition::VECCALC> rawBuf, accumBuf, tempBuf, metaBuf;
    pipe.InitBuffer(rawBuf, Max(queryWidth * sizeof(Q), BASE_TOPK * sizeof(int32_t)));
    pipe.InitBuffer(accumBuf, queryWidth * sizeof(float));
    pipe.InitBuffer(tempBuf, queryWidth * sizeof(float));
    pipe.InitBuffer(metaBuf,
                    Max(static_cast<uint32_t>(sizeof(CoarseWorkspace::Group)), CoarseWorkspace::DMA_BLOCK_BYTES));
    auto meta = metaBuf.Get<uint32_t>();
    constexpr uint32_t fields = sizeof(CoarseWorkspace::Group) / sizeof(uint32_t);
    for (uint32_t group = GetBlockIdx(); group < groups; group += GetBlockNum() * 2) {
      const uint32_t begin = group == 0 ? 0 : ends.GetValue(group - 1);
      const uint32_t end = ends.GetValue(group);
      const uint32_t qLen = end >= begin ? end - begin : 0;
      const uint32_t horizon = Min(keys.GetValue(group), tiling.s2Size);
      const uint32_t upper = horizon > qLen ? horizon - qLen : 0;
      const uint32_t windowBegin = upper >= tiling.groupSize ? upper - tiling.groupSize + 1 : 0;
      const uint32_t windowEnd =
          tiling.modifyLocalwindow ? Min(horizon, upper + 1) : Min(horizon, upper + tiling.groupSize);
      const uint32_t outRow = tiling.modifyLocalwindow ? begin : group;
      const bool validQuery = qLen > 0 && qLen <= tiling.groupSize && (!tiling.proxyMode || end <= tiling.queryRows);
      const bool shortRequest = tiling.modifyLocalwindow && horizon < qLen;
      const bool scan = validQuery && !shortRequest && windowBegin > 0;
      if (tiling.proxyMode && scan) {
        CoarseMeanRows<Q>(query, workspace, queryWidth, begin, group, qLen, rawBuf.Get<Q>(), accumBuf.Get<float>(),
                          tempBuf.Get<float>());
        CoarseMeanRows<W>(weights, workspace + layout.weights, heads, begin, group, qLen, rawBuf.Get<W>(),
                          accumBuf.Get<float>(), tempBuf.Get<float>());
      }
      const uint32_t copies = tiling.modifyLocalwindow ? qLen : 1;
      const CoarseWorkspace::Group row{outRow, copies, scan ? windowBegin : 0, validQuery ? windowEnd : 0};
      if (row.history == 0) {
        const uint32_t outputRows = tiling.modifyLocalwindow ? tiling.queryRows : groups;
        const uint32_t count = outRow < outputRows ? Min(copies, outputRows - outRow) : 0;
        for (uint32_t r = 0; r < count; ++r) {
          const uint32_t visible =
              shortRequest ? (horizon + r + 1 > qLen ? horizon + r + 1 - qLen : 0) : Min(horizon, row.windowEnd + r);
          const CoarseWorkspace::Group emptyHistory{outRow + r, 1, 0, visible};
          WriteCoarseOutput(emptyHistory, budget, tiling.outW, accumBuf.Get<float>(), tempBuf.Get<float>(),
                            rawBuf.Get<uint32_t>(), meta.ReinterpretCast<int32_t>(), output, outputLengths);
        }
      }
      meta.SetValue(0, row.outputRow);
      meta.SetValue(1, row.copies);
      meta.SetValue(2, row.history);
      meta.SetValue(3, row.windowEnd);
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      DataCopyPad(metadata[group * fields], meta, {1, sizeof(CoarseWorkspace::Group), 0, 0});
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      meta.SetValue(0, row.history);
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      DataCopyPad(history[group], meta, {1, sizeof(uint32_t), 0, 0});
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    CrossCoreSetFlag<2, PIPE_MTE3>(COARSE_READY_EVENT);
    pipe.Reset();
  } else {
    CrossCoreWaitFlag(COARSE_READY_EVENT);
  }

  LITilingData proxy;
  proxy.bSize = groups;
  proxy.s1Size = 1;
  proxy.s2Size = tiling.s2Size;
  proxy.sparseCount = BASE_TOPK;
  proxy.gSize = heads;
  proxy.blockSize = tiling.blockSize;
  proxy.maxBlockNumPerBatch = tiling.maxBlockNumPerBatch;
  proxy.usedCoreNum = tiling.usedCoreNum;
  PivotCoarseIndexerKernel<PivotCoarseType<LIT>> scan;
  scan.Init(tiling.proxyMode ? workspace : query, key, tiling.proxyMode ? workspace + layout.weights : weights, nullptr,
            historyGm, table, indices, nullptr, workspace + layout.scan, &proxy, &pipe, nullptr, geometry,
            reinterpret_cast<__gm__ int32_t*>(aslk), tiling.outW, budget, tiling.modifyLocalwindow != 0);
  scan.Process();
}
}  // namespace LIKernel

#endif
