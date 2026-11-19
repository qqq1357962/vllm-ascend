/*
 * Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0.
 */
#ifndef INDEXER_COARSE_SCREEN_OUTPUT_H
#define INDEXER_COARSE_SCREEN_OUTPUT_H

namespace LIKernel {
__aicore__ inline void WriteCoarseOutput(const CoarseWorkspace::Group& group, uint32_t budget, uint32_t outWidth,
                                         const LocalTensor<float>& sorted, LocalTensor<float> scores,
                                         LocalTensor<uint32_t> ids, LocalTensor<int32_t> lengths,
                                         GlobalTensor<int32_t> output, GlobalTensor<int32_t> aslk) {
  const uint32_t window = group.windowEnd - group.history;
  const uint32_t selected = Min(group.history, outWidth - budget);
  const uint32_t valid = selected + window;
  constexpr uint32_t align = CoarseWorkspace::DMA_BLOCK_BYTES / sizeof(int32_t);
  constexpr uint32_t SORT_BLOCK_ENTRIES = 32;
  for (uint32_t chunk = 0; chunk < outWidth; chunk += BASE_TOPK) {
    auto signedIds = ids.ReinterpretCast<int32_t>();
    if (selected > chunk) {
      Extract(scores, ids, sorted[chunk * 2], BASE_TOPK / SORT_BLOCK_ENTRIES);
    }
    const uint32_t validInChunk = valid > chunk ? Min(BASE_TOPK, valid - chunk) : 0;
    const uint32_t pad = (validInChunk + align - 1) / align * align;
    if (pad < BASE_TOPK) {
      PipeBarrier<PIPE_V>();
      Duplicate(signedIds[pad], int32_t(-1), BASE_TOPK - pad);
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (uint32_t idx = Max(chunk, selected); idx < chunk + pad; ++idx) {
      signedIds.SetValue(idx - chunk, idx < valid ? group.history + idx - selected : -1);
    }
    for (uint32_t r = 0; r < group.copies; ++r) {
      // The first token's pool is reused. Subsequent tokens replace its
      // tail with the positions that became visible since the first token.
      const uint32_t replace = Min(r, valid);
      for (uint32_t idx = Max(chunk, valid - replace); idx < Min(chunk + BASE_TOPK, valid); ++idx) {
        signedIds.SetValue(idx - chunk, group.windowEnd + r - (valid - idx));
      }
      SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
      SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
      DataCopy(output[(static_cast<uint64_t>(group.outputRow) + r) * outWidth + chunk], signedIds, BASE_TOPK);
      SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
  }
  lengths.SetValue(0, valid);
  for (uint32_t r = 1; r < group.copies; ++r) {
    lengths.SetValue(r, valid);
  }
  SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
  DataCopyPad(aslk[group.outputRow], lengths, {1, static_cast<uint16_t>(group.copies * sizeof(int32_t)), 0, 0});
  SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}
}  // namespace LIKernel

#endif
