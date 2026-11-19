/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "indexer_coarse_screen_template_tiling_key.h"

#include "arch22/indexer_coarse_screen_kernel_arch22.h"
#include "arch22/indexer_coarse_screen_pivot_coarse.h"
#include "arch22/indexer_coarse_screen_reuse.h"

using namespace LIKernel;

template <typename LIT>
__aicore__ inline void RunIndexerCoarseScreen(__gm__ uint8_t* query, __gm__ uint8_t* weights, __gm__ uint8_t* key,
                                              __gm__ uint8_t* queryEnds, __gm__ uint8_t* keyLengths,
                                              __gm__ uint8_t* table, __gm__ uint8_t* candidates, __gm__ uint8_t* aslk,
                                              __gm__ uint8_t* workspace, const IndexerCoarseScreenTilingData& tiling,
                                              TPipe& pipe) {
  if (!CoarseWorkspace::UseReuse(tiling.outW, tiling.modifyLocalwindow, tiling.gSize, tiling.groupSize)) {
    RunPivotCoarse<LIT>(query, key, weights, queryEnds, keyLengths, table, candidates, aslk, workspace, tiling, pipe);
    return;
  }

  LITilingData native;
  native.bSize = tiling.bSize;
  native.s1Size = tiling.queryRows;
  native.s2Size = tiling.s2Size;
  native.sparseCount = tiling.outW;
  native.sparseMode = 3;
  native.gSize = tiling.gSize;
  native.blockSize = tiling.blockSize;
  native.maxBlockNumPerBatch = tiling.maxBlockNumPerBatch;
  native.usedCoreNum = tiling.usedCoreNum;
  if (TryPivotReuse<LIT>(query, key, weights, queryEnds, keyLengths, table, candidates, nullptr, workspace, native,
                         pipe, aslk)) {
    return;
  }

  if ASCEND_IS_AIV {
    GlobalTensor<int32_t> ends, keys, lengths;
    ends.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryEnds));
    keys.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keyLengths));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(aslk));
    // Scalar GM stores to neighboring lengths can race on the same cache line.
    constexpr int32_t lengthsPerBlock = CoarseWorkspace::DMA_BLOCK_BYTES / sizeof(int32_t);
    TBuf<TPosition::VECCALC> lengthsBuffer;
    pipe.InitBuffer(lengthsBuffer, CoarseWorkspace::DMA_BLOCK_BYTES);
    auto local = lengthsBuffer.Get<int32_t>();
    for (uint32_t b = GetBlockIdx(); b < native.bSize; b += GetBlockNum() * 2) {
      const int32_t totalRows = tiling.queryRows;
      const int32_t start = b == 0 ? 0 : Min(totalRows, Max(0, ends.GetValue(b - 1)));
      const int32_t end = Min(totalRows, Max(start, ends.GetValue(b)));
      const int32_t visible = keys.GetValue(b) - (end - start);
      for (int32_t row = start; row < end; row += lengthsPerBlock) {
        const int32_t count = Min(lengthsPerBlock, end - row);
        for (int32_t i = 0; i < count; ++i) {
          local.SetValue(i, Min<int32_t>(tiling.outW, Max(0, visible + row + i - start + 1)));
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(lengths[row], local, {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      }
    }
    pipe.Reset();
  }
  PivotCoarseIndexerKernel<LIT> indexer;
  indexer.Init(query, key, weights, queryEnds, keyLengths, table, candidates, nullptr, workspace, &native, &pipe);
  indexer.Process();
}

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void indexer_coarse_screen(__gm__ uint8_t* qBar, __gm__ uint8_t* wBar, __gm__ uint8_t* key,
                                                 __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
                                                 __gm__ uint8_t* blocktable, __gm__ uint8_t* candidatesOut,
                                                 __gm__ uint8_t* aslkOut, __gm__ uint8_t* workspace,
                                                 __gm__ uint8_t* tiling) {
  TPipe tPipe;
  __gm__ uint8_t* user = GetUserWorkspace(workspace);
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA_WITH_STRUCT(IndexerCoarseScreenTilingData, tilingData, tiling);
  using Q = std::conditional_t<DT_Q == LI_TPL_FP16, half, bfloat16_t>;
  using Type = LIType<Q, Q, int32_t, PAGE_ATTENTION, LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG>;
  RunIndexerCoarseScreen<Type>(qBar, wBar, key, actualSeqLengthsQ, actualSeqLengths, blocktable, candidatesOut, aslkOut,
                               user, tilingData, tPipe);
}
