/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pivot_coarse_select.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "pivot_coarse_select_template_tiling_key.h"

#include "arch22/pivot_coarse_select_kernel.h"
#if (__CCE_AICORE__ == 220)
#include "arch22/pivot_coarse_select_pivot_reuse.h"
#include "arch22/pivot_coarse_select_grouped.h"
#endif

using namespace PivotCoarseSelectKernelImpl;

template <typename LIT>
__aicore__ inline void RunPivotCoarseSelect(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *queryLengths, __gm__ uint8_t *keyLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *indices, __gm__ uint8_t *values, __gm__ uint8_t *aslk, __gm__ uint8_t *workspace,
    const PivotCoarseSelectTilingData &tiling, TPipe &pipe)
{
#if (__CCE_AICORE__ == 220)
    if (tiling.groupSize != 0 && !tiling.modifyLocalwindow) {
        RunGroupedCoarse<LIT>(query, key, weights, queryLengths, keyLengths, blockTable, indices, values, aslk,
            workspace, tiling, pipe);
        return;
    }
    const bool reused = TryPivotReuse<LIT>(query, key, weights, queryLengths, keyLengths, blockTable,
        indices, values, workspace, tiling, pipe);
    if (!reused) {
#endif
        PivotCoarseSelectKernel<LIT> op;
        op.Init(query, key, weights, queryLengths, keyLengths, blockTable, indices, values, workspace, &tiling, &pipe);
        op.Process();
#if (__CCE_AICORE__ == 220)
    }
    if (tiling.modifyLocalwindow) {
        WriteLocalLengths(queryLengths, keyLengths, aslk, tiling, pipe);
    }
#endif
}

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void pivot_coarse_select(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                             __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                             __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
                                             __gm__ uint8_t *sparseValues, __gm__ uint8_t *aslk,
                                             __gm__ uint8_t *workspace,
                                             __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(PivotCoarseSelectTilingData, tilingData, tiling);
    if constexpr (DT_Q == LI_TPL_FP16 && DT_K == LI_TPL_FP16 && DT_OUT == LI_TPL_INT32) {
        using LIT = LIType<half, half, int32_t, PAGE_ATTENTION, LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T), DT_W_FLAG>;
        RunPivotCoarseSelect<LIT>(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, sparseIndices,
            sparseValues, aslk, user, tilingData, tPipe);
    } else {
        using LIT = LIType<bfloat16_t, bfloat16_t, int32_t, PAGE_ATTENTION, LI_LAYOUT(LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
            DT_W_FLAG>;
        RunPivotCoarseSelect<LIT>(query, key, weights, actualSeqLengthsQ, actualSeqLengths, blocktable, sparseIndices,
            sparseValues, aslk, user, tilingData, tPipe);
    }
}
