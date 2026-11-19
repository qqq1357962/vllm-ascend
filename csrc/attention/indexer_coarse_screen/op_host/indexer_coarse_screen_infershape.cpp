/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t CANDIDATES_INDEX = 0;
constexpr uint32_t ASLK_OUT_INDEX = 1;
constexpr uint32_t ATTR_COARSE_COUNT_INDEX = 0;
constexpr uint32_t ATTR_HAS_WINDOW_INDEX = 1;

static ge::graphStatus InferShapeIndexerCoarseScreen(gert::InferShapeContext* context) {
  OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerCoarseScreen", "InferShapeContext is nullptr!"),
              return ge::GRAPH_FAILED);
  const gert::Shape* qBarShape = context->GetInputShape(QBAR_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, qBarShape);
  const gert::Shape* actualSeqQShape = context->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqQShape);

  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* coarseCount = attrs->GetInt(ATTR_COARSE_COUNT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, coarseCount);
  OP_CHECK_IF((*coarseCount != 2048) && (*coarseCount != 4096),
              OP_LOGE(context, "coarse_count must be 2048 or 4096, including the local window."),
              return ge::GRAPH_FAILED);
  const int64_t* hasWindow = attrs->GetInt(ATTR_HAS_WINDOW_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, hasWindow);
  OP_CHECK_IF(*hasWindow != 1, OP_LOGE(context, "has_window must be 1; other modes are unsupported."),
              return ge::GRAPH_FAILED);
  const int64_t* gSize = attrs->GetInt(2);
  OP_CHECK_NULL_WITH_CONTEXT(context, gSize);
  const int64_t* proxyMode = attrs->GetInt(3);
  OP_CHECK_NULL_WITH_CONTEXT(context, proxyMode);
  OP_CHECK_IF((*gSize <= 0) || (*gSize > 16),
              OP_LOGE(context, "group_size must be in (0, 16], but now is %ld.", *gSize), return ge::GRAPH_FAILED);

  OP_CHECK_IF(qBarShape->GetDimNum() != 3,
              OP_LOGE(context, "Layout TND, q_bar dims (%zu) must be 3!", qBarShape->GetDimNum()),
              return ge::GRAPH_FAILED);
  const int64_t reqNum = actualSeqQShape->GetDim(0);
  const bool* modifyLocalwindow = attrs->GetBool(4);
  OP_CHECK_NULL_WITH_CONTEXT(context, modifyLocalwindow);
  OP_CHECK_IF(*modifyLocalwindow && *proxyMode == 0,
              OP_LOGE(context, "is_modify_localwindow requires raw queries (proxy_mode=1)."), return ge::GRAPH_FAILED);
  const int64_t outputRows = *modifyLocalwindow ? qBarShape->GetDim(0) : reqNum;

  OP_CHECK_IF((*proxyMode == 0) && (qBarShape->GetDim(0) != reqNum),
              OP_LOGE(context, "q_bar dim0 (%ld) must equal actual_seq_lengths_query dim0 (%ld).", qBarShape->GetDim(0),
                      reqNum),
              return ge::GRAPH_FAILED);
  OP_CHECK_IF((*proxyMode != 0) && (qBarShape->GetDim(0) < reqNum),
              OP_LOGE(context, "raw query dim0 (%ld) must cover actual_seq_lengths_query dim0 (%ld).",
                      qBarShape->GetDim(0), reqNum),
              return ge::GRAPH_FAILED);

  gert::Shape* candidatesShape = context->GetOutputShape(CANDIDATES_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, candidatesShape);
  gert::Shape* aslkOutShape = context->GetOutputShape(ASLK_OUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, aslkOutShape);

  int64_t outW = *coarseCount;
  candidatesShape->SetDimNum(2);
  candidatesShape->SetDim(0, outputRows);
  candidatesShape->SetDim(1, outW);

  aslkOutShape->SetDimNum(1);
  aslkOutShape->SetDim(0, outputRows);
  OP_LOGI(context->GetNodeName(), "IndexerCoarseScreen InferShape end.");

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeIndexerCoarseScreen(gert::InferDataTypeContext* context) {
  OP_CHECK_IF(context == nullptr, OP_LOGE("IndexerCoarseScreen", "InferDataTypeContext is nullptr!"),
              return ge::GRAPH_FAILED);
  OP_LOGI(context->GetNodeName(), "Enter IndexerCoarseScreen InferDataType impl.");

  context->SetOutputDataType(CANDIDATES_INDEX, ge::DT_INT32);
  context->SetOutputDataType(ASLK_OUT_INDEX, ge::DT_INT32);
  OP_LOGI(context->GetNodeName(), "IndexerCoarseScreen InferDataType end.");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IndexerCoarseScreen)
    .InferShape(InferShapeIndexerCoarseScreen)
    .InferDataType(InferDataTypeIndexerCoarseScreen);
}  // namespace ops
