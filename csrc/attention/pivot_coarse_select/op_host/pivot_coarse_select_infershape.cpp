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
 * \file pivot_coarse_select_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"


using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t ATTR_GROUP_SIZE_INDEX = 7;
constexpr uint32_t ATTR_MODIFY_LOCALWINDOW_INDEX = 9;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;

static ge::graphStatus InferShapePivotCoarseSelect(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("PivotCoarseSelect", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *seleced_count = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seleced_count);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
        OP_LOGE(context, "The attr layout_query should be TND or BSND, but got %s.", inputLayoutQueryPtrStr.c_str()),
        return ge::GRAPH_FAILED);

    auto *aslkShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, aslkShape);
    aslkShape->SetDimNum(1);
    const auto groupSize = attrs->GetInt(ATTR_GROUP_SIZE_INDEX);
    const auto modifyLocalwindow = attrs->GetBool(ATTR_MODIFY_LOCALWINDOW_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, modifyLocalwindow);
    aslkShape->SetDim(0, 0);
    if (*groupSize != 0) {
        const auto *queryEnds = context->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, queryEnds);
        const auto rows = *modifyLocalwindow ? queryShape->GetDim(0) : queryEnds->GetDim(0);
        aslkShape->SetDim(0, rows);
        sparseIndicesShape->SetDimNum(3);
        sparseIndicesShape->SetDim(0, rows);
        sparseIndicesShape->SetDim(1, 1);
        sparseIndicesShape->SetDim(2, *seleced_count);
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
        return ge::GRAPH_SUCCESS;
    }

    sparseIndicesShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        OP_CHECK_IF(
            queryShape->GetDimNum() != 4,
            OP_LOGE(context, "Layout BSND, queryDims (%zu) must be 4!", queryShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        sparseIndicesShape->SetDim(2, keyShape->GetDim(2));   // 2:Dim N
        sparseIndicesShape->SetDim(3, *seleced_count);        // 3:Dim K
    } else {
        OP_CHECK_IF(
            queryShape->GetDimNum() != 3,
            OP_LOGE(context, "Layout TND, queryDims (%zu) must be 3!", queryShape->GetDimNum()),
            return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));                      // 0:Dim T
        int32_t nDimIndex = (inputLayoutKeyPtrStr == "PA_BSND") ? 2 : 1; // 2:Key Dim N
        sparseIndicesShape->SetDim(1, keyShape->GetDim(nDimIndex));                // 1:Dim N
        sparseIndicesShape->SetDim(2, *seleced_count);                             // 2:Dim K
    }

    const bool *return_value = attrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, return_value);
    bool returnValueFlag = (return_value != nullptr) ? *return_value : false;
    if (returnValueFlag) {
        *sparseValuesShape = *sparseIndicesShape;
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }
    OP_LOGI(context->GetNodeName(), "PivotCoarseSelect InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePivotCoarseSelect(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("PivotCoarseSelect", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter PivotCoarseSelect InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = ge::DT_INT32;
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    context->SetOutputDataType(1, context->GetInputDataType(QUERY_INDEX));
    context->SetOutputDataType(2, ge::DT_INT32);
    OP_LOGI(context->GetNodeName(), "PivotCoarseSelect InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PivotCoarseSelect)
    .InferShape(InferShapePivotCoarseSelect)
    .InferDataType(InferDataTypePivotCoarseSelect);
} // namespace ops
