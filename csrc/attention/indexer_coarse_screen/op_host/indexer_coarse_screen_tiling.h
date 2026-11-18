/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEXER_COARSE_SCREEN_TILING_H_
#define INDEXER_COARSE_SCREEN_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "../op_kernel/indexer_coarse_screen_workspace.h"

namespace optiling {
struct TilingRequiredParaInfo {
  const gert::CompileTimeTensorDesc* desc;
  const gert::StorageShape* shape;
};

struct TilingOptionalParaInfo {
  const gert::CompileTimeTensorDesc* desc;
  const gert::Tensor* tensor;
};

enum class DataLayout : uint32_t { BSND = 0, TND = 1, BnBsND = 2 };

constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t WBAR_INDEX = 1;
constexpr uint32_t KEY_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;

constexpr uint32_t CANDIDATES_INDEX = 0;
constexpr uint32_t ASLK_OUT_INDEX = 1;

constexpr uint32_t ATTR_COARSE_COUNT_INDEX = 0;
constexpr uint32_t ATTR_HAS_WINDOW_INDEX = 1;
constexpr uint32_t ATTR_GROUP_SIZE_INDEX = 2;
constexpr uint32_t ATTR_PROXY_MODE_INDEX = 3;
constexpr uint32_t ATTR_MODIFY_LOCALWINDOW_INDEX = 4;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;

constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t HEAD_DIM_LIMIT = CoarseWorkspace::HEAD_DIM;
constexpr uint32_t COARSE_SORT_CAPACITY = 4096;
constexpr uint32_t QUERY_HEAD_NUM_LIMIT = 64;
constexpr uint32_t GROUP_SIZE_LIMIT = CoarseWorkspace::MAX_GROUP_SIZE;

BEGIN_TILING_DATA_DEF(IndexerCoarseScreenTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, groupSize)
TILING_DATA_FIELD_DEF(uint32_t, windowG)
TILING_DATA_FIELD_DEF(uint32_t, outW)
TILING_DATA_FIELD_DEF(uint32_t, hasWindow)
TILING_DATA_FIELD_DEF(uint32_t, proxyMode)
TILING_DATA_FIELD_DEF(uint32_t, queryRows)
TILING_DATA_FIELD_DEF(uint32_t, modifyLocalwindow)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IndexerCoarseScreen, IndexerCoarseScreenTilingData)

struct IndexerCoarseScreenCompileInfo {};

struct LiParaInfo {
  TilingRequiredParaInfo query = {nullptr, nullptr};
  TilingRequiredParaInfo weights = {nullptr, nullptr};
  TilingRequiredParaInfo key = {nullptr, nullptr};
  TilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
  TilingOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
  TilingOptionalParaInfo blockTable = {nullptr, nullptr};
  TilingRequiredParaInfo candidatesOut = {nullptr, nullptr};
  TilingRequiredParaInfo aslkOut = {nullptr, nullptr};

  const int32_t* coarseCount = nullptr;
  const int32_t* hasWindow = nullptr;
  const int32_t* groupSizeAttr = nullptr;
  const int32_t* proxyMode = nullptr;
  const bool* modifyLocalwindow = nullptr;
};

class IndexerCoarseScreenTilingInfo {
 public:
  const char* opName = nullptr;
  fe::PlatFormInfos* platformInfo = nullptr;
  LiParaInfo opParamInfo;

  platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
  uint32_t bSize = 0;
  int64_t s2Size = 0;
  uint32_t gSize = 0;
  uint32_t groupSize = 0;

  bool pageAttentionFlag = true;
  int32_t blockSize = 0;
  uint32_t maxBlockNumPerBatch = 0;

  uint32_t sparseCount = 0;
  uint32_t windowG = 0;
  uint32_t outW = 0;
  uint32_t hasWindow = 1;
  uint32_t proxyMode = 0;
  uint32_t queryRows = 0;
  bool modifyLocalwindow = false;

  ge::DataType inputQType = ge::DT_FLOAT16;
  ge::DataType inputKType = ge::DT_FLOAT16;
  ge::DataType weightsType = ge::DT_FLOAT16;
  ge::DataType outputType = ge::DT_INT32;

  DataLayout inputQLayout = DataLayout::TND;
  DataLayout inputKLayout = DataLayout::BnBsND;
};

class IndexerCoarseScreenInfoParser {
 public:
  explicit IndexerCoarseScreenInfoParser(gert::TilingContext* context) : context_(context) {}
  ~IndexerCoarseScreenInfoParser() = default;

  ge::graphStatus CheckRequiredInOutExistence() const;
  ge::graphStatus CheckRequiredAttrExistence() const;
  ge::graphStatus CheckRequiredParaExistence() const;
  ge::graphStatus GetActualSeqLenSize(uint32_t& size, const gert::Tensor* tensor,
                                      const std::string& actualSeqLenName) const;
  ge::graphStatus GetOpName();
  ge::graphStatus GetNpuInfo();
  void GetOptionalInputParaInfo();
  void GetInputParaInfo();
  void GetOutputParaInfo();
  ge::graphStatus GetAndCheckAttrParaInfo();
  ge::graphStatus GetOpParaInfo();
  ge::graphStatus ValidateInputShapesMatch();
  ge::graphStatus GetAndCheckInOutDataType();
  ge::graphStatus GetBatchSize();
  ge::graphStatus GetHeadDim();
  ge::graphStatus GetAndCheckOptionalInput();
  ge::graphStatus CheckShapeDim();
  ge::graphStatus GetAndCheckBlockSize();
  ge::graphStatus CheckBlockCount();
  ge::graphStatus GetS2SizeForPageAttention();
  ge::graphStatus GetN1Size();
  ge::graphStatus GetAndCheckN2Size();
  ge::graphStatus GetGSize();
  ge::graphStatus GetGroupSize();
  void GenerateInfo(IndexerCoarseScreenTilingInfo& liInfo);
  ge::graphStatus ParseAndCheck(IndexerCoarseScreenTilingInfo& liInfo);

 public:
  gert::TilingContext* context_ = nullptr;
  const char* opName_;
  fe::PlatFormInfos* platformInfo_;
  LiParaInfo opParamInfo_;

  uint32_t bSize_ = 0;
  uint32_t n1Size_ = 0;
  uint32_t n2Size_ = 0;
  uint32_t gSize_ = 0;
  uint32_t groupSize_ = 0;
  int64_t s2Size_ = 0;
  uint32_t headDim_ = 0;

  uint32_t maxBlockNumPerBatch_ = 0;
  int32_t blockSize_ = 0;
  platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
  ge::DataType inputQType_ = ge::DT_FLOAT16;
  ge::DataType inputKType_ = ge::DT_FLOAT16;
  ge::DataType weightsType_ = ge::DT_FLOAT16;
  ge::DataType outputType_ = ge::DT_INT32;
};

class IndexerCoarseScreenTiling {
 public:
  explicit IndexerCoarseScreenTiling(gert::TilingContext* context) : context_(context) {};
  ge::graphStatus DoTiling(IndexerCoarseScreenTilingInfo* tilingInfo);

 private:
  gert::TilingContext* context_ = nullptr;
  IndexerCoarseScreenTilingData tilingData_;
};
}  // namespace optiling
#endif
