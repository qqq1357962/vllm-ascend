# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the real arch22 length method with bounds-checked host tensors."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_sparse_prefix_length_and_dispatch(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required")
    source = Path(__file__).resolve().parents[3] / (
        "csrc/attention/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_kernel_mla.h"
    )
    text = source.read_text()
    start = text.index(
        "template <typename SFAT>\n__aicore__ inline void SparseFlashAttentionMla<SFAT>::GetSparseActualSeqLen"
    )
    end = text.index("\ntemplate <typename SFAT>", start + 1)
    method = text[start:end]
    stub = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
enum class SFA_LAYOUT { TND, BSND, PA_BSND };
struct Tensor {
    std::vector<int32_t> values;
    unsigned reads = 0;
    int32_t GetValue(uint64_t i) { ++reads; return values.at(i); }
};
template<class SFAT> struct SparseFlashAttentionMla {
    static constexpr auto LAYOUT_T = SFAT::q;
    static constexpr auto KV_LAYOUT_T = SFAT::kv;
    struct {
        int32_t nextTokensPerBatch = 0;
        uint64_t curActualSeqLenOri = 8192, curActualSeqLen = 0;
    } tempLoopInfo;
    struct {
        uint32_t sparseMode = 0, sparseBlockCount = 2048, sparseBlockSize = 1;
        uint32_t mBaseSize = 64, gSize = 64;
    } constInfo;
    uint32_t kvHeadNum = 1;
    Tensor actualSeqLengthsQGm{{3, 8}}, topKGm;
    void GetSparseActualSeqLen(uint32_t, uint32_t, uint32_t);
};
template<SFA_LAYOUT Q, SFA_LAYOUT KV> struct Layout {
    static constexpr auto q = Q, kv = KV;
};
"""
    checks = r"""
template<SFA_LAYOUT Q, SFA_LAYOUT KV> void check_layout(bool eligible) {
    SparseFlashAttentionMla<Layout<Q, KV>> op;
    op.topKGm.values.assign(8 * 2048, -1);
    // Batch 1, query 2 maps to flat query 5, not tile 2 or batch 1.
    for (int i = 0; i < 257; ++i) op.topKGm.values[5 * 2048 + i] = 8191 - i;
    op.GetSparseActualSeqLen(1, 2, 0);
    assert(op.tempLoopInfo.curActualSeqLen == (eligible ? 257 : 2048));
    assert((op.topKGm.reads != 0) == eligible);
}
int main() {
    check_layout<SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND>(true);
    check_layout<SFA_LAYOUT::BSND, SFA_LAYOUT::PA_BSND>(false);
    check_layout<SFA_LAYOUT::TND, SFA_LAYOUT::TND>(false);
    for (unsigned count : {0, 1, 127, 128, 129, 255, 256, 257, 2047, 2048}) {
        for (unsigned kv : {0, 128, 8192}) {
            SparseFlashAttentionMla<Layout<SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND>> op;
            op.topKGm.values.assign(2048, -1);
            for (unsigned i = 0; i < count; ++i) op.topKGm.values[i] = count - i - 1;
            op.tempLoopInfo.curActualSeqLenOri = kv;
            op.GetSparseActualSeqLen(0, 0, 0);
            assert(op.tempLoopInfo.curActualSeqLen == std::min({count, kv, 2048u}));
            if (kv == 0) assert(op.topKGm.reads == 0);
            if (count == 2048 && kv != 0) assert(op.topKGm.reads == 1);
        }
    }
    for (unsigned excluded = 0; excluded < 3; ++excluded) {
        SparseFlashAttentionMla<Layout<SFA_LAYOUT::TND, SFA_LAYOUT::PA_BSND>> op;
        if (excluded == 0) op.constInfo.sparseMode = 3;
        if (excluded == 1) op.constInfo.sparseBlockSize = 128;
        if (excluded == 2) op.constInfo.mBaseSize = 128;
        // Any access to the deliberately empty tensor would fail.
        op.GetSparseActualSeqLen(0, 0, 0);
        assert(op.topKGm.reads == 0);
    }
}
"""
    program = tmp_path / "prefix.cpp"
    program.write_text(stub + method + checks)
    binary = tmp_path / "prefix"
    subprocess.run([compiler, "-std=c++17", "-Wall", str(program), "-o", str(binary)], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=30)
