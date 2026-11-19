/* Copyright (c) 2026. Licensed under CANN Open Software License Agreement v2.0. */
#ifndef PIVOT_COARSE_SELECT_GROUPED_H
#define PIVOT_COARSE_SELECT_GROUPED_H

namespace PivotCoarseSelectKernelImpl {

struct CoarseWindow {
    uint32_t begin;
    uint32_t count;
    uint32_t upper;
    uint32_t end;
};

__aicore__ inline CoarseWindow GetCoarseWindow(GlobalTensor<int32_t> &ends, GlobalTensor<int32_t> &lengths,
    uint32_t row, const PivotCoarseSelectTilingData &tiling)
{
    const int32_t begin = row == 0 ? 0 : ends.GetValue(row - 1);
    const int32_t end = ends.GetValue(row);
    if (begin < 0 || end <= begin || end - begin > tiling.groupSize ||
        (tiling.proxyMode != 0 && end > tiling.s1Size)) {
        return {0, 0, 0, 0};
    }

    const uint32_t count = end - begin;
    const uint32_t horizon = Min(Max(lengths.GetValue(row), 0), tiling.s2Size);
    const uint32_t upper = horizon > count ? horizon - count : 0;
    return {static_cast<uint32_t>(begin), count, upper, horizon};
}

template <typename T>
__aicore__ inline void MeanCoarseRows(GlobalTensor<T> &src, GlobalTensor<T> &dst, uint32_t width,
    const CoarseWindow &group, uint32_t row, const LocalTensor<T> &raw,
    const LocalTensor<float> &sum, const LocalTensor<float> &converted)
{
    for (uint32_t i = 0; i < group.count; ++i) {
        const DataCopyExtParams copy{1, static_cast<uint32_t>(width * sizeof(T)), 0, 0, 0};
        DataCopyPad(raw, src[static_cast<uint64_t>(group.begin + i) * width], copy,
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        auto out = i == 0 ? sum : converted;
        if constexpr (std::is_same_v<T, float>) {
            Adds(out, raw, 0.0f, width);
        } else {
            Cast(out, raw, RoundMode::CAST_NONE, width);
        }
        PipeBarrier<PIPE_V>();
        if (i != 0) {
            Add(sum, sum, converted, width);
            PipeBarrier<PIPE_V>();
        }
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    }
    Muls(sum, sum, 1.0f / group.count, width);
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same_v<T, float>) {
        Adds(raw, sum, 0.0f, width);
    } else {
        Cast(raw, sum, RoundMode::CAST_RINT, width);
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyPad(dst[static_cast<uint64_t>(row) * width], raw,
        {1, static_cast<uint16_t>(width * sizeof(T)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

// Reduce exact integer membership over the retained prefix, including partial vector blocks.
__aicore__ inline bool WindowKeyAlreadySelected(const LocalTensor<int32_t> &ids, const LocalTensor<int32_t> &work,
    int32_t key, uint32_t retained, uint32_t width)
{
    Duplicate(work, 1, width);
    PipeBarrier<PIPE_V>();
    Adds(work, ids, -key, retained);
    PipeBarrier<PIPE_V>();
    Mins(work, work, 1, width);
    PipeBarrier<PIPE_V>();
    Maxs(work, work, -1, width);
    PipeBarrier<PIPE_V>();
    Mul(work, work, work, width);
    PipeBarrier<PIPE_V>();

    constexpr uint32_t lanes = PivotCoarseSelectVec::BLOCK_BYTES / sizeof(int32_t); // One aligned vector block.
    for (uint32_t half = width / 2; half >= lanes; half /= 2) {
        Add(work, work, work[half], half);
        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    uint32_t mismatches = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        mismatches += work.GetValue(lane);
    }
    return mismatches != width;
}

// A Cube key load can serve two proxies only when every shared logical page maps to the same physical page.
constexpr uint32_t PAGE_COMPARE_TILE = 256;
__aicore__ inline bool CanShareKeyPages(const GlobalTensor<int32_t> &table, uint64_t first, uint64_t second,
    uint32_t pages, const LocalTensor<int32_t> &left, const LocalTensor<int32_t> &right,
    const LocalTensor<int32_t> &difference)
{
    if (pages == 0 || table.GetValue(first) != table.GetValue(second) ||
        table.GetValue(first + pages - 1) != table.GetValue(second + pages - 1)) {
        return false;
    }
    constexpr uint32_t block = PivotCoarseSelectVec::BLOCK_BYTES / sizeof(int32_t);
    for (uint32_t offset = 0; offset < pages; offset += PAGE_COMPARE_TILE) {
        const uint32_t count = Min(pages - offset, PAGE_COMPARE_TILE);
        const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        const DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(left, table[first + offset], copy, pad);
        DataCopyPad(right, table[second + offset], copy, pad);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Duplicate(difference, 0, PAGE_COMPARE_TILE);
        PipeBarrier<PIPE_V>();
        Sub(difference, left, right, count);
        PipeBarrier<PIPE_V>();
        Mins(difference, difference, 1, PAGE_COMPARE_TILE);
        PipeBarrier<PIPE_V>();
        Maxs(difference, difference, -1, PAGE_COMPARE_TILE);
        PipeBarrier<PIPE_V>();
        Mul(difference, difference, difference, PAGE_COMPARE_TILE);
        PipeBarrier<PIPE_V>();
        for (uint32_t half = PAGE_COMPARE_TILE / 2; half >= block; half /= 2) {
            Add(difference, difference, difference[half], half);
            PipeBarrier<PIPE_V>();
        }
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        for (uint32_t lane = 0; lane < block; ++lane) {
            if (difference.GetValue(lane) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <typename LIT>
__aicore__ inline void RunGroupedCoarse(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *queryEnds, __gm__ uint8_t *keyLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *indices, __gm__ uint8_t *values, __gm__ uint8_t *aslk,
    __gm__ uint8_t *workspace, const PivotCoarseSelectTilingData &tiling, TPipe &pipe)
{
    using Q = typename LIT::queryType;
    using W = typename std::conditional<LIT::weightsTypeFlag, float, Q>::type;
    const bool raw = tiling.proxyMode != 0;
    const uint32_t rows = tiling.bSize;
    const uint32_t heads = tiling.gSize;
    const uint32_t qWidth = heads * INDEXER_HEAD_DIM;
    const uint32_t topk = tiling.sparseCount;
    const auto layout = GroupedWorkspace(rows, heads, topk, raw);
    GlobalTensor<int32_t> ends, lengths;
    ends.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryEnds));
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(keyLengths));

    if ASCEND_IS_AIV {
        GlobalTensor<int32_t> proxyEnds, proxyLengths, pages;
        GlobalTensor<uint32_t> sharedPairs;
        proxyEnds.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + layout.queryEnds));
        proxyLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + layout.history));
        sharedPairs.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace + layout.sharedPairs));

        TBuf<TPosition::VECCALC> rawBuf, sumBuf, convertedBuf, metaBuf, pageCompareBuf;
        const bool checkPairs = topk == COARSE_TOPK && rows >= 2 * tiling.usedCoreNum;
        if (checkPairs) {
            pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
            pipe.InitBuffer(pageCompareBuf, 3 * PAGE_COMPARE_TILE * sizeof(int32_t));
        }
        pipe.InitBuffer(metaBuf, PivotCoarseSelectVec::BLOCK_BYTES);
        auto meta = metaBuf.Get<int32_t>();
        GlobalTensor<Q> rawQ, proxyQ;
        GlobalTensor<W> rawW, proxyW;
        if (raw) {
            pipe.InitBuffer(rawBuf, Max(qWidth * sizeof(Q), heads * sizeof(W)));
            pipe.InitBuffer(sumBuf, qWidth * sizeof(float));
            pipe.InitBuffer(convertedBuf, qWidth * sizeof(float));
            rawQ.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(query));
            proxyQ.SetGlobalBuffer(reinterpret_cast<__gm__ Q *>(workspace));
            rawW.SetGlobalBuffer(reinterpret_cast<__gm__ W *>(weights));
            proxyW.SetGlobalBuffer(reinterpret_cast<__gm__ W *>(workspace + layout.weights));
        }
        for (uint32_t row = GetBlockIdx(); row < rows; row += GetBlockNum() * 2) {
            const auto group = GetCoarseWindow(ends, lengths, row, tiling);
            if (raw && group.upper != 0) {
                MeanCoarseRows(rawQ, proxyQ, qWidth, group, row, rawBuf.Get<Q>(),
                    sumBuf.Get<float>(), convertedBuf.Get<float>());
                MeanCoarseRows(rawW, proxyW, heads, group, row, rawBuf.Get<W>(),
                    sumBuf.Get<float>(), convertedBuf.Get<float>());
            }
            if (checkPairs && row % 2 == 0) {
                const auto tableStride = tiling.maxBlockNumPerBatch;
                uint32_t pagesInCommon = 0;
                if (row + 1 < rows) {
                    const auto next = GetCoarseWindow(ends, lengths, row + 1, tiling);
                    pagesInCommon = CeilDiv(Min(group.upper, next.upper), tiling.blockSize);
                }
                auto pageUb = pageCompareBuf.Get<int32_t>();
                const bool shared = CanShareKeyPages(pages, static_cast<uint64_t>(row) * tableStride,
                    static_cast<uint64_t>(row + 1) * tableStride, pagesInCommon, pageUb, pageUb[PAGE_COMPARE_TILE],
                    pageUb[2 * PAGE_COMPARE_TILE]);
                meta.SetValue(0, shared);
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(sharedPairs[row / 2], meta.ReinterpretCast<uint32_t>(), {1, sizeof(uint32_t), 0, 0});
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
            meta.SetValue(0, row + 1);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(proxyEnds[row], meta, {1, sizeof(int32_t), 0, 0});
            // A data copy starts at an aligned UB block.
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            meta.SetValue(0, group.upper);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(proxyLengths[row], meta, {1, sizeof(int32_t), 0, 0});
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        CrossCoreSetFlag<2, PIPE_MTE3>(PIVOT_READY_EVENT);
        pipe.Reset();
    } else {
        CrossCoreWaitFlag(PIVOT_READY_EVENT);
    }

    PivotCoarseSelectTilingData scan = tiling;
    scan.s1Size = rows;
    scan.sparseMode = 0;
    PivotCoarseSelectKernel<LIT> backend;
    backend.Init(raw ? workspace : query, key, raw ? workspace + layout.weights : weights,
        workspace + layout.queryEnds, workspace + layout.history, blockTable, workspace + layout.picks,
        values, workspace + layout.scan, &scan, &pipe, nullptr,
        reinterpret_cast<__gm__ const uint32_t *>(workspace + layout.sharedPairs));
    backend.Process();

    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        pipe.Reset();
        TBuf<TPosition::VECCALC> idsBuf, scratchBuf, lengthBuf;
        pipe.InitBuffer(idsBuf, topk * sizeof(int32_t));
        pipe.InitBuffer(scratchBuf, topk * sizeof(int32_t));
        pipe.InitBuffer(lengthBuf, PivotCoarseSelectVec::BLOCK_BYTES);
        auto ids = idsBuf.Get<int32_t>();
        auto scratch = scratchBuf.Get<int32_t>();
        auto length = lengthBuf.Get<int32_t>();
        GlobalTensor<int32_t> picks, out, lengthsOut;
        picks.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + layout.picks));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));
        lengthsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aslk));
        const uint32_t budget = 2 * tiling.groupSize - 1;
        for (uint32_t row = GetBlockIdx(); row < rows; row += GetBlockNum() * 2) {
            const auto group = GetCoarseWindow(ends, lengths, row, tiling);
            const uint32_t prefix = Min(group.upper, topk - budget);
            if (prefix > 0) {
                DataCopy(ids, picks[static_cast<uint64_t>(row) * topk], topk);
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            }
            uint32_t size = prefix;
            const uint32_t past = tiling.groupSize - 1;
            const uint32_t start = group.upper > past ? group.upper - past : 0;
            for (uint32_t pos = start; pos < group.end; ++pos) {
                if (pos >= group.upper || prefix == 0 || !WindowKeyAlreadySelected(ids, scratch, pos, prefix, topk)) {
                    ids.SetValue(size++, pos);
                }
            }
            // Fill the aligned suffix with the vector pipe, plus any scalar tail in the prefix block.
            const uint32_t lanes = PivotCoarseSelectVec::BLOCK_BYTES / sizeof(int32_t);
            const uint32_t alignedSize = PivotCoarseSelectCommon::Align(size, lanes);
            for (uint32_t i = size; i < alignedSize; ++i) {
                ids.SetValue(i, -1);
            }
            if (alignedSize < topk) {
                Duplicate(ids[alignedSize], -1, topk - alignedSize);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            }
            length.SetValue(0, size);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopy(out[static_cast<uint64_t>(row) * topk], ids, topk);
            DataCopyPad(lengthsOut[row], length, {1, sizeof(int32_t), 0, 0});
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}

__aicore__ inline void WriteLocalLengths(__gm__ uint8_t *queryEnds, __gm__ uint8_t *keyLengths,
    __gm__ uint8_t *aslk, const PivotCoarseSelectTilingData &tiling, TPipe &pipe)
{
    if ASCEND_IS_AIV {
        pipe.Reset();
        constexpr uint32_t LENGTH_TILE = PivotCoarseSelectVec::BLOCK_BYTES / sizeof(int32_t);
        TBuf<TPosition::VECCALC> lengthBuf;
        pipe.InitBuffer(lengthBuf, LENGTH_TILE * sizeof(int32_t));
        auto lengths = lengthBuf.Get<int32_t>();
        GlobalTensor<int32_t> ends, keys, output;
        ends.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryEnds));
        keys.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(keyLengths));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(aslk));
        int32_t begin = 0;
        const uint32_t core = GetBlockIdx();
        const uint32_t coreCount = GetBlockNum() * 2;
        for (uint32_t b = 0; b < tiling.bSize; ++b) {
            const int32_t end = Min(ends.GetValue(b), static_cast<int32_t>(tiling.s1Size));
            const int32_t kv = keys.GetValue(b);
            for (int32_t row = begin + core * LENGTH_TILE; row < end; row += coreCount * LENGTH_TILE) {
                const uint32_t count = Min<int32_t>(end - row, LENGTH_TILE);
                for (uint32_t lane = 0; lane < count; ++lane) {
                    const int32_t visible = kv - (end - row) + lane + 1;
                    lengths.SetValue(lane, Min<int32_t>(Max(visible, 0), tiling.sparseCount));
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(output[row], lengths, {1, static_cast<uint16_t>(count * sizeof(int32_t)), 0, 0});
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
            begin = Max(begin, end);
        }
    }
}

} // namespace PivotCoarseSelectKernelImpl
#endif
