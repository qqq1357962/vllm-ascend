// SPDX-License-Identifier: Apache-2.0

#include "kernel_operator.h"
#include "shmem.h"
#include <stdint.h>

using namespace AscendC;
constexpr uint64_t SLOT_BYTES = 10ULL << 20;
constexpr uint64_t LSE_OFFSET = 8ULL << 20;
constexpr uint64_t FLAG_OFFSET = 9ULL << 20;
constexpr uint64_t EPOCH_OFFSET = 20ULL << 20;
constexpr uint32_t MAX_T = 128;
constexpr uint32_t H = 8;
constexpr uint32_t D = 512;
constexpr uint32_t MAX_LANES = 4;

static_assert(8ULL * MAX_T * H * D * sizeof(bfloat16_t) <= LSE_OFFSET);
static_assert(LSE_OFFSET + 8ULL * MAX_T * H * sizeof(float) <= FLAG_OFFSET);
static_assert(FLAG_OFFSET + 8ULL * MAX_LANES * 64 <= SLOT_BYTES);
static_assert(2 * SLOT_BYTES == EPOCH_OFFSET);

template <typename T>
__aicore__ inline void xcopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t n) {
    DataCopyExtParams params(1, n * sizeof(T), 0, 0, 0);
    DataCopyPad(dst, src, params);
}
template <typename T>
__aicore__ inline void xcopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t n) {
    DataCopyExtParams params(1, n * sizeof(T), 0, 0, 0);
    DataCopyPadExtParams<T> padding;
    DataCopyPad(dst, src, params, padding);
}

template <typename T>
__aicore__ inline GlobalTensor<T> gm(uint64_t address) {
    GlobalTensor<T> x;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    x.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    return x;
}

template <HardEvent E>
__aicore__ inline void event() {
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

extern "C" __global__ __aicore__ void dcp_push_merge(
    GM_ADDR input, GM_ADDR lse, GM_ADDR output, GM_ADDR peers,
    uint32_t rank, uint32_t tokens) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t lanes = MAX_LANES;
    const uint32_t bid = GetBlockIdx();
    const uint32_t part = bid % lanes;
    const uint32_t head = bid / lanes;
    const uint32_t dst = (head + rank) % 8;
    const uint64_t local = reinterpret_cast<__gm__ uint64_t*>(peers)[rank];
    auto epoch_gm = gm<uint32_t>(local + EPOCH_OFFSET + bid * 64);
    // Unsigned rollover is defined, including epoch zero. Two slots are
    // sufficient: completing call k+1 requires every peer to have consumed k.
    uint32_t epoch = epoch_gm.GetValue(0) + 1U;
    epoch_gm.SetValue(0, epoch);
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(epoch_gm);
    const uint64_t slot = ((epoch - 1U) & 1U) * SLOT_BYTES;

    TPipe pipe;
    TBuf<TPosition::VECCALC> copy_buf, stat_buf, flag_buf, acc_buf, tmp_buf, row_buf;
    pipe.InitBuffer(copy_buf, H * D * sizeof(bfloat16_t));
    pipe.InitBuffer(stat_buf, 64 * sizeof(float));
    pipe.InitBuffer(flag_buf, 32);
    pipe.InitBuffer(acc_buf, D * sizeof(float));
    pipe.InitBuffer(tmp_buf, 8 * D * sizeof(float));
    pipe.InitBuffer(row_buf, 8 * D * sizeof(bfloat16_t));
    auto copy = copy_buf.Get<bfloat16_t>();
    auto stats = stat_buf.Get<float>();
    auto flag = flag_buf.Get<uint32_t>();
    auto acc = acc_buf.Get<float>();
    auto tmp = tmp_buf.Get<float>();
    auto row = row_buf.Get<bfloat16_t>();
    // Each sender owns one destination and one token stripe; no shared flag writers.
    for (uint32_t t = part; t < tokens; t += lanes) {
        aclshmemx_mte_put_nbi(
            reinterpret_cast<__gm__ bfloat16_t*>(local + slot + ((rank * MAX_T + t) * H * D) * 2ULL),
            reinterpret_cast<__gm__ bfloat16_t*>(input) + (t * 64 + dst * H) * D,
            reinterpret_cast<__ubuf__ bfloat16_t*>(copy.GetPhyAddr()), H * D * 2, H * D, dst, EVENT_ID0);
        event<HardEvent::MTE3_MTE2>();
        aclshmemx_mte_put_nbi(
            reinterpret_cast<__gm__ float*>(local + slot + LSE_OFFSET + (rank * MAX_T + t) * H * 4ULL),
            reinterpret_cast<__gm__ float*>(lse) + t * 64 + dst * H,
            reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()), 256, H, dst, EVENT_ID0);
        event<HardEvent::MTE3_MTE2>();
    }
    event<HardEvent::MTE3_S>();
    for (uint32_t i = 0; i < 8; ++i) flag.SetValue(i, epoch);
    event<HardEvent::S_MTE3>();
    aclshmemx_mte_put_nbi(
        reinterpret_cast<__gm__ uint32_t*>(local + slot + FLAG_OFFSET + (rank * MAX_LANES + part) * 64ULL),
        reinterpret_cast<__ubuf__ uint32_t*>(flag.GetPhyAddr()), 8, dst, EVENT_ID0);
    event<HardEvent::MTE3_MTE2>();
    // Only this stripe's source flags are dependencies of this worker's output rows.
    for (uint32_t src = 0; src < 8; ++src) {
        auto ready = gm<uint32_t>(local + slot + FLAG_OFFSET + (src * MAX_LANES + part) * 64ULL);
        uint32_t seen;
        uint32_t polls = 0;
        do {
            xcopy(flag, ready, 8);
            event<HardEvent::MTE2_S>();
            seen = flag.GetValue(0);
            if (++polls == 1000000) {
                auto error = gm<int32_t>(local + EPOCH_OFFSET + 4096 + bid * 64);
                error.SetValue(0, src + 1);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(error);
                // Report through the stream error path; never consume stale data.
                trap();
                return;
            }
        } while (seen != epoch);
    }
    auto out = gm<bfloat16_t>(reinterpret_cast<uint64_t>(output));
    for (uint32_t t = part; t < tokens; t += lanes) {
        auto recv_ls = gm<float>(local + slot + LSE_OFFSET + t * H * 4ULL);
        DataCopyExtParams ls_params(8, H * 4, (MAX_T - 1) * H * 4, 0, 0);
        DataCopyPadExtParams<float> ls_padding;
        DataCopyPad(stats, recv_ls, ls_params, ls_padding);
        event<HardEvent::MTE2_S>();
        float max_lse = -3.402823466e38F;
        uint32_t valid_mask = 0;
        for (uint32_t src = 0; src < 8; ++src) {
            float x = stats.GetValue(src * 8 + head);
            // Match the HCCL combine: NaN and either infinity are invalid.
            if (x >= -3.402823466e38F && x <= 3.402823466e38F) {
                valid_mask |= 1U << src;
                if (x > max_lse) max_lse = x;
            }
        }
        event<HardEvent::S_V>();
        Adds(stats, stats, -max_lse, 64);
        PipeBarrier<PIPE_V>();
        Exp(stats, stats, 64);
        event<HardEvent::V_S>();
        float sum = 0.0f;
        for (uint32_t src = 0; src < 8; ++src) {
            float weight = (valid_mask & (1U << src)) ? stats.GetValue(src * 8 + head) : 0.0f;
            stats.SetValue(src * 8 + head, weight);
            sum += weight;
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        auto recv_o = gm<bfloat16_t>(local + slot + (t * H + head) * D * 2ULL);
        DataCopyExtParams o_params(8, D * 2, (MAX_T * H - 1) * D * 2, 0, 0);
        DataCopyPadExtParams<bfloat16_t> o_padding;
        DataCopyPad(row, recv_o, o_params, o_padding);
        event<HardEvent::MTE2_V>();
        Cast(tmp, row, RoundMode::CAST_NONE, 8 * D);
        Duplicate(acc, 0.0f, D);
        for (uint32_t src = 0; src < 8; ++src) {
            const float weight = stats.GetValue(src * 8 + head) * inv;
            // An empty shard may expose an undefined O. Never evaluate 0 * NaN.
            if (weight == 0.0f) continue;
            PipeBarrier<PIPE_V>();
            Muls(tmp[src * D], tmp[src * D], weight, D);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, tmp[src * D], D);
        }
        PipeBarrier<PIPE_V>();
        Cast(row, acc, RoundMode::CAST_RINT, D);
        event<HardEvent::V_MTE3>();
        xcopy(out[(t * H + head) * D], row, D);
        event<HardEvent::MTE3_MTE2>();
    }
}

extern "C" void launch_dcp(void* stream, void* input, void* lse, void* output,
                           void* peers, uint32_t rank, uint32_t tokens) {
    dcp_push_merge<<<8 * MAX_LANES, nullptr, stream>>>(
        static_cast<uint8_t*>(input), static_cast<uint8_t*>(lse),
        static_cast<uint8_t*>(output), static_cast<uint8_t*>(peers),
        rank, tokens);
}
