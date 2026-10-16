// SPDX-License-Identifier: Apache-2.0
#include "shmem.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr int kRanks = 8;
constexpr uint64_t kHeapBytes = 32ULL << 20;
constexpr uint64_t kWorkspaceBytes = 10ULL << 20;
}

extern "C" size_t dcp_unique_id_size() { return sizeof(aclshmemx_uniqueid_t); }

extern "C" int dcp_shmem_status() { return aclshmemx_init_status(); }

extern "C" int dcp_get_unique_id(void* bytes) {
    aclshmemx_uniqueid_t uid{};
    int rc = aclshmemx_get_uniqueid(&uid);
    if (rc == 0) std::memcpy(bytes, &uid, sizeof(uid));
    return rc;
}

extern "C" int init_dcp_shmem(int rank, const void* bytes, uint64_t* peers) {
    // Own one SDK instance per worker. Do not finalize another user's instance.
    if (aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED) return -102;
    aclshmemx_uniqueid_t uid{};
    std::memcpy(&uid, bytes, sizeof(uid));
    aclshmemx_init_attr_t attr{};
    int rc = aclshmemx_set_attr_uniqueid_args(rank, kRanks, kHeapBytes, &uid, &attr);
    if (rc) return rc;
    attr.option_attr.shm_init_timeout = 60;
    attr.option_attr.shm_create_timeout = 60;
    attr.option_attr.control_operation_timeout = 60;
    rc = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (rc) return rc;
    void* base = aclshmem_malloc(kWorkspaceBytes);
    if (!base) {
        aclshmem_finalize();
        return -100;
    }
    for (int p = 0; p < kRanks; ++p) {
        peers[p] = reinterpret_cast<uint64_t>(aclshmem_ptr(base, p));
        if (!peers[p]) {
            aclshmem_free(base);
            aclshmem_finalize();
            return -101;
        }
    }
    return 0;
}

extern "C" int close_dcp_shmem(void* base) {
    aclshmem_free(base);
    return aclshmem_finalize();
}
