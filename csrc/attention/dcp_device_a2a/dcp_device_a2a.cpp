// SPDX-License-Identifier: Apache-2.0

#include <acl/acl.h>
#include <dlfcn.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "shmem.h"

extern "C" void launch_dcp(void* stream, void* input, void* lse, void* output,
                           void* peers, uint32_t rank, uint32_t tokens);

namespace vllm_ascend {
namespace {
constexpr int64_t kRanks = 8;
constexpr int64_t kHeads = 64;
constexpr int64_t kHeadDim = 512;
constexpr int64_t kMaxTokens = 128;
constexpr int64_t kHeapBytes = 32LL << 20;
constexpr int64_t kWorkspaceBytes = 22LL << 20;

// The external SDK exports Hybm symbols also exported by MemFabric. Loading it
// lazily with DEEPBIND preserves its internal bindings when MemFabric was
// imported first. These immutable function pointers and their handle remain
// alive for the process; workspace ownership/finalization is explicit below.
struct ShmemApi {
    void* runtime_handle;
    void* acl_handle;
    void* handle;
    decltype(&aclshmemx_init_status) status;
    decltype(&aclshmemx_get_uniqueid) unique_id;
    decltype(&aclshmemx_set_attr_uniqueid_args) set_attributes;
    decltype(&aclshmemx_init_attr) initialize;
    decltype(&aclshmem_malloc) allocate;
    decltype(&aclshmem_ptr) peer_pointer;
    decltype(&aclshmem_free) free;
    decltype(&aclshmem_finalize) finalize;

    template <typename T>
    T symbol(const char* name)
    {
        void* address = dlsym(handle, name);
        TORCH_CHECK(address != nullptr, "DCP SHMEM SDK is missing symbol ", name);
        return reinterpret_cast<T>(address);
    }

    ShmemApi()
    {
        // Register the CANN runtime dependency before the SDK opens ACL. CANN
        // 9.2 resolves internal rt* entry points lazily; opening ACL explicitly
        // here gives the SDK the same dependency scope as its standalone use.
        runtime_handle = dlopen(VLLM_ASCEND_DCP_CANN_ROOT "/lib64/libruntime.so", RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(runtime_handle != nullptr, "Cannot load CANN runtime for DCP SHMEM: ", dlerror());
        acl_handle = dlopen(VLLM_ASCEND_DCP_CANN_ROOT "/lib64/libascendcl.so", RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(acl_handle != nullptr, "Cannot load CANN ACL for DCP SHMEM: ", dlerror());
        TORCH_CHECK(dlsym(acl_handle, "rtDeviceGetBareTgid") != nullptr,
                    "CANN ACL cannot resolve the rtDeviceGetBareTgid entry point required by SHMEM.");
        handle = dlopen(VLLM_ASCEND_DCP_SHMEM_LIBRARY, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
        TORCH_CHECK(handle != nullptr, "Cannot load the SHMEM SDK used to build vllm_ascend_C: ", dlerror());
        status = symbol<decltype(status)>("aclshmemx_init_status");
        unique_id = symbol<decltype(unique_id)>("aclshmemx_get_uniqueid");
        set_attributes = symbol<decltype(set_attributes)>("aclshmemx_set_attr_uniqueid_args");
        initialize = symbol<decltype(initialize)>("aclshmemx_init_attr");
        allocate = symbol<decltype(allocate)>("aclshmem_malloc");
        peer_pointer = symbol<decltype(peer_pointer)>("aclshmem_ptr");
        free = symbol<decltype(free)>("aclshmem_free");
        finalize = symbol<decltype(finalize)>("aclshmem_finalize");
    }
};

const ShmemApi& shmem_api()
{
    static const ShmemApi api;
    return api;
}

void check_sdk(int code, const char* operation)
{
    TORCH_CHECK(code == 0, "DCP device A2A: ", operation, " failed with code ", code);
}

int64_t dcp_device_a2a_status()
{
    return shmem_api().status();
}

at::Tensor dcp_device_a2a_unique_id()
{
    aclshmemx_uniqueid_t uid{};
    check_sdk(shmem_api().unique_id(&uid), "get SHMEM unique ID");
    auto result = at::empty({static_cast<int64_t>(sizeof(uid))}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(result.data_ptr(), &uid, sizeof(uid));
    return result;
}

std::tuple<at::Tensor, std::vector<int64_t>> dcp_device_a2a_init(int64_t rank, const at::Tensor& unique_id)
{
    TORCH_CHECK(rank >= 0 && rank < kRanks, "DCP device A2A rank must be in [0, 8).");
    TORCH_CHECK(unique_id.device().is_cpu() && unique_id.scalar_type() == at::kByte &&
                unique_id.is_contiguous() && unique_id.numel() == sizeof(aclshmemx_uniqueid_t),
                "DCP device A2A requires a CPU SHMEM unique ID.");
    const auto& sdk = shmem_api();
    TORCH_CHECK(sdk.status() == ACLSHMEM_STATUS_NOT_INITIALIZED,
                "DCP device A2A requires exclusive ownership of the SHMEM SDK instance.");
    aclshmemx_uniqueid_t uid{};
    std::memcpy(&uid, unique_id.data_ptr(), sizeof(uid));
    aclshmemx_init_attr_t attr{};
    check_sdk(sdk.set_attributes(rank, kRanks, kHeapBytes, &uid, &attr), "set SHMEM attributes");
    attr.option_attr.shm_init_timeout = 60;
    attr.option_attr.shm_create_timeout = 60;
    attr.option_attr.control_operation_timeout = 60;
    check_sdk(sdk.initialize(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr), "initialize SHMEM");
    void* base = nullptr;
    try {
        base = sdk.allocate(kWorkspaceBytes);
        TORCH_CHECK(base != nullptr, "Cannot allocate DCP SHMEM workspace.");
        std::vector<int64_t> addresses(kRanks);
        for (int p = 0; p < kRanks; ++p) {
            addresses[p] = reinterpret_cast<int64_t>(sdk.peer_pointer(base, p));
            TORCH_CHECK(addresses[p] != 0, "Cannot map DCP SHMEM peer ", p);
        }
        check_sdk(aclrtMemset(base, kWorkspaceBytes, 0, kWorkspaceBytes), "clear SHMEM workspace");
        const c10::Device device(c10::DeviceType::PrivateUse1, c10_npu::current_device());
        // The group-owned Python runtime closes the SDK collectively after the
        // last graph replay. Tensor destruction must never enter a collective.
        auto workspace = at_npu::native::from_blob(
            base, {kWorkspaceBytes}, [](void*) {}, at::TensorOptions().dtype(at::kByte).device(device), device);
        return {workspace, addresses};
    } catch (...) {
        if (base != nullptr) sdk.free(base);
        sdk.finalize();
        throw;
    }
}

void dcp_device_a2a_close(const at::Tensor& workspace)
{
    TORCH_CHECK(workspace.device().is_privateuseone() && workspace.scalar_type() == at::kByte &&
                workspace.is_contiguous() && workspace.numel() == kWorkspaceBytes,
                "Invalid DCP SHMEM workspace.");
    const c10_npu::NPUGuard guard(workspace.device());
    const auto& sdk = shmem_api();
    sdk.free(workspace.data_ptr());
    check_sdk(sdk.finalize(), "finalize SHMEM");
}

at::Tensor dcp_device_a2a(const at::Tensor& input, const at::Tensor& lse, const at::Tensor& peers,
                         const at::Tensor& workspace, int64_t rank)
{
    TORCH_CHECK(input.device().is_privateuseone() && lse.device() == input.device() &&
                peers.device() == input.device() && workspace.device() == input.device(),
                "DCP device A2A tensors must be on the same NPU.");
    TORCH_CHECK(input.dim() == 3 && input.size(0) >= 1 && input.size(0) <= kMaxTokens &&
                input.size(1) == kHeads && input.size(2) == kHeadDim && input.scalar_type() == at::kBFloat16,
                "DCP device A2A expects BF16 [T,64,512], 1 <= T <= 128.");
    TORCH_CHECK(lse.sizes() == at::IntArrayRef({input.size(0), kHeads, 1}) && lse.scalar_type() == at::kFloat,
                "DCP device A2A expects FP32 LSE [T,64,1].");
    TORCH_CHECK(peers.dim() == 1 && peers.numel() == kRanks && peers.scalar_type() == at::kLong &&
                workspace.scalar_type() == at::kByte && workspace.numel() == kWorkspaceBytes,
                "Invalid DCP device A2A peers or workspace.");
    TORCH_CHECK(input.is_contiguous() && lse.is_contiguous() && peers.is_contiguous() && workspace.is_contiguous(),
                "DCP device A2A tensors must be contiguous.");
    TORCH_CHECK(rank >= 0 && rank < kRanks, "DCP device A2A rank must be in [0, 8).");
    const c10_npu::NPUGuard guard(input.device());
    auto output = at::empty({input.size(0), kHeads / kRanks, kHeadDim}, input.options());
    // Drain torch_npu's host task queue so preceding copies/contiguous
    // conversions reach this stream before the direct AscendC launch.
    launch_dcp(c10_npu::getCurrentNPUStream().stream(), input.data_ptr(), lse.data_ptr(),
               output.data_ptr(), peers.data_ptr(), rank, input.size(0));
    return output;
}
}  // namespace
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def("dcp_device_a2a_status() -> int", &vllm_ascend::dcp_device_a2a_status);
    ops.def("dcp_device_a2a_unique_id() -> Tensor", &vllm_ascend::dcp_device_a2a_unique_id);
    ops.def("dcp_device_a2a_init(int rank, Tensor unique_id) -> (Tensor, int[])",
            &vllm_ascend::dcp_device_a2a_init);
    ops.def("dcp_device_a2a_close(Tensor workspace) -> ()", &vllm_ascend::dcp_device_a2a_close);
    ops.def("dcp_device_a2a(Tensor input, Tensor lse, Tensor peers, Tensor(a!) workspace, int rank) -> Tensor");
    ops.impl("dcp_device_a2a", c10::DispatchKey::PrivateUse1, &vllm_ascend::dcp_device_a2a);
}
