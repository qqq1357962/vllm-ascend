# SPDX-License-Identifier: Apache-2.0

# This is part of the normal vllm_ascend_C build. SHMEM remains an optional,
# externally installed SDK, and no additional runtime library path is exposed.
set(VLLM_ASCEND_SHMEM_ROOT "" CACHE PATH "Optional Ascend SHMEM SDK prefix")
set(_shmem_root "${VLLM_ASCEND_SHMEM_ROOT}")
if(NOT _shmem_root)
  set(_shmem_root "/usr/local/Ascend/shmem/latest")
endif()
if(EXISTS "${_shmem_root}/shmem/include/shmem.h")
  string(APPEND _shmem_root "/shmem")
endif()

if(NOT EXISTS "${_shmem_root}/include/shmem.h" OR
   NOT EXISTS "${_shmem_root}/src/device/shmemi_device_meta.h" OR
   NOT EXISTS "${_shmem_root}/lib/libshmem.so")
  if(VLLM_ASCEND_SHMEM_ROOT)
    message(FATAL_ERROR "VLLM_ASCEND_SHMEM_ROOT must contain the installed SHMEM include, src and lib directories")
  endif()
  message(STATUS "Ascend SHMEM SDK not found: skipping optional DCP device exchange")
  return()
endif()

if(NOT SOC_VERSION MATCHES "^ascend910b")
  message(STATUS "DCP device exchange supports Ascend 910B only: skipping for ${SOC_VERSION}")
  return()
endif()

ascendc_library(vllm_ascend_dcp_device_kernel STATIC dcp_push_merge.cpp)
ascendc_include_directories(vllm_ascend_dcp_device_kernel PRIVATE
  "${_shmem_root}/include" "${_shmem_root}/src"
  "${_shmem_root}/src/device" "${_shmem_root}/src/device_simt"
  "${ASCEND_HOME_PATH}/include")
ascendc_compile_options(vllm_ascend_dcp_device_kernel PRIVATE
  -O2 "SHELL:-mllvm -cce-aicore-addr-transform"
  "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
target_sources(vllm_ascend_C PRIVATE dcp_device_a2a.cpp)
target_include_directories(vllm_ascend_C PRIVATE "${_shmem_root}/include")
target_compile_definitions(vllm_ascend_C PRIVATE
  VLLM_ASCEND_DCP_SHMEM_LIBRARY="${_shmem_root}/lib/libshmem.so"
  VLLM_ASCEND_DCP_CANN_ROOT="${ASCEND_HOME_PATH}")
target_link_libraries(vllm_ascend_C PRIVATE vllm_ascend_dcp_device_kernel)
message(STATUS "Ascend SHMEM detected: enabling DCP device exchange in vllm_ascend_C")
