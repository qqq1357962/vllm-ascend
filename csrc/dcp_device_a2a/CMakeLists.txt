# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "")
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "")
find_package(ASC REQUIRED)
project(dcp_device_a2a LANGUAGES ASC CXX)

# Optional, standalone build. SHMEM is not a dependency of the normal wheel.
set(SHMEM_ROOT "" CACHE PATH "SHMEM SDK prefix containing include, src and lib")
if(NOT EXISTS "${SHMEM_ROOT}/include/shmem.h" OR
   NOT EXISTS "${SHMEM_ROOT}/lib/libshmem.so")
  message(FATAL_ERROR "Set SHMEM_ROOT to the installed SHMEM SDK prefix")
endif()
add_library(dcp_device_a2a SHARED kernel.asc shmem_bridge.cpp)
set_target_properties(dcp_device_a2a PROPERTIES LINKER_LANGUAGE ASC)
target_include_directories(dcp_device_a2a PRIVATE
  "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src"
  "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt"
  "$ENV{ASCEND_HOME_PATH}/include")
target_link_libraries(dcp_device_a2a PRIVATE "${SHMEM_ROOT}/lib/libshmem.so")
target_compile_options(dcp_device_a2a PRIVATE
  $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201;-O2>
  "$<$<COMPILE_LANGUAGE:ASC>:SHELL:-mllvm -cce-aicore-addr-transform>"
  "$<$<COMPILE_LANGUAGE:ASC>:SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false>")
install(TARGETS dcp_device_a2a LIBRARY DESTINATION lib)
