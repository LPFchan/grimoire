cmake_minimum_required(VERSION 3.18)
set(PFLASH_USER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
project(pflash LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(GGML_CUDA         ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_METAL        OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN       OFF CACHE BOOL "" FORCE)
set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL       OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/CMakeLists.txt")
    message(FATAL_ERROR "deps/llama.cpp submodule missing. Run: git submodule update --init --recursive")
endif()
option(PFLASH_FA_ALL_QUANTS "Compile ggml-cuda fattn kernels for all KV-quant pairs" ON)
set(GGML_CUDA_FA_ALL_QUANTS ${PFLASH_FA_ALL_QUANTS} CACHE BOOL "" FORCE)

if(PFLASH_USER_CUDA_ARCHITECTURES)
    set(_pflash_archs "${PFLASH_USER_CUDA_ARCHITECTURES}")
else()
    set(_pflash_archs "75;86")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND _pflash_archs "120")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        list(APPEND _pflash_archs "110")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
        list(APPEND _pflash_archs "121")
    endif()
endif()

add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)
find_package(CUDAToolkit REQUIRED)

add_library(pflash27b STATIC
    src/errors.cpp
    src/qwen3_drafter.cpp
    src/qwen3_0p6b_loader.cpp
    src/qwen3_0p6b_graph.cpp
    src/qwen3_5_0p8b_loader.cpp
    src/qwen3_5_0p8b_graph.cpp
    src/flashprefill_q8.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
)

set_target_properties(pflash27b PROPERTIES CUDA_ARCHITECTURES "${_pflash_archs}")

list(GET _pflash_archs 0 _pflash_min_sm)
string(REGEX REPLACE "[^0-9]" "" _pflash_min_sm "${_pflash_min_sm}")

if(_pflash_min_sm GREATER_EQUAL 80)
    target_sources(pflash27b PRIVATE
        src/flashprefill_kernels.cu
        src/flashprefill_select.cpp
        src/flashprefill.cpp)
    target_compile_definitions(pflash27b PRIVATE DFLASH27B_HAVE_FLASHPREFILL=1)
endif()

option(PFLASH_ENABLE_BSA "Enable Block-Sparse Attention for spec-prefill scoring" ON)
if(PFLASH_ENABLE_BSA)
    foreach(_arch IN LISTS _pflash_archs)
        if(_arch LESS 80)
            set(PFLASH_ENABLE_BSA OFF)
            break()
        endif()
    endforeach()
endif()
if(PFLASH_ENABLE_BSA AND NOT EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include/cutlass/numeric_types.h")
    message(WARNING "Block-Sparse-Attention submodule missing. Disabling BSA.")
    set(PFLASH_ENABLE_BSA OFF)
endif()
if(PFLASH_ENABLE_BSA)
    target_sources(pflash27b PRIVATE src/bsa_fwd_inst.cu src/bsa_launcher.cu)
endif()

target_include_directories(pflash27b
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)
if(PFLASH_ENABLE_BSA)
    target_include_directories(pflash27b PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/bsa_stubs
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/block_sparse_attn/src)
    target_compile_options(pflash27b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(pflash27b PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASH_NAMESPACE=flash DFLASH27B_HAVE_BSA=1)
endif()

target_link_libraries(pflash27b
    PUBLIC
        ggml
        ggml-cuda
        ggml-base
        CUDA::cuda_driver
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(pflash27b PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-function>)
endif()

add_executable(pflash_daemon src/pflash_daemon.cpp)
target_include_directories(pflash_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(pflash_daemon PRIVATE pflash27b ggml ggml-cuda)
