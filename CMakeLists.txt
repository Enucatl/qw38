cmake_minimum_required(VERSION 3.28)

# nvcc 13.4 ignores -std=c++23 unless the host compiler is GCC 14+.
# Select that compiler before languages are enabled.
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  find_program(QW38_HOST_CXX NAMES g++-14)
  if(NOT QW38_HOST_CXX)
    message(FATAL_ERROR "g++-14 is required for CUDA 13.4 C++23")
  endif()
  set(CMAKE_CXX_COMPILER "${QW38_HOST_CXX}" CACHE FILEPATH "Host C++ compiler")
endif()
if(NOT CMAKE_CUDA_HOST_COMPILER)
  if(CMAKE_CXX_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
        "nvcc host compiler")
  elseif(DEFINED ENV{CXX})
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{CXX}" CACHE FILEPATH "nvcc host compiler")
  endif()
endif()

project(qw38 VERSION 0.1.0 LANGUAGES CXX CUDA)

# CMake 3.28's NVIDIA module records CUDA standards only through C++20.
# CUDA 13.3+ nvcc supports -std=c++23 with GCC 14; teach CMake the flag.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
   AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3)
  set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
  set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(QW38_CUDA_ARCHITECTURES "120-real" CACHE STRING
    "Native RTX 5090 / Blackwell CUDA architecture (sm_120 cubin only)")
set(CMAKE_CUDA_ARCHITECTURES "${QW38_CUDA_ARCHITECTURES}")
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "120")
  message(FATAL_ERROR
    "V0 requires native sm_120; got '${CMAKE_CUDA_ARCHITECTURES}'")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR
    "nvcc 13.4 C++23 requires GCC 14+ as host compiler; got ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "qw38 C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "qw38 CUDA compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "qw38 CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "qw38 CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "qw38 build type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(cuda)
add_subdirectory(src)
add_subdirectory(benchmarks)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
