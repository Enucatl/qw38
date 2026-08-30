cmake_minimum_required(VERSION 3.25)
project(qw38_llama_authority LANGUAGES CXX)

if(NOT DEFINED LLAMA_SOURCE OR NOT DEFINED LLAMA_BUILD)
  message(FATAL_ERROR "LLAMA_SOURCE and LLAMA_BUILD are required")
endif()

add_executable(qw38-llama-token-oracle token_oracle.cpp)
target_compile_features(qw38-llama-token-oracle PRIVATE cxx_std_17)
target_compile_options(qw38-llama-token-oracle PRIVATE -Wall -Wextra -Werror)
target_include_directories(
  qw38-llama-token-oracle
  PRIVATE "${LLAMA_SOURCE}/include" "${LLAMA_SOURCE}/ggml/include")
target_link_directories(qw38-llama-token-oracle PRIVATE "${LLAMA_BUILD}/bin")
target_link_libraries(qw38-llama-token-oracle PRIVATE llama ggml ggml-base)
set_target_properties(
  qw38-llama-token-oracle
  PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    RUNTIME_OUTPUT_DIRECTORY "${LLAMA_BUILD}/bin")
