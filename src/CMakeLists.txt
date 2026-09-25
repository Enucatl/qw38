add_library(qw38_format STATIC
  format/error.cpp
  format/constants.cpp
  format/wire.cpp
  format/layout.cpp
  format/schema.cpp
  format/sha256.cpp
  format/writer.cpp
  format/reader.cpp
  format/pack.cpp
  format/unpack.cpp
)
target_include_directories(qw38_format PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(qw38_format PUBLIC cxx_std_23)
target_compile_options(qw38_format PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_library(qw38_compiler STATIC
  compiler/error.cpp
  compiler/json.cpp
  compiler/identity.cpp
  compiler/transforms.cpp
  compiler/checkpoint.cpp
  compiler/compile.cpp
  compiler/quantization/quantizer.cpp
  compiler/quantization/reference.cpp
  compiler/quantization/q4k.cpp
)
# Reference fitting and reconstruction use the pinned scalar operation order.
set_source_files_properties(compiler/quantization/q4k.cpp
  compiler/quantization/reference.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
target_include_directories(qw38_compiler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(qw38_compiler PUBLIC qw38_format)
target_compile_features(qw38_compiler PUBLIC cxx_std_23)
target_compile_options(qw38_compiler PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_compile compiler/main.cpp)
target_link_libraries(qw38_compile PRIVATE qw38_compiler)
target_compile_features(qw38_compile PRIVATE cxx_std_23)
target_compile_options(qw38_compile PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
set_target_properties(qw38_compile PROPERTIES OUTPUT_NAME qw38-compile)

add_library(qw38_reference STATIC
  reference/error.cpp
  reference/math.cpp
)
target_include_directories(qw38_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(qw38_reference PUBLIC cxx_std_23)
target_compile_options(qw38_reference PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_library(qw38_runtime STATIC
  runtime/language_layer.cpp
  runtime/language_model.cpp
  runtime/error.cpp
  runtime/sizes.cpp
  runtime/arena.cpp
  runtime/model.cpp
  runtime/session.cpp
  runtime/runtime.cpp
  runtime/mlp.cpp
  runtime/prefill.cpp
  runtime/prefill_gdn.cpp
  runtime/gdn.cpp
  runtime/attention.cpp
)

add_executable(qw38_decode runtime/main.cpp)
target_link_libraries(qw38_decode PRIVATE qw38_runtime)
target_compile_features(qw38_decode PRIVATE cxx_std_23)
set_target_properties(qw38_decode PROPERTIES OUTPUT_NAME qw38-decode)

add_executable(qw38_bf16_diagnostic runtime/bf16_diagnostic.cpp)
target_link_libraries(qw38_bf16_diagnostic PRIVATE qw38_runtime)
target_compile_features(qw38_bf16_diagnostic PRIVATE cxx_std_23)
set_target_properties(qw38_bf16_diagnostic PROPERTIES
  OUTPUT_NAME qw38-bf16-diagnostic)

add_executable(qw38_evaluate runtime/evaluate.cpp)
target_link_libraries(qw38_evaluate PRIVATE qw38_runtime)
target_compile_features(qw38_evaluate PRIVATE cxx_std_23)
target_compile_options(qw38_evaluate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
set_target_properties(qw38_evaluate PROPERTIES OUTPUT_NAME qw38-evaluate)

add_executable(qw38_task022_trace runtime/task022_trace.cpp)
target_link_libraries(qw38_task022_trace PRIVATE qw38_runtime)
target_compile_features(qw38_task022_trace PRIVATE cxx_std_23)
set_target_properties(qw38_task022_trace PROPERTIES OUTPUT_NAME qw38-task022-trace)

add_executable(qw38_state_replay runtime/state_replay.cpp)
target_link_libraries(qw38_state_replay PRIVATE qw38_runtime)
target_compile_features(qw38_state_replay PRIVATE cxx_std_23)
target_compile_options(qw38_state_replay PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
set_target_properties(qw38_state_replay PROPERTIES OUTPUT_NAME qw38-state-replay)
target_include_directories(qw38_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(qw38_runtime PUBLIC qw38_format qw38_cuda)
target_compile_features(qw38_runtime PUBLIC cxx_std_23)
target_compile_options(qw38_runtime PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_host_expected_smoke host_expected_smoke.cpp)
target_compile_features(qw38_host_expected_smoke PRIVATE cxx_std_23)
target_compile_options(qw38_host_expected_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
