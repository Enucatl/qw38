add_executable(qw38_format_schema_test format_schema_test.cpp)
target_link_libraries(qw38_format_schema_test PRIVATE qw38_format)
target_compile_features(qw38_format_schema_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_schema_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_schema_integration_test
  format_schema_integration_test.cpp)
target_link_libraries(qw38_format_schema_integration_test PRIVATE qw38_format)
target_compile_features(qw38_format_schema_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_schema_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_writer_test format_writer_test.cpp)
target_link_libraries(qw38_format_writer_test PRIVATE qw38_format)
target_compile_features(qw38_format_writer_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_writer_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_sha256_test format_sha256_test.cpp)
target_link_libraries(qw38_format_sha256_test PRIVATE qw38_format)
target_compile_features(qw38_format_sha256_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_sha256_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_writer_integration_test
  format_writer_integration_test.cpp)
target_link_libraries(qw38_format_writer_integration_test PRIVATE qw38_format)
target_compile_features(qw38_format_writer_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_writer_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_reader_test format_reader_test.cpp)
target_link_libraries(qw38_format_reader_test PRIVATE qw38_format)
target_compile_features(qw38_format_reader_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_reader_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_reader_digest_test format_reader_digest_test.cpp)
target_link_libraries(qw38_format_reader_digest_test PRIVATE qw38_format)
target_compile_features(qw38_format_reader_digest_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_reader_digest_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_format_reader_integration_test
  format_reader_integration_test.cpp)
target_link_libraries(qw38_format_reader_integration_test PRIVATE qw38_format)
target_compile_features(qw38_format_reader_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_format_reader_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME host_expected_smoke COMMAND qw38_host_expected_smoke)
add_test(NAME format_schema COMMAND qw38_format_schema_test)
add_test(NAME format_schema_integration COMMAND qw38_format_schema_integration_test)
add_test(NAME format_writer COMMAND qw38_format_writer_test)
add_test(NAME format_sha256 COMMAND qw38_format_sha256_test)
add_test(NAME format_writer_integration COMMAND qw38_format_writer_integration_test)
add_test(NAME format_reader COMMAND qw38_format_reader_test)
add_test(NAME format_reader_digest COMMAND qw38_format_reader_digest_test)
add_test(NAME format_reader_integration COMMAND qw38_format_reader_integration_test)
add_test(NAME cuda_runtime_smoke COMMAND qw38_cuda_runtime_smoke)
add_test(
  NAME cuda_sm120_cubin
  COMMAND bash "${CMAKE_SOURCE_DIR}/tests/check_sm120_cubin.sh"
          $<TARGET_FILE:qw38_cuda_runtime_smoke>
)

set_tests_properties(
  format_schema format_schema_integration
  format_writer format_sha256 format_writer_integration
  format_reader format_reader_digest format_reader_integration
  PROPERTIES
  LABELS "format"
)
set_tests_properties(cuda_runtime_smoke PROPERTIES
  LABELS "cuda;gpu"
)
add_executable(qw38_compiler_identity_test compiler_identity_test.cpp)
target_link_libraries(qw38_compiler_identity_test PRIVATE qw38_compiler)
target_compile_features(qw38_compiler_identity_test PRIVATE cxx_std_23)
target_compile_options(qw38_compiler_identity_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_compiler_transform_test compiler_transform_test.cpp)
target_link_libraries(qw38_compiler_transform_test PRIVATE qw38_compiler)
target_compile_features(qw38_compiler_transform_test PRIVATE cxx_std_23)
target_compile_options(qw38_compiler_transform_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_compiler_integration_test compiler_integration_test.cpp)
target_link_libraries(qw38_compiler_integration_test PRIVATE qw38_compiler)
target_compile_features(qw38_compiler_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_compiler_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
target_compile_definitions(qw38_compiler_integration_test PRIVATE
  QW38_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(NAME compiler_identity COMMAND qw38_compiler_identity_test)
add_test(NAME compiler_transform COMMAND qw38_compiler_transform_test)
add_test(NAME compiler_integration COMMAND qw38_compiler_integration_test)

add_executable(qw38_quantizer_test quantizer_test.cpp)
target_link_libraries(qw38_quantizer_test PRIVATE qw38_compiler)
target_compile_features(qw38_quantizer_test PRIVATE cxx_std_23)
target_compile_options(qw38_quantizer_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_pack_layout_test pack_layout_test.cpp)
target_link_libraries(qw38_pack_layout_test PRIVATE qw38_format)
target_compile_features(qw38_pack_layout_test PRIVATE cxx_std_23)
target_compile_options(qw38_pack_layout_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_quant_reference_test quant_reference_test.cpp)
target_link_libraries(qw38_quant_reference_test PRIVATE qw38_compiler)
target_compile_features(qw38_quant_reference_test PRIVATE cxx_std_23)
target_compile_options(qw38_quant_reference_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_quant_compiler_integration_test
  quant_compiler_integration_test.cpp)
target_link_libraries(qw38_quant_compiler_integration_test PRIVATE qw38_compiler)
target_compile_features(qw38_quant_compiler_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_quant_compiler_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
target_compile_definitions(qw38_quant_compiler_integration_test PRIVATE
  QW38_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

add_test(NAME quantizer COMMAND qw38_quantizer_test)
add_test(NAME pack_layout COMMAND qw38_pack_layout_test)
add_test(NAME quant_reference COMMAND qw38_quant_reference_test)
add_test(NAME quant_compiler_integration COMMAND qw38_quant_compiler_integration_test)

set_tests_properties(
  compiler_identity compiler_transform compiler_integration
  PROPERTIES
  LABELS "compiler"
)

set_tests_properties(
  quantizer pack_layout quant_reference quant_compiler_integration
  PROPERTIES
  LABELS "quantizer;compiler"
)

set_tests_properties(cuda_sm120_cubin PROPERTIES
  LABELS "cuda"
)

add_executable(qw38_runtime_plan_test runtime_plan_test.cpp)
target_link_libraries(qw38_runtime_plan_test PRIVATE qw38_runtime)
target_compile_features(qw38_runtime_plan_test PRIVATE cxx_std_23)
target_compile_options(qw38_runtime_plan_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_runtime_raii_test runtime_raii_test.cpp)
target_link_libraries(qw38_runtime_raii_test PRIVATE qw38_runtime)
target_compile_features(qw38_runtime_raii_test PRIVATE cxx_std_23)
target_compile_options(qw38_runtime_raii_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_runtime_state_index_test runtime_state_index_test.cpp)
target_link_libraries(qw38_runtime_state_index_test PRIVATE qw38_runtime)
target_compile_features(qw38_runtime_state_index_test PRIVATE cxx_std_23)
target_compile_options(qw38_runtime_state_index_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_runtime_session_integration_test
  runtime_session_integration_test.cpp)
target_link_libraries(qw38_runtime_session_integration_test PRIVATE qw38_runtime)
target_compile_features(qw38_runtime_session_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_runtime_session_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME runtime_plan COMMAND qw38_runtime_plan_test)
add_test(NAME runtime_raii COMMAND qw38_runtime_raii_test)
add_test(NAME runtime_state_index COMMAND qw38_runtime_state_index_test)
add_test(NAME runtime_session_integration COMMAND qw38_runtime_session_integration_test)

set_tests_properties(runtime_plan PROPERTIES LABELS "runtime")
set_tests_properties(
  runtime_raii runtime_state_index runtime_session_integration
  PROPERTIES
  LABELS "runtime;cuda;gpu"
)

add_executable(qw38_reference_math_unit_test reference_math_unit_test.cpp)
target_link_libraries(qw38_reference_math_unit_test PRIVATE qw38_reference)
target_compile_features(qw38_reference_math_unit_test PRIVATE cxx_std_23)
target_compile_options(qw38_reference_math_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_activation_reference_test activation_reference_test.cpp)
target_link_libraries(qw38_activation_reference_test PRIVATE qw38_reference qw38_cuda)
target_compile_features(qw38_activation_reference_test PRIVATE cxx_std_23)
target_compile_options(qw38_activation_reference_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_activation_integration_test activation_integration_test.cpp)
target_link_libraries(qw38_activation_integration_test PRIVATE qw38_reference qw38_cuda)
target_compile_features(qw38_activation_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_activation_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME reference_math_unit COMMAND qw38_reference_math_unit_test)
add_test(NAME activation_reference COMMAND qw38_activation_reference_test)
add_test(NAME activation_integration COMMAND qw38_activation_integration_test)

set_tests_properties(reference_math_unit PROPERTIES LABELS "reference")
set_tests_properties(
  activation_reference activation_integration
  PROPERTIES
  LABELS "reference;cuda;gpu"
)

add_executable(qw38_decode_mmv_unit_test decode_mmv_unit_test.cpp)
target_link_libraries(qw38_decode_mmv_unit_test PRIVATE qw38_compiler qw38_cuda)
target_compile_features(qw38_decode_mmv_unit_test PRIVATE cxx_std_23)
target_compile_options(qw38_decode_mmv_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_decode_mmv_reference_test decode_mmv_reference_test.cpp)
target_link_libraries(qw38_decode_mmv_reference_test PRIVATE qw38_compiler qw38_cuda)
target_compile_features(qw38_decode_mmv_reference_test PRIVATE cxx_std_23)
target_compile_options(qw38_decode_mmv_reference_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_decode_mmv_integration_test decode_mmv_integration_test.cpp)
target_link_libraries(qw38_decode_mmv_integration_test PRIVATE qw38_compiler qw38_cuda)
target_compile_features(qw38_decode_mmv_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_decode_mmv_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME decode_mmv_unit COMMAND qw38_decode_mmv_unit_test)
add_test(NAME decode_mmv_reference COMMAND qw38_decode_mmv_reference_test)
add_test(NAME decode_mmv_integration COMMAND qw38_decode_mmv_integration_test)

set_tests_properties(
  decode_mmv_unit decode_mmv_reference decode_mmv_integration
  PROPERTIES
  LABELS "cuda;gpu;decode_mmv"
)

add_executable(qw38_mlp_unit_test mlp_unit_test.cpp)
target_link_libraries(qw38_mlp_unit_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_mlp_unit_test PRIVATE cxx_std_23)
target_compile_options(qw38_mlp_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_mlp_reference_test mlp_reference_test.cpp)
target_link_libraries(qw38_mlp_reference_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_mlp_reference_test PRIVATE cxx_std_23)
target_compile_options(qw38_mlp_reference_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_mlp_integration_test mlp_integration_test.cpp)
target_link_libraries(qw38_mlp_integration_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_mlp_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_mlp_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME mlp_unit COMMAND qw38_mlp_unit_test)
add_test(NAME mlp_reference COMMAND qw38_mlp_reference_test)
add_test(NAME mlp_integration COMMAND qw38_mlp_integration_test)

set_tests_properties(
  mlp_unit mlp_reference mlp_integration
  PROPERTIES
  LABELS "cuda;gpu;mlp"
)
