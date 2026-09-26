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
add_executable(qw38_language_model_plan_test language_model_plan_test.cpp)
target_link_libraries(qw38_language_model_plan_test PRIVATE qw38_runtime)
target_compile_features(qw38_language_model_plan_test PRIVATE cxx_std_23)
add_test(NAME language_model_plan COMMAND qw38_language_model_plan_test)
add_executable(qw38_language_model_integration_test
  language_model_integration_test.cpp)
target_link_libraries(qw38_language_model_integration_test PRIVATE qw38_runtime)
target_compile_features(qw38_language_model_integration_test PRIVATE cxx_std_23)
add_test(NAME language_model_integration COMMAND qw38_language_model_integration_test)
set_tests_properties(language_model_integration PROPERTIES
  LABELS "runtime;gpu;extended;checkpoint;authority")
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
          $<TARGET_FILE:qw38_decode_mmv_unit_test>
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
add_executable(qw38_compiler_bf16_verification_test compiler_bf16_verification_test.cpp)
target_link_libraries(qw38_compiler_bf16_verification_test PRIVATE qw38_compiler)
target_compile_features(qw38_compiler_bf16_verification_test PRIVATE cxx_std_23)
target_compile_options(qw38_compiler_bf16_verification_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
add_test(NAME compiler_bf16_verification COMMAND qw38_compiler_bf16_verification_test)
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
add_executable(qw38_q4k_reference_test q4k_reference_test.cpp)
target_link_libraries(qw38_q4k_reference_test PRIVATE qw38_compiler)
target_compile_options(qw38_q4k_reference_test PRIVATE -ffp-contract=off -fno-fast-math)
add_test(NAME q4k_reference COMMAND qw38_q4k_reference_test)
add_executable(qw38_q4k_projection_screen q4k_projection_screen.cpp)
target_link_libraries(qw38_q4k_projection_screen PRIVATE qw38_compiler qw38_cuda)
add_executable(qw38_q4k_artifact_audit q4k_artifact_audit.cpp)
target_link_libraries(qw38_q4k_artifact_audit PRIVATE qw38_format)
add_test(NAME pack_layout COMMAND qw38_pack_layout_test)
add_test(NAME quant_reference COMMAND qw38_quant_reference_test)
add_test(NAME quant_compiler_integration COMMAND qw38_quant_compiler_integration_test)

add_executable(qw38_candidate_failure_test candidate_failure_test.cpp)
target_link_libraries(qw38_candidate_failure_test PRIVATE qw38_compiler qw38_runtime)
target_compile_features(qw38_candidate_failure_test PRIVATE cxx_std_23)
target_compile_options(qw38_candidate_failure_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
add_test(NAME candidate_failure COMMAND qw38_candidate_failure_test)
set_tests_properties(candidate_failure PROPERTIES LABELS "compiler;quantizer;runtime;gpu")

set_tests_properties(
  compiler_identity compiler_transform compiler_bf16_verification compiler_integration
  PROPERTIES
  LABELS "compiler"
)

set_tests_properties(
  compiler_integration
  PROPERTIES
  LABELS "compiler;extended;checkpoint;authority"
)

set_tests_properties(
  quantizer pack_layout quant_reference quant_compiler_integration
  PROPERTIES
  LABELS "quantizer;compiler"
)

set_tests_properties(
  quant_compiler_integration
  PROPERTIES
  LABELS "quantizer;compiler;extended;checkpoint"
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

add_executable(qw38_language_layer_integration_test
  language_layer_integration_test.cpp)
target_link_libraries(qw38_language_layer_integration_test PRIVATE
  qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_language_layer_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_language_layer_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
add_test(NAME language_layer_integration COMMAND qw38_language_layer_integration_test)
set_tests_properties(language_layer_integration PROPERTIES
  LABELS "cuda;gpu;runtime;gdn;attention;mlp"
)

add_executable(qw38_runtime_allocation_failure_test
  runtime_allocation_failure_test.cpp)
target_link_libraries(qw38_runtime_allocation_failure_test PRIVATE qw38_runtime)
target_compile_features(qw38_runtime_allocation_failure_test PRIVATE cxx_std_23)
target_compile_options(qw38_runtime_allocation_failure_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME runtime_plan COMMAND qw38_runtime_plan_test)
add_test(NAME runtime_raii COMMAND qw38_runtime_raii_test)
add_test(NAME runtime_state_index COMMAND qw38_runtime_state_index_test)
add_test(NAME runtime_session_integration COMMAND qw38_runtime_session_integration_test)
add_test(NAME runtime_allocation_failure COMMAND qw38_runtime_allocation_failure_test)

set_tests_properties(runtime_plan PROPERTIES LABELS "runtime")
set_tests_properties(
  runtime_raii runtime_state_index runtime_session_integration
  runtime_allocation_failure
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
add_custom_target(qw38_activation_unit_test
  DEPENDS qw38_activation_reference_test
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
add_custom_target(qw38_cuda_sm120_cubin_test
  COMMAND bash "${CMAKE_SOURCE_DIR}/tests/check_sm120_cubin.sh"
          $<TARGET_FILE:qw38_decode_mmv_unit_test>
  DEPENDS qw38_decode_mmv_unit_test
  VERBATIM
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

add_executable(qw38_prefill_projection_test prefill_projection_test.cpp)
target_link_libraries(qw38_prefill_projection_test PRIVATE qw38_cuda qw38_compiler)
target_compile_features(qw38_prefill_projection_test PRIVATE cxx_std_23)
add_test(NAME prefill_projection COMMAND qw38_prefill_projection_test)
set_tests_properties(prefill_projection PROPERTIES LABELS "cuda;gpu;prefill")

add_executable(qw38_prefill_artifact_integration_test prefill_artifact_integration_test.cpp)
target_link_libraries(qw38_prefill_artifact_integration_test PRIVATE
  qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_prefill_artifact_integration_test PRIVATE cxx_std_23)
add_test(NAME prefill_artifact_integration COMMAND qw38_prefill_artifact_integration_test)
set_tests_properties(prefill_artifact_integration PROPERTIES LABELS "cuda;gpu;prefill;extended")

set_tests_properties(
  decode_mmv_unit decode_mmv_reference decode_mmv_integration
  PROPERTIES
  LABELS "cuda;gpu;decode_mmv"
)

set_tests_properties(
  decode_mmv_integration
  PROPERTIES
  LABELS "cuda;gpu;decode_mmv;extended;full_vocab"
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

add_executable(qw38_gdn_unit_test gdn_unit_test.cpp)
target_link_libraries(qw38_gdn_unit_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_gdn_unit_test PRIVATE cxx_std_23)
target_compile_options(qw38_gdn_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_gdn_reference_test gdn_reference_test.cpp)
target_link_libraries(qw38_gdn_reference_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_gdn_reference_test PRIVATE cxx_std_23)
target_compile_options(qw38_gdn_reference_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_gdn_integration_test gdn_integration_test.cpp)
target_link_libraries(qw38_gdn_integration_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_gdn_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_gdn_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME gdn_unit COMMAND qw38_gdn_unit_test)
add_test(NAME gdn_reference COMMAND qw38_gdn_reference_test)
add_test(NAME gdn_integration COMMAND qw38_gdn_integration_test)

add_executable(qw38_gdn_prefill_test gdn_prefill_test.cpp)
target_link_libraries(qw38_gdn_prefill_test PRIVATE qw38_runtime qw38_reference)
target_compile_features(qw38_gdn_prefill_test PRIVATE cxx_std_23)
add_test(NAME gdn_prefill COMMAND qw38_gdn_prefill_test)
set_tests_properties(gdn_prefill PROPERTIES LABELS "cuda;gpu;gdn;prefill")

add_executable(qw38_gdn_prefill_artifact_test gdn_prefill_artifact_test.cpp)
target_link_libraries(qw38_gdn_prefill_artifact_test PRIVATE qw38_runtime)
target_compile_features(qw38_gdn_prefill_artifact_test PRIVATE cxx_std_23)
add_test(NAME gdn_prefill_artifact COMMAND qw38_gdn_prefill_artifact_test)
set_tests_properties(gdn_prefill_artifact PROPERTIES
  LABELS "cuda;gpu;gdn;prefill;extended")

set_tests_properties(
  gdn_unit gdn_reference gdn_integration
  PROPERTIES
  LABELS "cuda;gpu;gdn"
)

add_executable(qw38_attention_unit_test attention_unit_test.cpp)
add_executable(qw38_attention_prefill_test attention_prefill_test.cpp)
target_link_libraries(qw38_attention_prefill_test PRIVATE qw38_cuda qw38_format)
target_compile_features(qw38_attention_prefill_test PRIVATE cxx_std_23)
add_test(NAME attention_prefill COMMAND qw38_attention_prefill_test)
set_tests_properties(attention_prefill PROPERTIES LABELS "cuda;gpu;attention;prefill")
add_executable(qw38_attention_prefill_artifact_test attention_prefill_artifact_test.cpp)
target_link_libraries(qw38_attention_prefill_artifact_test PRIVATE qw38_runtime)
target_compile_features(qw38_attention_prefill_artifact_test PRIVATE cxx_std_23)
add_test(NAME attention_prefill_artifact COMMAND qw38_attention_prefill_artifact_test)
set_tests_properties(attention_prefill_artifact PROPERTIES LABELS "cuda;gpu;attention;prefill;extended")
target_link_libraries(qw38_attention_unit_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_attention_unit_test PRIVATE cxx_std_23)
target_compile_options(qw38_attention_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_attention_core_unit_test attention_core_unit_test.cpp)
target_link_libraries(qw38_attention_core_unit_test PRIVATE qw38_cuda qw38_reference)
target_compile_features(qw38_attention_core_unit_test PRIVATE cxx_std_23)
target_compile_options(qw38_attention_core_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_attention_reference_test attention_reference_test.cpp)
target_link_libraries(qw38_attention_reference_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_attention_reference_test PRIVATE cxx_std_23)
target_compile_options(qw38_attention_reference_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_attention_integration_test attention_integration_test.cpp)
target_link_libraries(qw38_attention_integration_test PRIVATE qw38_runtime qw38_compiler qw38_reference)
target_compile_features(qw38_attention_integration_test PRIVATE cxx_std_23)
target_compile_options(qw38_attention_integration_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_test(NAME attention_unit COMMAND qw38_attention_unit_test)
add_test(NAME attention_core_unit COMMAND qw38_attention_core_unit_test)
add_test(NAME attention_reference COMMAND qw38_attention_reference_test)
add_test(NAME attention_integration COMMAND qw38_attention_integration_test)

set_tests_properties(
  attention_unit attention_core_unit attention_reference attention_integration
  PROPERTIES
  LABELS "cuda;gpu;attention"
)

add_executable(qw38_nvfp4_test nvfp4_test.cpp)
target_link_libraries(qw38_nvfp4_test PRIVATE qw38_cuda qw38_compiler)
target_include_directories(qw38_nvfp4_test SYSTEM PRIVATE "${QW38_CUTLASS_DIR}/include"
  "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl")
add_test(NAME nvfp4 COMMAND qw38_nvfp4_test)
set_tests_properties(nvfp4 PROPERTIES LABELS "cuda;gpu;compiler;prefill")
