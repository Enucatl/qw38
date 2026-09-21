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

add_test(NAME host_expected_smoke COMMAND qw38_host_expected_smoke)
add_test(NAME format_schema COMMAND qw38_format_schema_test)
add_test(NAME format_schema_integration COMMAND qw38_format_schema_integration_test)
add_test(NAME format_writer COMMAND qw38_format_writer_test)
add_test(NAME format_sha256 COMMAND qw38_format_sha256_test)
add_test(NAME format_writer_integration COMMAND qw38_format_writer_integration_test)
add_test(NAME cuda_runtime_smoke COMMAND qw38_cuda_runtime_smoke)
add_test(
  NAME cuda_sm120_cubin
  COMMAND bash "${CMAKE_SOURCE_DIR}/tests/check_sm120_cubin.sh"
          $<TARGET_FILE:qw38_cuda_runtime_smoke>
)

set_tests_properties(
  format_schema format_schema_integration
  format_writer format_sha256 format_writer_integration
  PROPERTIES
  LABELS "format"
)
set_tests_properties(cuda_runtime_smoke PROPERTIES
  LABELS "cuda;gpu"
)
set_tests_properties(cuda_sm120_cubin PROPERTIES
  LABELS "cuda"
)
