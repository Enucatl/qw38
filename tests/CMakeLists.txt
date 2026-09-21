add_test(NAME host_expected_smoke COMMAND qw38_host_expected_smoke)
add_test(NAME cuda_runtime_smoke COMMAND qw38_cuda_runtime_smoke)
add_test(
  NAME cuda_sm120_cubin
  COMMAND bash "${CMAKE_SOURCE_DIR}/tests/check_sm120_cubin.sh"
          $<TARGET_FILE:qw38_cuda_runtime_smoke>
)

set_tests_properties(cuda_runtime_smoke PROPERTIES
  LABELS "cuda;gpu"
)
set_tests_properties(cuda_sm120_cubin PROPERTIES
  LABELS "cuda"
)
