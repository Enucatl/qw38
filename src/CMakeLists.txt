add_executable(qw38_host_expected_smoke host_expected_smoke.cpp)
target_compile_features(qw38_host_expected_smoke PRIVATE cxx_std_23)
target_compile_options(qw38_host_expected_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
