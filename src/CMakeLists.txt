add_library(qw38_format STATIC
  format/error.cpp
  format/constants.cpp
  format/wire.cpp
  format/layout.cpp
  format/schema.cpp
)
target_include_directories(qw38_format PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(qw38_format PUBLIC cxx_std_23)
target_compile_options(qw38_format PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_host_expected_smoke host_expected_smoke.cpp)
target_compile_features(qw38_host_expected_smoke PRIVATE cxx_std_23)
target_compile_options(qw38_host_expected_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
