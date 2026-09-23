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
)
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
  runtime/error.cpp
  runtime/sizes.cpp
  runtime/arena.cpp
  runtime/model.cpp
  runtime/session.cpp
  runtime/runtime.cpp
  runtime/mlp.cpp
  runtime/gdn.cpp
  runtime/attention.cpp
)
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
