# Benchmark targets are distinct from correctness tests and must not be
# registered with CTest. EXCLUDE_FROM_ALL keeps them off the default build.
add_executable(qw38_bench_placeholder EXCLUDE_FROM_ALL placeholder.cpp)
target_compile_features(qw38_bench_placeholder PRIVATE cxx_std_23)

add_executable(qw38_bench_decode_mmv EXCLUDE_FROM_ALL decode_mmv_bench.cpp)
target_link_libraries(qw38_bench_decode_mmv PRIVATE qw38_cuda)
target_compile_features(qw38_bench_decode_mmv PRIVATE cxx_std_23)
target_compile_options(qw38_bench_decode_mmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_bench_mlp EXCLUDE_FROM_ALL mlp_bench.cpp)
target_link_libraries(qw38_bench_mlp PRIVATE qw38_runtime)
target_compile_features(qw38_bench_mlp PRIVATE cxx_std_23)
target_compile_options(qw38_bench_mlp PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_bench_gdn_recurrence EXCLUDE_FROM_ALL gdn_recurrence_bench.cpp)
target_link_libraries(qw38_bench_gdn_recurrence PRIVATE qw38_cuda)
target_compile_features(qw38_bench_gdn_recurrence PRIVATE cxx_std_23)
target_compile_options(qw38_bench_gdn_recurrence PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

add_executable(qw38_bench_gdn_mixer EXCLUDE_FROM_ALL gdn_mixer_bench.cpp)
target_link_libraries(qw38_bench_gdn_mixer PRIVATE qw38_runtime)
target_compile_features(qw38_bench_gdn_mixer PRIVATE cxx_std_23)
target_compile_options(qw38_bench_gdn_mixer PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
