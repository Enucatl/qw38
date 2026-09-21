# Benchmark targets are distinct from correctness tests and must not be
# registered with CTest. EXCLUDE_FROM_ALL keeps them off the default build.
add_executable(qw38_bench_placeholder EXCLUDE_FROM_ALL placeholder.cpp)
target_compile_features(qw38_bench_placeholder PRIVATE cxx_std_23)
