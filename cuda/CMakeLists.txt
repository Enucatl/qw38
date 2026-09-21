add_executable(qw38_cuda_runtime_smoke cuda_runtime_smoke.cu)
set_target_properties(qw38_cuda_runtime_smoke PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(qw38_cuda_runtime_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
