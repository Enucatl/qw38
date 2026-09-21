find_package(CUDAToolkit REQUIRED)

add_library(qw38_cuda STATIC
  error.cpp
  alloc.cpp
  buffer.cpp
  stream.cpp
  event.cpp
  upload.cpp
  copy.cu
  activation.cu
  decode_mmv.cu
)
set_target_properties(qw38_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
target_include_directories(qw38_cuda PUBLIC "${CMAKE_SOURCE_DIR}")
target_compile_features(qw38_cuda PUBLIC cxx_std_23)
target_link_libraries(qw38_cuda PUBLIC CUDA::cudart)
target_compile_options(qw38_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_executable(qw38_cuda_runtime_smoke cuda_runtime_smoke.cu)
set_target_properties(qw38_cuda_runtime_smoke PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(qw38_cuda_runtime_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
