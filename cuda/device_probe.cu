#include <cuda_runtime.h>

#include <cstdio>

namespace {

int fail(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

}  // namespace

int main() {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return fail("cudaGetDevice", error);
  }

  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return fail("cudaGetDeviceProperties", error);
  }
  if (properties.major != 12 || properties.minor != 0) {
    std::fprintf(stderr, "expected compute capability 12.0, found %d.%d\n",
                 properties.major, properties.minor);
    return 2;
  }

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) {
    return fail("cudaMemGetInfo", error);
  }

  std::printf("name=%s\n", properties.name);
  std::printf("compute_capability=%d.%d\n", properties.major, properties.minor);
  std::printf("total_bytes=%zu\n", total_bytes);
  std::printf("free_bytes=%zu\n", free_bytes);
  return 0;
}
