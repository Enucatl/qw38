#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

static_assert(__cplusplus >= 202302L, "CUDA compilation must use C++23");

namespace {

[[noreturn]] void fail(char const* what, cudaError_t err) {
  std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
  std::exit(1);
}

void check(cudaError_t err, char const* what) {
  if (err != cudaSuccess) {
    fail(what, err);
  }
}

struct DeviceDeleter {
  void operator()(void* ptr) const noexcept {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

void print_cuda_version(char const* label, int version) {
  std::printf("%s=%d.%d\n", label, version / 1000, (version % 1000) / 10);
}

}  // namespace

__global__ void add_one_and_report_arch(int* value, int* arch) {
  *value += 1;
#if defined(__CUDA_ARCH__)
  *arch = __CUDA_ARCH__;
#else
  *arch = 0;
#endif
}

int main() {
  int device = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int driver_version = 0;
  int runtime_version = 0;
  check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

  std::printf("device_name=%s\n", prop.name);
  std::printf("device_compute_capability=%d.%d\n", prop.major, prop.minor);
  print_cuda_version("cuda_driver_version", driver_version);
  print_cuda_version("cuda_runtime_version", runtime_version);

  int host_value = 41;
  int host_arch = 0;

  void* raw_value = nullptr;
  void* raw_arch = nullptr;
  check(cudaMalloc(&raw_value, sizeof(int)), "cudaMalloc value");
  std::unique_ptr<void, DeviceDeleter> device_value(raw_value);
  check(cudaMalloc(&raw_arch, sizeof(int)), "cudaMalloc arch");
  std::unique_ptr<void, DeviceDeleter> device_arch(raw_arch);

  check(cudaMemcpy(device_value.get(), &host_value, sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H2D value");
  check(cudaMemcpy(device_arch.get(), &host_arch, sizeof(int),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H2D arch");

  add_one_and_report_arch<<<1, 1>>>(static_cast<int*>(device_value.get()),
                                    static_cast<int*>(device_arch.get()));
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check(cudaMemcpy(&host_value, device_value.get(), sizeof(int),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H value");
  check(cudaMemcpy(&host_arch, device_arch.get(), sizeof(int),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H arch");

  std::printf("compiled_cuda_arch=%d\n", host_arch);
  std::printf("kernel_result=%d\n", host_value);

  if (host_arch != 1200) {
    std::fprintf(stderr, "compiled CUDA arch %d is not sm_120 (1200)\n",
                 host_arch);
    return 1;
  }
  if (host_value != 42) {
    std::fprintf(stderr, "kernel result %d != 42\n", host_value);
    return 1;
  }

  std::printf("cuda runtime smoke ok\n");
  return 0;
}
