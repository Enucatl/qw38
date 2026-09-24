// Offline real-checkpoint activation proxies for TASK-019 shape probes.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_OK(call) do { auto error = (call); if (error != cudaSuccess) { \
  std::cerr << #call << ": " << cudaGetErrorString(error) << '\n'; return 2; \
} } while (false)
#define CUBLAS_OK(call) do { auto status = (call); if (status != CUBLAS_STATUS_SUCCESS) { \
  std::cerr << #call << ": cuBLAS status " << int(status) << '\n'; return 2; \
} } while (false)

namespace {

std::vector<uint16_t> read_prefix(const std::string& path, size_t elements) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < std::streamoff(elements * 2))
    throw std::runtime_error("BF16 source shorter than selected extent: " + path);
  input.seekg(0);
  std::vector<uint16_t> values(elements);
  input.read(reinterpret_cast<char*>(values.data()), elements * 2);
  if (!input) throw std::runtime_error("BF16 source read failed: " + path);
  return values;
}

}  // namespace

__global__ void apply_proxy(const __nv_bfloat16* first,
                            const __nv_bfloat16* second,
                            __nv_bfloat16* output, size_t count,
                            bool multiply_second) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float x = __bfloat162float(first[index]);
  float value = x / (1.f + expf(-x));
  if (multiply_second) value *= __bfloat162float(second[index]);
  output[index] = __float2bfloat16(value);
}

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage: task019_derive_activations silu|swiglu A B1 B2-or-- M N K output\n";
    return 2;
  }
  std::string mode = argv[1], a_path = argv[2], b1_path = argv[3];
  std::string b2_path = argv[4], output_path = argv[8];
  int m = std::stoi(argv[5]), n = std::stoi(argv[6]), k = std::stoi(argv[7]);
  bool swiglu = mode == "swiglu";
  if ((mode != "silu" && !swiglu) || (swiglu && b2_path == "-") ||
      (!swiglu && b2_path != "-") || m <= 0 || n <= 0 || k != 5120 ||
      m > 1024 || n > 17408) return 2;
  auto a = read_prefix(a_path, size_t(m) * k);
  auto b1 = read_prefix(b1_path, size_t(n) * k);
  std::vector<uint16_t> b2;
  if (swiglu) b2 = read_prefix(b2_path, size_t(n) * k);
  uint16_t *d_a = nullptr, *d_b1 = nullptr, *d_b2 = nullptr;
  uint16_t *d_c1 = nullptr, *d_c2 = nullptr, *d_out = nullptr;
  CUDA_OK(cudaMalloc(&d_a, a.size() * 2));
  CUDA_OK(cudaMalloc(&d_b1, b1.size() * 2));
  CUDA_OK(cudaMalloc(&d_c1, size_t(m) * n * 2));
  CUDA_OK(cudaMalloc(&d_out, size_t(m) * n * 2));
  if (swiglu) {
    CUDA_OK(cudaMalloc(&d_b2, b2.size() * 2));
    CUDA_OK(cudaMalloc(&d_c2, size_t(m) * n * 2));
  }
  CUDA_OK(cudaMemcpy(d_a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_b1, b1.data(), b1.size() * 2, cudaMemcpyHostToDevice));
  if (swiglu)
    CUDA_OK(cudaMemcpy(d_b2, b2.data(), b2.size() * 2, cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  CUBLAS_OK(cublasCreate(&handle));
  float alpha = 1.f, beta = 0.f;
  auto gemm = [&](uint16_t* weight, uint16_t* output) {
    return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, weight, CUDA_R_16BF, k, d_a, CUDA_R_16BF, k,
        &beta, output, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };
  CUBLAS_OK(gemm(d_b1, d_c1));
  if (swiglu) CUBLAS_OK(gemm(d_b2, d_c2));
  size_t count = size_t(m) * n;
  apply_proxy<<<(count + 255) / 256, 256>>>(
      reinterpret_cast<__nv_bfloat16*>(d_c1),
      reinterpret_cast<__nv_bfloat16*>(d_c2),
      reinterpret_cast<__nv_bfloat16*>(d_out), count, swiglu);
  CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> result(count);
  CUDA_OK(cudaMemcpy(result.data(), d_out, count * 2, cudaMemcpyDeviceToHost));
  std::ofstream output(output_path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(result.data()), count * 2);
  if (!output) throw std::runtime_error("failed writing output: " + output_path);
  std::cout << "Derived BF16 activation proxy: mode=" << mode
            << " m=" << m << " k=" << n
            << " bytes=" << count * 2 << " output=" << output_path << '\n';
  CUBLAS_OK(cublasDestroy(handle));
  CUDA_OK(cudaFree(d_a));
  CUDA_OK(cudaFree(d_b1));
  CUDA_OK(cudaFree(d_c1));
  CUDA_OK(cudaFree(d_out));
  if (swiglu) {
    CUDA_OK(cudaFree(d_b2));
    CUDA_OK(cudaFree(d_c2));
  }
  return 0;
}
