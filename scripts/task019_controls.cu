// TASK-019 bounded Q4G64-to-BF16 and full BF16 cuBLAS projection controls.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

float bf16_value(uint16_t code) {
  uint32_t bits = uint32_t(code) << 16;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

std::vector<uint16_t> read_bf16(const std::string& path, size_t elements) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < std::streamoff(elements * 2))
    throw std::runtime_error("BF16 input is shorter than the selected prefix: " + path);
  input.seekg(0);
  std::vector<uint16_t> result(elements);
  input.read(reinterpret_cast<char*>(result.data()), elements * 2);
  if (!input) throw std::runtime_error("BF16 input read failed: " + path);
  return result;
}

struct Q4 {
  std::vector<uint8_t> codes;
  std::vector<__half> scales;
};

Q4 quantize_q4(const std::vector<uint16_t>& weights, int rows, int k) {
  Q4 result;
  result.codes.resize(size_t(rows) * k / 2);
  result.scales.resize(size_t(rows) * k / 64);
  for (int row = 0; row < rows; ++row) {
    for (int base = 0; base < k; base += 64) {
      float peak = 0;
      for (int offset = 0; offset < 64; ++offset)
        peak = std::max(peak, std::abs(bf16_value(
            weights[size_t(row) * k + base + offset])));
      float need = std::max(peak / 7.f, 0x1p-14f);
      __half scale = peak == 0 ? __float2half(0.f) : __float2half_ru(need);
      result.scales[size_t(row) * (k / 64) + base / 64] = scale;
      for (int offset = 0; offset < 64; offset += 2) {
        auto code_for = [&](int position) {
          if (peak == 0) return 0;
          float value = bf16_value(weights[size_t(row) * k + base + position]);
          return std::clamp(int(std::nearbyint(value / __half2float(scale))), -7, 7);
        };
        int low = code_for(offset) & 15;
        int high = code_for(offset + 1) & 15;
        result.codes[(size_t(row) * k + base + offset) / 2] =
            uint8_t(low | high << 4);
      }
    }
  }
  return result;
}

__global__ void unpack_q4(const uint8_t* codes, const __half* scales,
                          __nv_bfloat16* dense, int rows, int k) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(rows) * k) return;
  unsigned byte = codes[index / 2];
  int code = index & 1 ? byte >> 4 : byte & 15;
  if (code >= 8) code -= 16;
  size_t row = index / k;
  size_t group = (index % k) / 64;
  dense[index] = __float2bfloat16(float(code) *
                                  __half2float(scales[row * (k / 64) + group]));
}

int check_output(const std::vector<uint16_t>& a,
                 const std::vector<uint16_t>& b, const Q4* q4,
                 const std::vector<uint16_t>& output, int m, int n, int k) {
  float max_abs = 0;
  int mismatches = 0;
  int checked = 0;
  for (int row = 0; row < m; ++row) {
    if (row >= 4 && row != m / 2 && row != m - 1) continue;
    for (int column = 0; column < n; ++column) {
      if (column >= 8 && column != n / 4 && column != n / 2 &&
          column != 3 * n / 4 && column != n - 1) continue;
      ++checked;
      float sum = 0;
      for (int inner = 0; inner < k; ++inner) {
        float weight = bf16_value(b[size_t(column) * k + inner]);
        if (q4) {
          size_t index = size_t(column) * k + inner;
          unsigned byte = q4->codes[index / 2];
          int code = index & 1 ? byte >> 4 : byte & 15;
          if (code >= 8) code -= 16;
          weight = bf16_value(__bfloat16_as_ushort(__float2bfloat16(
              float(code) * __half2float(q4->scales[size_t(column) *
                                                     (k / 64) + inner / 64]))));
        }
        sum = std::fma(bf16_value(a[size_t(row) * k + inner]), weight, sum);
      }
      float expected = bf16_value(__bfloat16_as_ushort(__float2bfloat16(sum)));
      float actual = bf16_value(output[size_t(row) * n + column]);
      float error = std::abs(expected - actual);
      max_abs = std::max(max_abs, error);
      if (error > std::max(.0078125f * std::abs(expected), .0001f)) {
        if (mismatches < 5)
          std::cerr << "control mismatch row=" << row << " column=" << column
                    << " expected=" << expected << " actual=" << actual << '\n';
        ++mismatches;
      }
    }
  }
  std::cout << "Independent " << (q4 ? "Q4G64" : "BF16")
            << " contraction: " << (mismatches ? "FAILED" : "PASS")
            << " sampled_outputs=" << checked
            << " max_abs=" << max_abs << " mismatches=" << mismatches << '\n';
  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6 || argc > 7) {
    std::cerr << "usage: task019_controls A.bf16le B.bf16le M N K [samples]\n";
    return 2;
  }
  std::string a_path = argv[1], b_path = argv[2];
  int m = std::stoi(argv[3]), n = std::stoi(argv[4]), k = std::stoi(argv[5]);
  int samples = argc == 7 ? std::stoi(argv[6]) : 5;
  if (m <= 0 || n <= 0 || k <= 0 || k % 64 != 0 ||
      m > 1024 || n > 248320 || k > 17408 || samples < 1 || samples > 100)
    return 2;
  auto a = read_bf16(a_path, size_t(m) * k);
  auto b = read_bf16(b_path, size_t(n) * k);
  int tile_n = std::min(n, 128);
  auto qbegin = std::chrono::steady_clock::now();
  Q4 q4 = quantize_q4(b, tile_n, k);
  auto qend = std::chrono::steady_clock::now();
  std::cout << "Q4G64 weight pack_ms="
            << std::chrono::duration<double, std::milli>(qend-qbegin).count()
            << " bounded_rows=" << tile_n << " scales=FP16 codes=INT4\n";

  uint16_t *device_a = nullptr, *device_b = nullptr, *device_b_q4 = nullptr;
  uint16_t *device_c = nullptr, *device_c_q4 = nullptr;
  uint8_t* device_codes = nullptr;
  __half* device_scales = nullptr;
  CUDA_OK(cudaMalloc(&device_a, a.size() * 2));
  CUDA_OK(cudaMalloc(&device_b, b.size() * 2));
  CUDA_OK(cudaMalloc(&device_b_q4, size_t(tile_n) * k * 2));
  CUDA_OK(cudaMalloc(&device_c, size_t(m) * n * 2));
  CUDA_OK(cudaMalloc(&device_c_q4, size_t(m) * tile_n * 2));
  CUDA_OK(cudaMalloc(&device_codes, q4.codes.size()));
  CUDA_OK(cudaMalloc(&device_scales, q4.scales.size() * 2));
  CUDA_OK(cudaMemcpy(device_b, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(device_codes, q4.codes.data(), q4.codes.size(),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(device_scales, q4.scales.data(), q4.scales.size() * 2,
                     cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  CUBLAS_OK(cublasCreate(&handle));
  float alpha = 1.f, beta = 0.f;
  auto gemm = [&](int cols, const uint16_t* weights, uint16_t* output) {
    return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, m, k,
        &alpha, weights, CUDA_R_16BF, k, device_a, CUDA_R_16BF, k,
        &beta, output, CUDA_R_16BF, cols, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };
  std::vector<uint16_t> c(size_t(m) * n), c_q4(size_t(m) * tile_n);
  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start));
  CUDA_OK(cudaEventCreate(&stop));
  for (int warmup = 0; warmup < 5; ++warmup) {
    CUDA_OK(cudaMemcpy(device_a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CUBLAS_OK(gemm(n, device_b, device_c));
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(c.data(), device_c, c.size() * 2, cudaMemcpyDeviceToHost));
  }
  for (int sample = 0; sample < samples; ++sample) {
    auto begin = std::chrono::steady_clock::now();
    CUDA_OK(cudaMemcpy(device_a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(start));
    CUBLAS_OK(gemm(n, device_b, device_c));
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_OK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_OK(cudaMemcpy(c.data(), device_c, c.size() * 2, cudaMemcpyDeviceToHost));
    auto end = std::chrono::steady_clock::now();
    std::cout << "BF16 sample=" << sample << " gemm_ms=" << kernel_ms
              << " host_complete_ms="
              << std::chrono::duration<double, std::milli>(end-begin).count()
              << '\n';
  }
  if (check_output(a, b, nullptr, c, m, n, k)) return 1;
  for (int warmup = 0; warmup < 5; ++warmup) {
    CUDA_OK(cudaMemcpy(device_a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    unpack_q4<<<(size_t(tile_n) * k + 255) / 256, 256>>>(
        device_codes, device_scales,
        reinterpret_cast<__nv_bfloat16*>(device_b_q4), tile_n, k);
    CUBLAS_OK(gemm(tile_n, device_b_q4, device_c_q4));
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(c_q4.data(), device_c_q4, c_q4.size() * 2,
                       cudaMemcpyDeviceToHost));
  }
  for (int sample = 0; sample < samples; ++sample) {
    auto begin = std::chrono::steady_clock::now();
    CUDA_OK(cudaMemcpy(device_a, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(start));
    unpack_q4<<<(size_t(tile_n) * k + 255) / 256, 256>>>(
        device_codes, device_scales,
        reinterpret_cast<__nv_bfloat16*>(device_b_q4), tile_n, k);
    CUBLAS_OK(gemm(tile_n, device_b_q4, device_c_q4));
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float kernel_ms = 0;
    CUDA_OK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_OK(cudaMemcpy(c_q4.data(), device_c_q4, c_q4.size() * 2,
                       cudaMemcpyDeviceToHost));
    auto end = std::chrono::steady_clock::now();
    std::cout << "Q4G64 sample=" << sample << " unpack_gemm_ms=" << kernel_ms
              << " host_complete_ms="
              << std::chrono::duration<double, std::milli>(end-begin).count()
              << '\n';
  }
  if (check_output(a, b, &q4, c_q4, m, tile_n, k)) return 1;
  float q4_error_max = 0;
  float q4_error_abs_over_max1 = 0;
  for (int row = 0; row < m; ++row) {
    if (row >= 4 && row != m / 2 && row != m - 1) continue;
    for (int column = 0; column < tile_n; ++column) {
      if (column >= 8 && column != tile_n / 4 &&
          column != tile_n / 2 && column != 3 * tile_n / 4 &&
          column != tile_n - 1) continue;
      float bf16 = bf16_value(c[size_t(row) * n + column]);
      float quantized = bf16_value(c_q4[size_t(row) * tile_n + column]);
      float error = std::abs(bf16 - quantized);
      q4_error_max = std::max(q4_error_max, error);
      q4_error_abs_over_max1 = std::max(q4_error_abs_over_max1,
                                    error / std::max(1.f, std::abs(bf16)));
    }
  }
  std::cout << "Q4G64 quantization error against BF16 (sampled outputs):"
            << " max_abs=" << q4_error_max
            << " max_abs_over_max1=" << q4_error_abs_over_max1 << '\n';
  size_t free_bytes, total_bytes;
  CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "Control bytes: bf16_weights=" << b.size()*2
            << " q4_codes=" << q4.codes.size()
            << " q4_scales=" << q4.scales.size()*2
            << " q4_bounded_unpack=" << size_t(tile_n)*k*2
            << " bf16_output=" << c.size()*2
            << " q4_output=" << c_q4.size()*2
            << " free_after=" << free_bytes << '\n';
  CUDA_OK(cudaEventDestroy(start));
  CUDA_OK(cudaEventDestroy(stop));
  CUBLAS_OK(cublasDestroy(handle));
  CUDA_OK(cudaFree(device_a));
  CUDA_OK(cudaFree(device_b));
  CUDA_OK(cudaFree(device_b_q4));
  CUDA_OK(cudaFree(device_c));
  CUDA_OK(cudaFree(device_c_q4));
  CUDA_OK(cudaFree(device_codes));
  CUDA_OK(cudaFree(device_scales));
  return 0;
}
