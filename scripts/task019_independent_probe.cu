// Bounded independent check of the pinned CUTLASS SM120 example's device output.
// The included example supplies its kernel and physical layouts; reconstruction
// below reads packed bytes and implements E2M1/scale decoding independently.
#define main task019_cutlass_example_main
#ifdef TASK019_MX
#include "../.cache/task019/build-v4.8.0-sm120/task019/79x_blackwell_geforce_mxfp4_mxfp4_bf16_gemm.cu"
#else
#include "../.cache/task019/cutlass-v4.8.0/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu"
#endif
#undef main

#include <algorithm>
#include <cuda_bf16.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float e2m1(unsigned code) {
  constexpr float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  float value = values[code & 7];
  return code & 8 ? -value : value;
}

float scale_value(unsigned code) {
  if constexpr (cute::is_same_v<ElementA::ScaleFactorType,
                                cutlass::float_ue4m3_t>) {
    if (code == 0x7f) throw std::runtime_error("invalid UE4M3 scale");
    unsigned exponent = code >> 3;
    unsigned mantissa = code & 7;
    return exponent == 0 ? std::ldexp(float(mantissa), -9)
                         : std::ldexp(1.f + float(mantissa) / 8.f,
                                      int(exponent) - 7);
  } else {
    if (code == 0xff) throw std::runtime_error("invalid UE8M0 scale");
    return std::ldexp(1.f, int(code) - 127);
  }
}

float bf16_value(uint16_t code) {
  uint32_t bits = uint32_t(code) << 16;
  float value;
  std::memcpy(&value, &bits, sizeof(value));
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

unsigned e2m1_code(float value) {
  float magnitude = std::abs(value);
  unsigned sign = std::signbit(value) ? 8 : 0;
  unsigned code = magnitude <= .25f ? 0 : magnitude < .75f ? 1 :
                  magnitude <= 1.25f ? 2 : magnitude < 1.75f ? 3 :
                  magnitude <= 2.5f ? 4 : magnitude < 3.5f ? 5 :
                  magnitude <= 5.f ? 6 : 7;
  return sign | code;
}

unsigned scale_code(float peak) {
  if (peak == 0) return cute::is_same_v<ElementA::ScaleFactorType,
                                        cutlass::float_ue4m3_t> ? 0 : 0;
  float desired = peak / 6.f;
  if constexpr (cute::is_same_v<ElementA::ScaleFactorType,
                                cutlass::float_ue4m3_t>) {
    auto encoded = cutlass::float_ue4m3_t(desired);
    unsigned char code;
    std::memcpy(&code, &encoded, 1);
    return std::max(1u, unsigned(code));
  } else {
    int exponent = int(std::ceil(std::log2(desired)));
    return unsigned(std::clamp(exponent + 127, 0, 254));
  }
}

template<class Layout, class SfLayout>
void pack_real(const std::vector<uint16_t>& source, int rows, int logical_k,
               int physical_k, void* payload, const Layout& layout,
               void* scales, const SfLayout& sf_layout) {
  auto* data = static_cast<unsigned char*>(payload);
  auto* sf_data = static_cast<unsigned char*>(scales);
  constexpr int block = cute::is_same_v<ElementA::ScaleFactorType,
                                        cutlass::float_ue4m3_t> ? 16 : 32;
  for (int row = 0; row < rows; ++row) {
    for (int begin = 0; begin < physical_k; begin += block) {
      float peak = 0;
      for (int k = begin; k < std::min(begin + block, logical_k); ++k)
        peak = std::max(peak, std::abs(bf16_value(source[size_t(row) *
                                                        logical_k + k])));
      unsigned sf = scale_code(peak);
      float decoded_scale = scale_value(sf);
      for (int k = begin; k < begin + block; ++k) {
        sf_data[size_t(sf_layout(row, k, 0))] = sf;
        float value = k < logical_k ? bf16_value(source[size_t(row) *
                                                        logical_k + k]) : 0.f;
        unsigned code = peak == 0 ? 0 : e2m1_code(value / decoded_scale);
        auto index = size_t(layout(row, k, 0));
        auto& byte = data[index / 2];
        byte = index & 1 ? (byte & 0x0f) | (code << 4)
                         : (byte & 0xf0) | code;
      }
    }
  }
}

template<class Layout, class SfLayout>
void pack_real_b(const std::vector<uint16_t>& source, int rows, int logical_k,
                 int physical_k, void* payload, const Layout& layout,
                 void* scales, const SfLayout& sf_layout) {
  constexpr int block = cute::is_same_v<ElementA::ScaleFactorType,
                                        cutlass::float_ue4m3_t> ? 16 : 32;
  int blocks_per_row = physical_k / block;
  std::vector<uint8_t> codes(size_t(rows) * physical_k);
  std::vector<uint8_t> scale_codes(size_t(rows) * blocks_per_row);
  for (int row = 0; row < rows; ++row) {
    for (int begin = 0; begin < physical_k; begin += block) {
      float peak = 0;
      for (int k = begin; k < std::min(begin + block, logical_k); ++k)
        peak = std::max(peak, std::abs(bf16_value(source[size_t(row) *
                                                        logical_k + k])));
      unsigned sf = scale_code(peak);
      float decoded_scale = scale_value(sf);
      scale_codes[size_t(row) * blocks_per_row + begin / block] = sf;
      for (int k = begin; k < begin + block; ++k) {
        float value = k < logical_k ? bf16_value(source[size_t(row) *
                                                        logical_k + k]) : 0.f;
        codes[size_t(row) * physical_k + k] =
            peak == 0 ? 0 : e2m1_code(value / decoded_scale);
      }
    }
  }
  auto* data = static_cast<uint8_t*>(payload);
  auto* sf_data = static_cast<uint8_t*>(scales);
  for (int row_base = 0; row_base < rows; row_base += 128) {
    int row_end = std::min(row_base + 128, rows);
    for (int k_base = 0; k_base < physical_k; k_base += 128) {
      for (int k = k_base; k < k_base + 128; ++k) {
        for (int row = row_base; row < row_end; ++row) {
          auto index = size_t(layout(row, k, 0));
          unsigned code = codes[size_t(row) * physical_k + k];
          auto& byte = data[index / 2];
          byte = index & 1 ? (byte & 0x0f) | (code << 4)
                           : (byte & 0xf0) | code;
          if (k % block == 0)
            sf_data[size_t(sf_layout(row, k, 0))] =
                scale_codes[size_t(row) * blocks_per_row + k / block];
        }
      }
    }
  }
}

template<class Layout>
float decoded(const void* payload, const Layout& layout, const void* scales,
              const LayoutSFA& sf_layout, int row, int k) {
  auto element = size_t(layout(row, k, 0));
  auto byte = static_cast<const unsigned char*>(payload)[element / 2];
  unsigned code = element & 1 ? byte >> 4 : byte & 15;
  auto sf_index = size_t(sf_layout(row, k, 0));
  unsigned sf_code = static_cast<const unsigned char*>(scales)[sf_index];
  return e2m1(code) * scale_value(sf_code);
}

template<class Layout>
void zero_range(void* payload, const Layout& layout, int row, int begin,
                int end) {
  auto* bytes = static_cast<unsigned char*>(payload);
  for (int k = begin; k < end; ++k) {
    auto element = size_t(layout(row, k, 0));
    auto& byte = bytes[element / 2];
    byte = element & 1 ? byte & 0x0f : byte & 0xf0;
  }
}

__device__ float e2m1_device(unsigned code) {
  constexpr float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  float value = values[code & 7];
  return code & 8 ? -value : value;
}

__device__ float scale_device(unsigned code) {
  if constexpr (cute::is_same_v<ElementA::ScaleFactorType,
                                cutlass::float_ue4m3_t>) {
    unsigned exponent = code >> 3;
    unsigned mantissa = code & 7;
    return exponent == 0 ? ldexpf(float(mantissa), -9)
                         : ldexpf(1.f + float(mantissa) / 8.f,
                                  int(exponent) - 7);
  } else {
    return ldexpf(1.f, int(code) - 127);
  }
}

}  // namespace

__global__ void same_weight_gemv(const unsigned char* packed_b,
                                 const unsigned char* scales_b,
                                 LayoutB b_layout, LayoutSFB sf_layout,
                                 const uint16_t* activation, uint16_t* output,
                                 int logical_n, int logical_k) {
  int row = blockIdx.x;
  if (row >= logical_n) return;
  __shared__ float partial[256];
  float sum = 0;
  for (int k = threadIdx.x; k < logical_k; k += blockDim.x) {
    auto index = size_t(b_layout(row, k, 0));
    unsigned byte = packed_b[index / 2];
    unsigned code = index & 1 ? byte >> 4 : byte & 15;
    unsigned scale = scales_b[size_t(sf_layout(row, k, 0))];
    float a = __bfloat162float(__ushort_as_bfloat16(activation[k]));
    sum = fmaf(a, e2m1_device(code) * scale_device(scale), sum);
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset)
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[row] = __bfloat16_as_ushort(__float2bfloat16(partial[0]));
}

int main(int argc, char const** argv) {
  Options options;
  options.parse(argc, argv);
  std::string a_file, b_file;
  int samples = 0;
  int logical_k = options.k;
  int logical_n = options.n;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--logical-k=", 12) == 0)
      logical_k = std::stoi(argv[i] + 12);
    else if (std::strncmp(argv[i], "--logical-n=", 12) == 0)
      logical_n = std::stoi(argv[i] + 12);
    else if (std::strncmp(argv[i], "--a-file=", 9) == 0)
      a_file = argv[i] + 9;
    else if (std::strncmp(argv[i], "--b-file=", 9) == 0)
      b_file = argv[i] + 9;
    else if (std::strncmp(argv[i], "--samples=", 10) == 0)
      samples = std::stoi(argv[i] + 10);
  }
  bool real = !a_file.empty() || !b_file.empty();
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 ||
      options.m > (real ? 1024 : 32) ||
      options.n > (real ? 248320 : 128) ||
      options.k > (real ? 17408 : 512) ||
      options.k % 128 != 0 || options.alpha != 1.f || options.beta != 0.f ||
      samples < 0 || samples > 100 || (real && (a_file.empty() || b_file.empty()))) {
    std::cerr << "unsupported probe shape, files, samples or epilogue\n";
    return 2;
  }
  if (logical_k <= 0 || logical_k > options.k ||
      logical_n <= 0 || logical_n > options.n) return 2;

  size_t free_before, total_bytes;
  CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
  initialize(options);
  std::vector<uint16_t> source_a, source_b;
  if (real) {
    source_a = read_bf16(a_file, size_t(options.m) * logical_k);
    source_b = read_bf16(b_file, size_t(logical_n) * logical_k);
    auto begin = std::chrono::steady_clock::now();
    pack_real_b(source_b, logical_n, logical_k, options.k, block_B.host_data(),
                layout_B, block_SFB.host_data(), layout_SFB);
    for (int n = logical_n; n < options.n; ++n)
      zero_range(block_B.host_data(), layout_B, n, 0, options.k);
    auto packed = std::chrono::steady_clock::now();
    block_B.sync_device();
    block_SFB.sync_device();
    auto uploaded = std::chrono::steady_clock::now();
    std::cout << "Weight prepare: pack_ms="
              << std::chrono::duration<double, std::milli>(packed - begin).count()
              << " upload_ms="
              << std::chrono::duration<double, std::milli>(uploaded - packed).count()
              << " source=BF16 tensor_scale=1\n";
    pack_real(source_a, options.m, logical_k, options.k, block_A.host_data(),
              layout_A, block_SFA.host_data(), layout_SFA);
    block_A.sync_device();
    block_SFA.sync_device();
  } else {
    // Bounded edge fixture: a logical K/N tail and all-zero first blocks.
    for (int m = 0; m < options.m; ++m)
      zero_range(block_A.host_data(), layout_A, m, logical_k, options.k);
    for (int n = 0; n < options.n; ++n)
      zero_range(block_B.host_data(), layout_B, n, logical_k, options.k);
    for (int n = logical_n; n < options.n; ++n)
      zero_range(block_B.host_data(), layout_B, n, 0, options.k);
    zero_range(block_A.host_data(), layout_A, 0, 0,
               cute::is_same_v<ElementA::ScaleFactorType,
                               cutlass::float_ue4m3_t> ? 16 : 32);
    zero_range(block_B.host_data(), layout_B, 0, 0,
               cute::is_same_v<ElementA::ScaleFactorType,
                               cutlass::float_ue4m3_t> ? 16 : 32);
    block_A.sync_device();
    block_B.sync_device();
  }

  Gemm gemm;
  auto arguments = args_from_options(options);
  auto workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  size_t free_after;
  CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
  auto supported = gemm.can_implement(arguments);
  if (supported != cutlass::Status::kSuccess) {
    std::cerr << "unsupported by pinned CUTLASS example: "
              << cutlass::cutlassGetStatusString(supported) << '\n';
    return 3;
  }
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  block_D.sync_host();

  float max_abs = 0;
  float max_abs_over_max1 = 0;
  float quantization_max_abs = 0;
  float quantization_max_abs_over_max1 = 0;
  int mismatches = 0;
  int checked = 0;
  for (int m = 0; m < options.m; ++m) {
    if (real && m >= 4 && m != options.m / 2 && m != options.m - 1)
      continue;
    for (int n = 0; n < options.n; ++n) {
      if (real && n >= 8 && n != logical_n / 4 &&
          n != logical_n / 2 && n != 3 * logical_n / 4 &&
          n != logical_n - 1 && n != options.n - 1)
        continue;
      float reference = 0;
      for (int k = 0; k < options.k; ++k) {
        float a = decoded(block_A.host_data(), layout_A, block_SFA.host_data(),
                          layout_SFA, m, k);
        float b = decoded(block_B.host_data(), layout_B, block_SFB.host_data(),
                          layout_SFB, n, k);
        reference = std::fma(a, b, reference);
      }
      float expected = float(cutlass::bfloat16_t(reference));
      float actual = float(block_D.host_data()[size_t(layout_D(m, n, 0))]);
      ++checked;
      float error = std::abs(expected - actual);
      max_abs = std::max(max_abs, error);
      max_abs_over_max1 = std::max(max_abs_over_max1,
                              error / std::max(1.f, std::abs(expected)));
      // Tensor Core accumulation order may differ; allow one BF16 ULP plus
      // a small FP32 accumulation margin, while requiring exact zero blocks.
      float tolerance = std::max(.0078125f * std::abs(expected), .0001f);
      if (error > tolerance) {
        if (mismatches < 5)
          std::cerr << "mismatch m=" << m << " n=" << n
                    << " expected=" << expected << " actual=" << actual
                    << " error=" << error << '\n';
        ++mismatches;
      }
      if (real && n < logical_n) {
        float original = 0;
        for (int k = 0; k < logical_k; ++k)
          original = std::fma(bf16_value(source_a[size_t(m) * logical_k + k]),
                              bf16_value(source_b[size_t(n) * logical_k + k]),
                              original);
        float bf16_original = float(cutlass::bfloat16_t(original));
        float q_error = std::abs(bf16_original - expected);
        quantization_max_abs = std::max(quantization_max_abs, q_error);
        quantization_max_abs_over_max1 = std::max(
            quantization_max_abs_over_max1,
            q_error / std::max(1.f, std::abs(bf16_original)));
      }
    }
  }
  std::cout << "Independent reconstructed contraction: "
            << (mismatches ? "FAILED" : "PASS") << " m=" << options.m
            << " logical_n=" << logical_n << " padded_n=" << options.n
            << " logical_k=" << logical_k
            << " padded_k=" << options.k << " output=BF16"
            << " workspace_bytes=" << workspace_size
            << " checked_outputs=" << checked
            << " max_abs=" << max_abs
            << " max_abs_over_max1=" << max_abs_over_max1
            << " mismatches=" << mismatches << '\n';
  if (real) {
    std::cout << "Quantization error against BF16 input (sampled outputs):"
              << " max_abs=" << quantization_max_abs
              << " max_abs_over_max1=" << quantization_max_abs_over_max1
              << '\n';
    std::cout << "Bytes: operand_a=" << (cute::size(layout_A) + 1) / 2
              << " scale_a=" << cute::size(cute::filter_zeros(layout_SFA))
              << " operand_b=" << (cute::size(layout_B) + 1) / 2
              << " scale_b=" << cute::size(cute::filter_zeros(layout_SFB))
              << " output=" << size_t(options.m) * options.n * 2
              << " workspace=" << workspace_size
              << " free_before=" << free_before
              << " free_after=" << free_after << '\n';
  }
  if (!mismatches && real && samples > 0) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int warmup = 0; warmup < 5; ++warmup) {
      pack_real(source_a, options.m, logical_k, options.k, block_A.host_data(),
                layout_A, block_SFA.host_data(), layout_SFA);
      block_A.sync_device();
      block_SFA.sync_device();
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
      CUDA_CHECK(cudaDeviceSynchronize());
      block_D.sync_host();
    }
    for (int sample = 0; sample < samples; ++sample) {
      auto begin = std::chrono::steady_clock::now();
      pack_real(source_a, options.m, logical_k, options.k, block_A.host_data(),
                layout_A, block_SFA.host_data(), layout_SFA);
      block_A.sync_device();
      block_SFA.sync_device();
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUDA_CHECK(cudaEventRecord(start));
      CUTLASS_CHECK(gemm.run());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float kernel_ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
      block_D.sync_host();
      auto end = std::chrono::steady_clock::now();
      std::cout << "Sample: index=" << sample
                << " kernel_ms=" << kernel_ms
                << " host_complete_ms="
                << std::chrono::duration<double, std::milli>(end - begin).count()
                << '\n';
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    uint16_t *device_activation = nullptr, *device_gemv_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_activation, size_t(logical_k) * 2));
    CUDA_CHECK(cudaMalloc(&device_gemv_output, size_t(logical_n) * 2));
    std::vector<uint16_t> gemv_output(logical_n);
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int warmup = 0; warmup < 5; ++warmup) {
      CUDA_CHECK(cudaMemcpy(device_activation, source_a.data(),
                            size_t(logical_k) * 2, cudaMemcpyHostToDevice));
      same_weight_gemv<<<logical_n, 256>>>(
          reinterpret_cast<const unsigned char*>(block_B.device_data()),
          reinterpret_cast<const unsigned char*>(block_SFB.device_data()),
          layout_B, layout_SFB, device_activation, device_gemv_output,
          logical_n, logical_k);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int sample = 0; sample < samples; ++sample) {
      auto begin = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaMemcpy(device_activation, source_a.data(),
                            size_t(logical_k) * 2, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventRecord(start));
      same_weight_gemv<<<logical_n, 256>>>(
          reinterpret_cast<const unsigned char*>(block_B.device_data()),
          reinterpret_cast<const unsigned char*>(block_SFB.device_data()),
          layout_B, layout_SFB, device_activation, device_gemv_output,
          logical_n, logical_k);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float kernel_ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
      CUDA_CHECK(cudaMemcpy(gemv_output.data(), device_gemv_output,
                            size_t(logical_n) * 2, cudaMemcpyDeviceToHost));
      auto end = std::chrono::steady_clock::now();
      std::cout << "GEMV Sample: index=" << sample
                << " kernel_ms=" << kernel_ms
                << " host_complete_ms="
                << std::chrono::duration<double, std::milli>(end - begin).count()
                << '\n';
    }
    int gemv_mismatches = 0;
    float gemv_max_abs = 0;
    for (int n = 0; n < logical_n; ++n) {
      if (n >= 8 && n != logical_n / 4 && n != logical_n / 2 &&
          n != 3 * logical_n / 4 && n != logical_n - 1) continue;
      float reference = 0;
      for (int k = 0; k < logical_k; ++k) {
        float b = decoded(block_B.host_data(), layout_B,
                          block_SFB.host_data(), layout_SFB, n, k);
        reference = std::fma(bf16_value(source_a[k]), b, reference);
      }
      float expected = float(cutlass::bfloat16_t(reference));
      float actual = bf16_value(gemv_output[n]);
      float error = std::abs(expected - actual);
      gemv_max_abs = std::max(gemv_max_abs, error);
      if (error > std::max(.0078125f * std::abs(expected), .0001f))
        ++gemv_mismatches;
    }
    std::cout << "Same-weight BF16-activation GEMV: "
              << (gemv_mismatches ? "FAILED" : "PASS")
              << " n=" << logical_n << " k=" << logical_k
              << " max_abs=" << gemv_max_abs
              << " mismatches=" << gemv_mismatches << '\n';
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_activation));
    CUDA_CHECK(cudaFree(device_gemv_output));
    if (gemv_mismatches) return 1;
  }
  return mismatches ? 1 : 0;
}
