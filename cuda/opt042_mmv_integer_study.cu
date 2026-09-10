#include "full_scheduler.h"
#include "mixer.h"
#include "quant.h"
#include "quant_mmv.h"
#include "scheduler.h"
#include "scheduler_primitives.h"
#include "sha256.h"
#include "weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT042_MMV_INTEGER_STUDY_RESULT=";
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr int kWarpSize = 32;
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kValuesPerBlock = 256;
constexpr std::size_t kCapacity = 131072;
constexpr float kCud001Abs = 3.0e-4F;
constexpr float kCud001Rms = 2.0e-4F;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr const char* kProof =
    "claims no performance improvement; fixed CUD-001 envelope; unchanged "
    "FP32-scale Q8 staging; production dispatch unchanged; accepted keep "
    "denominators remain unchanged; does not substitute for the 2K llama.cpp "
    "parity gate; integer-dot is diagnostic only";
constexpr const char* kReportPath =
    "evidence/optimization/opt042-mmv-integer-study/REPORT.md";
constexpr int kSynGateWeight = 128;
constexpr int kSynDownWeight = 64;
constexpr int kRealGateWeight = 128;
constexpr int kRealDownWeight = 64;

struct Shape {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  int weight;
  bool probe;
};

constexpr Shape kSynthetic[] = {
    {"q4_k_17x256", 17, 256, 0, true},
    {"q4_k_257x512", 257, 512, 0, true},
    {"q4k_gate_up", 17408, 5120, kSynGateWeight, false},
    {"q4k_down", 5120, 17408, kSynDownWeight, false},
};
constexpr int kSyntheticCount =
    static_cast<int>(sizeof(kSynthetic) / sizeof(kSynthetic[0]));

constexpr int kRealLayers[] = {0, 3, 63};
constexpr int kRealLayerCount = 3;

enum class Projection { kGate, kUp, kDown };

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

void print_array(FILE* file, const char* key, const std::vector<float>& values) {
  std::fprintf(file, "\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::fprintf(file, ",");
    std::fprintf(file, "%.9g", static_cast<double>(values[index]));
  }
  std::fprintf(file, "]");
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

float unit_normal(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  const float u1 =
      (static_cast<float>(x & 0x00FFFFFFU) + 1.0F) / 16777217.0F;
  x = x * 1664525U + 1013904223U;
  const float u2 = static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

void fill_weights(std::size_t rows, std::size_t columns,
                  std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kValuesPerBlock) * kQ4KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ4KBytes) {
    write_u16(weights->data() + offset, 0x2400U);
    write_u16(weights->data() + offset + 2, 0x1C00U);
  }
}

void fill_activation(std::size_t columns,
                     std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    (*activation)[column] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(column), 0xA11CE5u));
  }
}

void q4_scale_min_host(const std::uint8_t* packed, int index, int* scale,
                       int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
}

float read_half_host(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

bool host_cud001(const std::vector<std::uint8_t>& weights, std::size_t rows,
                 std::size_t columns,
                 const std::vector<qw38::cuda::Q8Block>& activation,
                 std::vector<float>* output) {
  output->assign(rows, 0.0F);
  std::vector<float> decoded(kValuesPerBlock);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::size_t block = 0; block < columns / kValuesPerBlock; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / kValuesPerBlock) + block) *
                                kQ4KBytes;
      const qw38::Status status = qw38::internal::decode_q4_k(
          packed, kQ4KBytes, decoded.data(), decoded.size());
      if (!status.is_ok()) return false;
      for (std::size_t within = 0; within < kValuesPerBlock; ++within) {
        const std::size_t column = block * kValuesPerBlock + within;
        const auto& q8 = activation[column / 32];
        sum += decoded[within] *
               (q8.scale * static_cast<float>(q8.values[column % 32]));
      }
    }
    (*output)[row] = sum;
  }
  return true;
}

void host_integer(const std::vector<std::uint8_t>& weights, std::size_t rows,
                  std::size_t columns,
                  const std::vector<qw38::cuda::Q8Block>& activation,
                  std::vector<float>* output) {
  output->assign(rows, 0.0F);
  for (std::size_t row = 0; row < rows; ++row) {
    float acc = 0.0F;
    for (std::size_t block = 0; block < columns / kValuesPerBlock; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / kValuesPerBlock) + block) *
                                kQ4KBytes;
      const float d = read_half_host(packed);
      const float dmin = read_half_host(packed + 2);
      float folded = 0.0F;
      for (int s = 0; s < 8; ++s) {
        int scale = 0;
        int minimum = 0;
        q4_scale_min_host(packed + 4, s, &scale, &minimum);
        const qw38::cuda::Q8Block& q8 = activation[block * 8 + s];
        std::int32_t dot = 0;
        std::int32_t sum_q8 = 0;
        const int high = s & 1;
        const std::uint8_t* qs = packed + 16 + (s / 2) * 32;
        for (int i = 0; i < 32; ++i) {
          const int quant = high == 0 ? (qs[i] & 15) : (qs[i] >> 4);
          const std::int32_t q8v = static_cast<std::int32_t>(q8.values[i]);
          dot += quant * q8v;
          sum_q8 += q8v;
        }
        const float sd = d * static_cast<float>(scale * dot);
        const float md = dmin * static_cast<float>(minimum * sum_q8);
        const float inner = sd - md;
        const float c = q8.scale * inner;
        folded = s == 0 ? c : folded + c;
      }
      acc = block == 0 ? folded : acc + folded;
    }
    (*output)[row] = acc;
  }
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  float max_rel = 0.0F;
  float mean_rel = 0.0F;
  float cosine = 0.0F;
  int first_fail = -1;
  std::size_t nonfinite = 0;
  std::size_t rel_gt_1em3 = 0;
  std::size_t rel_gt_1em2 = 0;
  std::size_t rel_gt_1em1 = 0;
  std::size_t mmq_fail = 0;
};

Envelope compare_cud001(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        std::size_t columns) {
  Envelope metrics;
  double squared = 0.0;
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  double rel_sum = 0.0;
  std::size_t rel_count = 0;
  const float abs_tol =
      0.20F * std::sqrt(static_cast<float>(columns));
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float got = actual[index];
    const float ref = expected[index];
    if (!std::isfinite(got) || !std::isfinite(ref)) {
      ++metrics.nonfinite;
      if (metrics.first_fail < 0) {
        metrics.first_fail = static_cast<int>(index);
      }
      continue;
    }
    const float error = std::fabs(got - ref);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
    if (metrics.first_fail < 0 && error > kCud001Abs) {
      metrics.first_fail = static_cast<int>(index);
    }
    const float relative =
        ref != 0.0F ? error / std::fabs(ref)
                    : (error > 0.0F ? std::numeric_limits<float>::infinity()
                                    : 0.0F);
    if (std::isfinite(relative)) {
      metrics.max_rel = std::max(metrics.max_rel, relative);
      rel_sum += static_cast<double>(relative);
      ++rel_count;
      if (relative > 1.0e-3F) ++metrics.rel_gt_1em3;
      if (relative > 1.0e-2F) ++metrics.rel_gt_1em2;
      if (relative > 1.0e-1F) ++metrics.rel_gt_1em1;
    }
    if (error > abs_tol && relative > 0.05F) ++metrics.mmq_fail;
    dot += static_cast<double>(got) * ref;
    na += static_cast<double>(got) * got;
    nb += static_cast<double>(ref) * ref;
  }
  metrics.rms =
      actual.empty()
          ? 0.0F
          : static_cast<float>(std::sqrt(squared / static_cast<double>(actual.size())));
  metrics.mean_rel =
      rel_count == 0 ? 0.0F : static_cast<float>(rel_sum / rel_count);
  const double denom = std::sqrt(na) * std::sqrt(nb);
  metrics.cosine = denom == 0.0 ? 1.0F : static_cast<float>(dot / denom);
  return metrics;
}

Envelope compare_simple(const std::vector<float>& actual,
                        const std::vector<float>& expected) {
  Envelope metrics;
  double squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      ++metrics.nonfinite;
      continue;
    }
    const float error = std::fabs(actual[index] - expected[index]);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
  }
  metrics.rms =
      actual.empty()
          ? 0.0F
          : static_cast<float>(std::sqrt(squared / static_cast<double>(actual.size())));
  return metrics;
}

bool checksum_bytes(const void* data, std::size_t bytes, std::string* digest) {
  return qw38::internal::sha256_bytes(
             static_cast<const unsigned char*>(data), bytes, digest)
      .is_ok();
}

__device__ float read_half_dev(const std::uint8_t* bytes) {
  const unsigned short bits = static_cast<unsigned short>(bytes[0]) |
                              (static_cast<unsigned short>(bytes[1]) << 8U);
  return __half2float(__ushort_as_half(bits));
}

__device__ void q4_scale_min_dev(const std::uint8_t* packed, int index,
                                 int* scale, int* minimum) {
  if (index < 4) {
    *scale = packed[index] & 63;
    *minimum = packed[index + 4] & 63;
    return;
  }
  *scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
  *minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
}

template <int Warps>
__global__ void q4k_integer_mmv(const std::uint8_t* weights, std::size_t rows,
                                std::size_t columns, const qw38::cuda::Q8Block* q8,
                                float* output) {
  const int warp = threadIdx.x / kWarpSize;
  const int lid = threadIdx.x & (kWarpSize - 1);
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * Warps + warp;
  if (row >= rows) return;

  const int s = lid / 4;
  const int pack = lid % 4;
  const std::uint8_t* row_weights =
      weights + row * (columns / kValuesPerBlock) * kQ4KBytes;
  float acc = 0.0F;
  const std::size_t n_blocks = columns / kValuesPerBlock;
  for (std::size_t weight_block = 0; weight_block < n_blocks; ++weight_block) {
    const std::uint8_t* block = row_weights + weight_block * kQ4KBytes;
    const float d = read_half_dev(block);
    const float dmin = read_half_dev(block + 2);
    int scale = 0;
    int minimum = 0;
    q4_scale_min_dev(block + 4, s, &scale, &minimum);
    const std::uint8_t* qs = block + 16 + (s / 2) * 32;
    const int high = s & 1;
    const qw38::cuda::Q8Block& q8b = q8[weight_block * 8 + s];
    std::int8_t q4_lo[4];
    std::int8_t q4_hi[4];
    std::int8_t u_lo[4];
    std::int8_t u_hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int e0 = pack * 4 + i;
      const int e1 = 16 + pack * 4 + i;
      const std::uint8_t b0 = qs[e0];
      const std::uint8_t b1 = qs[e1];
      q4_lo[i] = static_cast<std::int8_t>(high == 0 ? (b0 & 15) : (b0 >> 4));
      q4_hi[i] = static_cast<std::int8_t>(high == 0 ? (b1 & 15) : (b1 >> 4));
      u_lo[i] = q8b.values[e0];
      u_hi[i] = q8b.values[e1];
    }
    auto pack4 = [](const std::int8_t* bytes) {
      return static_cast<int>(static_cast<std::uint8_t>(bytes[0])) |
             (static_cast<int>(static_cast<std::uint8_t>(bytes[1])) << 8) |
             (static_cast<int>(static_cast<std::uint8_t>(bytes[2])) << 16) |
             (static_cast<int>(static_cast<std::uint8_t>(bytes[3])) << 24);
    };
    const int q4p0 = pack4(q4_lo);
    const int q4p1 = pack4(q4_hi);
    const int u0 = pack4(u_lo);
    const int u1 = pack4(u_hi);
    int dot = 0;
    dot = __dp4a(q4p0, u0, dot);
    dot = __dp4a(q4p1, u1, dot);
    int sum_q8 = 0;
    sum_q8 = __dp4a(0x01010101, u0, sum_q8);
    sum_q8 = __dp4a(0x01010101, u1, sum_q8);
    dot += __shfl_xor_sync(0xFFFFFFFFU, dot, 1, kWarpSize);
    dot += __shfl_xor_sync(0xFFFFFFFFU, dot, 2, kWarpSize);
    sum_q8 += __shfl_xor_sync(0xFFFFFFFFU, sum_q8, 1, kWarpSize);
    sum_q8 += __shfl_xor_sync(0xFFFFFFFFU, sum_q8, 2, kWarpSize);
    float c = 0.0F;
    if (pack == 0) {
      const float sd = __fmul_rn(d, static_cast<float>(scale * dot));
      const float md = __fmul_rn(dmin, static_cast<float>(minimum * sum_q8));
      const float inner = __fsub_rn(sd, md);
      c = __fmul_rn(q8b.scale, inner);
    }
    const float c0 = __shfl_sync(0xFFFFFFFFU, c, 0, kWarpSize);
    const float c1 = __shfl_sync(0xFFFFFFFFU, c, 4, kWarpSize);
    const float c2 = __shfl_sync(0xFFFFFFFFU, c, 8, kWarpSize);
    const float c3 = __shfl_sync(0xFFFFFFFFU, c, 12, kWarpSize);
    const float c4 = __shfl_sync(0xFFFFFFFFU, c, 16, kWarpSize);
    const float c5 = __shfl_sync(0xFFFFFFFFU, c, 20, kWarpSize);
    const float c6 = __shfl_sync(0xFFFFFFFFU, c, 24, kWarpSize);
    const float c7 = __shfl_sync(0xFFFFFFFFU, c, 28, kWarpSize);
    if (lid == 0) {
      float folded = c0;
      folded = __fadd_rn(folded, c1);
      folded = __fadd_rn(folded, c2);
      folded = __fadd_rn(folded, c3);
      folded = __fadd_rn(folded, c4);
      folded = __fadd_rn(folded, c5);
      folded = __fadd_rn(folded, c6);
      folded = __fadd_rn(folded, c7);
      acc = weight_block == 0 ? folded : __fadd_rn(acc, folded);
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    acc = __fadd_rn(acc, __shfl_down_sync(0xFFFFFFFFU, acc, offset, kWarpSize));
  }
  if (lid == 0) output[row] = acc;
}

cudaError_t launch_integer_mmv(const std::uint8_t* weights, std::size_t rows,
                               std::size_t columns, const qw38::cuda::Q8Block* q8,
                               float* output, cudaStream_t stream) {
  const unsigned int warps = qw38::cuda::selected_mmv_warps(rows);
  const unsigned int row_blocks =
      static_cast<unsigned int>((rows + warps - 1) / warps);
  const int threads = static_cast<int>(warps) * kWarpSize;
  if (warps == 4) {
    q4k_integer_mmv<4><<<row_blocks, threads, 0, stream>>>(
        weights, rows, columns, q8, output);
  } else if (warps == 8) {
    q4k_integer_mmv<8><<<row_blocks, threads, 0, stream>>>(
        weights, rows, columns, q8, output);
  } else {
    q4k_integer_mmv<16><<<row_blocks, threads, 0, stream>>>(
        weights, rows, columns, q8, output);
  }
  return cudaPeekAtLastError();
}

int integer_occupancy(unsigned int warps) {
  int occupancy = 0;
  cudaError_t error = cudaErrorInvalidValue;
  if (warps == 4) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_integer_mmv<4>, 128, 0);
  } else if (warps == 8) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_integer_mmv<8>, 256, 0);
  } else if (warps == 16) {
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, q4k_integer_mmv<16>, 512, 0);
  }
  if (error != cudaSuccess) return 0;
  return occupancy;
}

struct TimedView {
  std::vector<float> samples;
  std::vector<float> warmup;
  float mean_ms = 0.0F;
};

struct CandidateResult {
  TimedView complete;
  TimedView prequant;
  int occupancy = 0;
  bool launch_ok = false;
  bool staging_equal = false;
  bool vs_host_integer_byte_equal = false;
  bool cud001_eligible = false;
  Envelope vs_host{};
  Envelope vs_packed{};
};

struct CaseResult {
  std::string id;
  std::size_t rows = 0;
  std::size_t columns = 0;
  unsigned int warps = 0;
  int weight = 0;
  bool probe = false;
  const char* activation_source = "";
  std::string weights_sha;
  std::string activation_sha;
  std::size_t weights_bytes = 0;
  std::size_t activation_bytes = 0;
  bool staging_equal = false;
  CandidateResult packed;
  CandidateResult integer;
};

void finish_mean(TimedView* view) {
  double acc = 0.0;
  for (float sample : view->samples) acc += static_cast<double>(sample);
  view->mean_ms = static_cast<float>(acc / kMeasured);
}

cudaError_t record_ms(cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaError_t error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return error;
  return cudaEventElapsedTime(ms, start, stop);
}

cudaError_t launch_packed_complete(const std::uint8_t* weights, std::size_t rows,
                                   std::size_t columns,
                                   const __nv_bfloat16* activation,
                                   qw38::cuda::Q8Block* q8, float* output) {
  cudaError_t error =
      qw38::cuda::launch_quantize_bf16_q8(activation, q8, columns, nullptr);
  if (error != cudaSuccess) return error;
  return qw38::cuda::launch_quant_mmv_prequant(
      qw38::cuda::QuantKind::kQ4K, weights, rows, columns, q8, output, nullptr);
}

cudaError_t launch_integer_complete(const std::uint8_t* weights,
                                    std::size_t rows, std::size_t columns,
                                    const __nv_bfloat16* activation,
                                    qw38::cuda::Q8Block* q8, float* output) {
  cudaError_t error =
      qw38::cuda::launch_quantize_bf16_q8(activation, q8, columns, nullptr);
  if (error != cudaSuccess) return error;
  return launch_integer_mmv(weights, rows, columns, q8, output, nullptr);
}

cudaError_t run_case(const char* id, const std::uint8_t* host_weights,
                     std::size_t weight_bytes, const std::uint8_t* device_weights,
                     std::size_t rows, std::size_t columns, int weight,
                     bool probe, const char* activation_source,
                     const std::vector<__nv_bfloat16>& activation, FILE* raw,
                     CaseResult* result) {
  result->id = id;  // copy into owned string; callers reuse stack buffers
  result->rows = rows;
  result->columns = columns;
  result->warps = qw38::cuda::selected_mmv_warps(rows);
  result->weight = weight;
  result->probe = probe;
  result->activation_source = activation_source;
  result->weights_bytes = weight_bytes;
  result->activation_bytes = activation.size() * sizeof(activation[0]);
  if (!checksum_bytes(host_weights, weight_bytes, &result->weights_sha)) {
    std::fprintf(stderr, "weight checksum failed for %s\n", id);
    return cudaErrorUnknown;
  }
  if (!checksum_bytes(activation.data(), result->activation_bytes,
                      &result->activation_sha)) {
    std::fprintf(stderr, "activation checksum failed for %s\n", id);
    return cudaErrorUnknown;
  }

  const std::size_t staged_count = columns / 32;
  std::uint8_t* owned_weights = nullptr;
  const std::uint8_t* gpu_weights = device_weights;
  cudaError_t error = cudaSuccess;
  if (gpu_weights == nullptr) {
    error = cudaMalloc(&owned_weights, weight_bytes);
    if (error != cudaSuccess) return error;
    error = cudaMemcpy(owned_weights, host_weights, weight_bytes,
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      cudaFree(owned_weights);
      return error;
    }
    gpu_weights = owned_weights;
  }

  __nv_bfloat16* device_activation = nullptr;
  qw38::cuda::Q8Block* device_q8[2] = {nullptr, nullptr};
  qw38::cuda::Q8Block* device_q8_shared = nullptr;
  float* device_out[2] = {nullptr, nullptr};
  error = cudaMalloc(&device_activation, result->activation_bytes);
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_q8_shared,
                       staged_count * sizeof(qw38::cuda::Q8Block));
  }
  for (int cand = 0; cand < 2 && error == cudaSuccess; ++cand) {
    error = cudaMalloc(&device_q8[cand],
                       staged_count * sizeof(qw38::cuda::Q8Block));
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_out[cand], rows * sizeof(float));
    }
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_activation, activation.data(),
                       result->activation_bytes, cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(device_out[1]);
    cudaFree(device_out[0]);
    cudaFree(device_q8[1]);
    cudaFree(device_q8[0]);
    cudaFree(device_q8_shared);
    cudaFree(device_activation);
    cudaFree(owned_weights);
    return error;
  }

  result->packed.occupancy =
      qw38::cuda::mmv_packed_occupancy(qw38::cuda::QuantKind::kQ4K, result->warps);
  result->integer.occupancy = integer_occupancy(result->warps);
  result->packed.complete.samples.assign(kMeasured, 0.0F);
  result->packed.complete.warmup.assign(kWarmups, 0.0F);
  result->packed.prequant.samples.assign(kMeasured, 0.0F);
  result->packed.prequant.warmup.assign(kWarmups, 0.0F);
  result->integer.complete.samples.assign(kMeasured, 0.0F);
  result->integer.complete.warmup.assign(kWarmups, 0.0F);
  result->integer.prequant.samples.assign(kMeasured, 0.0F);
  result->integer.prequant.warmup.assign(kWarmups, 0.0F);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaFree(device_out[1]);
    cudaFree(device_out[0]);
    cudaFree(device_q8[1]);
    cudaFree(device_q8[0]);
    cudaFree(device_q8_shared);
    cudaFree(device_activation);
    cudaFree(owned_weights);
    return error;
  }

  auto time_complete = [&](int round, bool warmup_round) -> cudaError_t {
    cudaError_t inner = cudaSuccess;
    for (int cand = 0; cand < 2; ++cand) {
      inner = cudaMemset(device_out[cand], 0, rows * sizeof(float));
      if (inner != cudaSuccess) return inner;
      inner = cudaEventRecord(start);
      if (inner == cudaSuccess) {
        inner = cand == 0
                    ? launch_packed_complete(gpu_weights, rows, columns,
                                             device_activation, device_q8[0],
                                             device_out[0])
                    : launch_integer_complete(gpu_weights, rows, columns,
                                              device_activation, device_q8[1],
                                              device_out[1]);
      }
      if (inner == cudaSuccess) inner = cudaEventRecord(stop);
      float ms = 0.0F;
      if (inner == cudaSuccess) inner = record_ms(start, stop, &ms);
      if (inner != cudaSuccess) return inner;
      CandidateResult& slot = cand == 0 ? result->packed : result->integer;
      TimedView& view = slot.complete;
      if (warmup_round) {
        view.warmup[static_cast<std::size_t>(round)] = ms;
        std::fprintf(raw, "case=%s view=complete candidate=%s warmup=%d ms=%.9g\n",
                     id, cand == 0 ? "packed" : "integer", round,
                     static_cast<double>(ms));
      } else {
        view.samples[static_cast<std::size_t>(round)] = ms;
        std::fprintf(raw, "case=%s view=complete candidate=%s sample=%d ms=%.9g\n",
                     id, cand == 0 ? "packed" : "integer", round,
                     static_cast<double>(ms));
      }
    }
    return cudaSuccess;
  };

  auto time_prequant = [&](int round, bool warmup_round) -> cudaError_t {
    cudaError_t inner =
        qw38::cuda::launch_quantize_bf16_q8(device_activation, device_q8_shared,
                                            columns, nullptr);
    if (inner != cudaSuccess) return inner;
    inner = cudaDeviceSynchronize();
    if (inner != cudaSuccess) return inner;
    for (int cand = 0; cand < 2; ++cand) {
      inner = cudaMemset(device_out[cand], 0, rows * sizeof(float));
      if (inner != cudaSuccess) return inner;
      inner = cudaEventRecord(start);
      if (inner == cudaSuccess) {
        inner = cand == 0
                    ? qw38::cuda::launch_quant_mmv_prequant(
                          qw38::cuda::QuantKind::kQ4K, gpu_weights, rows,
                          columns, device_q8_shared, device_out[0], nullptr)
                    : launch_integer_mmv(gpu_weights, rows, columns,
                                         device_q8_shared, device_out[1],
                                         nullptr);
      }
      if (inner == cudaSuccess) inner = cudaEventRecord(stop);
      float ms = 0.0F;
      if (inner == cudaSuccess) inner = record_ms(start, stop, &ms);
      if (inner != cudaSuccess) return inner;
      CandidateResult& slot = cand == 0 ? result->packed : result->integer;
      TimedView& view = slot.prequant;
      if (warmup_round) {
        view.warmup[static_cast<std::size_t>(round)] = ms;
        std::fprintf(raw, "case=%s view=prequant candidate=%s warmup=%d ms=%.9g\n",
                     id, cand == 0 ? "packed" : "integer", round,
                     static_cast<double>(ms));
      } else {
        view.samples[static_cast<std::size_t>(round)] = ms;
        std::fprintf(raw, "case=%s view=prequant candidate=%s sample=%d ms=%.9g\n",
                     id, cand == 0 ? "packed" : "integer", round,
                     static_cast<double>(ms));
      }
    }
    return cudaSuccess;
  };

  for (int warmup = 0; warmup < kWarmups && error == cudaSuccess; ++warmup) {
    error = time_complete(warmup, true);
  }
  for (int sample = 0; sample < kMeasured && error == cudaSuccess; ++sample) {
    error = time_complete(sample, false);
  }
  for (int warmup = 0; warmup < kWarmups && error == cudaSuccess; ++warmup) {
    error = time_prequant(warmup, true);
  }
  for (int sample = 0; sample < kMeasured && error == cudaSuccess; ++sample) {
    error = time_prequant(sample, false);
  }

  if (error == cudaSuccess) {
    error = cudaMemset(device_out[0], 0, rows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(device_out[1], 0, rows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = launch_packed_complete(gpu_weights, rows, columns, device_activation,
                                   device_q8[0], device_out[0]);
  }
  if (error == cudaSuccess) {
    error = launch_integer_complete(gpu_weights, rows, columns,
                                    device_activation, device_q8[1],
                                    device_out[1]);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();

  std::vector<float> host_out[2];
  std::vector<qw38::cuda::Q8Block> host_q8[2];
  if (error == cudaSuccess) {
    for (int cand = 0; cand < 2; ++cand) {
      host_out[cand].assign(rows, 0.0F);
      host_q8[cand].assign(staged_count, qw38::cuda::Q8Block{});
      error = cudaMemcpy(host_out[cand].data(), device_out[cand],
                         rows * sizeof(float), cudaMemcpyDeviceToHost);
      if (error == cudaSuccess) {
        error = cudaMemcpy(host_q8[cand].data(), device_q8[cand],
                           staged_count * sizeof(qw38::cuda::Q8Block),
                           cudaMemcpyDeviceToHost);
      }
      if (error != cudaSuccess) break;
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device_out[1]);
  cudaFree(device_out[0]);
  cudaFree(device_q8[1]);
  cudaFree(device_q8[0]);
  cudaFree(device_q8_shared);
  cudaFree(device_activation);
  cudaFree(owned_weights);
  if (error != cudaSuccess) return error;

  finish_mean(&result->packed.complete);
  finish_mean(&result->packed.prequant);
  finish_mean(&result->integer.complete);
  finish_mean(&result->integer.prequant);
  result->packed.launch_ok = true;
  result->integer.launch_ok = true;
  result->staging_equal =
      std::memcmp(host_q8[0].data(), host_q8[1].data(),
                  staged_count * sizeof(qw38::cuda::Q8Block)) == 0;
  result->packed.staging_equal = result->staging_equal;
  result->integer.staging_equal = result->staging_equal;
  std::fprintf(raw, "case=%s staging_equal=%s\n", id,
               json_bool(result->staging_equal));

  std::vector<std::uint8_t> host_weight_copy(
      host_weights, host_weights + weight_bytes);
  std::vector<float> expected_cud001;
  std::vector<float> expected_integer;
  if (!host_cud001(host_weight_copy, rows, columns, host_q8[0],
                   &expected_cud001)) {
    std::fprintf(stderr, "host CUD-001 reference failed for %s\n", id);
    return cudaErrorUnknown;
  }
  host_integer(host_weight_copy, rows, columns, host_q8[1], &expected_integer);
  result->packed.vs_host =
      compare_cud001(host_out[0], expected_cud001, columns);
  result->integer.vs_host =
      compare_cud001(host_out[1], expected_cud001, columns);
  result->integer.vs_packed = compare_simple(host_out[1], host_out[0]);
  result->integer.vs_host_integer_byte_equal =
      std::memcmp(host_out[1].data(), expected_integer.data(),
                  rows * sizeof(float)) == 0;
  auto eligible = [](const Envelope& env) {
    return env.nonfinite == 0 && env.max_abs <= kCud001Abs &&
           env.rms <= kCud001Rms;
  };
  result->packed.cud001_eligible = eligible(result->packed.vs_host);
  result->integer.cud001_eligible = eligible(result->integer.vs_host);
  std::fprintf(raw,
               "case=%s packed vs_host max_abs=%.9g rms=%.9g nonfinite=%zu "
               "eligible=%s\n",
               id, static_cast<double>(result->packed.vs_host.max_abs),
               static_cast<double>(result->packed.vs_host.rms),
               result->packed.vs_host.nonfinite,
               json_bool(result->packed.cud001_eligible));
  std::fprintf(raw,
               "case=%s integer vs_host max_abs=%.9g rms=%.9g nonfinite=%zu "
               "byte_equal_host_integer=%s eligible=%s\n",
               id, static_cast<double>(result->integer.vs_host.max_abs),
               static_cast<double>(result->integer.vs_host.rms),
               result->integer.vs_host.nonfinite,
               json_bool(result->integer.vs_host_integer_byte_equal),
               json_bool(result->integer.cud001_eligible));
  return cudaSuccess;
}

void print_envelope(const Envelope& env, bool include_rel) {
  std::printf("\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu,"
              "\"cosine\":%.9g,\"first_fail\":%d",
              static_cast<double>(env.max_abs), static_cast<double>(env.rms),
              env.nonfinite, static_cast<double>(env.cosine), env.first_fail);
  if (include_rel) {
    std::printf(",\"relative\":{\"max\":%.9g,\"mean\":%.9g,\"gt_1e-3\":%zu,"
                "\"gt_1e-2\":%zu,\"gt_1e-1\":%zu},\"mmq_association_fail\":%zu",
                static_cast<double>(env.max_rel),
                static_cast<double>(env.mean_rel), env.rel_gt_1em3,
                env.rel_gt_1em2, env.rel_gt_1em1, env.mmq_fail);
  }
}

void print_timed(const TimedView& view) {
  std::printf("{\"mean_ms\":%.9g,", static_cast<double>(view.mean_ms));
  print_array(stdout, "samples", view.samples);
  std::printf(",");
  print_array(stdout, "warmup_ms", view.warmup);
  std::printf("}");
}

void print_candidate(const char* name, const CandidateResult& slot,
                     bool integer) {
  std::printf("\"%s\":{\"id\":\"%s\",\"occupancy\":%d,\"launch_ok\":%s,"
              "\"staging_equal\":%s,\"cud001_eligible\":%s,\"complete\":",
              name, name, slot.occupancy, json_bool(slot.launch_ok),
              json_bool(slot.staging_equal), json_bool(slot.cud001_eligible));
  print_timed(slot.complete);
  std::printf(",\"prequant\":");
  print_timed(slot.prequant);
  std::printf(",\"vs_host_cud001\":{");
  print_envelope(slot.vs_host, true);
  std::printf("}");
  if (integer) {
    std::printf(",\"vs_host_integer_byte_equal\":%s,\"vs_packed\":{",
                json_bool(slot.vs_host_integer_byte_equal));
    print_envelope(slot.vs_packed, false);
    std::printf("}");
  }
  std::printf("}");
}

void print_case(const CaseResult& result) {
  std::printf("\"%s\":{\"id\":\"%s\",\"rows\":%zu,\"columns\":%zu,\"warps\":%u,"
              "\"weight\":%d,\"probe\":%s,\"activation_source\":\"%s\","
              "\"checksums\":{\"weights\":\"%s\",\"activation\":\"%s\","
              "\"weights_bytes\":%zu,\"activation_bytes\":%zu},"
              "\"staging_equal\":%s,\"candidates\":{",
              result.id.c_str(), result.id.c_str(), result.rows, result.columns,
              result.warps,
              result.weight, json_bool(result.probe), result.activation_source,
              result.weights_sha.c_str(), result.activation_sha.c_str(),
              result.weights_bytes, result.activation_bytes,
              json_bool(result.staging_equal));
  print_candidate("packed", result.packed, false);
  std::printf(",");
  print_candidate("integer", result.integer, true);
  std::printf("}}");
}

const qw38::cuda::DeviceTensor& projection_tensor(
    const qw38::cuda::DeviceCommonLayer& layer, Projection projection) {
  if (projection == Projection::kGate) return layer.ffn_gate;
  if (projection == Projection::kUp) return layer.ffn_up;
  return layer.ffn_down;
}

const qw38::internal::TensorView& projection_host(
    const qw38::internal::CommonLayerWeights& layer, Projection projection) {
  if (projection == Projection::kGate) return layer.ffn_gate;
  if (projection == Projection::kUp) return layer.ffn_up;
  return layer.ffn_down;
}

const char* projection_name(Projection projection) {
  if (projection == Projection::kGate) return "gate";
  if (projection == Projection::kUp) return "up";
  return "down";
}

int projection_weight(Projection projection) {
  return projection == Projection::kDown ? kRealDownWeight : kRealGateWeight;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL.gguf RAW_PATH\n", argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  const char* raw_path = argv[2];
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("cudaDeviceSynchronize", error);
  cudaDeviceProp props{};
  error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) return fail_cuda("cudaGetDeviceProperties", error);

  const int occ4 = integer_occupancy(4);
  const int occ8 = integer_occupancy(8);
  const int occ16 = integer_occupancy(16);
  const int packed4 =
      qw38::cuda::mmv_packed_occupancy(qw38::cuda::QuantKind::kQ4K, 4);
  const int packed8 =
      qw38::cuda::mmv_packed_occupancy(qw38::cuda::QuantKind::kQ4K, 8);
  const int packed16 =
      qw38::cuda::mmv_packed_occupancy(qw38::cuda::QuantKind::kQ4K, 16);
  if (occ4 < 1 || occ8 < 1 || occ16 < 1) {
    std::fprintf(stderr, "integer-dot occupancy is below 1\n");
    return 1;
  }

  CaseResult synthetic[kSyntheticCount];
  for (int index = 0; index < kSyntheticCount; ++index) {
    const Shape& shape = kSynthetic[index];
    std::vector<std::uint8_t> weights;
    fill_weights(shape.rows, shape.columns, &weights);
    std::vector<__nv_bfloat16> activation;
    fill_activation(shape.columns, &activation);
    error = run_case(shape.id, weights.data(), weights.size(), nullptr,
                     shape.rows, shape.columns, shape.weight, shape.probe,
                     "unit_normal", activation, raw, &synthetic[index]);
    if (error != cudaSuccess) return fail_cuda("synthetic case", error);
  }

  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights host_weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &host_weights);
  }
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) {
    status = model.upload(host_weights, mapping.data(), mapping.size());
  }
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(kCapacity);
  qw38::cuda::SchedulerWorkspace workspace;
  if (status.is_ok()) status = workspace.create(kCapacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);

  std::vector<std::size_t> tokens(2048);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), tokens.size(), &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs,
      qw38::cuda::GdnScanPath::kFusedTokenLoop, nullptr);
  if (!status.is_ok()) return fail_status(status);

  float* device_hidden = nullptr;
  __nv_bfloat16* device_normed = nullptr;
  float* device_gate = nullptr;
  float* device_up = nullptr;
  qw38::cuda::Q8Block* device_q8_norm = nullptr;
  error = cudaMalloc(&device_hidden, hidden.size() * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_normed,
                       qw38::internal::kResidualWidth * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_gate, qw38::internal::kFfnWidth * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_up, qw38::internal::kFfnWidth * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&device_q8_norm, qw38::cuda::q8_workspace_bytes(
                                            qw38::internal::kResidualWidth));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device_hidden, hidden.data(), hidden.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("real capture malloc", error);

  CaseResult real_cases[kRealLayerCount * 3];
  int real_index = 0;
  char case_id[32];
  for (int layer_i = 0; layer_i < kRealLayerCount; ++layer_i) {
    const int layer = kRealLayers[layer_i];
    const qw38::cuda::DeviceCommonLayer& device_layer =
        model.common_layer(static_cast<std::size_t>(layer));
    const qw38::internal::CommonLayerWeights& host_layer =
        host_weights.layers[static_cast<std::size_t>(layer)].common;
    error = qw38::cuda::launch_rms_norm_rows_fp32_to_bf16(
        device_hidden, device_layer.ffn_norm, qw38::internal::kResidualWidth, 1,
        device_normed, nullptr);
    if (error != cudaSuccess) return fail_cuda("rms_norm", error);
    std::vector<__nv_bfloat16> gate_up_act(qw38::internal::kResidualWidth);
    error = cudaMemcpy(gate_up_act.data(), device_normed,
                       gate_up_act.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail_cuda("copy normed", error);

    for (Projection projection : {Projection::kGate, Projection::kUp}) {
      const qw38::cuda::DeviceTensor& tensor =
          projection_tensor(device_layer, projection);
      const qw38::internal::TensorView& host = projection_host(host_layer, projection);
      std::snprintf(case_id, sizeof(case_id), "real_L%d_%s", layer,
                    projection_name(projection));
      error = run_case(case_id, host.data, host.storage_bytes, tensor.data,
                       tensor.rows, tensor.columns, projection_weight(projection),
                       false, "post_prefix_hidden", gate_up_act, raw,
                       &real_cases[real_index]);
      if (error != cudaSuccess) return fail_cuda(case_id, error);
      ++real_index;
    }

    error = launch_packed_complete(device_layer.ffn_gate.data,
                                   device_layer.ffn_gate.rows,
                                   device_layer.ffn_gate.columns, device_normed,
                                   device_q8_norm, device_gate);
    if (error == cudaSuccess) {
      error = launch_packed_complete(device_layer.ffn_up.data,
                                     device_layer.ffn_up.rows,
                                     device_layer.ffn_up.columns, device_normed,
                                     device_q8_norm, device_up);
    }
    __nv_bfloat16* device_down_act = nullptr;
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_down_act,
                         qw38::internal::kFfnWidth * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_swiglu_bf16(device_gate, device_up,
                                             qw38::internal::kFfnWidth,
                                             device_down_act, nullptr);
    }
    if (error != cudaSuccess) return fail_cuda("down activation", error);
    std::vector<__nv_bfloat16> down_act(qw38::internal::kFfnWidth);
    error = cudaMemcpy(down_act.data(), device_down_act,
                       down_act.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
    cudaFree(device_down_act);
    if (error != cudaSuccess) return fail_cuda("copy down act", error);
    const qw38::cuda::DeviceTensor& down = device_layer.ffn_down;
    const qw38::internal::TensorView& host_down = host_layer.ffn_down;
    std::snprintf(case_id, sizeof(case_id), "real_L%d_down", layer);
    error = run_case(case_id, host_down.data, host_down.storage_bytes, down.data,
                     down.rows, down.columns, kRealDownWeight, false,
                     "post_prefix_hidden", down_act, raw,
                     &real_cases[real_index]);
    if (error != cudaSuccess) return fail_cuda(case_id, error);
    ++real_index;
  }

  cudaFree(device_q8_norm);
  cudaFree(device_up);
  cudaFree(device_gate);
  cudaFree(device_normed);
  cudaFree(device_hidden);

  double syn_packed = 0.0;
  double syn_integer = 0.0;
  int syn_weight = 0;
  bool numeric_ok = true;
  bool staging_ok = true;
  for (int index = 0; index < kSyntheticCount; ++index) {
    const CaseResult& slot = synthetic[index];
    numeric_ok = numeric_ok && slot.packed.cud001_eligible &&
                 slot.integer.cud001_eligible;
    staging_ok = staging_ok && slot.staging_equal;
    if (slot.weight == 0) continue;
    syn_weight += slot.weight;
    syn_packed += static_cast<double>(slot.weight) *
                  static_cast<double>(slot.packed.complete.mean_ms);
    syn_integer += static_cast<double>(slot.weight) *
                   static_cast<double>(slot.integer.complete.mean_ms);
  }
  const float t_syn_packed =
      static_cast<float>(syn_packed / syn_weight);
  const float t_syn_integer =
      static_cast<float>(syn_integer / syn_weight);

  double real_packed_sum = 0.0;
  double real_integer_sum = 0.0;
  for (int layer_i = 0; layer_i < kRealLayerCount; ++layer_i) {
    double packed_layer = 0.0;
    double integer_layer = 0.0;
    int layer_weight = 0;
    for (int proj = 0; proj < 3; ++proj) {
      const CaseResult& slot = real_cases[layer_i * 3 + proj];
      numeric_ok = numeric_ok && slot.packed.cud001_eligible &&
                   slot.integer.cud001_eligible;
      staging_ok = staging_ok && slot.staging_equal;
      layer_weight += slot.weight;
      packed_layer += static_cast<double>(slot.weight) *
                      static_cast<double>(slot.packed.complete.mean_ms);
      integer_layer += static_cast<double>(slot.weight) *
                       static_cast<double>(slot.integer.complete.mean_ms);
    }
    real_packed_sum += packed_layer / layer_weight;
    real_integer_sum += integer_layer / layer_weight;
  }
  const float t_real_packed =
      static_cast<float>(real_packed_sum / kRealLayerCount);
  const float t_real_integer =
      static_cast<float>(real_integer_sum / kRealLayerCount);

  const char* admissibility = "performance_reject";
  bool promote = false;
  if (!numeric_ok) {
    admissibility = "numeric_reject";
  } else if (t_syn_integer < t_syn_packed && t_real_integer < t_real_packed) {
    admissibility = "eligible_and_faster";
    promote = true;
  }

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf("\"schema_version\":1,\"task\":\"OPT-042\",\"status\":\"measured\",");
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\","
              "\"compute_capability\":\"%d.%d\",",
              utc_text, props.name, props.major, props.minor);
  std::printf("\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\",", kLlamaRev,
              kGgufSha);
  std::printf("\"claims_performance_improvement\":false,"
              "\"publishes_successor_oracle\":false,"
              "\"promote_to_production_ab\":%s,\"admissibility\":\"%s\","
              "\"selected_mmv_load_path\":\"packed\",\"reverted\":false,",
              json_bool(promote), admissibility);
  std::printf("\"accepted_keep_denominators\":{"
              "\"source_keep_fixture\":\"fixtures/opt041_fattn_warp_qk.json\","
              "\"p\":{\"quartz_mean_tok_s\":2076.98315},"
              "\"d128\":{\"quartz_mean_tok_s\":26.1599541,"
              "\"token_latency_p95_ms\":38.3730087,"
              "\"run_mean_token_latency_p95_ms\":38.2578201},"
              "\"d2048\":{\"quartz_mean_tok_s\":25.2924843,"
              "\"token_latency_p95_ms\":39.6526222,"
              "\"run_mean_token_latency_p95_ms\":39.5695038}},");
  std::printf("\"synthetic\":{");
  for (int index = 0; index < kSyntheticCount; ++index) {
    if (index != 0) std::printf(",");
    print_case(synthetic[index]);
  }
  std::printf("},\"real\":{");
  for (int index = 0; index < kRealLayerCount * 3; ++index) {
    if (index != 0) std::printf(",");
    print_case(real_cases[index]);
  }
  std::printf("},\"weighted_complete_ms\":{"
              "\"synthetic\":{\"packed\":%.9g,\"integer\":%.9g},"
              "\"real\":{\"packed\":%.9g,\"integer\":%.9g}},",
              static_cast<double>(t_syn_packed),
              static_cast<double>(t_syn_integer),
              static_cast<double>(t_real_packed),
              static_cast<double>(t_real_integer));
  std::printf("\"staging_equal\":%s,", json_bool(staging_ok));
  std::printf("\"occupancy\":{\"integer\":{\"4\":%d,\"8\":%d,\"16\":%d},"
              "\"packed\":{\"4\":%d,\"8\":%d,\"16\":%d}},",
              occ4, occ8, occ16, packed4, packed8, packed16);
  std::printf("\"q8_0_mixer_sibling\":\"not_run\",\"q6k_logits_sibling\":\"not_run\",");
  std::printf("\"owns_opt016_parity_gate\":false,\"substitutes_for_opt016\":false,");
  std::printf("\"nsight_systems\":\"not_used\",\"nsight_compute\":\"not_used\",");
  std::printf("\"proof_limit\":\"%s\",\"report_path\":\"%s\"}\n", kProof,
              kReportPath);
  std::printf("status=passed admissibility=%s syn_packed=%.9g syn_integer=%.9g "
              "real_packed=%.9g real_integer=%.9g\n",
              admissibility, static_cast<double>(t_syn_packed),
              static_cast<double>(t_syn_integer),
              static_cast<double>(t_real_packed),
              static_cast<double>(t_real_integer));
  std::fclose(raw);
  return 0;
}
