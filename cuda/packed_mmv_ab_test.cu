#include "quant_mmv.h"

#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_PACKED_MMV_AB_RESULT=";
constexpr const char* kCandidates[] = {"elementwise", "packed"};
constexpr int kCandidateCount = 2;
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;

struct Shape {
  const char* id;
  qw38::cuda::QuantKind kind;
  std::size_t rows;
  std::size_t columns;
  int weight;
  bool probe;
};

constexpr Shape kShapes[] = {
    {"q4_k_17x256", qw38::cuda::QuantKind::kQ4K, 17, 256, 0, true},
    {"q4_k_257x512", qw38::cuda::QuantKind::kQ4K, 257, 512, 0, true},
    {"q6_k_17x256", qw38::cuda::QuantKind::kQ6K, 17, 256, 0, true},
    {"q6_k_257x512", qw38::cuda::QuantKind::kQ6K, 257, 512, 0, true},
    {"q4k_gate_up", qw38::cuda::QuantKind::kQ4K, 17408, 5120, 128, false},
    {"q4k_down", qw38::cuda::QuantKind::kQ4K, 5120, 17408, 64, false},
    {"q6k_attn_out", qw38::cuda::QuantKind::kQ6K, 5120, 6144, 16, false},
    {"q6k_logits", qw38::cuda::QuantKind::kQ6K, 248320, 5120, 1, false},
};
constexpr int kShapeCount = static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
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

void fill_weights(qw38::cuda::QuantKind kind, std::size_t rows,
                  std::size_t columns, std::vector<std::uint8_t>* weights) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? 144
      : kind == qw38::cuda::QuantKind::kQ6K ? 210
                                           : 34;
  const std::size_t block_values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  weights->assign(rows * (columns / block_values) * block_bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size();
       offset += block_bytes) {
    if (kind == qw38::cuda::QuantKind::kQ4K) {
      write_u16(weights->data() + offset, 0x2400U);
      write_u16(weights->data() + offset + 2, 0x1C00U);
    } else {
      write_u16(weights->data() + offset + 208, 0x1C00U);
    }
  }
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
  const float u2 =
      static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

void fill_activation(std::size_t columns, std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    (*activation)[column] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(column), 0xA11CE5u));
  }
}

bool host_reference(qw38::cuda::QuantKind kind,
                    const std::vector<std::uint8_t>& weights, std::size_t rows,
                    std::size_t columns,
                    const std::vector<qw38::cuda::Q8Block>& activation,
                    std::vector<float>* output) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K ? 144 : 210;
  output->assign(rows, 0.0F);
  std::vector<float> decoded(256);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::size_t block = 0; block < columns / 256; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / 256) + block) * block_bytes;
      const qw38::Status status =
          kind == qw38::cuda::QuantKind::kQ4K
              ? qw38::internal::decode_q4_k(packed, block_bytes, decoded.data(),
                                            decoded.size())
              : qw38::internal::decode_q6_k(packed, block_bytes, decoded.data(),
                                            decoded.size());
      if (!status.is_ok()) return false;
      for (std::size_t within = 0; within < 256; ++within) {
        const std::size_t column = block * 256 + within;
        const auto& q8 = activation[column / 32];
        sum += decoded[within] *
               (q8.scale * static_cast<float>(q8.values[column % 32]));
      }
    }
    (*output)[row] = sum;
  }
  return true;
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
};

Envelope compare(const std::vector<float>& actual,
                 const std::vector<float>& expected) {
  Envelope metrics;
  double squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) ++metrics.nonfinite;
    const float error = std::fabs(actual[index] - expected[index]);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
  }
  metrics.rms = actual.empty()
                    ? 0.0F
                    : static_cast<float>(std::sqrt(
                          squared / static_cast<double>(actual.size())));
  return metrics;
}

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt034-packed-mmv/mmv-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaDeviceProp props{};
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) return fail_cuda("cudaGetDeviceProperties", error);

  struct Candidate {
    std::vector<float> samples;
    std::vector<float> warmup;
    float mean_ms = 0.0F;
    int occupancy = 0;
    bool launch_ok = false;
    bool byte_equal = false;
    bool q8_equal = false;
    bool eligible = false;
    Envelope vs_host{};
  };

  struct ShapeResult {
    Candidate candidates[2];
  };
  ShapeResult results[kShapeCount];

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("cudaEventCreate", error);

  for (int shape_index = 0; shape_index < kShapeCount; ++shape_index) {
    const Shape& shape = kShapes[shape_index];
    std::vector<std::uint8_t> weights;
    fill_weights(shape.kind, shape.rows, shape.columns, &weights);
    std::vector<__nv_bfloat16> activation;
    fill_activation(shape.columns, &activation);
    const std::size_t staged_count = shape.columns / 32;
    const unsigned int warps = qw38::cuda::selected_mmv_warps(shape.rows);

    std::uint8_t* device_weights = nullptr;
    __nv_bfloat16* device_activation = nullptr;
    qw38::cuda::Q8Block* device_q8[2] = {nullptr, nullptr};
    float* device_out[2] = {nullptr, nullptr};
    error = cudaMalloc(&device_weights, weights.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_activation,
                         activation.size() * sizeof(activation[0]));
    }
    for (int cand = 0; cand < kCandidateCount && error == cudaSuccess; ++cand) {
      error = cudaMalloc(&device_q8[cand],
                         staged_count * sizeof(qw38::cuda::Q8Block));
      if (error == cudaSuccess) {
        error = cudaMalloc(&device_out[cand], shape.rows * sizeof(float));
      }
    }
    if (error != cudaSuccess) return fail_cuda("cudaMalloc", error);
    error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_activation, activation.data(),
                         activation.size() * sizeof(activation[0]),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) return fail_cuda("cudaMemcpy H2D", error);

    for (int cand = 0; cand < kCandidateCount; ++cand) {
      results[shape_index].candidates[cand].occupancy =
          cand == 1 ? qw38::cuda::mmv_packed_occupancy(shape.kind, warps) : 1;
      results[shape_index].candidates[cand].samples.assign(kMeasured, 0.0F);
      results[shape_index].candidates[cand].warmup.assign(kWarmups, 0.0F);
    }
    if (results[shape_index].candidates[0].occupancy < 1) {
      results[shape_index].candidates[0].occupancy = 1;
    }

    auto launch = [&](int cand) -> cudaError_t {
      return qw38::cuda::launch_quant_mmv_path(
          shape.kind, device_weights, shape.rows, shape.columns,
          device_activation, device_q8[cand], device_out[cand],
          kCandidates[cand], nullptr);
    };

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
      for (int cand = 0; cand < kCandidateCount; ++cand) {
        error = cudaEventRecord(start);
        if (error == cudaSuccess) error = launch(cand);
        if (error == cudaSuccess) error = cudaEventRecord(stop);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        if (error != cudaSuccess) return fail_cuda("warmup launch", error);
        float ms = 0.0F;
        error = cudaEventElapsedTime(&ms, start, stop);
        if (error != cudaSuccess) return fail_cuda("warmup elapsed", error);
        results[shape_index].candidates[cand].warmup[static_cast<std::size_t>(
            warmup)] = ms;
        std::fprintf(raw, "shape=%s candidate=%s warmup=%d ms=%.9g\n",
                     shape.id, kCandidates[cand], warmup,
                     static_cast<double>(ms));
      }
    }
    for (int sample = 0; sample < kMeasured; ++sample) {
      for (int cand = 0; cand < kCandidateCount; ++cand) {
        error = cudaEventRecord(start);
        if (error == cudaSuccess) error = launch(cand);
        if (error == cudaSuccess) error = cudaEventRecord(stop);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        if (error != cudaSuccess) return fail_cuda("measured launch", error);
        float ms = 0.0F;
        error = cudaEventElapsedTime(&ms, start, stop);
        if (error != cudaSuccess) return fail_cuda("measured elapsed", error);
        results[shape_index].candidates[cand].samples[static_cast<std::size_t>(
            sample)] = ms;
        std::fprintf(raw, "shape=%s candidate=%s sample=%d ms=%.9g\n",
                     shape.id, kCandidates[cand], sample,
                     static_cast<double>(ms));
      }
    }

    std::vector<float> host_out[2];
    std::vector<qw38::cuda::Q8Block> host_q8[2];
    for (int cand = 0; cand < kCandidateCount; ++cand) {
      host_out[cand].assign(shape.rows, 0.0F);
      host_q8[cand].assign(staged_count, qw38::cuda::Q8Block{});
      error = cudaMemcpy(host_out[cand].data(), device_out[cand],
                         shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
      if (error == cudaSuccess) {
        error = cudaMemcpy(host_q8[cand].data(), device_q8[cand],
                           staged_count * sizeof(qw38::cuda::Q8Block),
                           cudaMemcpyDeviceToHost);
      }
      if (error != cudaSuccess) return fail_cuda("cudaMemcpy D2H", error);
      results[shape_index].candidates[cand].launch_ok = true;
      double acc = 0.0;
      for (float sample : results[shape_index].candidates[cand].samples) {
        acc += static_cast<double>(sample);
      }
      results[shape_index].candidates[cand].mean_ms =
          static_cast<float>(acc / kMeasured);
    }
    const bool output_eq =
        std::memcmp(host_out[0].data(), host_out[1].data(),
                    shape.rows * sizeof(float)) == 0;
    const bool q8_eq =
        std::memcmp(host_q8[0].data(), host_q8[1].data(),
                    staged_count * sizeof(qw38::cuda::Q8Block)) == 0;
    results[shape_index].candidates[0].byte_equal = true;
    results[shape_index].candidates[0].q8_equal = true;
    results[shape_index].candidates[1].byte_equal = output_eq;
    results[shape_index].candidates[1].q8_equal = q8_eq;
    std::fprintf(raw, "shape=%s byte_equal=%s q8_equal=%s\n", shape.id,
                 json_bool(output_eq), json_bool(q8_eq));

    if (shape.probe) {
      std::vector<float> expected;
      if (!host_reference(shape.kind, weights, shape.rows, shape.columns,
                          host_q8[0], &expected)) {
        std::fprintf(stderr, "host reference failed for %s\n", shape.id);
        return 1;
      }
      for (int cand = 0; cand < kCandidateCount; ++cand) {
        results[shape_index].candidates[cand].vs_host =
            compare(host_out[cand], expected);
        std::fprintf(raw,
                     "shape=%s candidate=%s vs_host max_abs=%.9g rms=%.9g "
                     "nonfinite=%zu\n",
                     shape.id, kCandidates[cand],
                     static_cast<double>(
                         results[shape_index].candidates[cand].vs_host.max_abs),
                     static_cast<double>(
                         results[shape_index].candidates[cand].vs_host.rms),
                     results[shape_index].candidates[cand].vs_host.nonfinite);
      }
    }

    for (int cand = 0; cand < kCandidateCount; ++cand) {
      Candidate& slot = results[shape_index].candidates[cand];
      const bool envelope_ok =
          !shape.probe ||
          (slot.vs_host.nonfinite == 0 && slot.vs_host.max_abs <= 3.0e-4F &&
           slot.vs_host.rms <= 2.0e-4F);
      slot.eligible = slot.launch_ok && slot.occupancy >= 1 &&
                      slot.q8_equal && envelope_ok &&
                      (cand == 0 || slot.byte_equal);
    }

    cudaFree(device_out[1]);
    cudaFree(device_out[0]);
    cudaFree(device_q8[1]);
    cudaFree(device_q8[0]);
    cudaFree(device_activation);
    cudaFree(device_weights);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::fclose(raw);

  double weighted[2] = {0.0, 0.0};
  int weight_sum = 0;
  bool packed_eligible = true;
  for (int shape_index = 0; shape_index < kShapeCount; ++shape_index) {
    packed_eligible =
        packed_eligible && results[shape_index].candidates[1].eligible;
    const int weight = kShapes[shape_index].weight;
    if (weight == 0) continue;
    weight_sum += weight;
    for (int cand = 0; cand < kCandidateCount; ++cand) {
      weighted[cand] += static_cast<double>(weight) *
                        static_cast<double>(
                            results[shape_index].candidates[cand].mean_ms);
    }
  }
  const float weighted_mean[2] = {
      static_cast<float>(weighted[0] / weight_sum),
      static_cast<float>(weighted[1] / weight_sum)};
  const bool win = packed_eligible && weighted_mean[1] < weighted_mean[0];
  const char* winner = win ? "packed" : "elementwise";

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf("\"winner\":\"%s\",\"win\":%s,", winner, json_bool(win));
  std::printf("\"weighted_mean_ms\":{\"elementwise\":%.9g,\"packed\":%.9g},",
              static_cast<double>(weighted_mean[0]),
              static_cast<double>(weighted_mean[1]));
  std::printf("\"measurement_utc\":\"%s\",", utc_text);
  std::printf("\"device\":\"%s\",\"compute_capability\":\"%d.%d\",", props.name,
              props.major, props.minor);
  std::printf("\"shapes\":{");
  for (int shape_index = 0; shape_index < kShapeCount; ++shape_index) {
    if (shape_index != 0) std::printf(",");
    const Shape& shape = kShapes[shape_index];
    std::printf("\"%s\":{\"id\":\"%s\",\"kind\":\"%s\",\"rows\":%zu,"
                "\"columns\":%zu,\"warps\":%u,\"weight\":%d,\"probe\":%s,"
                "\"candidates\":{",
                shape.id, shape.id,
                shape.kind == qw38::cuda::QuantKind::kQ4K ? "q4_k" : "q6_k",
                shape.rows, shape.columns,
                qw38::cuda::selected_mmv_warps(shape.rows), shape.weight,
                json_bool(shape.probe));
    for (int cand = 0; cand < kCandidateCount; ++cand) {
      if (cand != 0) std::printf(",");
      const Candidate& slot = results[shape_index].candidates[cand];
      std::printf("\"%s\":{\"id\":\"%s\",\"mean_ms\":%.9g,", kCandidates[cand],
                  kCandidates[cand], static_cast<double>(slot.mean_ms));
      print_array(stdout, "samples", slot.samples);
      std::printf(",");
      print_array(stdout, "warmup_ms", slot.warmup);
      std::printf(",\"occupancy\":%d,\"launch_ok\":%s,\"eligible\":%s,"
                  "\"byte_equal\":%s,\"q8_equal\":%s,\"vs_host\":{"
                  "\"max_abs\":%.9g,\"rms\":%.9g,\"nonfinite\":%zu}}",
                  slot.occupancy, json_bool(slot.launch_ok),
                  json_bool(slot.eligible), json_bool(slot.byte_equal),
                  json_bool(slot.q8_equal),
                  static_cast<double>(slot.vs_host.max_abs),
                  static_cast<double>(slot.vs_host.rms), slot.vs_host.nonfinite);
    }
    std::printf("}}");
  }
  std::printf("}}\n");
  std::printf("status=passed winner=%s weighted_elementwise=%.9g "
              "weighted_packed=%.9g\n",
              winner, static_cast<double>(weighted_mean[0]),
              static_cast<double>(weighted_mean[1]));
  return 0;
}
