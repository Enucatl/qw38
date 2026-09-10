#include "quant.h"
#include "quant_mmv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT046_Q4_DECODE_AB_RESULT=";
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kValuesPerBlock = 256;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kSynGateWeight = 128;
constexpr int kSynDownWeight = 64;

struct Shape {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  int weight;
  bool probe;
  float abs_orig;
  float rms_orig;
  float cos_orig;
  float abs_staged;
  float rms_staged;
  float cos_staged;
};

constexpr Shape kSynthetic[] = {
    {"q4_k_17x256", 17, 256, 0, true, 0.921789042F, 0.379745845F,
     1.15251221e-5F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"q4_k_257x512", 257, 512, 0, true, 0.998716683F, 0.523951687F,
     2.46477834e-5F, 3.0e-4F, 2.0e-4F, 1.0000000002331468e-6F},
    {"q4k_gate_up", 17408, 5120, kSynGateWeight, false, 3.27488041F,
     1.81794829F, 1.97071521e-5F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"q4k_down", 5120, 17408, kSynDownWeight, false, 3.13864238F, 1.84476077F,
     5.47266476e-6F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
};
constexpr int kSyntheticCount =
    static_cast<int>(sizeof(kSynthetic) / sizeof(kSynthetic[0]));

struct Candidate {
  const char* id;
  const char* path;
  unsigned int warps_per_row;
  bool q8_1;
  bool packed;
};

constexpr Candidate kCandidates[] = {
    {"packed", "packed", 0, false, true},
    {"integer_q8_w1", "integer_q8", 1, false, false},
    {"integer_q8_w2", "integer_q8", 2, false, false},
    {"integer_q8_w4", "integer_q8", 4, false, false},
    {"integer_q8_w8", "integer_q8", 8, false, false},
    {"integer_q8_1_w1", "integer_q8_1", 1, true, false},
    {"integer_q8_1_w2", "integer_q8_1", 2, true, false},
    {"integer_q8_1_w4", "integer_q8_1", 4, true, false},
    {"integer_q8_1_w8", "integer_q8_1", 8, true, false},
};
constexpr int kCandidateCount =
    static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

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

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  float one_minus_cosine = 0.0F;
  std::size_t nonfinite = 0;
};

Envelope compare_vec(const std::vector<float>& actual,
                     const std::vector<double>& expected) {
  Envelope metrics;
  double squared = 0.0;
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float got = actual[index];
    const double ref = expected[index];
    if (!std::isfinite(got) || !std::isfinite(ref)) {
      ++metrics.nonfinite;
      continue;
    }
    const float error = std::fabs(got - static_cast<float>(ref));
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
    dot += static_cast<double>(got) * ref;
    na += static_cast<double>(got) * got;
    nb += ref * ref;
  }
  metrics.rms = actual.empty()
                    ? 0.0F
                    : static_cast<float>(std::sqrt(squared / actual.size()));
  if (na == 0.0 && nb == 0.0) {
    metrics.one_minus_cosine = 0.0F;
  } else if (na == 0.0 || nb == 0.0) {
    metrics.one_minus_cosine = 1.0F;
  } else {
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb));
    metrics.one_minus_cosine = static_cast<float>(
        1.0 - std::min(1.0, std::max(-1.0, cosine)));
  }
  return metrics;
}

bool host_fp64(const std::vector<std::uint8_t>& weights, std::size_t rows,
               std::size_t columns, const std::vector<float>& activation,
               std::vector<double>* output) {
  output->assign(rows, 0.0);
  std::vector<float> decoded(kValuesPerBlock);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / kValuesPerBlock; ++block) {
      const std::uint8_t* packed =
          weights.data() + (row * (columns / kValuesPerBlock) + block) *
                                kQ4KBytes;
      const qw38::Status status = qw38::internal::decode_q4_k(
          packed, kQ4KBytes, decoded.data(), decoded.size());
      if (!status.is_ok()) return false;
      for (std::size_t within = 0; within < kValuesPerBlock; ++within) {
        sum += static_cast<double>(decoded[within]) *
               static_cast<double>(
                   activation[block * kValuesPerBlock + within]);
      }
    }
    (*output)[row] = sum;
  }
  return true;
}

void host_q8_stage(const std::vector<__nv_bfloat16>& activation,
                   std::vector<float>* staged) {
  staged->assign(activation.size(), 0.0F);
  for (std::size_t group = 0; group < activation.size() / 32; ++group) {
    float maximum = 0.0F;
    for (int i = 0; i < 32; ++i) {
      maximum = std::max(
          maximum, std::fabs(__bfloat162float(activation[group * 32 + i])));
    }
    const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    for (int i = 0; i < 32; ++i) {
      const float value = __bfloat162float(activation[group * 32 + i]);
      const float quant =
          scale == 0.0F ? 0.0F : std::round(value / scale);
      (*staged)[group * 32 + i] = scale * quant;
    }
  }
}

cudaError_t launch_candidate(const Candidate& cand, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation, void* workspace,
                             float* output) {
  if (cand.packed) {
    return qw38::cuda::launch_quant_mmv_path(
        qw38::cuda::QuantKind::kQ4K, weights, rows, columns, activation,
        static_cast<qw38::cuda::Q8Block*>(workspace), output, "packed",
        nullptr);
  }
  return qw38::cuda::launch_q4k_coop_mmv(weights, rows, columns, activation,
                                         workspace, output, cand.warps_per_row,
                                         cand.q8_1, nullptr);
}

cudaError_t launch_prequant(const Candidate& cand, const std::uint8_t* weights,
                            std::size_t rows, std::size_t columns,
                            const void* staged, float* output) {
  if (cand.packed) {
    return qw38::cuda::launch_quant_mmv_prequant(
        qw38::cuda::QuantKind::kQ4K, weights, rows, columns,
        static_cast<const qw38::cuda::Q8Block*>(staged), output, nullptr);
  }
  return qw38::cuda::launch_q4k_coop_mmv_prequant(
      weights, rows, columns, staged, output, cand.warps_per_row, cand.q8_1,
      nullptr);
}

struct Timed {
  std::vector<float> warmup;
  std::vector<float> samples;
  float mean_ms = 0.0F;
  bool launch_ok = false;
};

struct CandResult {
  Timed complete;
  Timed prequant;
  int occupancy = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  Envelope vs_fp64_orig{};
  Envelope vs_fp64_staged{};
  bool eligible = false;
  bool cooperative = false;
  bool dp4a = false;
  std::size_t staging_bytes = 0;
};

void finish_mean(Timed* timed) {
  if (timed->samples.size() != static_cast<std::size_t>(kMeasured)) {
    timed->launch_ok = false;
    return;
  }
  double acc = 0.0;
  for (float sample : timed->samples) acc += static_cast<double>(sample);
  timed->mean_ms = static_cast<float>(acc / kMeasured);
  timed->launch_ok = true;
}

cudaError_t record_ms(cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaError_t error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return error;
  return cudaEventElapsedTime(ms, start, stop);
}

void emit_timed(FILE* file, const Timed& timed) {
  std::fprintf(file, "\"mean_ms\":%.9g,\"launch_ok\":%s,",
               static_cast<double>(timed.mean_ms), json_bool(timed.launch_ok));
  print_array(file, "warmup_ms", timed.warmup);
  std::fprintf(file, ",");
  print_array(file, "samples", timed.samples);
}

void emit_env(FILE* file, const Envelope& env) {
  std::fprintf(file,
               "\"max_abs\":%.9g,\"rms\":%.9g,\"one_minus_cosine\":%.9g,"
               "\"nonfinite\":%zu",
               static_cast<double>(env.max_abs), static_cast<double>(env.rms),
               static_cast<double>(env.one_minus_cosine), env.nonfinite);
}

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt046-q4-decode/q4-decode-ab-raw.txt";
  const char* model_path = argc > 2 ? argv[2] : nullptr;
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaDeviceProp prop{};
  cudaError_t error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaGetDeviceProperties", error);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaEventCreate", error);
  }

  std::vector<std::vector<CandResult>> results(
      static_cast<std::size_t>(kSyntheticCount),
      std::vector<CandResult>(static_cast<std::size_t>(kCandidateCount)));

  for (int s = 0; s < kSyntheticCount && error == cudaSuccess; ++s) {
    const Shape& shape = kSynthetic[s];
    std::vector<std::uint8_t> weights;
    fill_weights(shape.rows, shape.columns, &weights);
    std::vector<__nv_bfloat16> activation;
    fill_activation(shape.columns, &activation);
    std::vector<float> act_f32(shape.columns);
    for (std::size_t i = 0; i < shape.columns; ++i) {
      act_f32[i] = __bfloat162float(activation[i]);
    }
    std::vector<float> staged_f32;
    host_q8_stage(activation, &staged_f32);
    std::vector<double> fp64_orig;
    std::vector<double> fp64_staged;
    if (!host_fp64(weights, shape.rows, shape.columns, act_f32, &fp64_orig) ||
        !host_fp64(weights, shape.rows, shape.columns, staged_f32,
                   &fp64_staged)) {
      std::fclose(raw);
      return 1;
    }

    std::uint8_t* device_weights = nullptr;
    __nv_bfloat16* device_act = nullptr;
    std::vector<void*> workspaces(static_cast<std::size_t>(kCandidateCount),
                                  nullptr);
    std::vector<float*> outputs(static_cast<std::size_t>(kCandidateCount),
                                nullptr);
    error = cudaMalloc(&device_weights, weights.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_act, activation.size() * sizeof(activation[0]));
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      error = cudaMalloc(&workspaces[static_cast<std::size_t>(c)],
                         qw38::cuda::q8_workspace_bytes(shape.columns));
      if (error == cudaSuccess) {
        error = cudaMalloc(&outputs[static_cast<std::size_t>(c)],
                           shape.rows * sizeof(float));
      }
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_weights, weights.data(), weights.size(),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_act, activation.data(),
                         activation.size() * sizeof(activation[0]),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("synthetic malloc", error);
    }

    for (int c = 0; c < kCandidateCount; ++c) {
      CandResult& slot = results[static_cast<std::size_t>(s)]
                                 [static_cast<std::size_t>(c)];
      const Candidate& cand = kCandidates[c];
      slot.cooperative = !cand.packed;
      slot.dp4a = !cand.packed;
      slot.staging_bytes = qw38::cuda::q8_workspace_bytes(shape.columns);
      if (cand.packed) {
        slot.occupancy = qw38::cuda::mmv_packed_occupancy(
            qw38::cuda::QuantKind::kQ4K,
            qw38::cuda::selected_mmv_warps(shape.rows));
        slot.registers = 0;
        slot.local_bytes = 0;
      } else {
        qw38::cuda::q4k_coop_kernel_attributes(
            cand.warps_per_row, cand.q8_1, &slot.registers, &slot.local_bytes,
            &slot.occupancy);
      }
    }

    auto time_pair = [&](int cand_index, bool complete, bool warmup,
                         int sample) -> cudaError_t {
      const Candidate& cand = kCandidates[cand_index];
      void* ws = workspaces[static_cast<std::size_t>(cand_index)];
      float* out = outputs[static_cast<std::size_t>(cand_index)];
      cudaError_t local = cudaEventRecord(start);
      if (local == cudaSuccess) {
        if (complete) {
          local = launch_candidate(cand, device_weights, shape.rows,
                                   shape.columns, device_act, ws, out);
        } else {
          local = launch_prequant(cand, device_weights, shape.rows,
                                  shape.columns, ws, out);
        }
      }
      if (local == cudaSuccess) local = cudaEventRecord(stop);
      float ms = 0.0F;
      if (local == cudaSuccess) local = record_ms(start, stop, &ms);
      if (local != cudaSuccess) return local;
      Timed& timed =
          complete ? results[static_cast<std::size_t>(s)]
                             [static_cast<std::size_t>(cand_index)]
                                 .complete
                   : results[static_cast<std::size_t>(s)]
                             [static_cast<std::size_t>(cand_index)]
                                 .prequant;
      if (warmup) timed.warmup.push_back(ms);
      else timed.samples.push_back(ms);
      std::fprintf(raw, "shape=%s cand=%s %s sample=%d ms=%.9g\n", shape.id,
                   cand.id, complete ? "complete" : "prequant", sample,
                   static_cast<double>(ms));
      return cudaSuccess;
    };

    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      for (int w = 0; w < kWarmups && error == cudaSuccess; ++w) {
        error = time_pair(c, true, true, w);
      }
    }
    for (int sample = 0; sample < kMeasured && error == cudaSuccess; ++sample) {
      for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
        error = time_pair(c, true, false, sample);
      }
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      error = launch_candidate(kCandidates[c], device_weights, shape.rows,
                               shape.columns, device_act,
                               workspaces[static_cast<std::size_t>(c)],
                               outputs[static_cast<std::size_t>(c)]);
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      for (int w = 0; w < kWarmups && error == cudaSuccess; ++w) {
        error = time_pair(c, false, true, w);
      }
    }
    for (int sample = 0; sample < kMeasured && error == cudaSuccess; ++sample) {
      for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
        error = time_pair(c, false, false, sample);
      }
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("timing", error);
    }

    for (int c = 0; c < kCandidateCount; ++c) {
      CandResult& slot = results[static_cast<std::size_t>(s)]
                                 [static_cast<std::size_t>(c)];
      finish_mean(&slot.complete);
      finish_mean(&slot.prequant);
      std::vector<float> host_out(shape.rows);
      error = cudaMemcpy(host_out.data(),
                         outputs[static_cast<std::size_t>(c)],
                         shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("D2H", error);
      }
      slot.vs_fp64_orig = compare_vec(host_out, fp64_orig);
      slot.vs_fp64_staged = compare_vec(host_out, fp64_staged);
      const CandResult& packed_slot =
          results[static_cast<std::size_t>(s)][0];
      const bool orig_ok =
          slot.vs_fp64_orig.nonfinite == 0 &&
          (slot.vs_fp64_orig.max_abs <= shape.abs_orig ||
           slot.vs_fp64_orig.max_abs <= packed_slot.vs_fp64_orig.max_abs * 1.05F) &&
          (slot.vs_fp64_orig.rms <= shape.rms_orig ||
           slot.vs_fp64_orig.rms <= packed_slot.vs_fp64_orig.rms * 1.05F) &&
          (slot.vs_fp64_orig.one_minus_cosine <= shape.cos_orig ||
           slot.vs_fp64_orig.one_minus_cosine <=
               packed_slot.vs_fp64_orig.one_minus_cosine * 1.05F + 1.0e-7F);
      const bool staged_ok =
          slot.vs_fp64_staged.nonfinite == 0 &&
          slot.vs_fp64_staged.max_abs <= shape.abs_staged &&
          slot.vs_fp64_staged.rms <= shape.rms_staged &&
          slot.vs_fp64_staged.one_minus_cosine <= shape.cos_staged;
      const bool numeric =
          orig_ok && slot.vs_fp64_staged.nonfinite == 0 &&
          (shape.probe && !kCandidates[c].q8_1 ? staged_ok : true);
      slot.eligible = slot.complete.launch_ok && slot.prequant.launch_ok &&
                      slot.occupancy >= 1 && numeric &&
                      (kCandidates[c].packed || slot.local_bytes == 0);
    }

    cudaFree(device_act);
    cudaFree(device_weights);
    for (int c = 0; c < kCandidateCount; ++c) {
      cudaFree(workspaces[static_cast<std::size_t>(c)]);
      cudaFree(outputs[static_cast<std::size_t>(c)]);
    }
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double packed_weighted = 0.0;
  double best_integer_weighted = 1.0e30;
  int best_integer = -1;
  int weight_sum = 0;
  bool packed_eligible = true;
  for (int s = 0; s < kSyntheticCount; ++s) {
    const int weight = kSynthetic[s].weight;
    if (weight == 0) continue;
    weight_sum += weight;
    packed_eligible =
        packed_eligible &&
        results[static_cast<std::size_t>(s)][0].eligible;
    packed_weighted +=
        weight * results[static_cast<std::size_t>(s)][0].complete.mean_ms;
  }
  for (int c = 1; c < kCandidateCount; ++c) {
    bool ok = packed_eligible;
    double weighted = 0.0;
    for (int s = 0; s < kSyntheticCount; ++s) {
      if (kSynthetic[s].weight == 0) {
        ok = ok && results[static_cast<std::size_t>(s)]
                           [static_cast<std::size_t>(c)]
                               .eligible;
        continue;
      }
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      ok = ok && slot.eligible;
      weighted += kSynthetic[s].weight * slot.complete.mean_ms;
    }
    if (ok && weighted < best_integer_weighted) {
      best_integer_weighted = weighted;
      best_integer = c;
    }
  }
  packed_weighted /= weight_sum;
  if (best_integer >= 0) best_integer_weighted /= weight_sum;
  const bool win = best_integer >= 0 && best_integer_weighted < packed_weighted;
  const int winner = win ? best_integer : 0;

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf("\"schema_version\":1,\"task\":\"OPT-046\",\"status\":\"measured\",");
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\","
              "\"compute_capability\":\"%d.%d\",",
              utc_text, prop.name, prop.major, prop.minor);
  std::printf("\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\",", kLlamaRev,
              kGgufSha);
  std::printf("\"winner\":\"%s\",\"win\":%s,\"packed_weighted_ms\":%.9g,"
              "\"winner_weighted_ms\":%.9g,",
              kCandidates[winner].id, json_bool(win), packed_weighted,
              win ? best_integer_weighted : packed_weighted);
  std::printf("\"selected_q4_decode_path\":\"packed\","
              "\"selected_q4_decode_warps_per_row\":4,");
  std::printf("\"ab\":{\"winner\":\"%s\",\"win\":%s,\"candidates\":{",
              kCandidates[winner].id, json_bool(win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    const Candidate& cand = kCandidates[c];
    int registers = 0;
    std::size_t local_bytes = 0;
    int occupancy = 0;
    bool eligible_all = true;
    double complete_ms = 0.0;
    int complete_n = 0;
    for (int s = 0; s < kSyntheticCount; ++s) {
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      eligible_all = eligible_all && slot.eligible;
      if (kSynthetic[s].weight > 0) {
        complete_ms += kSynthetic[s].weight * slot.complete.mean_ms;
        complete_n += kSynthetic[s].weight;
      }
      registers = slot.registers;
      local_bytes = slot.local_bytes;
      occupancy = slot.occupancy;
    }
    std::printf("\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"warps_per_row\":%u,"
                "\"q8_1\":%s,\"packed\":%s,\"occupancy\":%d,\"registers\":%d,"
                "\"local_bytes\":%zu,\"cooperative\":%s,\"dp4a\":%s,"
                "\"eligible\":%s,\"weighted_complete_ms\":%.9g,\"shapes\":{",
                cand.id, cand.id, cand.path, cand.warps_per_row,
                json_bool(cand.q8_1), json_bool(cand.packed), occupancy,
                registers, local_bytes, json_bool(!cand.packed),
                json_bool(!cand.packed), json_bool(eligible_all),
                complete_n == 0 ? 0.0 : complete_ms / complete_n);
    for (int s = 0; s < kSyntheticCount; ++s) {
      if (s != 0) std::printf(",");
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      std::printf("\"%s\":{\"staging_bytes\":%zu,\"complete\":{",
                  kSynthetic[s].id, slot.staging_bytes);
      emit_timed(stdout, slot.complete);
      std::printf("},\"prequant\":{");
      emit_timed(stdout, slot.prequant);
      std::printf("},\"vs_fp64_original_bf16\":{");
      emit_env(stdout, slot.vs_fp64_orig);
      std::printf("},\"vs_fp64_staged_activation\":{");
      emit_env(stdout, slot.vs_fp64_staged);
      std::printf("},\"eligible\":%s}", json_bool(slot.eligible));
    }
    std::printf("}}");
  }
  std::printf("}},\"model_path\":%s%s%s,\"graphs_recapture_required\":%s}\n",
              model_path == nullptr ? "null" : "\"",
              model_path == nullptr ? "" : model_path,
              model_path == nullptr ? "" : "\"", json_bool(win));
  std::printf("status=passed winner=%s packed_ms=%.9g winner_ms=%.9g\n",
              kCandidates[winner].id, packed_weighted,
              win ? best_integer_weighted : packed_weighted);
  std::fclose(raw);
  (void)model_path;
  return 0;
}
