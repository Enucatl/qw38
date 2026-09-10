#include "quant.h"
#include "quant_mmv.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT047_Q8_DECODE_AB_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ80Bytes = 34;
constexpr std::size_t kQ80Values = 32;

struct Shape {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  int weight;
  const char* role;
  bool probe;
  float abs_orig;
  float rms_orig;
  float cos_orig;
  float abs_staged;
  float rms_staged;
  float cos_staged;
};

constexpr Shape kShapes[] = {
    {"q8_0_17x256", 17, 256, 0, "probe", true, 8.822089429F, 4.6501625335F,
     4.837330126e-6F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"q8_0_257x512", 257, 512, 0, "probe", true, 18.6608887F, 8.139086731F,
     8.571427969e-6F, 3.0e-4F, 2.0e-4F, 1.0000000001165733e-6F},
    {"gdn_alpha", 48, 5120, 48, "skinny", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"gdn_beta", 48, 5120, 48, "skinny", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"attn_kv", 1024, 5120, 32, "medium", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"gdn_value_gate", 6144, 5120, 48, "wide", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"gdn_packed_qkv", 10240, 5120, 48, "wide", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"gdn_output", 5120, 6144, 48, "wide", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
    {"attn_query_gate", 12288, 5120, 16, "wide", false, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
};
constexpr int kShapeCount = static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));

struct Candidate {
  const char* id;
  const char* path;
  unsigned int warps_per_row;
  bool direct;
};

constexpr Candidate kCandidates[] = {
    {"direct_bf16", "direct_bf16", 0, true},
    {"dp4a_w1", "dp4a_q8_1", 1, false},
    {"dp4a_w2", "dp4a_q8_1", 2, false},
    {"dp4a_w4", "dp4a_q8_1", 4, false},
    {"dp4a_w8", "dp4a_q8_1", 8, false},
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
  weights->assign(rows * (columns / kQ80Values) * kQ80Bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ80Bytes) {
    write_u16(weights->data() + offset, 0x3C00U);
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
  std::vector<float> decoded(kQ80Values);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / kQ80Values; ++block) {
      const std::uint8_t* packed =
          weights.data() +
          (row * (columns / kQ80Values) + block) * kQ80Bytes;
      const qw38::Status status = qw38::internal::decode_q8_0(
          packed, kQ80Bytes, decoded.data(), decoded.size());
      if (!status.is_ok()) return false;
      for (std::size_t within = 0; within < kQ80Values; ++within) {
        sum += static_cast<double>(decoded[within]) *
               static_cast<double>(activation[block * kQ80Values + within]);
      }
    }
    (*output)[row] = sum;
  }
  return true;
}

void host_q8_1_stage(const std::vector<__nv_bfloat16>& activation,
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
      const float quant = scale == 0.0F ? 0.0F : std::round(value / scale);
      (*staged)[group * 32 + i] = scale * quant;
    }
  }
}

cudaError_t launch_candidate(const Candidate& cand, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation, void* workspace,
                             float* output) {
  if (cand.direct) {
    return qw38::cuda::launch_q8_mmv_bf16(weights, rows, columns, activation,
                                          output, nullptr);
  }
  return qw38::cuda::launch_q8_coop_mmv(weights, rows, columns, activation,
                                        workspace, output, cand.warps_per_row,
                                        nullptr);
}

cudaError_t launch_prequant(const Candidate& cand, const std::uint8_t* weights,
                            std::size_t rows, std::size_t columns,
                            const void* staged, const __nv_bfloat16* activation,
                            float* output) {
  if (cand.direct) {
    return qw38::cuda::launch_q8_mmv_bf16(weights, rows, columns, activation,
                                          output, nullptr);
  }
  return qw38::cuda::launch_q8_coop_mmv_prequant(
      weights, rows, columns, staged, output, cand.warps_per_row, nullptr);
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

struct GroupTimed {
  Timed complete;
  bool launch_ok = false;
};

void finish_mean(Timed* timed, int measured) {
  if (static_cast<int>(timed->samples.size()) != measured) {
    timed->launch_ok = false;
    return;
  }
  double acc = 0.0;
  for (float sample : timed->samples) acc += static_cast<double>(sample);
  timed->mean_ms = static_cast<float>(acc / measured);
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

void emit_phase(const char* name) {
  const std::time_t now = std::time(nullptr);
  std::fprintf(stderr, "phase=%s epoch=%lld\n", name,
               static_cast<long long>(now));
}

int shape_limit(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return 1;
  return kShapeCount;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr,
                 "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
                 "or acceptance\n");
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  const int n_shapes = shape_limit(tier);
  const bool run_groups = tier != qw38::cuda::TestTier::kSmoke;
  emit_phase("start");
  std::fprintf(stderr, "tier=%s warmups=%d samples=%d shapes=%d\n",
               qw38::cuda::test_tier_name(), warmups, measured, n_shapes);

  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt047-q8-decode/q8-decode-ab-raw.txt";
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
      static_cast<std::size_t>(kShapeCount),
      std::vector<CandResult>(static_cast<std::size_t>(kCandidateCount)));
  std::vector<GroupTimed> gdn_groups(static_cast<std::size_t>(kCandidateCount));
  std::vector<GroupTimed> attn_groups(static_cast<std::size_t>(kCandidateCount));

  emit_phase("shapes");
  for (int s = 0; s < n_shapes && error == cudaSuccess; ++s) {
    const Shape& shape = kShapes[s];
    std::vector<std::uint8_t> weights;
    fill_weights(shape.rows, shape.columns, &weights);
    std::vector<__nv_bfloat16> activation;
    fill_activation(shape.columns, &activation);
    std::vector<float> act_f32(shape.columns);
    for (std::size_t i = 0; i < shape.columns; ++i) {
      act_f32[i] = __bfloat162float(activation[i]);
    }
    std::vector<float> staged_f32;
    host_q8_1_stage(activation, &staged_f32);
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
                         qw38::cuda::q8_1_workspace_bytes(shape.columns));
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
      slot.cooperative = !cand.direct;
      slot.dp4a = !cand.direct;
      slot.staging_bytes = qw38::cuda::q8_1_workspace_bytes(shape.columns);
      if (cand.direct) {
        qw38::cuda::q8_mmv_bf16_kernel_attributes(
            &slot.registers, &slot.local_bytes, &slot.occupancy);
      } else {
        qw38::cuda::q8_coop_kernel_attributes(
            cand.warps_per_row, &slot.registers, &slot.local_bytes,
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
          if (!cand.direct) {
            local = qw38::cuda::launch_quantize_bf16_q8_1(device_act, ws,
                                                          shape.columns,
                                                          nullptr);
          }
          if (local == cudaSuccess) {
            local = launch_prequant(cand, device_weights, shape.rows,
                                    shape.columns, ws, device_act, out);
          }
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
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        error = time_pair(c, true, true, w);
      }
    }
    for (int sample = 0; sample < measured && error == cudaSuccess; ++sample) {
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
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        error = time_pair(c, false, true, w);
      }
    }
    for (int sample = 0; sample < measured && error == cudaSuccess; ++sample) {
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
      finish_mean(&slot.complete, measured);
      finish_mean(&slot.prequant, measured);
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
      const CandResult& bf16_slot = results[static_cast<std::size_t>(s)][0];
      const bool orig_ok =
          slot.vs_fp64_orig.nonfinite == 0 &&
          (slot.vs_fp64_orig.max_abs <= shape.abs_orig ||
           slot.vs_fp64_orig.max_abs <=
               bf16_slot.vs_fp64_orig.max_abs * 1.05F + 1.0e-6F) &&
          (slot.vs_fp64_orig.rms <= shape.rms_orig ||
           slot.vs_fp64_orig.rms <=
               bf16_slot.vs_fp64_orig.rms * 1.05F + 1.0e-6F) &&
          (slot.vs_fp64_orig.one_minus_cosine <= shape.cos_orig ||
           slot.vs_fp64_orig.one_minus_cosine <=
               bf16_slot.vs_fp64_orig.one_minus_cosine * 1.05F + 1.0e-7F);
      const bool staged_ok = slot.vs_fp64_staged.nonfinite == 0;
      slot.eligible = slot.complete.launch_ok && slot.prequant.launch_ok &&
                      slot.occupancy >= 1 && orig_ok && staged_ok &&
                      (kCandidates[c].direct || slot.local_bytes == 0);
    }

    cudaFree(device_act);
    cudaFree(device_weights);
    for (int c = 0; c < kCandidateCount; ++c) {
      cudaFree(workspaces[static_cast<std::size_t>(c)]);
      cudaFree(outputs[static_cast<std::size_t>(c)]);
    }
  }

  if (run_groups && error == cudaSuccess) {
    emit_phase("groups");
    auto alloc_mat = [&](std::size_t rows, std::size_t cols, std::uint8_t** dev)
        -> cudaError_t {
      std::vector<std::uint8_t> host;
      fill_weights(rows, cols, &host);
      cudaError_t local = cudaMalloc(dev, host.size());
      if (local == cudaSuccess) {
        local = cudaMemcpy(*dev, host.data(), host.size(),
                           cudaMemcpyHostToDevice);
      }
      return local;
    };
    std::uint8_t* packed = nullptr;
    std::uint8_t* value_gate = nullptr;
    std::uint8_t* alpha = nullptr;
    std::uint8_t* beta = nullptr;
    std::uint8_t* gdn_out = nullptr;
    std::uint8_t* query_gate = nullptr;
    std::uint8_t* key = nullptr;
    std::uint8_t* value = nullptr;
    __nv_bfloat16* act5120 = nullptr;
    __nv_bfloat16* act6144 = nullptr;
    void* staged = nullptr;
    float* out_a = nullptr;
    float* out_b = nullptr;
    float* out_c = nullptr;
    float* out_d = nullptr;
    float* out_e = nullptr;
    std::vector<__nv_bfloat16> host5120;
    std::vector<__nv_bfloat16> host6144;
    fill_activation(5120, &host5120);
    fill_activation(6144, &host6144);
    error = alloc_mat(10240, 5120, &packed);
    if (error == cudaSuccess) error = alloc_mat(6144, 5120, &value_gate);
    if (error == cudaSuccess) error = alloc_mat(48, 5120, &alpha);
    if (error == cudaSuccess) error = alloc_mat(48, 5120, &beta);
    if (error == cudaSuccess) error = alloc_mat(5120, 6144, &gdn_out);
    if (error == cudaSuccess) error = alloc_mat(12288, 5120, &query_gate);
    if (error == cudaSuccess) error = alloc_mat(1024, 5120, &key);
    if (error == cudaSuccess) error = alloc_mat(1024, 5120, &value);
    if (error == cudaSuccess) {
      error = cudaMalloc(&act5120, 5120 * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&act6144, 6144 * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&staged, qw38::cuda::q8_1_workspace_bytes(6144));
    }
    if (error == cudaSuccess) error = cudaMalloc(&out_a, 12288 * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(&out_b, 6144 * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(&out_c, 1024 * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(&out_d, 1024 * sizeof(float));
    if (error == cudaSuccess) error = cudaMalloc(&out_e, 5120 * sizeof(float));
    if (error == cudaSuccess) {
      error = cudaMemcpy(act5120, host5120.data(),
                         5120 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(act6144, host6144.data(),
                         6144 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("group malloc", error);
    }

    auto gdn_seq = [&](const Candidate& cand) -> cudaError_t {
      cudaError_t local = cudaSuccess;
      if (cand.direct) {
        local = qw38::cuda::launch_q8_mmv_bf16(packed, 10240, 5120, act5120,
                                               out_a, nullptr);
        if (local == cudaSuccess) {
          local = qw38::cuda::launch_q8_mmv_bf16(value_gate, 6144, 5120, act5120,
                                                 out_b, nullptr);
        }
        if (local == cudaSuccess) {
          local = qw38::cuda::launch_q8_mmv_bf16(alpha, 48, 5120, act5120, out_c,
                                                 nullptr);
        }
        if (local == cudaSuccess) {
          local = qw38::cuda::launch_q8_mmv_bf16(beta, 48, 5120, act5120, out_d,
                                                 nullptr);
        }
        if (local == cudaSuccess) {
          local = qw38::cuda::launch_q8_mmv_bf16(gdn_out, 5120, 6144, act6144,
                                                 out_e, nullptr);
        }
        return local;
      }
      local = qw38::cuda::launch_quantize_bf16_q8_1(act5120, staged, 5120,
                                                    nullptr);
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            packed, 10240, 5120, staged, out_a, cand.warps_per_row, nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            value_gate, 6144, 5120, staged, out_b, cand.warps_per_row, nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            alpha, 48, 5120, staged, out_c, cand.warps_per_row, nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            beta, 48, 5120, staged, out_d, cand.warps_per_row, nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_quantize_bf16_q8_1(act6144, staged, 6144,
                                                      nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            gdn_out, 5120, 6144, staged, out_e, cand.warps_per_row, nullptr);
      }
      return local;
    };
    auto attn_seq = [&](const Candidate& cand) -> cudaError_t {
      cudaError_t local = cudaSuccess;
      if (cand.direct) {
        local = qw38::cuda::launch_q8_mmv_bf16(query_gate, 12288, 5120, act5120,
                                               out_a, nullptr);
        if (local == cudaSuccess) {
          local = qw38::cuda::launch_q8_mmv_bf16(key, 1024, 5120, act5120, out_c,
                                                 nullptr);
        }
        if (local == cudaSuccess) {
          local = qw38::cuda::launch_q8_mmv_bf16(value, 1024, 5120, act5120,
                                                 out_d, nullptr);
        }
        return local;
      }
      local = qw38::cuda::launch_quantize_bf16_q8_1(act5120, staged, 5120,
                                                    nullptr);
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            query_gate, 12288, 5120, staged, out_a, cand.warps_per_row,
            nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            key, 1024, 5120, staged, out_c, cand.warps_per_row, nullptr);
      }
      if (local == cudaSuccess) {
        local = qw38::cuda::launch_q8_coop_mmv_prequant(
            value, 1024, 5120, staged, out_d, cand.warps_per_row, nullptr);
      }
      return local;
    };

    auto time_group = [&](int cand_index, bool gdn, bool warmup,
                          int sample) -> cudaError_t {
      const Candidate& cand = kCandidates[cand_index];
      cudaError_t local = cudaEventRecord(start);
      if (local == cudaSuccess) {
        local = gdn ? gdn_seq(cand) : attn_seq(cand);
      }
      if (local == cudaSuccess) local = cudaEventRecord(stop);
      float ms = 0.0F;
      if (local == cudaSuccess) local = record_ms(start, stop, &ms);
      if (local != cudaSuccess) return local;
      Timed& timed = gdn ? gdn_groups[static_cast<std::size_t>(cand_index)].complete
                         : attn_groups[static_cast<std::size_t>(cand_index)].complete;
      if (warmup) timed.warmup.push_back(ms);
      else timed.samples.push_back(ms);
      std::fprintf(raw, "group=%s cand=%s sample=%d ms=%.9g\n",
                   gdn ? "gdn" : "attn", cand.id, sample,
                   static_cast<double>(ms));
      return cudaSuccess;
    };

    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        error = time_group(c, true, true, w);
        if (error == cudaSuccess) error = time_group(c, false, true, w);
      }
    }
    for (int sample = 0; sample < measured && error == cudaSuccess; ++sample) {
      for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
        error = time_group(c, true, false, sample);
        if (error == cudaSuccess) error = time_group(c, false, false, sample);
      }
    }
    for (int c = 0; c < kCandidateCount; ++c) {
      finish_mean(&gdn_groups[static_cast<std::size_t>(c)].complete, measured);
      finish_mean(&attn_groups[static_cast<std::size_t>(c)].complete, measured);
      gdn_groups[static_cast<std::size_t>(c)].launch_ok =
          gdn_groups[static_cast<std::size_t>(c)].complete.launch_ok;
      attn_groups[static_cast<std::size_t>(c)].launch_ok =
          attn_groups[static_cast<std::size_t>(c)].complete.launch_ok;
    }
    cudaFree(packed);
    cudaFree(value_gate);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(gdn_out);
    cudaFree(query_gate);
    cudaFree(key);
    cudaFree(value);
    cudaFree(act5120);
    cudaFree(act6144);
    cudaFree(staged);
    cudaFree(out_a);
    cudaFree(out_b);
    cudaFree(out_c);
    cudaFree(out_d);
    cudaFree(out_e);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  emit_phase("reduce");

  constexpr int kGdnLayers = 48;
  constexpr int kAttnLayers = 16;
  bool bf16_eligible = true;
  for (int s = 0; s < n_shapes; ++s) {
    bf16_eligible = bf16_eligible && results[static_cast<std::size_t>(s)][0].eligible;
  }
  double bf16_group = 0.0;
  if (run_groups) {
    bf16_group = kGdnLayers * gdn_groups[0].complete.mean_ms +
                 kAttnLayers * attn_groups[0].complete.mean_ms;
  }
  double best_group = 1.0e30;
  int best = -1;
  for (int c = 1; c < kCandidateCount; ++c) {
    bool ok = bf16_eligible &&
              (!run_groups || (gdn_groups[static_cast<std::size_t>(c)].launch_ok &&
                               attn_groups[static_cast<std::size_t>(c)].launch_ok));
    for (int s = 0; s < n_shapes; ++s) {
      ok = ok && results[static_cast<std::size_t>(s)]
                        [static_cast<std::size_t>(c)]
                            .eligible;
    }
    if (!ok) continue;
    double weighted = 0.0;
    if (run_groups) {
      weighted = kGdnLayers *
                     gdn_groups[static_cast<std::size_t>(c)].complete.mean_ms +
                 kAttnLayers *
                     attn_groups[static_cast<std::size_t>(c)].complete.mean_ms;
    } else {
      weighted = results[0][static_cast<std::size_t>(c)].complete.mean_ms;
    }
    if (weighted < best_group) {
      best_group = weighted;
      best = c;
    }
  }
  const bool win = best >= 0 && run_groups && best_group < bf16_group;
  const int winner = win ? best : 0;

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf("\"schema_version\":1,\"task\":\"OPT-047\",\"status\":\"measured\",");
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\","
              "\"compute_capability\":\"%d.%d\",",
              utc_text, prop.name, prop.major, prop.minor);
  std::printf("\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\",", kLlamaRev,
              kGgufSha);
  std::printf("\"tier\":\"%s\",\"winner\":\"%s\",\"win\":%s,"
              "\"direct_bf16_group_ms\":%.9g,\"winner_group_ms\":%.9g,",
              qw38::cuda::test_tier_name(), kCandidates[winner].id,
              json_bool(win), bf16_group, win ? best_group : bf16_group);
  std::printf("\"ab\":{\"winner\":\"%s\",\"win\":%s,\"candidates\":{",
              kCandidates[winner].id, json_bool(win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    const Candidate& cand = kCandidates[c];
    int registers = 0;
    std::size_t local_bytes = 0;
    int occupancy = 0;
    bool eligible_all = true;
    for (int s = 0; s < n_shapes; ++s) {
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      eligible_all = eligible_all && slot.eligible;
      registers = slot.registers;
      local_bytes = slot.local_bytes;
      occupancy = slot.occupancy;
    }
    const double group_ms =
        run_groups
            ? kGdnLayers * gdn_groups[static_cast<std::size_t>(c)].complete.mean_ms +
                  kAttnLayers *
                      attn_groups[static_cast<std::size_t>(c)].complete.mean_ms
            : 0.0;
    std::printf("\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"warps_per_row\":%u,"
                "\"direct\":%s,\"occupancy\":%d,\"registers\":%d,"
                "\"local_bytes\":%zu,\"cooperative\":%s,\"dp4a\":%s,"
                "\"eligible\":%s,\"weighted_group_ms\":%.9g,\"shapes\":{",
                cand.id, cand.id, cand.path, cand.warps_per_row,
                json_bool(cand.direct), occupancy, registers, local_bytes,
                json_bool(!cand.direct), json_bool(!cand.direct),
                json_bool(eligible_all), group_ms);
    for (int s = 0; s < n_shapes; ++s) {
      if (s != 0) std::printf(",");
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      std::printf("\"%s\":{\"role\":\"%s\",\"staging_bytes\":%zu,\"complete\":{",
                  kShapes[s].id, kShapes[s].role, slot.staging_bytes);
      emit_timed(stdout, slot.complete);
      std::printf("},\"prequant\":{");
      emit_timed(stdout, slot.prequant);
      std::printf("},\"vs_fp64_original_bf16\":{");
      emit_env(stdout, slot.vs_fp64_orig);
      std::printf("},\"vs_fp64_staged_activation\":{");
      emit_env(stdout, slot.vs_fp64_staged);
      std::printf("},\"eligible\":%s}", json_bool(slot.eligible));
    }
    std::printf("}");
    if (run_groups) {
      std::printf(",\"gdn_group\":{");
      emit_timed(stdout, gdn_groups[static_cast<std::size_t>(c)].complete);
      std::printf("},\"attn_group\":{");
      emit_timed(stdout, attn_groups[static_cast<std::size_t>(c)].complete);
      std::printf("}");
    }
    std::printf("}");
  }
  std::printf("}},\"model_path\":%s%s%s,\"activation_approximation\":"
              "\"Q8_1 block quantization of BF16 mixer activations\","
              "\"opt046_q4_path\":\"packed\"}\n",
              model_path == nullptr ? "null" : "\"",
              model_path == nullptr ? "" : model_path,
              model_path == nullptr ? "" : "\"");
  std::printf("status=passed winner=%s bf16_ms=%.9g winner_ms=%.9g\n",
              kCandidates[winner].id, bf16_group,
              win ? best_group : bf16_group);
  std::fclose(raw);
  emit_phase("done");
  (void)model_path;
  return 0;
}
