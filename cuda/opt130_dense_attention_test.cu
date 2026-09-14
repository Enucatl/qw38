#include "attention_decode.h"
#include "decode_launch_state.cuh"
#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT130_DENSE_ATTENTION_RESULT=";
constexpr char kCounts[] = "QW38_OPT130_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kRotary = 64;
constexpr int kDefaultPrefixes[] = {128, 1023, 1024, 2048, 8192, 32768};
constexpr int kDefaultPrefixCount = 6;
constexpr float kVsControlAbs = 2.0e-2F;
constexpr std::size_t kMinCapacity = 4096;
constexpr int kDefaultWarmups = 3;
constexpr int kDefaultPairs = 10;
constexpr std::size_t kDefaultDecodeTokens = 256;

struct Options final {
  const char* workload = "inspect";
  std::size_t prefix = 128;
  std::size_t warmups = kDefaultWarmups;
  std::size_t samples = kDefaultPairs;
  std::size_t tokens = kDefaultDecodeTokens;
  std::size_t capacity = 0;
};

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
  std::chrono::steady_clock::time_point cancel_requested{};
  bool requested = false;
};

struct ArmResult final {
  float setup_ms = 0.0F;
  float prefill_ms = 0.0F;
  float decode_only_ms = 0.0F;
  float request_ms = 0.0F;
  float ttft_ms = 0.0F;
  float decode_only_tok_s = 0.0F;
  float request_tok_s = 0.0F;
  float p50_ms = 0.0F;
  float p95_ms = 0.0F;
  std::size_t graph_bytes = 0;
  std::uint32_t capture = 0;
  std::uint32_t instantiate = 0;
  std::uint32_t upload = 0;
  std::uint32_t destroy = 0;
  std::uint32_t topology_recapture = 0;
};

struct DeviceCase final {
  qw38::cuda::AttentionConfig config{};
  std::size_t position = 0;
  std::size_t qn = 0;
  std::size_t kn = 0;
  std::size_t cache = 0;
  float* d_q = nullptr;
  float* d_k = nullptr;
  float* d_v = nullptr;
  float* d_g = nullptr;
  float* d_qs = nullptr;
  float* d_ks = nullptr;
  float* d_nq = nullptr;
  float* d_nk = nullptr;
  float* d_scores = nullptr;
  float* d_out = nullptr;
  float* d_vkq = nullptr;
  float* d_meta = nullptr;
  __nv_bfloat16* d_ck = nullptr;
  __nv_bfloat16* d_cv = nullptr;
  __nv_bfloat16* d_candk = nullptr;
  __nv_bfloat16* d_candv = nullptr;
  void free_all() {
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_qs);
    cudaFree(d_ks);
    cudaFree(d_nq);
    cudaFree(d_nk);
    cudaFree(d_scores);
    cudaFree(d_out);
    cudaFree(d_vkq);
    cudaFree(d_meta);
    cudaFree(d_ck);
    cudaFree(d_cv);
    cudaFree(d_candk);
    cudaFree(d_candv);
  }
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s [--workload inspect|identity|primitive|same-math|"
      "cancellation|handoff|graph-ab|state-memory] "
      "[--prefix N] [--warmups N] [--samples N] [--tokens N] [--capacity N] "
      "[MODEL] independently_restored=true same_binary=true\n",
      argv0);
  return 2;
}

bool parse_size(const char* text, std::size_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prefix)) return -2;
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->warmups)) return -2;
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->samples)) return -2;
    } else if (std::strcmp(arg, "--tokens") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->tokens)) return -2;
    } else if (std::strcmp(arg, "--capacity") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->capacity)) return -2;
    } else if (arg[0] == '-') {
      return -2;
    } else {
      model_index = index;
    }
  }
  return model_index;
}

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

float unit(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  return (static_cast<float>(x & 0x00FFFFFFU) / 8388608.0F - 1.0F) * 0.35F;
}

void fill_case(const qw38::cuda::AttentionConfig& config, std::size_t position,
               std::uint32_t seed, std::vector<float>* query,
               std::vector<float>* key, std::vector<float>* value,
               std::vector<float>* gate, std::vector<float>* qscale,
               std::vector<float>* kscale,
               std::vector<__nv_bfloat16>* committed_key,
               std::vector<__nv_bfloat16>* committed_value) {
  const std::size_t qn = qw38::cuda::attention_query_values(config);
  const std::size_t kn = qw38::cuda::attention_kv_row_values(config);
  const std::size_t cache = qw38::cuda::attention_cache_values(config);
  query->resize(qn);
  key->resize(kn);
  value->resize(kn);
  gate->resize(qn);
  qscale->assign(config.head_width, 1.0F);
  kscale->assign(config.head_width, 1.0F);
  committed_key->assign(cache, __float2bfloat16_rn(0.0F));
  committed_value->assign(cache, __float2bfloat16_rn(0.0F));
  for (std::size_t index = 0; index < qn; ++index) {
    (*query)[index] = unit(static_cast<std::uint32_t>(index), seed);
    (*gate)[index] = unit(static_cast<std::uint32_t>(index), seed + 17U);
  }
  for (std::size_t index = 0; index < kn; ++index) {
    (*key)[index] = unit(static_cast<std::uint32_t>(index), seed + 31U);
    (*value)[index] = unit(static_cast<std::uint32_t>(index), seed + 47U);
  }
  for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
    (*qscale)[dim] = 0.85F + 0.002F * static_cast<float>(dim);
    (*kscale)[dim] = 0.90F + 0.0015F * static_cast<float>(dim);
  }
  for (std::size_t token = 0; token < position; ++token) {
    for (std::uint32_t kv_head = 0; kv_head < config.kv_heads; ++kv_head) {
      for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
        const std::size_t phys = qw38::cuda::attention_kv_physical_index(
            token, kv_head, dim, config.capacity, config.head_width);
        const float kitem =
            unit(static_cast<std::uint32_t>(phys), seed + 101U);
        const float vitem =
            unit(static_cast<std::uint32_t>(phys), seed + 131U);
        (*committed_key)[phys] = __float2bfloat16_rn(kitem);
        (*committed_value)[phys] = __float2bfloat16_rn(vitem);
      }
    }
  }
}

int check(cudaError_t error, const char* op) {
  if (error != cudaSuccess) return fail_cuda(op, error);
  return 0;
}

int alloc_device(DeviceCase* device, const std::vector<float>& query,
                 const std::vector<float>& key, const std::vector<float>& value,
                 const std::vector<float>& gate,
                 const std::vector<float>& qscale,
                 const std::vector<float>& kscale,
                 const std::vector<__nv_bfloat16>& committed_key,
                 const std::vector<__nv_bfloat16>& committed_value) {
  const std::size_t partial = qw38::cuda::decode_kv_partial_vkq_values(16);
  const std::size_t meta = qw38::cuda::decode_kv_partial_meta_values(16);
  const std::size_t scores =
      qw38::cuda::attention_score_values(device->config, device->position);
  int rc = 0;
  rc |= check(cudaMalloc(&device->d_q, device->qn * sizeof(float)), "q");
  rc |= check(cudaMalloc(&device->d_k, device->kn * sizeof(float)), "k");
  rc |= check(cudaMalloc(&device->d_v, device->kn * sizeof(float)), "v");
  rc |= check(cudaMalloc(&device->d_g, device->qn * sizeof(float)), "g");
  rc |= check(cudaMalloc(&device->d_qs, kWidth * sizeof(float)), "qs");
  rc |= check(cudaMalloc(&device->d_ks, kWidth * sizeof(float)), "ks");
  rc |= check(cudaMalloc(&device->d_nq, device->qn * sizeof(float)), "nq");
  rc |= check(cudaMalloc(&device->d_nk, device->kn * sizeof(float)), "nk");
  rc |= check(cudaMalloc(&device->d_scores, scores * sizeof(float)), "scores");
  rc |= check(cudaMalloc(&device->d_out, device->qn * sizeof(float)), "out");
  rc |= check(cudaMalloc(&device->d_vkq, partial * sizeof(float)), "vkq");
  rc |= check(cudaMalloc(&device->d_meta, meta * sizeof(float)), "meta");
  rc |= check(cudaMalloc(&device->d_ck, device->cache * sizeof(__nv_bfloat16)),
              "ck");
  rc |= check(cudaMalloc(&device->d_cv, device->cache * sizeof(__nv_bfloat16)),
              "cv");
  rc |= check(cudaMalloc(&device->d_candk, device->kn * sizeof(__nv_bfloat16)),
              "candk");
  rc |= check(cudaMalloc(&device->d_candv, device->kn * sizeof(__nv_bfloat16)),
              "candv");
  if (rc != 0) {
    device->free_all();
    return rc;
  }
  rc |= check(cudaMemcpy(device->d_q, query.data(), device->qn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D q");
  rc |= check(cudaMemcpy(device->d_k, key.data(), device->kn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D k");
  rc |= check(cudaMemcpy(device->d_v, value.data(), device->kn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D v");
  rc |= check(cudaMemcpy(device->d_g, gate.data(), device->qn * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D g");
  rc |= check(cudaMemcpy(device->d_qs, qscale.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D qs");
  rc |= check(cudaMemcpy(device->d_ks, kscale.data(), kWidth * sizeof(float),
                         cudaMemcpyHostToDevice),
              "H2D ks");
  rc |= check(cudaMemcpy(device->d_ck, committed_key.data(),
                         device->cache * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice),
              "H2D ck");
  rc |= check(cudaMemcpy(device->d_cv, committed_value.data(),
                         device->cache * sizeof(__nv_bfloat16),
                         cudaMemcpyHostToDevice),
              "H2D cv");
  if (rc != 0) device->free_all();
  return rc;
}

cudaError_t launch_arm(DeviceCase* device, bool candidate) {
  qw38::cuda::Opt130DenseAttentionScope scope(
      candidate ? qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open
                : qw38::cuda::kLegalDecodeAttentionHybridCrossover);
  qw38::cuda::AttentionCache committed{device->d_ck, device->d_cv};
  qw38::cuda::AttentionCache cand{device->d_candk, device->d_candv};
  const int n_parts =
      qw38::cuda::decode_kv_parts_for_position(device->position);
  return qw38::cuda::launch_attention_prepare_partitioned(
      device->config, device->position, device->d_q, device->d_k, device->d_v,
      device->d_qs, device->d_ks, device->d_g, committed, cand, device->d_nq,
      device->d_nk, device->d_scores, device->d_out, device->d_vkq,
      device->d_meta, n_parts, nullptr);
}

float mean_ms(const std::vector<float>& values) {
  if (values.empty()) return 0.0F;
  double sum = 0.0;
  for (float item : values) sum += item;
  return static_cast<float>(sum / static_cast<double>(values.size()));
}

int time_arm(DeviceCase* device, bool candidate, int warmups, int samples,
             std::vector<float>* ms) {
  ms->clear();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int round = 0; round < warmups + samples; ++round) {
    cudaEventRecord(start, nullptr);
    const cudaError_t error = launch_arm(device, candidate);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda(candidate ? "candidate" : "parent", error);
    }
    if (round >= warmups) {
      float elapsed = 0.0F;
      cudaEventElapsedTime(&elapsed, start, stop);
      ms->push_back(elapsed);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

int compare_outputs(const std::vector<float>& left,
                    const std::vector<float>& right, float* max_abs,
                    std::size_t* nonfinite) {
  *max_abs = 0.0F;
  *nonfinite = 0;
  const std::size_t n = std::min(left.size(), right.size());
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index])) {
      ++(*nonfinite);
    }
    *max_abs = std::max(*max_abs, std::fabs(left[index] - right[index]));
  }
  return 0;
}

qw38::cuda::AttentionConfig make_config(std::size_t position) {
  qw38::cuda::AttentionConfig config{};
  config.query_heads = kQueryHeads;
  config.kv_heads = kKvHeads;
  config.head_width = kWidth;
  config.rotary_width = kRotary;
  config.capacity = position + 1;
  return config;
}

int setup_case(std::size_t position, DeviceCase* device,
               std::vector<float>* host_out) {
  *device = DeviceCase{};
  device->config = make_config(position);
  device->position = position;
  device->qn = qw38::cuda::attention_query_values(device->config);
  device->kn = qw38::cuda::attention_kv_row_values(device->config);
  device->cache = qw38::cuda::attention_cache_values(device->config);
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> gate;
  std::vector<float> qscale;
  std::vector<float> kscale;
  std::vector<__nv_bfloat16> ck;
  std::vector<__nv_bfloat16> cv;
  fill_case(device->config, position, 89U, &query, &key, &value, &gate, &qscale,
            &kscale, &ck, &cv);
  host_out->assign(device->qn, 0.0F);
  return alloc_device(device, query, key, value, gate, qscale, kscale, ck, cv);
}

int run_inspect() {
  int occupancy = 0;
  int nsm = 0;
  const int raw2048 =
      qw38::cuda::occupancy_raw_vec128_n_parts(2048, &occupancy, &nsm);
  const int snap2048 = qw38::cuda::occupancy_snapped_vec128_n_parts(2048);
  const int snap128 = qw38::cuda::occupancy_snapped_vec128_n_parts(128);
  const int snap8192 = qw38::cuda::occupancy_snapped_vec128_n_parts(8192);
  const int parent_crossover =
      qw38::cuda::selected_decode_attention_crossover_threshold();
  const int parent_verified =
      qw38::cuda::selected_decode_attention_verified_max();
  const int parent_parts = qw38::cuda::selected_vec128_n_parts();
  int cand_kernel = 0;
  int cand_parts = 0;
  {
    qw38::cuda::Opt130DenseAttentionScope scope(
        qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open);
    cand_kernel =
        qw38::cuda::decode_attention_vec128_uses_online_at(8192) ? 1 : 0;
    cand_parts = qw38::cuda::decode_kv_parts_for_position(2048);
  }
  const bool parent_8192_warp =
      !qw38::cuda::decode_attention_vec128_uses_online_at(8192);
  const bool ok = parent_crossover == 1024 && parent_verified == 4096 &&
                  parent_parts == 16 && cand_kernel == 1 && cand_parts == 8 &&
                  parent_8192_warp;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"workload\":\"inspect\","
      "\"ok\":%s,\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"parent\":\"hybrid_crossover\",\"candidate\":\"occupancy_vec128_open\","
      "\"frozen_dispatch\":\"hybrid_crossover@1024_vec128_open_nparts8\","
      "\"parent_crossover\":%d,\"parent_verified_max\":%d,"
      "\"parent_vec128_n_parts\":%d,\"candidate_verified_max\":%d,"
      "\"candidate_vec128_n_parts\":%d,\"occupancy\":%d,\"nsm\":%d,"
      "\"raw_n_parts_2048\":%d,\"snap_n_parts_128\":%d,"
      "\"snap_n_parts_2048\":%d,\"snap_n_parts_8192\":%d,"
      "\"candidate_vec128_at_8192\":%s,\"gqa6_admitted\":false,"
      "\"mma_launched\":false,\"second_candidate\":\"llama_mma_decode_consumer\","
      "\"dense_bf16\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), kLlamaRev, kGgufSha, parent_crossover,
      parent_verified, parent_parts, qw38::cuda::kOpt130CandidateVerifiedMax,
      qw38::cuda::kOpt130CandidateVec128NParts, occupancy, nsm, raw2048,
      snap128, snap2048, snap8192, json_bool(cand_kernel == 1));
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_identity() {
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"workload\":\"identity\","
      "\"cases\":[",
      kPrefix);
  for (int i = 0; i < kDefaultPrefixCount; ++i) {
    const std::size_t position =
        static_cast<std::size_t>(kDefaultPrefixes[i]);
    DeviceCase device;
    std::vector<float> unused;
    int rc = setup_case(position, &device, &unused);
    std::vector<float> parent(device.qn);
    std::vector<float> candidate(device.qn);
    if (rc == 0) rc = check(launch_arm(&device, false), "parent launch");
    if (rc == 0) {
      rc = check(cudaMemcpy(parent.data(), device.d_out,
                            device.qn * sizeof(float), cudaMemcpyDeviceToHost),
                 "D2H parent");
    }
    if (rc == 0) rc = check(launch_arm(&device, true), "candidate launch");
    if (rc == 0) {
      rc = check(cudaMemcpy(candidate.data(), device.d_out,
                            device.qn * sizeof(float), cudaMemcpyDeviceToHost),
                 "D2H candidate");
    }
    float max_abs = 0.0F;
    std::size_t nonfinite = 0;
    if (rc == 0) compare_outputs(parent, candidate, &max_abs, &nonfinite);
    const bool below = position < 1024;
    const bool ok =
        rc == 0 && nonfinite == 0 &&
        (below ? max_abs < 1.0e-6F : max_abs <= kVsControlAbs);
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"prefix\":%zu,\"ok\":%s,\"max_abs\":%.9g,\"nonfinite\":%zu,"
        "\"same_kernel_below_crossover\":%s}",
        i == 0 ? "" : ",", position, json_bool(ok),
        static_cast<double>(max_abs), nonfinite, json_bool(below));
    device.free_all();
  }
  std::printf("],\"ok\":%s,\"keep\":false}\n", json_bool(all_ok));
  print_counts(0, 1, 2, false);
  return all_ok ? 0 : 1;
}

int run_primitive(const Options& options) {
  const int warmups = static_cast<int>(options.warmups);
  const int samples = static_cast<int>(options.samples);
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"workload\":\"primitive\","
      "\"warmups\":%d,\"samples\":%d,\"rows\":[",
      kPrefix, warmups, samples);
  for (int i = 0; i < kDefaultPrefixCount; ++i) {
    const std::size_t position =
        static_cast<std::size_t>(kDefaultPrefixes[i]);
    DeviceCase device;
    std::vector<float> unused;
    int rc = setup_case(position, &device, &unused);
    std::vector<float> parent_ms;
    std::vector<float> cand_ms;
    if (rc == 0) rc = time_arm(&device, false, warmups, samples, &parent_ms);
    if (rc == 0) rc = time_arm(&device, true, warmups, samples, &cand_ms);
    const float parent = mean_ms(parent_ms);
    const float cand = mean_ms(cand_ms);
    const float saving = parent - cand;
    const bool ok = rc == 0;
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"prefix\":%zu,\"ok\":%s,\"parent_ms\":%.9g,\"candidate_ms\":%.9g,"
        "\"saving_ms\":%.9g}",
        i == 0 ? "" : ",", position, json_bool(ok),
        static_cast<double>(parent), static_cast<double>(cand),
        static_cast<double>(saving));
    device.free_all();
  }
  std::printf("],\"ok\":%s,\"keep\":false}\n", json_bool(all_ok));
  print_counts(static_cast<std::size_t>(warmups),
               static_cast<std::size_t>(samples), 2, false);
  return all_ok ? 0 : 1;
}

std::size_t session_capacity(std::size_t prefix, std::size_t outputs,
                             std::size_t requested) {
  if (requested != 0) return requested;
  return std::max(prefix + outputs + 32, kMinCapacity);
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

float percentile(std::vector<float> values, float fraction) {
  if (values.empty()) return 0.0F;
  std::sort(values.begin(), values.end());
  const float position = fraction * static_cast<float>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const float weight = position - static_cast<float>(lower);
  return values[lower] * (1.0F - weight) + values[upper] * weight;
}

float wall_ms(const std::chrono::steady_clock::time_point started) {
  return static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count());
}

qw38::Status load_model(const char* path, qw38::internal::MappedFile* mapping,
                        qw38::cuda::ResidentModel* model) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  if (status.is_ok()) status = mapping->open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, *mapping, &weights);
  }
  if (status.is_ok()) {
    status = model->upload(weights, mapping->data(), mapping->size());
  }
  return status;
}

qw38::Status run_token(const qw38::cuda::ResidentModel& model, std::size_t token,
                       qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace, float* logits,
                       float* hidden, float* elapsed,
                       qw38::cuda::SchedulerGraphs* graphs,
                       const qw38::cuda::EvalControl* control = nullptr) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, nullptr,
      qw38::cuda::PointwisePath::kFused, graphs, nullptr);
}

qw38::Status json_escape(const std::string& input, std::string* output) {
  output->clear();
  output->reserve(input.size());
  for (char ch : input) {
    if (ch == '"' || ch == '\\') {
      output->push_back('\\');
      output->push_back(ch);
    } else if (ch == '\n') {
      output->append("\\n");
    } else {
      output->push_back(ch);
    }
  }
  return qw38::Status::ok();
}

qw38::Status create_graphs(const qw38::cuda::ResidentModel& model,
                           qw38::cuda::SchedulerSession* session,
                           qw38::cuda::SchedulerWorkspace* workspace,
                           qw38::cuda::SchedulerGraphs* graphs) {
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  return graphs->create(model, workspace, session);
}

qw38::Status prefill(const qw38::cuda::ResidentModel& model,
                     const std::vector<std::size_t>& tokens, std::size_t prefix,
                     qw38::cuda::SchedulerSession* session,
                     qw38::cuda::SchedulerWorkspace* workspace,
                     qw38::cuda::SchedulerGraphs* graphs, float* logits,
                     float* hidden) {
  qw38::cuda::SyncResult sync{};
  return qw38::cuda::sync_tokens(model, tokens.data(), prefix, session,
                                 workspace, logits, qw38::internal::kVocabularySize,
                                 hidden, qw38::internal::kResidualWidth, &sync,
                                 nullptr, graphs);
}

bool exact_buffers(
    const std::vector<float>& left, const std::vector<float>& right,
    const std::array<float, qw38::internal::kResidualWidth>& hidden_left,
    const std::array<float, qw38::internal::kResidualWidth>& hidden_right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0 &&
         std::memcmp(hidden_left.data(), hidden_right.data(),
                     hidden_left.size() * sizeof(float)) == 0;
}

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    if (!context->requested) {
      context->requested = true;
      context->cancel_requested = std::chrono::steady_clock::now();
    }
    return {qw38::StatusCode::kCancelled, "opt130 graph cancellation"};
  }
  return qw38::Status::ok();
}

qw38::Status run_request_arm(const qw38::cuda::ResidentModel& model,
                             const std::vector<std::size_t>& tokens,
                             std::size_t prefix, std::size_t outputs,
                             bool candidate, ArmResult* out) {
  qw38::cuda::Opt130DenseAttentionScope attn(
      candidate ? qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open
                : qw38::cuda::kLegalDecodeAttentionHybridCrossover);
  const std::size_t capacity = session_capacity(prefix, outputs, 0);
  const auto setup_started = std::chrono::steady_clock::now();
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &session, &workspace, &graphs);
  out->setup_ms = wall_ms(setup_started);
  if (!status.is_ok()) return status;
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  const auto request_started = std::chrono::steady_clock::now();
  if (prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  out->prefill_ms = wall_ms(request_started);
  out->ttft_ms = out->setup_ms + out->prefill_ms;
  if (!status.is_ok()) return status;
  std::vector<float> latencies;
  latencies.reserve(outputs);
  const auto decode_started = std::chrono::steady_clock::now();
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float elapsed = 0.0F;
    const auto token_started = std::chrono::steady_clock::now();
    status = run_token(model, tokens[prefix + step], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs);
    if (!status.is_ok()) return status;
    latencies.push_back(wall_ms(token_started));
  }
  out->decode_only_ms = wall_ms(decode_started);
  out->request_ms = out->setup_ms + wall_ms(request_started);
  const std::size_t counted = outputs == 0 ? prefix : outputs;
  out->request_tok_s =
      out->request_ms > 0.0F
          ? static_cast<float>(counted) * 1000.0F / out->request_ms
          : 0.0F;
  out->decode_only_tok_s =
      outputs > 0 && out->decode_only_ms > 0.0F
          ? static_cast<float>(outputs) * 1000.0F / out->decode_only_ms
          : out->request_tok_s;
  out->p50_ms = percentile(latencies, 0.50F);
  out->p95_ms = percentile(latencies, 0.95F);
  const qw38::cuda::GraphLifecycleCounts counts = graphs.lifecycle_counts();
  out->graph_bytes = graphs.allocated_bytes();
  out->capture = counts.capture;
  out->instantiate = counts.instantiate;
  out->upload = counts.upload;
  out->destroy = counts.destroy;
  out->topology_recapture = counts.topology_recapture;
  return status;
}

void print_arm(const char* name, const ArmResult& arm, bool last) {
  std::printf(
      "\"%s\":{\"setup_ms\":%.9g,\"prefill_ms\":%.9g,\"decode_only_ms\":%.9g,"
      "\"request_ms\":%.9g,\"ttft_ms\":%.9g,\"decode_only_tok_s\":%.9g,"
      "\"request_tok_s\":%.9g,\"tok_s\":%.9g,\"p50_ms\":%.9g,\"p95_ms\":%.9g,"
      "\"graph_bytes\":%zu,\"capture\":%u,\"instantiate\":%u,\"upload\":%u,"
      "\"destroy\":%u,\"topology_recapture\":%u}%s",
      name, static_cast<double>(arm.setup_ms), static_cast<double>(arm.prefill_ms),
      static_cast<double>(arm.decode_only_ms),
      static_cast<double>(arm.request_ms), static_cast<double>(arm.ttft_ms),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.request_tok_s),
      static_cast<double>(arm.decode_only_tok_s),
      static_cast<double>(arm.p50_ms), static_cast<double>(arm.p95_ms),
      arm.graph_bytes, arm.capture, arm.instantiate, arm.upload, arm.destroy,
      arm.topology_recapture, last ? "" : ",");
}

int run_graph_ab(const qw38::cuda::ResidentModel& model, const Options& options) {
  const bool prefill_only = options.tokens == 0;
  const std::size_t prompt = options.prefix;
  const std::size_t outputs = prefill_only ? 0 : options.tokens;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  qw38::Status status = qw38::Status::ok();
  for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
    ArmResult discarded;
    status = run_request_arm(model, tokens, prompt, outputs, false, &discarded);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs, true, &discarded);
    }
    if (!status.is_ok()) return fail_status(status);
    std::printf(
        "quartz_warmup=%zu decode_only_tok_s=%.9g request_tok_s=%.9g "
        "independently_restored=true\n",
        warmup, static_cast<double>(discarded.decode_only_tok_s),
        static_cast<double>(discarded.request_tok_s));
  }
  std::vector<ArmResult> arm_a(options.samples);
  std::vector<ArmResult> arm_b(options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    const bool ba = (pair % 2) == 1;
    ArmResult first;
    ArmResult second;
    status = run_request_arm(model, tokens, prompt, outputs, ba, &first);
    if (status.is_ok()) {
      status = run_request_arm(model, tokens, prompt, outputs, !ba, &second);
    }
    if (!status.is_ok()) return fail_status(status);
    arm_a[pair] = ba ? second : first;
    arm_b[pair] = ba ? first : second;
    std::printf(
        "quartz_pair sample_index=%zu order=%s A_decode_only_tok_s=%.9g "
        "B_decode_only_tok_s=%.9g A_request_tok_s=%.9g B_request_tok_s=%.9g "
        "independently_restored=true same_binary=true\n",
        pair, ba ? "BA" : "AB",
        static_cast<double>(arm_a[pair].decode_only_tok_s),
        static_cast<double>(arm_b[pair].decode_only_tok_s),
        static_cast<double>(arm_a[pair].request_tok_s),
        static_cast<double>(arm_b[pair].request_tok_s));
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"workload\":\"graph-ab\","
      "\"prefix\":%zu,\"prefill\":%s,\"warmups\":%zu,\"samples\":%zu,"
      "\"tokens\":%zu,\"parent\":\"hybrid_crossover\","
      "\"candidate\":\"occupancy_vec128_open\","
      "\"metrics\":[\"decode_only\",\"complete_request\"],"
      "\"observed_warmups\":%zu,\"observed_samples\":%zu,"
      "\"observed_candidates\":2,\"independently_restored\":true,"
      "\"same_binary\":true,\"keep\":false,\"pairs\":[",
      kPrefix, prompt, json_bool(prefill_only), options.warmups, options.samples,
      outputs, options.warmups, options.samples);
  for (std::size_t pair = 0; pair < options.samples; ++pair) {
    if (pair != 0) std::printf(",");
    std::printf("{\"sample_index\":%zu,\"order\":\"%s\",", pair,
                (pair % 2) == 1 ? "BA" : "AB");
    print_arm("A", arm_a[pair], false);
    print_arm("B", arm_b[pair], true);
    std::printf("}");
  }
  std::printf("]}\n");
  print_counts(options.warmups, options.samples, 2, false);
  return 0;
}

int run_same_math(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  qw38::cuda::Opt130DenseAttentionScope attn(
      qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open);
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 8 : options.tokens;
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  std::vector<std::vector<float>> eager_token_logits;
  std::vector<std::array<float, qw38::internal::kResidualWidth>> eager_token_hidden;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession eager_session;
    qw38::cuda::SchedulerWorkspace eager_workspace;
    status = eager_session.create(capacity);
    if (status.is_ok()) status = eager_workspace.create(capacity);
    std::vector<float> eager_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                       nullptr, eager_logits.data(), eager_hidden.data());
    }
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float eager_ms = 0.0F;
      status = run_token(model, tokens[prefix + step], &eager_session,
                         &eager_workspace, eager_logits.data(),
                         eager_hidden.data(), &eager_ms, nullptr);
      if (status.is_ok()) {
        eager_token_logits.push_back(eager_logits);
        eager_token_hidden.push_back(eager_hidden);
      }
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  bool exact = status.is_ok() && eager_token_logits.size() == outputs;
  std::size_t matched = 0;
  {
    qw38::cuda::SchedulerSession graph_session;
    qw38::cuda::SchedulerWorkspace graph_workspace;
    qw38::cuda::SchedulerGraphs segment_graphs;
    if (status.is_ok()) status = graph_session.create(capacity);
    if (status.is_ok()) status = graph_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &graph_session, &graph_workspace,
                             &segment_graphs);
    }
    std::vector<float> graph_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float graph_ms = 0.0F;
      status = run_token(model, tokens[prefix + step], &graph_session,
                         &graph_workspace, graph_logits.data(),
                         graph_hidden.data(), &graph_ms, &segment_graphs);
      if (status.is_ok() && step < eager_token_logits.size() &&
          exact_buffers(eager_token_logits[step], graph_logits,
                        eager_token_hidden[step], graph_hidden)) {
        ++matched;
      } else {
        exact = false;
      }
    }
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool ok = status.is_ok() && exact && matched == outputs;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"workload\":\"same-math\","
      "\"ok\":%s,\"exact\":%s,\"matched\":%zu,\"outputs\":%zu,\"prefix\":%zu,"
      "\"eager_fallback\":true,\"message\":\"%s\",\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), matched, outputs, prefix,
      message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_cancellation(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  qw38::cuda::Opt130DenseAttentionScope attn(
      qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open);
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  bool all_ok = true;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\","
      "\"workload\":\"cancellation\",\"cadence\":\"eight_layer_segment\","
      "\"cases\":[",
      kPrefix);
  const char* names[] = {"mid_segment", "commit"};
  const std::size_t stops[] = {1, 8};
  for (int case_index = 0; case_index < 2; ++case_index) {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::Status status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    const std::size_t frontier_before = session.frontier();
    PollContext poll_context{};
    poll_context.stop_after = stops[case_index];
    qw38::cuda::EvalControl control{};
    control.poll = poll_stop;
    control.context = &poll_context;
    float elapsed = 0.0F;
    qw38::Status token_status = qw38::Status::ok();
    const auto started = std::chrono::steady_clock::now();
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      token_status =
          run_token(model, tokens[prefix], &session, &workspace, logits.data(),
                    hidden.data(), &elapsed, &graphs, &control);
    }
    const float response_ms =
        poll_context.requested ? wall_ms(poll_context.cancel_requested)
                               : wall_ms(started);
    const bool cancelled = token_status.code() == qw38::StatusCode::kCancelled;
    const bool preserved = session.frontier() == frontier_before;
    const bool ok = status.is_ok() && cancelled && preserved &&
                    poll_context.calls >= 1;
    all_ok = all_ok && ok;
    std::printf(
        "%s{\"name\":\"%s\",\"ok\":%s,\"cancelled\":%s,\"frontier_preserved\":%s,"
        "\"poll_calls\":%zu,\"stop_after\":%zu,\"cancel_response_ms\":%.9g,"
        "\"atomic_commit\":%s}",
        case_index == 0 ? "" : ",", names[case_index], json_bool(ok),
        json_bool(cancelled), json_bool(preserved), poll_context.calls,
        stops[case_index], static_cast<double>(response_ms),
        json_bool(preserved));
  }
  std::printf("],\"ok\":%s,\"keep\":false}\n", json_bool(all_ok));
  print_counts(0, 1, 1, false);
  return all_ok ? 0 : 1;
}

int run_handoff(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 1, options.capacity);
  std::vector<std::size_t> tokens(prefix + 1);
  fill_tokens(&tokens);
  std::vector<float> parent_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> parent_hidden{};
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::Opt130DenseAttentionScope parent(
        qw38::cuda::kLegalDecodeAttentionHybridCrossover);
    qw38::cuda::SchedulerSession parent_session;
    qw38::cuda::SchedulerWorkspace parent_workspace;
    qw38::cuda::SchedulerGraphs parent_graphs;
    status = parent_session.create(capacity);
    if (status.is_ok()) status = parent_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &parent_session, &parent_workspace,
                             &parent_graphs);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &parent_session, &parent_workspace,
                       &parent_graphs, parent_logits.data(),
                       parent_hidden.data());
    }
    float parent_ms = 0.0F;
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix], &parent_session,
                         &parent_workspace, parent_logits.data(),
                         parent_hidden.data(), &parent_ms, &parent_graphs);
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::vector<float> cand_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> cand_hidden{};
  {
    qw38::cuda::Opt130DenseAttentionScope cand(
        qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open);
    qw38::cuda::SchedulerSession cand_session;
    qw38::cuda::SchedulerWorkspace cand_workspace;
    qw38::cuda::SchedulerGraphs cand_graphs;
    if (status.is_ok()) status = cand_session.create(capacity);
    if (status.is_ok()) status = cand_workspace.create(capacity);
    if (status.is_ok()) {
      status =
          create_graphs(model, &cand_session, &cand_workspace, &cand_graphs);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &cand_session, &cand_workspace,
                       &cand_graphs, cand_logits.data(), cand_hidden.data());
    }
    float cand_ms = 0.0F;
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix], &cand_session, &cand_workspace,
                         cand_logits.data(), cand_hidden.data(), &cand_ms,
                         &cand_graphs);
    }
  }
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  compare_outputs(parent_logits, cand_logits, &max_abs, &nonfinite);
  const bool finite = nonfinite == 0;
  const bool ok = status.is_ok() && finite;
  std::string message;
  json_escape(status.message(), &message);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\",\"workload\":\"handoff\","
      "\"ok\":%s,\"finite\":%s,\"max_abs\":%.9g,\"prefix\":%zu,"
      "\"short_output_tokens\":1,\"message\":\"%s\",\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(finite), static_cast<double>(max_abs),
      prefix, message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_state_memory(const qw38::cuda::ResidentModel& model,
                     const Options& options) {
  qw38::cuda::Opt130DenseAttentionScope attn(
      qw38::cuda::kLegalDecodeAttentionOccupancyVec128Open);
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession eager_session;
    qw38::cuda::SchedulerWorkspace eager_workspace;
    status = eager_session.create(capacity);
    if (status.is_ok()) status = eager_workspace.create(capacity);
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                       nullptr, eager_logits.data(), eager_hidden.data());
    }
    float eager_ms = 0.0F;
    if (status.is_ok()) {
      status = run_token(model, tokens[prefix], &eager_session, &eager_workspace,
                         eager_logits.data(), eager_hidden.data(), &eager_ms,
                         nullptr);
    }
  }
  cudaDeviceSynchronize();
  cudaGetLastError();
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::size_t graph_bytes = 0;
  std::size_t session_bytes = 0;
  std::size_t workspace_bytes = 0;
  {
    qw38::cuda::SchedulerSession graph_session;
    qw38::cuda::SchedulerWorkspace graph_workspace;
    qw38::cuda::SchedulerGraphs segment_graphs;
    if (status.is_ok()) status = graph_session.create(capacity);
    if (status.is_ok()) status = graph_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &graph_session, &graph_workspace,
                             &segment_graphs);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
    float graph_ms = 0.0F;
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix], &graph_session, &graph_workspace,
                         graph_logits.data(), graph_hidden.data(), &graph_ms,
                         &segment_graphs);
    }
    graph_bytes = segment_graphs.allocated_bytes();
    session_bytes = graph_session.allocated_bytes();
    workspace_bytes = graph_workspace.allocated_bytes();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool equal = status.is_ok() && exact_buffers(eager_logits, graph_logits,
                                                     eager_hidden, graph_hidden);
  const std::size_t peak = session_bytes + workspace_bytes + graph_bytes;
  const bool ok = status.is_ok() && equal;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-130\","
      "\"workload\":\"state-memory\",\"ok\":%s,\"state_equals\":%s,"
      "\"graph_bytes\":%zu,\"session_bytes\":%zu,\"workspace_bytes\":%zu,"
      "\"peak_bytes\":%zu,\"eager_fallback\":true,\"message\":\"%s\","
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(equal), graph_bytes, session_bytes,
      workspace_bytes, peak, message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

bool needs_model(const char* workload) {
  return std::strcmp(workload, "same-math") == 0 ||
         std::strcmp(workload, "cancellation") == 0 ||
         std::strcmp(workload, "handoff") == 0 ||
         std::strcmp(workload, "graph-ab") == 0 ||
         std::strcmp(workload, "state-memory") == 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  const int model_index = parse_args(argc, argv, &options);
  if (model_index == -2) return usage(argv[0]);
  if (std::strcmp(options.workload, "inspect") == 0) return run_inspect();
  if (std::strcmp(options.workload, "identity") == 0) return run_identity();
  if (std::strcmp(options.workload, "primitive") == 0) {
    return run_primitive(options);
  }
  if (!needs_model(options.workload)) return usage(argv[0]);
  if (model_index < 0) return usage(argv[0]);
  qw38::internal::MappedFile mapping;
  qw38::cuda::ResidentModel model;
  const qw38::Status status = load_model(argv[model_index], &mapping, &model);
  if (!status.is_ok()) return fail_status(status);
  if (std::strcmp(options.workload, "same-math") == 0) {
    return run_same_math(model, options);
  }
  if (std::strcmp(options.workload, "cancellation") == 0) {
    return run_cancellation(model, options);
  }
  if (std::strcmp(options.workload, "handoff") == 0) {
    return run_handoff(model, options);
  }
  if (std::strcmp(options.workload, "graph-ab") == 0) {
    return run_graph_ab(model, options);
  }
  if (std::strcmp(options.workload, "state-memory") == 0) {
    return run_state_memory(model, options);
  }
  return usage(argv[0]);
}
