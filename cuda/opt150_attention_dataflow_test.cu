#include "attention_decode.h"
#include "attention_decode_path.cuh"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT150_ATTENTION_DATAFLOW_RESULT=";
constexpr char kCounts[] = "QW38_OPT150_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 33024};
constexpr int kProductionCapacity = 131072;
constexpr int kAttentionLayers = 16;
constexpr int kDispatchPositions[] = {0,    128,  1023, 1024, 4096,  4097, 6144,
                                      7935, 7936, 8191, 8192, 8193, 32768};
constexpr int kDispatchCount = 13;
constexpr int kNumericalPositions[] = {128, 1024, 8192};
constexpr int kNumericalCount = 3;
constexpr int kLayers[] = {3, 7, 63};
constexpr int kLayerCount = 3;
constexpr int kTimingPositions[] = {2048, 8192, 32768};
constexpr int kTimingCount = 3;
constexpr float kFp64Abs = 5.0e-3F;
constexpr float kRmsEps = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;
constexpr int kFattnStride = 256;
constexpr int kAllocParts = 256;

struct Options final {
  const char* workload = "identity";
  int warmups = 1;
  int samples = 3;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload identity|correctness|replay] "
               "[--warmups N] [--samples N]\n",
               argv0);
  return 2;
}

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->warmups)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->samples)) return usage(argv[0]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

int llama_padded_nkv(int n_kv) {
  if (n_kv < kFattnStride) return kFattnStride;
  return ((n_kv + kFattnStride - 1) / kFattnStride) * kFattnStride;
}

const char* llama_selected_kernel(int padded) {
  if (padded % kFattnStride != 0) return "fattn-mma-f16";
  if (padded >= 8192) return "fattn-mma-f16_ncols1=1_ncols2=8";
  return "flash_attn_ext_vec<256,1>";
}

float unit(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  return (static_cast<float>(x & 0x00FFFFFFU) / 8388608.0F - 1.0F) * 0.35F;
}

void fill_case(std::size_t position, std::uint32_t seed,
               std::vector<float>* query, std::vector<float>* key,
               std::vector<float>* value, std::vector<float>* gate,
               std::vector<float>* qscale, std::vector<float>* kscale,
               std::vector<__nv_bfloat16>* committed_key,
               std::vector<__nv_bfloat16>* committed_value) {
  const std::size_t qn = qw38::cuda::attention_query_values(kConfig);
  const std::size_t kn = qw38::cuda::attention_kv_row_values(kConfig);
  const std::size_t cache = qw38::cuda::attention_cache_values(kConfig);
  query->resize(qn);
  key->resize(kn);
  value->resize(kn);
  gate->resize(qn);
  qscale->assign(kConfig.head_width, 1.0F);
  kscale->assign(kConfig.head_width, 1.0F);
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
  for (std::uint32_t dim = 0; dim < kConfig.head_width; ++dim) {
    (*qscale)[dim] = 0.85F + 0.002F * static_cast<float>(dim);
    (*kscale)[dim] = 0.90F + 0.0015F * static_cast<float>(dim);
  }
  for (std::size_t token = 0; token < position; ++token) {
    for (std::uint32_t kv_head = 0; kv_head < kConfig.kv_heads; ++kv_head) {
      for (std::uint32_t dim = 0; dim < kConfig.head_width; ++dim) {
        const std::size_t phys = qw38::cuda::attention_kv_physical_index(
            token, kv_head, dim, kConfig.capacity, kConfig.head_width);
        (*committed_key)[phys] = __float2bfloat16_rn(
            unit(static_cast<std::uint32_t>(phys), seed + 101U));
        (*committed_value)[phys] = __float2bfloat16_rn(
            unit(static_cast<std::uint32_t>(phys), seed + 131U));
      }
    }
  }
}

void host_rms_rope(const float* input, const float* scale, std::uint32_t heads,
                   std::uint32_t width, std::uint32_t rotary,
                   std::size_t position, std::vector<float>* out) {
  out->resize(static_cast<std::size_t>(heads) * width);
  for (std::uint32_t head = 0; head < heads; ++head) {
    const std::size_t base = static_cast<std::size_t>(head) * width;
    float sum = 0.0F;
    for (std::uint32_t dim = 0; dim < width; ++dim) {
      sum += input[base + dim] * input[base + dim];
    }
    const float inv =
        1.0F / std::sqrt(sum / static_cast<float>(width) + kRmsEps);
    for (std::uint32_t dim = 0; dim < width; ++dim) {
      (*out)[base + dim] = input[base + dim] * inv * scale[dim];
    }
    const std::uint32_t half = rotary / 2;
    std::vector<float> orig(out->begin() + static_cast<std::ptrdiff_t>(base),
                            out->begin() +
                                static_cast<std::ptrdiff_t>(base + rotary));
    for (std::uint32_t pair = 0; pair < half; ++pair) {
      const float exponent =
          static_cast<float>(pair * 2) / static_cast<float>(rotary);
      const float angle =
          static_cast<float>(position) / std::pow(kRopeTheta, exponent);
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      (*out)[base + pair] = orig[pair] * c - orig[half + pair] * s;
      (*out)[base + half + pair] = orig[half + pair] * c + orig[pair] * s;
    }
  }
}

void fp64_sample(std::size_t position, const std::vector<float>& nq,
                 const std::vector<float>& gate,
                 const std::vector<__nv_bfloat16>& committed_key,
                 const std::vector<__nv_bfloat16>& committed_value,
                 const std::vector<__nv_bfloat16>& cand_key,
                 const std::vector<__nv_bfloat16>& cand_value,
                 std::vector<double>* sampled) {
  const std::uint32_t group = kConfig.query_heads / kConfig.kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kConfig.head_width));
  const int heads[] = {0, 1, 12, 23};
  const int dims[] = {0, 1, 32, 64, 128, 200, 254, 255};
  sampled->clear();
  for (int hi = 0; hi < 4; ++hi) {
    const std::uint32_t qh = static_cast<std::uint32_t>(heads[hi]);
    const std::uint32_t kv = qh / group;
    const std::size_t qbase =
        static_cast<std::size_t>(qh) * kConfig.head_width;
    std::vector<double> scores(position + 1);
    double maximum = -1.0e300;
    for (std::size_t token = 0; token <= position; ++token) {
      double score = 0.0;
      for (std::uint32_t dim = 0; dim < kConfig.head_width; ++dim) {
        const float kitem =
            token < position
                ? __bfloat162float(committed_key[qw38::cuda::attention_kv_physical_index(
                      token, kv, dim, kConfig.capacity, kConfig.head_width)])
                : __bfloat162float(cand_key[static_cast<std::size_t>(kv) *
                                                kConfig.head_width +
                                            dim]);
        score += static_cast<double>(nq[qbase + dim]) *
                 static_cast<double>(kitem);
      }
      scores[token] = score * scale;
      maximum = std::max(maximum, scores[token]);
    }
    double denom = 0.0;
    for (std::size_t token = 0; token <= position; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      denom += scores[token];
    }
    for (int di = 0; di < 8; ++di) {
      const std::uint32_t dim = static_cast<std::uint32_t>(dims[di]);
      double acc = 0.0;
      for (std::size_t token = 0; token <= position; ++token) {
        const float vitem =
            token < position
                ? __bfloat162float(committed_value[qw38::cuda::attention_kv_physical_index(
                      token, kv, dim, kConfig.capacity, kConfig.head_width)])
                : __bfloat162float(cand_value[static_cast<std::size_t>(kv) *
                                                  kConfig.head_width +
                                              dim]);
        acc += scores[token] * static_cast<double>(vitem);
      }
      const double ungated = denom == 0.0 ? 0.0 : acc / denom;
      const double g = static_cast<double>(gate[qbase + dim]);
      const double sig = g >= 0.0 ? 1.0 / (1.0 + std::exp(-g))
                                  : std::exp(g) / (1.0 + std::exp(g));
      sampled->push_back(ungated * sig);
    }
  }
}

struct Buffers {
  float* query = nullptr;
  float* key = nullptr;
  float* value = nullptr;
  float* gate = nullptr;
  float* qscale = nullptr;
  float* kscale = nullptr;
  float* nq = nullptr;
  float* nk = nullptr;
  float* scores = nullptr;
  float* output = nullptr;
  float* partial = nullptr;
  float* meta = nullptr;
  __nv_bfloat16* committed_key = nullptr;
  __nv_bfloat16* committed_value = nullptr;
  __nv_bfloat16* candidate_key = nullptr;
  __nv_bfloat16* candidate_value = nullptr;
};

void free_buffers(Buffers* buffers) {
  cudaFree(buffers->query);
  cudaFree(buffers->key);
  cudaFree(buffers->value);
  cudaFree(buffers->gate);
  cudaFree(buffers->qscale);
  cudaFree(buffers->kscale);
  cudaFree(buffers->nq);
  cudaFree(buffers->nk);
  cudaFree(buffers->scores);
  cudaFree(buffers->output);
  cudaFree(buffers->partial);
  cudaFree(buffers->meta);
  cudaFree(buffers->committed_key);
  cudaFree(buffers->committed_value);
  cudaFree(buffers->candidate_key);
  cudaFree(buffers->candidate_value);
}

cudaError_t alloc_buffers(Buffers* buffers) {
  const std::size_t qn = qw38::cuda::attention_query_values(kConfig);
  const std::size_t kn = qw38::cuda::attention_kv_row_values(kConfig);
  const std::size_t cache = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t vkq = qw38::cuda::decode_kv_partial_vkq_values(kAllocParts);
  const std::size_t meta = qw38::cuda::decode_kv_partial_meta_values(kAllocParts);
  cudaError_t error = cudaMalloc(&buffers->query, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->key, kn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->value, kn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->gate, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->qscale, kConfig.head_width * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->kscale, kConfig.head_width * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->nq, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->nk, kn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->scores, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->output, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->partial, vkq * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->meta, meta * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->committed_key, cache * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error =
        cudaMalloc(&buffers->committed_value, cache * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->candidate_key, kn * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->candidate_value, kn * sizeof(__nv_bfloat16));
  return error;
}

cudaError_t upload(const Buffers& buffers, const std::vector<float>& query,
                   const std::vector<float>& key,
                   const std::vector<float>& value,
                   const std::vector<float>& gate,
                   const std::vector<float>& qscale,
                   const std::vector<float>& kscale,
                   const std::vector<__nv_bfloat16>& committed_key,
                   const std::vector<__nv_bfloat16>& committed_value) {
  cudaError_t error = cudaMemcpy(buffers.query, query.data(),
                                 query.size() * sizeof(float),
                                 cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.key, key.data(), key.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.value, value.data(),
                       value.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.gate, gate.data(), gate.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.qscale, qscale.data(),
                       qscale.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.kscale, kscale.data(),
                       kscale.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.committed_key, committed_key.data(),
                       committed_key.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.committed_value, committed_value.data(),
                       committed_value.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  return error;
}

cudaError_t launch_shipping(const Buffers& buffers, std::size_t position) {
  qw38::cuda::opt137_prepare_mma_runtime();
  qw38::cuda::AttentionCache committed{buffers.committed_key,
                                       buffers.committed_value};
  qw38::cuda::AttentionCache cand{buffers.candidate_key,
                                  buffers.candidate_value};
  const int n_parts = qw38::cuda::decode_kv_parts_for_position(position);
  return qw38::cuda::launch_attention_prepare_partitioned(
      kConfig, position, buffers.query, buffers.key, buffers.value,
      buffers.qscale, buffers.kscale, buffers.gate, committed, cand,
      buffers.nq, buffers.nk, buffers.scores, buffers.output, buffers.partial,
      buffers.meta, n_parts, nullptr);
}

void print_counts(int warmups, int samples, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-150\",\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":1,"
      "\"observed_shapes\":%d,\"keep\":%s}\n",
      kCounts, warmups, samples, kTimingCount, json_bool(keep));
}

int run_identity() {
  cudaDeviceProp prop{};
  cudaError_t error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) return fail_cuda("props", error);
  std::printf("%s{", kPrefix);
  std::printf(
      "\"schema_version\":1,\"task\":\"OPT-150\",\"ok\":true,"
      "\"pass\":true,\"workload\":\"identity\","
      "\"gpu\":\"%s\",\"allocated_capacity\":%d,\"probe_capacity\":%u,"
      "\"crossover\":%d,\"verified_max\":%d,\"mma_threshold\":%d,"
      "\"flash_vec\":%s,\"opt137_mma\":%s,\"n_parts_pin\":%d,"
      "\"gguf_sha256\":\"%s\",\"llama_revision\":\"%s\","
      "\"dispatch\":[",
      prop.name, kProductionCapacity, kConfig.capacity,
      qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold,
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax,
      qw38::cuda::kOpt137MmaThreshold,
      json_bool(qw38::cuda::kSelectedDecodeAttentionFlashVec),
      json_bool(qw38::cuda::kSelectedOpt137DenseMma),
      qw38::cuda::kSelectedVec128NParts, kGgufSha, kLlamaRev);
  for (int i = 0; i < kDispatchCount; ++i) {
    const std::size_t position =
        static_cast<std::size_t>(kDispatchPositions[i]);
    const int visible = static_cast<int>(position + 1);
    const int padded = llama_padded_nkv(visible);
    if (i) std::printf(",");
    std::printf(
        "{\"position\":%zu,\"path\":\"%s\",\"launch\":\"%s\","
        "\"topology\":%d,\"n_parts\":%d,\"visible\":%d,"
        "\"llama_padded\":%d,\"llama_kernel\":\"%s\"}",
        position, qw38::cuda::effective_decode_attention_dispatch_path(position),
        qw38::cuda::decode_attention_vec128_path_for_position(position),
        qw38::cuda::decode_graph_topology_index(position),
        qw38::cuda::decode_kv_parts_for_position(position), visible, padded,
        llama_selected_kernel(padded));
  }
  std::printf("],\"claims_throughput\":false}\n");
  print_counts(0, 1, false);
  return 0;
}

bool expected_path(std::size_t position, const char* path) {
  if (position >= 8192) {
    return std::strcmp(path, "dense_bf16_tile_f16_mma_decode_v1") == 0;
  }
  if (position >= 1024 && position <= 4096) {
    return std::strcmp(path, "decode_attention_flash_vec_v1") == 0;
  }
  return std::strcmp(path, "warp_query") == 0;
}

int run_one(Buffers* buffers, std::size_t position, std::uint32_t seed,
            bool sample_fp64) {
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(position, seed, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  cudaError_t error = upload(*buffers, query, key, value, gate, qscale, kscale,
                             committed_key, committed_value);
  if (error == cudaSuccess) error = launch_shipping(*buffers, position);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> output(qw38::cuda::attention_query_values(kConfig));
  if (error == cudaSuccess)
    error = cudaMemcpy(output.data(), buffers->output,
                       output.size() * sizeof(float), cudaMemcpyDeviceToHost);
  std::vector<__nv_bfloat16> cand_key(
      qw38::cuda::attention_kv_row_values(kConfig));
  std::vector<__nv_bfloat16> cand_value(cand_key.size());
  if (error == cudaSuccess)
    error = cudaMemcpy(cand_key.data(), buffers->candidate_key,
                       cand_key.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(cand_value.data(), buffers->candidate_value,
                       cand_value.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("run_one", error);
  const char* path =
      qw38::cuda::effective_decode_attention_dispatch_path(position);
  const char* launch = qw38::cuda::last_decode_query_prep_launch_variant();
  std::size_t nonfinite = 0;
  for (float value : output) {
    if (!std::isfinite(value)) ++nonfinite;
  }
  float fp64_abs = 0.0F;
  if (sample_fp64) {
    std::vector<float> nq;
    host_rms_rope(query.data(), qscale.data(), kConfig.query_heads,
                  kConfig.head_width, kConfig.rotary_width, position, &nq);
    std::vector<double> sampled;
    fp64_sample(position, nq, gate, committed_key, committed_value, cand_key,
                cand_value, &sampled);
    const int heads[] = {0, 1, 12, 23};
    const int dims[] = {0, 1, 32, 64, 128, 200, 254, 255};
    std::size_t si = 0;
    for (int hi = 0; hi < 4; ++hi) {
      for (int di = 0; di < 8; ++di, ++si) {
        const std::size_t index =
            static_cast<std::size_t>(heads[hi]) * kConfig.head_width +
            static_cast<std::size_t>(dims[di]);
        fp64_abs = std::max(
            fp64_abs,
            std::fabs(output[index] - static_cast<float>(sampled[si])));
      }
    }
  }
  const bool dispatch_ok = expected_path(position, path);
  const bool fp64_ok = !sample_fp64 || fp64_abs <= kFp64Abs;
  const bool pass = dispatch_ok && nonfinite == 0 && fp64_ok;
  std::printf(
      "%s{\"kind\":\"dispatch\",\"position\":%zu,\"path\":\"%s\","
      "\"launch\":\"%s\",\"n_parts\":%d,\"topology\":%d,"
      "\"visible\":%zu,\"llama_padded\":%d,\"llama_kernel\":\"%s\","
      "\"nonfinite\":%zu,\"fp64_abs\":%.9g,\"fp64_sampled\":%s,"
      "\"dispatch_ok\":%s,\"ok\":%s}\n",
      kPrefix, position, path, launch,
      qw38::cuda::decode_kv_parts_for_position(position),
      qw38::cuda::decode_graph_topology_index(position), position + 1,
      llama_padded_nkv(static_cast<int>(position + 1)),
      llama_selected_kernel(llama_padded_nkv(static_cast<int>(position + 1))),
      nonfinite, static_cast<double>(fp64_abs), json_bool(sample_fp64),
      json_bool(dispatch_ok), json_bool(pass));
  return pass ? 0 : 1;
}

int run_correctness(Buffers* buffers) {
  int rc = 0;
  bool dispatch_ok = true;
  bool finite = true;
  bool fp64_ok = true;
  for (int i = 0; i < kDispatchCount; ++i) {
    const std::size_t position =
        static_cast<std::size_t>(kDispatchPositions[i]);
    const int local = run_one(buffers, position, 17U + static_cast<std::uint32_t>(i),
                              false);
    rc |= local;
    if (local != 0) dispatch_ok = false;
  }
  for (int li = 0; li < kLayerCount; ++li) {
    for (int pi = 0; pi < kNumericalCount; ++pi) {
      const std::size_t position =
          static_cast<std::size_t>(kNumericalPositions[pi]);
      const std::uint32_t seed =
          static_cast<std::uint32_t>(kLayers[li]) * 10007U + 89U;
      const int local = run_one(buffers, position, seed, true);
      rc |= local;
      if (local != 0) {
        fp64_ok = false;
        finite = false;
      }
    }
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-150\",\"ok\":%s,\"pass\":%s,"
      "\"workload\":\"correctness\",\"dispatch_ok\":%s,\"finite\":%s,"
      "\"fp64_ok\":%s,\"positions\":13,\"numerical_cases\":9,"
      "\"claims_throughput\":false}\n",
      kPrefix, json_bool(rc == 0), json_bool(rc == 0), json_bool(dispatch_ok),
      json_bool(finite), json_bool(fp64_ok));
  print_counts(0, 1, false);
  return rc;
}

int time_layers(Buffers* buffers, std::size_t position, int layers,
                float* out_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  error = cudaEventRecord(start, nullptr);
  for (int layer = 0; layer < layers && error == cudaSuccess; ++layer) {
    error = launch_shipping(*buffers, position);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(out_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (error != cudaSuccess) return fail_cuda("time_layers", error);
  return 0;
}

int run_replay(Buffers* buffers, int warmups, int samples) {
  int rc = 0;
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-150\",\"ok\":true,"
              "\"pass\":true,\"workload\":\"replay\",\"shapes\":[",
              kPrefix);
  for (int si = 0; si < kTimingCount; ++si) {
    const std::size_t position =
        static_cast<std::size_t>(kTimingPositions[si]);
    std::vector<float> query, key, value, gate, qscale, kscale;
    std::vector<__nv_bfloat16> committed_key, committed_value;
    fill_case(position, 16U + static_cast<std::uint32_t>(si), &query, &key,
              &value, &gate, &qscale, &kscale, &committed_key,
              &committed_value);
    cudaError_t error =
        upload(*buffers, query, key, value, gate, qscale, kscale,
               committed_key, committed_value);
    if (error != cudaSuccess) return fail_cuda("replay upload", error);
    float ignore = 0.0F;
    for (int warm = 0; warm < warmups; ++warm) {
      rc |= time_layers(buffers, position, kAttentionLayers, &ignore);
    }
    std::vector<float> quartz_ms(static_cast<std::size_t>(samples), 0.0F);
    for (int sample = 0; sample < samples; ++sample) {
      const std::uint32_t seed =
          100U + static_cast<std::uint32_t>(si * 16 + sample);
      fill_case(position, seed, &query, &key, &value, &gate, &qscale, &kscale,
                &committed_key, &committed_value);
      error = upload(*buffers, query, key, value, gate, qscale, kscale,
                     committed_key, committed_value);
      if (error != cudaSuccess) return fail_cuda("replay rotate", error);
      rc |= time_layers(buffers, position, kAttentionLayers, &quartz_ms[static_cast<std::size_t>(sample)]);
    }
    float sum = 0.0F;
    for (float item : quartz_ms) sum += item;
    const float mean = samples > 0 ? sum / static_cast<float>(samples) : 0.0F;
    const int padded = llama_padded_nkv(static_cast<int>(position + 1));
    if (si) std::printf(",");
    std::printf(
        "{\"shape\":\"D%zu\",\"position\":%zu,\"quartz_mean_ms\":%.9g,"
        "\"quartz_round_ms\":[",
        position, position, static_cast<double>(mean));
    for (int sample = 0; sample < samples; ++sample) {
      if (sample) std::printf(",");
      std::printf("%.9g",
                  static_cast<double>(quartz_ms[static_cast<std::size_t>(sample)]));
    }
    std::printf(
        "],\"llama_enclosing_mean_ms\":null,\"adapter_mean_ms\":null,"
        "\"llama_native_or_selected_ms\":null,"
        "\"llama_selected_kernel\":\"%s\","
        "\"llama_vec_is_selected\":%s,\"path\":\"%s\","
        "\"attention_layers\":%d,\"warmups\":%d,\"samples\":%d,"
        "\"adapter_charged_separately\":true}",
        llama_selected_kernel(padded),
        json_bool(padded < 8192),
        qw38::cuda::effective_decode_attention_dispatch_path(position),
        kAttentionLayers, warmups, samples);
  }
  std::printf("],\"claims_throughput\":false}\n");
  print_counts(warmups, samples, false);
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "%s\n",
                 "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
                 "screen, or acceptance");
    return 2;
  }
  Options options;
  const int parsed = parse_args(argc, argv, &options);
  if (parsed != 0) return parsed;
  if (std::strcmp(options.workload, "identity") == 0) {
    return run_identity();
  }
  Buffers buffers{};
  const cudaError_t error = alloc_buffers(&buffers);
  if (error != cudaSuccess) {
    free_buffers(&buffers);
    return fail_cuda("alloc", error);
  }
  int rc = 0;
  if (std::strcmp(options.workload, "correctness") == 0) {
    rc = run_correctness(&buffers);
  } else if (std::strcmp(options.workload, "replay") == 0) {
    rc = run_replay(&buffers, options.warmups, options.samples);
  } else {
    free_buffers(&buffers);
    return usage(argv[0]);
  }
  free_buffers(&buffers);
  return rc;
}
