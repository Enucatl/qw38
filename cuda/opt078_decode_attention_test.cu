#include "attention_decode.h"
#include "scheduler_primitives.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT078_DECODE_ATTENTION_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kParts = 16;
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kRotary = 64;
constexpr float kRmsEps = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;

struct Options final {
  const char* workload = nullptr;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|screen|acceptance]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* default_workload(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kScreen) return "screen";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "acceptance";
  return "correctness";
}

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

float unit(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  return (static_cast<float>(x & 0x00FFFFFFU) / 8388608.0F - 1.0F) * 0.35F;
}

std::uint64_t fnv(const float* data, std::size_t count) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, data + index, sizeof(bits));
    hash ^= bits;
    hash *= 1099511628211ULL;
  }
  return hash;
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
    const float inv = 1.0F / std::sqrt(sum / static_cast<float>(width) + kRmsEps);
    for (std::uint32_t dim = 0; dim < width; ++dim) {
      (*out)[base + dim] = input[base + dim] * inv * scale[dim];
    }
    const std::uint32_t half = rotary / 2;
    std::vector<float> orig(out->begin() + static_cast<std::ptrdiff_t>(base),
                            out->begin() + static_cast<std::ptrdiff_t>(base + rotary));
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

void fp64_sample(const qw38::cuda::AttentionConfig& config, std::size_t position,
                 const std::vector<float>& nq, const std::vector<float>& nk,
                 const std::vector<float>& value, const std::vector<float>& gate,
                 const std::vector<__nv_bfloat16>& committed_key,
                 const std::vector<__nv_bfloat16>& committed_value,
                 const std::vector<__nv_bfloat16>& cand_key,
                 const std::vector<__nv_bfloat16>& cand_value,
                 std::vector<double>* sampled, const int* heads,
                 const int* dims, int n_heads, int n_dims) {
  (void)nk;
  (void)value;
  const std::uint32_t group = config.query_heads / config.kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(config.head_width));
  sampled->clear();
  for (int hi = 0; hi < n_heads; ++hi) {
    const std::uint32_t qh = static_cast<std::uint32_t>(heads[hi]);
    const std::uint32_t kv = qh / group;
    const std::size_t qbase = static_cast<std::size_t>(qh) * config.head_width;
    std::vector<double> scores(position + 1);
    double maximum = -1.0e300;
    for (std::size_t token = 0; token <= position; ++token) {
      double score = 0.0;
      for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
        const float kitem =
            token < position
                ? __bfloat162float(committed_key[qw38::cuda::attention_kv_physical_index(
                      token, kv, dim, config.capacity, config.head_width)])
                : __bfloat162float(cand_key[static_cast<std::size_t>(kv) *
                                                config.head_width +
                                            dim]);
        score += static_cast<double>(nq[qbase + dim]) * static_cast<double>(kitem);
      }
      scores[token] = score * scale;
      maximum = std::max(maximum, scores[token]);
    }
    double denom = 0.0;
    for (std::size_t token = 0; token <= position; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      denom += scores[token];
    }
    for (int di = 0; di < n_dims; ++di) {
      const std::uint32_t dim = static_cast<std::uint32_t>(dims[di]);
      double acc = 0.0;
      for (std::size_t token = 0; token <= position; ++token) {
        const float vitem =
            token < position
                ? __bfloat162float(committed_value[qw38::cuda::attention_kv_physical_index(
                      token, kv, dim, config.capacity, config.head_width)])
                : __bfloat162float(cand_value[static_cast<std::size_t>(kv) *
                                                  config.head_width +
                                              dim]);
        acc += scores[token] * static_cast<double>(vitem);
      }
      const double ungated = denom == 0.0 ? 0.0 : acc / denom;
      const double g = static_cast<double>(gate[qbase + dim]);
      const double sig =
          g >= 0.0 ? 1.0 / (1.0 + std::exp(-g)) : std::exp(g) / (1.0 + std::exp(g));
      sampled->push_back(ungated * sig);
    }
  }
}

struct LaunchResult {
  std::vector<float> output;
  std::vector<float> normalized;
  std::vector<__nv_bfloat16> candidate_key;
  std::vector<__nv_bfloat16> candidate_value;
  std::vector<__nv_bfloat16> bf16_out;
};

int launch_complete(const char* path, const qw38::cuda::AttentionConfig& config,
                    std::size_t position, const std::vector<float>& query,
                    const std::vector<float>& key, const std::vector<float>& value,
                    const std::vector<float>& gate,
                    const std::vector<float>& qscale,
                    const std::vector<float>& kscale,
                    const std::vector<__nv_bfloat16>& committed_key,
                    const std::vector<__nv_bfloat16>& committed_value,
                    LaunchResult* result, bool offset_kv) {
  qw38::cuda::DecodeQueryPrepPathScope scope(path);
  const std::size_t qn = query.size();
  const std::size_t kn = key.size();
  const std::size_t cache = committed_key.size();
  const std::size_t vkq = qw38::cuda::decode_kv_partial_vkq_values(kParts);
  const std::size_t meta = qw38::cuda::decode_kv_partial_meta_values(kParts);
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
  __nv_bfloat16* d_candk_base = nullptr;
  __nv_bfloat16* d_candv_base = nullptr;
  __nv_bfloat16* d_bf16 = nullptr;
  cudaError_t error = cudaMalloc(&d_q, qn * sizeof(float));
#define QW38_ALLOC(p, n, t) \
  if (error == cudaSuccess) error = cudaMalloc(&(p), (n) * sizeof(t))
  QW38_ALLOC(d_k, kn, float);
  QW38_ALLOC(d_v, kn, float);
  QW38_ALLOC(d_g, qn, float);
  QW38_ALLOC(d_qs, config.head_width, float);
  QW38_ALLOC(d_ks, config.head_width, float);
  QW38_ALLOC(d_nq, qn, float);
  QW38_ALLOC(d_nk, kn, float);
  QW38_ALLOC(d_scores, qn, float);
  QW38_ALLOC(d_out, qn, float);
  QW38_ALLOC(d_vkq, vkq, float);
  QW38_ALLOC(d_meta, meta, float);
  QW38_ALLOC(d_ck, cache + 8, __nv_bfloat16);
  QW38_ALLOC(d_cv, cache + 8, __nv_bfloat16);
  QW38_ALLOC(d_candk_base, kn + 8, __nv_bfloat16);
  QW38_ALLOC(d_candv_base, kn + 8, __nv_bfloat16);
  QW38_ALLOC(d_bf16, qn, __nv_bfloat16);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("malloc", error);
  const std::size_t kv_off = offset_kv ? 1 : 0;
  __nv_bfloat16* d_candk = d_candk_base + kv_off;
  __nv_bfloat16* d_candv = d_candv_base + kv_off;
#define QW38_H2D(d, h)                                                       \
  if (error == cudaSuccess)                                                  \
    error = cudaMemcpy((d), (h).data(), (h).size() * sizeof((h)[0]),         \
                       cudaMemcpyHostToDevice)
  QW38_H2D(d_q, query);
  QW38_H2D(d_k, key);
  QW38_H2D(d_v, value);
  QW38_H2D(d_g, gate);
  QW38_H2D(d_qs, qscale);
  QW38_H2D(d_ks, kscale);
  QW38_H2D(d_ck, committed_key);
  QW38_H2D(d_cv, committed_value);
#undef QW38_H2D
  if (error != cudaSuccess) return fail_cuda("H2D", error);
  qw38::cuda::AttentionCache committed{d_ck, d_cv};
  qw38::cuda::AttentionCache candidate{d_candk, d_candv};
  qw38::cuda::reset_decode_kv_unaligned_fallback();
  error = qw38::cuda::launch_attention_prepare_partitioned(
      config, position, d_q, d_k, d_v, d_qs, d_ks, d_g, committed, candidate,
      d_nq, d_nk, d_scores, d_out, d_vkq, d_meta, kParts, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_fp32_to_bf16(d_out, qn, d_bf16, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("complete attention", error);
  result->output.resize(qn);
  result->normalized.resize(qn);
  result->candidate_key.resize(kn);
  result->candidate_value.resize(kn);
  result->bf16_out.resize(qn);
  error = cudaMemcpy(result->output.data(), d_out, qn * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->normalized.data(), d_nq, qn * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->candidate_key.data(), d_candk,
                       kn * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->candidate_value.data(), d_candv,
                       kn * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->bf16_out.data(), d_bf16, qn * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(d_bf16);
  cudaFree(d_candv_base);
  cudaFree(d_candk_base);
  cudaFree(d_cv);
  cudaFree(d_ck);
  cudaFree(d_meta);
  cudaFree(d_vkq);
  cudaFree(d_out);
  cudaFree(d_scores);
  cudaFree(d_nk);
  cudaFree(d_nq);
  cudaFree(d_ks);
  cudaFree(d_qs);
  cudaFree(d_g);
  cudaFree(d_v);
  cudaFree(d_k);
  cudaFree(d_q);
  if (error != cudaSuccess) return fail_cuda("D2H", error);
  return 0;
}

int run_case(const char* name, std::size_t position, std::uint32_t capacity,
             bool sample_fp64) {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           capacity};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 9, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  const char* paths[] = {"warp_query", "prepared_q", "prepared_q_veckv"};
  LaunchResult results[3];
  int rc = 0;
  for (int index = 0; index < 3; ++index) {
    rc |= launch_complete(paths[index], config, position, query, key, value,
                          gate, qscale, kscale, committed_key, committed_value,
                          &results[index], false);
  }
  if (rc != 0) return rc;
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  for (int cand = 1; cand < 3; ++cand) {
    for (std::size_t index = 0; index < results[0].output.size(); ++index) {
      if (!std::isfinite(results[0].output[index]) ||
          !std::isfinite(results[cand].output[index])) {
        ++nonfinite;
      }
      max_abs = std::max(max_abs, std::fabs(results[0].output[index] -
                                            results[cand].output[index]));
    }
    for (std::size_t index = 0; index < results[0].normalized.size(); ++index) {
      max_abs = std::max(max_abs, std::fabs(results[0].normalized[index] -
                                            results[cand].normalized[index]));
    }
  }
  float fp64_abs = 0.0F;
  if (sample_fp64) {
    std::vector<float> nq;
    host_rms_rope(query.data(), qscale.data(), config.query_heads,
                  config.head_width, config.rotary_width, position, &nq);
    std::vector<float> nk;
    host_rms_rope(key.data(), kscale.data(), config.kv_heads, config.head_width,
                  config.rotary_width, position, &nk);
    const int heads[] = {0, 1, 12, 23};
    const int dims[] = {0, 1, 32, 64, 128, 200, 254, 255};
    std::vector<double> sampled;
    fp64_sample(config, position, nq, nk, value, gate, committed_key,
                committed_value, results[0].candidate_key,
                results[0].candidate_value, &sampled, heads, dims, 4, 8);
    std::size_t si = 0;
    for (int hi = 0; hi < 4; ++hi) {
      for (int di = 0; di < 8; ++di, ++si) {
        const std::size_t index =
            static_cast<std::size_t>(heads[hi]) * config.head_width +
            static_cast<std::size_t>(dims[di]);
        fp64_abs = std::max(
            fp64_abs, std::fabs(results[0].output[index] -
                                static_cast<float>(sampled[si])));
      }
    }
  }
  const bool pass = nonfinite == 0 && max_abs == 0.0F && fp64_abs <= 5.0e-3F;
  std::printf(
      "attn_case=%s position=%zu n_parts=16 seq_abs=%.9g seq_nonfinite=%zu "
      "fp64_abs=%.9g prep_launches_last=%u pass=%s\n",
      name, position, max_abs, nonfinite, fp64_abs,
      qw38::cuda::last_decode_attention_prep_launches(),
      pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_refresh_proof() {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query(qw38::cuda::attention_query_values(config), 0.1F);
  std::vector<float> scale(config.head_width, 1.0F);
  float* d_q = nullptr;
  float* d_s = nullptr;
  float* d_n = nullptr;
  cudaError_t error =
      cudaMalloc(&d_q, query.size() * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&d_s, scale.size() * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&d_n, query.size() * sizeof(float));
  if (error != cudaSuccess) return fail_cuda("refresh malloc", error);
  error = cudaMemcpy(d_q, query.data(), query.size() * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_s, scale.data(), scale.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_prepare_decode_query(config, 1, d_q, d_s, d_n,
                                                    nullptr);
  }
  std::vector<float> first(query.size());
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = cudaMemcpy(first.data(), d_n, first.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  for (float& item : query) item = 0.4F;
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_q, query.data(), query.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_prepare_decode_query(config, 1, d_q, d_s, d_n,
                                                    nullptr);
  }
  std::vector<float> second(query.size());
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = cudaMemcpy(second.data(), d_n, second.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  for (float& item : query) item = 0.1F;
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_q, query.data(), query.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_prepare_decode_query(config, 7, d_q, d_s, d_n,
                                                    nullptr);
  }
  std::vector<float> third(query.size());
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = cudaMemcpy(third.data(), d_n, third.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(d_n);
  cudaFree(d_s);
  cudaFree(d_q);
  if (error != cudaSuccess) return fail_cuda("refresh", error);
  const bool changed_input = fnv(first.data(), first.size()) !=
                             fnv(second.data(), second.size());
  const bool changed_pos = fnv(first.data(), first.size()) !=
                           fnv(third.data(), third.size());
  const bool pass = changed_input && changed_pos;
  std::printf(
      "prepared_q_refresh pointer_identity_skip=false input_changed=%s "
      "position_changed=%s pass=%s\n",
      changed_input ? "true" : "false", changed_pos ? "true" : "false",
      pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_candidate_boundary() {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, 4, 3, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  LaunchResult base;
  int rc = launch_complete("prepared_q", config, 4, query, key, value, gate,
                           qscale, kscale, committed_key, committed_value,
                           &base, false);
  if (rc != 0) return rc;
  for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
    const std::size_t phys = qw38::cuda::attention_kv_physical_index(
        4, 0, dim, config.capacity, config.head_width);
    committed_key[phys] = __float2bfloat16_rn(9.0F);
  }
  LaunchResult committed_mut;
  rc = launch_complete("prepared_q", config, 4, query, key, value, gate, qscale,
                       kscale, committed_key, committed_value, &committed_mut,
                       false);
  if (rc != 0) return rc;
  value[0] += 1.5F;
  LaunchResult cand_mut;
  rc = launch_complete("prepared_q", config, 4, query, key, value, gate, qscale,
                       kscale, committed_key, committed_value, &cand_mut,
                       false);
  if (rc != 0) return rc;
  const bool committed_unused = base.output == committed_mut.output;
  const bool candidate_used = base.output != cand_mut.output;
  const bool pass = committed_unused && candidate_used;
  std::printf(
      "candidate_kv_boundary committed_current_unused=%s "
      "candidate_current_used=%s pass=%s\n",
      committed_unused ? "true" : "false", candidate_used ? "true" : "false",
      pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_alignment_fallback() {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           32};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, 3, 11, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  LaunchResult aligned;
  LaunchResult unaligned;
  int rc = launch_complete("prepared_q_veckv", config, 3, query, key, value,
                           gate, qscale, kscale, committed_key, committed_value,
                           &aligned, false);
  const unsigned int aligned_flag = qw38::cuda::last_decode_kv_unaligned_fallback();
  if (rc != 0) return rc;
  rc = launch_complete("prepared_q_veckv", config, 3, query, key, value, gate,
                       qscale, kscale, committed_key, committed_value,
                       &unaligned, true);
  const unsigned int unaligned_flag =
      qw38::cuda::last_decode_kv_unaligned_fallback();
  if (rc != 0) return rc;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < aligned.output.size(); ++index) {
    max_abs = std::max(max_abs, std::fabs(aligned.output[index] -
                                          unaligned.output[index]));
  }
  const bool pass = unaligned_flag != 0 && max_abs == 0.0F;
  std::printf(
      "alignment_fallback aligned_flag=%u unaligned_flag=%u abs=%.9g "
      "vec_load_used=%s pass=%s\n",
      aligned_flag, unaligned_flag, max_abs,
      qw38::cuda::last_decode_vec_kv_used() ? "true" : "false",
      pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_attrs() {
  int regs = 0;
  std::size_t local_bytes = 1;
  int occ = 0;
  qw38::cuda::decode_attention_kernel_attributes("prepared_q", &regs,
                                                 &local_bytes, &occ);
  const int prep_occ = qw38::cuda::decode_query_prep_occupancy();
  const int warp_occ = qw38::cuda::decode_kv_warp_query_occupancy();
  std::printf(
      "prep_occupancy=%d warp_occupancy=%d prepared_occupancy=%d "
      "prepared_regs=%d prepared_local_bytes=%zu sequential_pin=%s "
      "n_parts_low=%d n_parts_high=%d\n",
      prep_occ, warp_occ, occ, regs, local_bytes,
      qw38::cuda::selected_decode_query_prep_path(),
      qw38::cuda::selected_decode_kv_parts_below_2048(),
      qw38::cuda::selected_decode_kv_parts_at_or_above_2048());
  FILE* pipe = popen(
      "cuobjdump -sass build/qw38-cuda-opt078-decode-attention-test "
      "2>/dev/null | head -c 4096",
      "r");
  int sass_bytes = 0;
  if (pipe != nullptr) {
    char buf[256];
    while (std::fread(buf, 1, sizeof(buf), pipe) > 0) sass_bytes += 256;
    pclose(pipe);
  }
  std::printf("sass_status=%s sass_bytes=%d\n",
              sass_bytes > 0 ? "excerpt" : "unavailable", sass_bytes);
  return (prep_occ > 0 && warp_occ > 0 &&
          std::strcmp(qw38::cuda::selected_decode_query_prep_path(),
                      "warp_query") == 0 &&
          qw38::cuda::selected_decode_kv_parts_below_2048() == 16 &&
          qw38::cuda::selected_decode_kv_parts_at_or_above_2048() == 16)
             ? 0
             : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload;
  if (workload == nullptr || workload[0] == '\0') {
    workload = default_workload(qw38::cuda::test_tier());
  }
  if (std::strcmp(qw38::cuda::selected_decode_query_prep_path(), "warp_query") !=
      0) {
    std::fprintf(stderr, "production pin is not warp_query\n");
    return 1;
  }
  int rc = 0;
  rc |= run_case("tiny_p0", 0, 32, false);
  rc |= run_case("tiny_p1", 1, 32, true);
  if (std::strcmp(workload, "smoke") != 0) {
    rc |= run_case("tiny_p31", 31, 64, false);
    rc |= run_case("tiny_p32", 32, 64, true);
    rc |= run_refresh_proof();
    rc |= run_candidate_boundary();
    rc |= run_alignment_fallback();
  }
  if (std::strcmp(workload, "smoke") != 0 &&
      std::strcmp(workload, "correctness") != 0) {
    rc |= run_case("prod_p127", 127, 256, false);
    rc |= run_case("prod_p128", 128, 256, true);
  } else if (std::strcmp(workload, "correctness") == 0) {
    rc |= run_case("prod_p127", 127, 256, true);
  }
  if (std::strcmp(workload, "acceptance") == 0) {
    rc |= run_case("prod_p2047", 2047, 2064, false);
    rc |= run_case("prod_p2048", 2048, 2064, true);
  }
  rc |= run_attrs();
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-078\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_decode_query_prep\":\"%s\",\"n_parts\":16,\"nonfinite\":%d,"
      "\"pass\":%s}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_decode_query_prep_path(), pass ? 0 : 1,
      pass ? "true" : "false");
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}
