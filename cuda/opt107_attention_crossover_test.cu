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

constexpr char kPrefix[] = "QW38_OPT107_ATTENTION_CROSSOVER_RESULT=";
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
constexpr float kFp64Abs = 5.0e-3F;
constexpr float kVsControlAbs = 2.0e-2F;
constexpr int kScreenThreshold = 2048;

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

void fp64_sample(const qw38::cuda::AttentionConfig& config, std::size_t position,
                 const std::vector<float>& nq, const std::vector<float>& gate,
                 const std::vector<__nv_bfloat16>& committed_key,
                 const std::vector<__nv_bfloat16>& committed_value,
                 const std::vector<__nv_bfloat16>& cand_key,
                 const std::vector<__nv_bfloat16>& cand_value,
                 std::vector<double>* sampled, const int* heads,
                 const int* dims, int n_heads, int n_dims) {
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
  const char* launch = "";
  const char* path = "";
  unsigned int prep_launches = 0;
  unsigned int n_parts = 0;
  unsigned int grid_x = 0;
  unsigned int block_y = 0;
};

int launch_once(const char* forced_path, int crossover_threshold, int n_parts,
                const qw38::cuda::AttentionConfig& config, std::size_t position,
                const std::vector<float>& query, const std::vector<float>& key,
                const std::vector<float>& value, const std::vector<float>& gate,
                const std::vector<float>& qscale,
                const std::vector<float>& kscale,
                const std::vector<__nv_bfloat16>& committed_key,
                const std::vector<__nv_bfloat16>& committed_value,
                LaunchResult* result) {
  if (forced_path != nullptr) {
    qw38::cuda::set_decode_attention_vec128_path_override(forced_path);
  } else {
    qw38::cuda::clear_decode_attention_vec128_path_override();
  }
  qw38::cuda::DecodeAttentionCrossoverScope crossover_scope(crossover_threshold);
  qw38::cuda::Vec128NPartsScope parts_scope(n_parts);
  const std::size_t qn = query.size();
  const std::size_t kn = key.size();
  const std::size_t cache = committed_key.size();
  const std::size_t vkq = qw38::cuda::decode_kv_partial_vkq_values(16);
  const std::size_t meta = qw38::cuda::decode_kv_partial_meta_values(16);
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
  QW38_ALLOC(d_ck, cache, __nv_bfloat16);
  QW38_ALLOC(d_cv, cache, __nv_bfloat16);
  QW38_ALLOC(d_candk, kn, __nv_bfloat16);
  QW38_ALLOC(d_candv, kn, __nv_bfloat16);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("malloc", error);
#define QW38_H2D(d, h)                                               \
  if (error == cudaSuccess)                                          \
    error = cudaMemcpy((d), (h).data(), (h).size() * sizeof((h)[0]), \
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
      d_nq, d_nk, d_scores, d_out, d_vkq, d_meta, n_parts, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("complete attention", error);
  result->output.resize(qn);
  result->normalized.resize(qn);
  result->candidate_key.resize(kn);
  result->candidate_value.resize(kn);
  result->launch = qw38::cuda::last_decode_query_prep_launch_variant();
  result->path =
      qw38::cuda::decode_attention_vec128_path_for_position(position);
  result->prep_launches = qw38::cuda::last_decode_attention_prep_launches();
  result->n_parts = qw38::cuda::last_decode_attention_n_parts();
  result->grid_x = qw38::cuda::last_decode_attention_gqa_grid_x();
  result->block_y = qw38::cuda::last_decode_attention_gqa_block_y();
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
  cudaFree(d_candv);
  cudaFree(d_candk);
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
  if (forced_path != nullptr) {
    qw38::cuda::clear_decode_attention_vec128_path_override();
  }
  if (error != cudaSuccess) return fail_cuda("D2H", error);
  return 0;
}

bool same_launch(const LaunchResult& left, const LaunchResult& right) {
  return std::strcmp(left.launch, right.launch) == 0 &&
         left.prep_launches == right.prep_launches &&
         left.n_parts == right.n_parts && left.grid_x == right.grid_x &&
         left.block_y == right.block_y;
}

int run_selector_host() {
  int rc = 0;
  const int thresholds[] = {0, 512, 1024, 1536, 2048};
  const std::size_t positions[] = {0, 1, 31, 32, 127, 128, 511, 512, 513,
                                   1023, 1024, 1025, 1535, 1536, 1537,
                                   2047, 2048, 2049, 4095, 4096, 4097,
                                   131072};
  for (int threshold : thresholds) {
    qw38::cuda::DecodeAttentionCrossoverScope scope(threshold);
    qw38::cuda::clear_decode_attention_vec128_path_override();
    for (std::size_t position : positions) {
      const char* path =
          qw38::cuda::decode_attention_vec128_path_for_position(position);
      const bool expect_vec =
          threshold > 0 &&
          position >= static_cast<std::size_t>(threshold) &&
          position <= static_cast<std::size_t>(
                          qw38::cuda::kSelectedDecodeAttentionVerifiedMax);
      const char* want = expect_vec ? "vec128_online" : "warp_query";
      if (std::strcmp(path, want) != 0) {
        std::fprintf(stderr,
                     "selector mismatch threshold=%d position=%zu got=%s "
                     "want=%s\n",
                     threshold, position, path, want);
        rc = 1;
      }
    }
  }
  {
    qw38::cuda::DecodeAttentionVec128PathScope forced("vec128_online");
    qw38::cuda::DecodeAttentionCrossoverScope scope(2048);
    if (std::strcmp(qw38::cuda::decode_attention_vec128_path_for_position(128),
                    "vec128_online") != 0) {
      std::fprintf(stderr, "path override must win at short prefix\n");
      rc = 1;
    }
  }
  const bool fallback =
      std::strcmp(qw38::cuda::decode_attention_vec128_path_for_position(
                      qw38::cuda::kDecodeAttentionFallback128K),
                  "warp_query") == 0;
  const bool pin_warp =
      std::strcmp(qw38::cuda::selected_decode_attention_vec128_path(),
                  "warp_query") == 0;
  const bool disabled =
      qw38::cuda::selected_decode_attention_crossover_threshold() == 0 ||
      qw38::cuda::legal_decode_attention_crossover_threshold(
          qw38::cuda::selected_decode_attention_crossover_threshold());
  std::printf(
      "selector_host fallback_128k=%s pin_warp_query=%s verified_max=%d "
      "threshold_pin=%d extra_kernel=false extra_sync=false extra_alloc=false "
      "extra_copy=false pass=%s\n",
      fallback ? "true" : "false", pin_warp ? "true" : "false",
      qw38::cuda::selected_decode_attention_verified_max(),
      qw38::cuda::selected_decode_attention_crossover_threshold(),
      (rc == 0 && fallback && pin_warp && disabled) ? "true" : "false");
  return (rc == 0 && fallback && pin_warp && disabled) ? 0 : 1;
}

int run_identity_case(const char* name, std::size_t position, int threshold,
                      bool expect_vec, bool sample_fp64) {
  const std::uint32_t capacity =
      static_cast<std::uint32_t>(std::max(position + 16, std::size_t{32}));
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           capacity};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  LaunchResult control;
  LaunchResult hybrid;
  int rc = launch_once("warp_query", 0, kParts, config, position, query, key,
                       value, gate, qscale, kscale, committed_key,
                       committed_value, &control);
  rc |= launch_once(nullptr, threshold, kParts, config, position, query, key,
                    value, gate, qscale, kscale, committed_key,
                    committed_value, &hybrid);
  if (rc != 0) return rc;
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < control.output.size(); ++index) {
    if (!std::isfinite(control.output[index]) ||
        !std::isfinite(hybrid.output[index])) {
      ++nonfinite;
    }
    max_abs = std::max(max_abs, std::fabs(control.output[index] -
                                          hybrid.output[index]));
  }
  float fp64_abs = 0.0F;
  if (sample_fp64) {
    std::vector<float> nq;
    host_rms_rope(query.data(), qscale.data(), config.query_heads,
                  config.head_width, config.rotary_width, position, &nq);
    const int heads[] = {0, 1, 12, 23};
    const int dims[] = {0, 1, 32, 64, 128, 200, 254, 255};
    std::vector<double> sampled;
    fp64_sample(config, position, nq, gate, committed_key, committed_value,
                hybrid.candidate_key, hybrid.candidate_value, &sampled, heads,
                dims, 4, 8);
    std::size_t si = 0;
    for (int hi = 0; hi < 4; ++hi) {
      for (int di = 0; di < 8; ++di, ++si) {
        const std::size_t index =
            static_cast<std::size_t>(heads[hi]) * config.head_width +
            static_cast<std::size_t>(dims[di]);
        fp64_abs = std::max(
            fp64_abs, std::fabs(hybrid.output[index] -
                                static_cast<float>(sampled[si])));
      }
    }
  }
  const bool path_ok =
      std::strcmp(hybrid.path, expect_vec ? "vec128_online" : "warp_query") ==
      0;
  const bool launch_ok =
      expect_vec
          ? std::strcmp(hybrid.launch, "vec128_online_decode_attention") == 0 &&
                hybrid.block_y == 4
          : (std::strcmp(hybrid.launch, "warp_query_decode_attention") == 0 &&
             same_launch(control, hybrid) && control.output == hybrid.output);
  const bool extra_ok = hybrid.prep_launches == 0 && control.prep_launches == 0;
  const bool pass = nonfinite == 0 && path_ok && launch_ok && extra_ok &&
                    fp64_abs <= kFp64Abs &&
                    (expect_vec ? max_abs <= kVsControlAbs : max_abs == 0.0F);
  std::printf(
      "attn_case=%s position=%zu threshold=%d path=%s launch=%s "
      "prep_launches=%u n_parts=%u seq_abs=%.9g seq_nonfinite=%zu "
      "fp64_abs=%.9g extra_kernel=false extra_sync=false extra_alloc=false "
      "extra_copy=false pass=%s\n",
      name, position, threshold, hybrid.path, hybrid.launch,
      hybrid.prep_launches, hybrid.n_parts, max_abs, nonfinite, fp64_abs,
      pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_zero_case(std::size_t position) {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  for (float& item : query) item = 0.0F;
  LaunchResult control;
  LaunchResult candidate;
  int rc = launch_once("warp_query", 0, kParts, config, position, query, key,
                       value, gate, qscale, kscale, committed_key,
                       committed_value, &control);
  rc |= launch_once("vec128_online", 0, kParts, config, position, query, key,
                    value, gate, qscale, kscale, committed_key,
                    committed_value, &candidate);
  if (rc != 0) return rc;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < control.output.size(); ++index) {
    max_abs = std::max(max_abs, std::fabs(control.output[index] -
                                          candidate.output[index]));
  }
  const bool pass = max_abs <= kVsControlAbs;
  std::printf("zero_case position=%zu abs=%.9g pass=%s\n", position, max_abs,
              pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_cancellation_case(std::size_t position) {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  for (std::size_t index = 0; index < query.size(); index += 2) {
    query[index] = 1.0F;
    if (index + 1 < query.size()) query[index + 1] = -1.0F;
  }
  LaunchResult control;
  LaunchResult candidate;
  int rc = launch_once("warp_query", 0, kParts, config, position, query, key,
                       value, gate, qscale, kscale, committed_key,
                       committed_value, &control);
  rc |= launch_once("vec128_online", 0, kParts, config, position, query, key,
                    value, gate, qscale, kscale, committed_key,
                    committed_value, &candidate);
  if (rc != 0) return rc;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < control.output.size(); ++index) {
    max_abs = std::max(max_abs, std::fabs(control.output[index] -
                                          candidate.output[index]));
  }
  const bool pass = max_abs <= kVsControlAbs;
  std::printf("cancellation_case position=%zu abs=%.9g pass=%s\n", position,
              max_abs, pass ? "true" : "false");
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
  int rc = launch_once("vec128_online", 0, kParts, config, 4, query, key, value,
                       gate, qscale, kscale, committed_key, committed_value,
                       &base);
  if (rc != 0) return rc;
  for (std::uint32_t dim = 0; dim < config.head_width; ++dim) {
    const std::size_t phys = qw38::cuda::attention_kv_physical_index(
        4, 0, dim, config.capacity, config.head_width);
    committed_key[phys] = __float2bfloat16_rn(9.0F);
  }
  LaunchResult committed_mut;
  rc = launch_once("vec128_online", 0, kParts, config, 4, query, key, value,
                   gate, qscale, kscale, committed_key, committed_value,
                   &committed_mut);
  if (rc != 0) return rc;
  value[0] += 1.5F;
  LaunchResult cand_mut;
  rc = launch_once("vec128_online", 0, kParts, config, 4, query, key, value,
                   gate, qscale, kscale, committed_key, committed_value,
                   &cand_mut);
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

int run_repeat_eager() {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, 32, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  LaunchResult first;
  LaunchResult second;
  int rc = launch_once(nullptr, kScreenThreshold, kParts, config, 32, query,
                       key, value, gate, qscale, kscale, committed_key,
                       committed_value, &first);
  rc |= launch_once(nullptr, kScreenThreshold, kParts, config, 32, query, key,
                    value, gate, qscale, kscale, committed_key,
                    committed_value, &second);
  if (rc != 0) return rc;
  const bool pass = first.output == second.output && same_launch(first, second);
  std::printf("graph_eager_repeat pass=%s\n", pass ? "true" : "false");
  return pass ? 0 : 1;
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
  if (!qw38::cuda::legal_decode_attention_vec128_path(
          qw38::cuda::selected_decode_attention_vec128_path()) ||
      std::strcmp(qw38::cuda::selected_decode_attention_vec128_path(),
                  "warp_query") != 0) {
    std::fprintf(stderr, "OPT-103 vec128 pin must remain warp_query\n");
    return 1;
  }
  if (std::strcmp(qw38::cuda::selected_decode_attention_gqa_path(),
                  "warp_query") != 0) {
    std::fprintf(stderr, "gqa production pin is not warp_query\n");
    return 1;
  }
  int rc = 0;
  rc |= run_selector_host();
  rc |= run_identity_case("tiny_p0", 0, kScreenThreshold, false, false);
  rc |= run_identity_case("tiny_p1", 1, kScreenThreshold, false, true);
  if (std::strcmp(workload, "smoke") != 0) {
    rc |= run_identity_case("tiny_p31", 31, kScreenThreshold, false, false);
    rc |= run_identity_case("tiny_p32", 32, kScreenThreshold, false, true);
    rc |= run_zero_case(1);
    rc |= run_cancellation_case(1);
    rc |= run_zero_case(32);
    rc |= run_cancellation_case(32);
    rc |= run_candidate_boundary();
    rc |= run_repeat_eager();
    rc |= run_identity_case("bound_tm1", kScreenThreshold - 1, kScreenThreshold,
                            false, true);
    rc |= run_identity_case("bound_t", kScreenThreshold, kScreenThreshold, true,
                            true);
    rc |= run_identity_case("bound_tp1", kScreenThreshold + 1, kScreenThreshold,
                            true, true);
  }
  if (std::strcmp(workload, "smoke") != 0 &&
      std::strcmp(workload, "correctness") != 0) {
    rc |= run_identity_case("prod_p127", 127, kScreenThreshold, false, false);
    rc |= run_identity_case("prod_p128", 128, kScreenThreshold, false, true);
  } else if (std::strcmp(workload, "correctness") == 0) {
    rc |= run_identity_case("prod_p127", 127, kScreenThreshold, false, true);
  }
  if (std::strcmp(workload, "acceptance") == 0) {
    rc |= run_identity_case("prod_p2047", 2047, kScreenThreshold, false, false);
    rc |= run_identity_case("prod_p2048", 2048, kScreenThreshold, true, true);
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-107\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_decode_attention_vec128\":\"%s\","
      "\"crossover_threshold\":%d,\"verified_max\":%d,"
      "\"n_parts\":16,\"nonfinite\":%d,\"pass\":%s}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_decode_attention_vec128_path(),
      qw38::cuda::selected_decode_attention_crossover_threshold(),
      qw38::cuda::selected_decode_attention_verified_max(), pass ? 0 : 1,
      pass ? "true" : "false");
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}
