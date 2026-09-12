#include "attention_decode.h"
#include "opt108_llama_vector_adapter.cuh"
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

constexpr char kPrefix[] = "QW38_OPT108_LLAMA_VECTOR_STACK_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kRotary = 64;
constexpr float kRmsEps = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;
constexpr float kFp64Abs = 5.0e-3F;
constexpr float kVsControlAbs = 2.0e-2F;
constexpr int kPrimitivePositions[] = {128, 512, 2048, 4096};

struct Options final {
  const char* workload = nullptr;
  int position = 0;
  int warmups = 0;
  int samples = 0;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|screen|acceptance|"
               "primitive] [--position N] [--warmups N] [--samples N]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--position") == 0 && index + 1 < argc) {
      options->position = std::atoi(argv[++index]);
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      options->warmups = std::atoi(argv[++index]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      options->samples = std::atoi(argv[++index]);
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
  int n_parts = 0;
  int occupancy = 0;
  int nsm = 0;
  int ntiles_kv = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  bool prepared_q_once = false;
  bool nvidia_float2 = false;
};

struct DeviceCase {
  const qw38::cuda::AttentionConfig* config = nullptr;
  std::size_t position = 0;
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
  __nv_bfloat16* d_candk_base = nullptr;
  __nv_bfloat16* d_candv_base = nullptr;
  std::size_t qn = 0;
  std::size_t kn = 0;

  void free_all() {
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
  }
};

int alloc_case(const qw38::cuda::AttentionConfig& config, std::size_t position,
               const std::vector<float>& query, const std::vector<float>& key,
               const std::vector<float>& value, const std::vector<float>& gate,
               const std::vector<float>& qscale,
               const std::vector<float>& kscale,
               const std::vector<__nv_bfloat16>& committed_key,
               const std::vector<__nv_bfloat16>& committed_value,
               DeviceCase* device) {
  device->config = &config;
  device->position = position;
  device->qn = query.size();
  device->kn = key.size();
  const std::size_t cache = committed_key.size();
  const int alloc_parts = qw38::cuda::opt108::kMaxParts;
  const std::size_t vkq = qw38::cuda::opt108::partial_vkq_values(alloc_parts);
  const std::size_t meta = qw38::cuda::opt108::partial_meta_values(alloc_parts);
  cudaError_t error = cudaMalloc(&device->d_q, device->qn * sizeof(float));
#define QW38_ALLOC(p, n, t) \
  if (error == cudaSuccess) error = cudaMalloc(&(device->p), (n) * sizeof(t))
  QW38_ALLOC(d_k, device->kn, float);
  QW38_ALLOC(d_v, device->kn, float);
  QW38_ALLOC(d_g, device->qn, float);
  QW38_ALLOC(d_qs, config.head_width, float);
  QW38_ALLOC(d_ks, config.head_width, float);
  QW38_ALLOC(d_nq, device->qn, float);
  QW38_ALLOC(d_nk, device->kn, float);
  QW38_ALLOC(d_scores, device->qn, float);
  QW38_ALLOC(d_out, device->qn, float);
  QW38_ALLOC(d_vkq, vkq, float);
  QW38_ALLOC(d_meta, meta, float);
  QW38_ALLOC(d_ck, cache + 8, __nv_bfloat16);
  QW38_ALLOC(d_cv, cache + 8, __nv_bfloat16);
  QW38_ALLOC(d_candk_base, device->kn + 8, __nv_bfloat16);
  QW38_ALLOC(d_candv_base, device->kn + 8, __nv_bfloat16);
#undef QW38_ALLOC
  if (error != cudaSuccess) {
    device->free_all();
    return fail_cuda("malloc", error);
  }
  device->d_candk = device->d_candk_base;
  device->d_candv = device->d_candv_base;
#define QW38_H2D(d, h)                                               \
  if (error == cudaSuccess)                                          \
    error = cudaMemcpy((d), (h).data(), (h).size() * sizeof((h)[0]), \
                       cudaMemcpyHostToDevice)
  QW38_H2D(device->d_q, query);
  QW38_H2D(device->d_k, key);
  QW38_H2D(device->d_v, value);
  QW38_H2D(device->d_g, gate);
  QW38_H2D(device->d_qs, qscale);
  QW38_H2D(device->d_ks, kscale);
  QW38_H2D(device->d_ck, committed_key);
  QW38_H2D(device->d_cv, committed_value);
#undef QW38_H2D
  if (error != cudaSuccess) {
    device->free_all();
    return fail_cuda("H2D", error);
  }
  return 0;
}

int launch_control(DeviceCase* device, LaunchResult* result) {
  qw38::cuda::AttentionCache committed{device->d_ck, device->d_cv};
  qw38::cuda::AttentionCache candidate{device->d_candk, device->d_candv};
  const int n_parts =
      qw38::cuda::decode_kv_parts_for_position(device->position);
  cudaError_t error = qw38::cuda::launch_attention_prepare_partitioned(
      *device->config, device->position, device->d_q, device->d_k, device->d_v,
      device->d_qs, device->d_ks, device->d_g, committed, candidate,
      device->d_nq, device->d_nk, device->d_scores, device->d_out, device->d_vkq,
      device->d_meta, n_parts, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("control attention", error);
  result->n_parts = n_parts;
  result->output.resize(device->qn);
  result->normalized.resize(device->qn);
  result->candidate_key.resize(device->kn);
  result->candidate_value.resize(device->kn);
  error = cudaMemcpy(result->output.data(), device->d_out,
                     device->qn * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->normalized.data(), device->d_nq,
                       device->qn * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->candidate_key.data(), device->d_candk,
                       device->kn * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->candidate_value.data(), device->d_candv,
                       device->kn * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("control D2H", error);
  return 0;
}

int launch_candidate(DeviceCase* device, LaunchResult* result) {
  qw38::cuda::AttentionCache committed{device->d_ck, device->d_cv};
  qw38::cuda::AttentionCache candidate{device->d_candk, device->d_candv};
  cudaError_t error = qw38::cuda::opt108::launch_llama_vec_stack(
      *device->config, device->position, device->d_q, device->d_k, device->d_v,
      device->d_qs, device->d_ks, device->d_g, committed, candidate,
      device->d_nq, device->d_nk, device->d_out, device->d_vkq, device->d_meta,
      0, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("candidate attention", error);
  const auto& info = qw38::cuda::opt108::last_launch();
  result->n_parts = info.n_parts;
  result->occupancy = info.occupancy_per_sm;
  result->nsm = info.nsm;
  result->ntiles_kv = info.ntiles_kv;
  result->registers = info.registers;
  result->local_bytes = info.local_bytes;
  result->prepared_q_once = info.prepared_q_once;
  result->nvidia_float2 = info.nvidia_float2_v_accum && !info.amd_half2_v_accum;
  result->output.resize(device->qn);
  result->normalized.resize(device->qn);
  result->candidate_key.resize(device->kn);
  result->candidate_value.resize(device->kn);
  error = cudaMemcpy(result->output.data(), device->d_out,
                     device->qn * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->normalized.data(), device->d_nq,
                       device->qn * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->candidate_key.data(), device->d_candk,
                       device->kn * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(result->candidate_value.data(), device->d_candv,
                       device->kn * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("candidate D2H", error);
  return 0;
}

int run_case(const char* name, std::size_t position, std::uint32_t capacity,
             bool sample_fp64) {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           capacity};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  DeviceCase device{};
  int rc = alloc_case(config, position, query, key, value, gate, qscale, kscale,
                      committed_key, committed_value, &device);
  if (rc != 0) return rc;
  LaunchResult control;
  LaunchResult candidate;
  rc = launch_control(&device, &control);
  rc |= launch_candidate(&device, &candidate);
  if (rc != 0) {
    device.free_all();
    return rc;
  }
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < control.output.size(); ++index) {
    if (!std::isfinite(control.output[index]) ||
        !std::isfinite(candidate.output[index])) {
      ++nonfinite;
    }
    max_abs = std::max(max_abs, std::fabs(control.output[index] -
                                          candidate.output[index]));
  }
  float fp64_abs = 0.0F;
  int fp64_bad = 0;
  if (sample_fp64) {
    std::vector<float> nq;
    host_rms_rope(query.data(), qscale.data(), config.query_heads,
                  config.head_width, config.rotary_width, position, &nq);
    const int heads[] = {0, 5, 11, 23};
    const int dims[] = {0, 17, 128, 255};
    std::vector<double> sampled;
    fp64_sample(config, position, nq, gate, committed_key, committed_value,
                candidate.candidate_key, candidate.candidate_value, &sampled,
                heads, dims, 4, 4);
    int si = 0;
    for (int hi = 0; hi < 4; ++hi) {
      for (int di = 0; di < 4; ++di, ++si) {
        const std::size_t index =
            static_cast<std::size_t>(heads[hi]) * config.head_width +
            static_cast<std::size_t>(dims[di]);
        const float diff = std::fabs(
            candidate.output[index] - static_cast<float>(sampled[si]));
        fp64_abs = std::max(fp64_abs, diff);
        if (diff > kFp64Abs) ++fp64_bad;
      }
    }
  }
  const bool pass =
      nonfinite == 0 && max_abs <= kVsControlAbs && fp64_bad == 0 &&
      candidate.prepared_q_once && candidate.nvidia_float2 &&
      candidate.n_parts >= 1 &&
      !qw38::cuda::opt108::kAmdHalf2VAccum &&
      !qw38::cuda::opt108::kF16CacheMigration;
  std::printf(
      "attn_case=%s position=%zu n_parts=%d occupancy=%d nsm=%d ntiles_kv=%d "
      "regs=%d local=%zu prepared_q_once=%s nvidia_float2=%s amd_half2=%s "
      "f16_cache=%s seq_abs=%.9g seq_nonfinite=%zu fp64_abs=%.9g fp64_bad=%d "
      "cpy_bytes=%d threads=%d pass=%s\n",
      name, position, candidate.n_parts, candidate.occupancy, candidate.nsm,
      candidate.ntiles_kv, candidate.registers, candidate.local_bytes,
      candidate.prepared_q_once ? "true" : "false",
      candidate.nvidia_float2 ? "true" : "false",
      qw38::cuda::opt108::kAmdHalf2VAccum ? "true" : "false",
      qw38::cuda::opt108::kF16CacheMigration ? "true" : "false", max_abs,
      nonfinite, fp64_abs, fp64_bad, qw38::cuda::opt108::kCpyBytes,
      qw38::cuda::opt108::kThreads, pass ? "true" : "false");
  device.free_all();
  return pass ? 0 : 1;
}

int run_zero_case(std::size_t position) {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 7, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  std::fill(query.begin(), query.end(), 0.0F);
  DeviceCase device{};
  int rc = alloc_case(config, position, query, key, value, gate, qscale, kscale,
                      committed_key, committed_value, &device);
  if (rc != 0) return rc;
  LaunchResult candidate;
  rc = launch_candidate(&device, &candidate);
  device.free_all();
  if (rc != 0) return rc;
  std::size_t nonfinite = 0;
  for (float item : candidate.output) {
    if (!std::isfinite(item)) ++nonfinite;
  }
  const bool pass = nonfinite == 0;
  std::printf("zero_case position=%zu nonfinite=%zu pass=%s\n", position,
              nonfinite, pass ? "true" : "false");
  return pass ? 0 : 1;
}

int run_repeat_eager() {
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           64};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, 32, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  DeviceCase device{};
  int rc = alloc_case(config, 32, query, key, value, gate, qscale, kscale,
                      committed_key, committed_value, &device);
  if (rc != 0) return rc;
  LaunchResult first;
  LaunchResult second;
  rc = launch_candidate(&device, &first);
  rc |= launch_candidate(&device, &second);
  device.free_all();
  if (rc != 0) return rc;
  const bool pass = first.output == second.output;
  std::printf("graph_eager_repeat pass=%s n_parts=%d\n",
              pass ? "true" : "false", first.n_parts);
  return pass ? 0 : 1;
}

int run_attrs() {
  int regs = 0;
  std::size_t local_bytes = 0;
  int occ = 0;
  qw38::cuda::opt108::kernel_attributes(&regs, &local_bytes, &occ);
  const int nsm = qw38::cuda::opt108::nsm_count();
  const int p128 = qw38::cuda::opt108::parallel_blocks_for_kv(129, occ, nsm);
  const int p2048 = qw38::cuda::opt108::parallel_blocks_for_kv(2049, occ, nsm);
  std::printf(
      "opt108_attrs occupancy=%d nsm=%d registers=%d local_bytes=%zu "
      "threads=%d cpy_bytes=%d nvidia_float2=%s amd_half2=%s "
      "f16_cache=%s prepared_q_once=%s n_parts_d128=%d n_parts_d2048=%d "
      "hardcoded_16=%s production_vec128=%s production_crossover=%d\n",
      occ, nsm, regs, local_bytes, qw38::cuda::opt108::kThreads,
      qw38::cuda::opt108::kCpyBytes,
      qw38::cuda::opt108::kNvidiaFloat2VAccum ? "true" : "false",
      qw38::cuda::opt108::kAmdHalf2VAccum ? "true" : "false",
      qw38::cuda::opt108::kF16CacheMigration ? "true" : "false", "true", p128,
      p2048, (p128 == 16 && p2048 == 16) ? "true" : "false",
      qw38::cuda::selected_decode_attention_vec128_path(),
      qw38::cuda::selected_decode_attention_crossover_threshold());
  return (occ > 0 && nsm > 0 && qw38::cuda::opt108::kNvidiaFloat2VAccum &&
          !qw38::cuda::opt108::kAmdHalf2VAccum && p128 == 1 && p2048 >= 1 &&
          p2048 != 16 &&
          std::strcmp(qw38::cuda::selected_decode_attention_vec128_path(),
                      "warp_query") == 0)
             ? 0
             : 1;
}

int time_path(DeviceCase* device, bool candidate, int warmups, int samples,
              std::vector<float>* ms) {
  ms->clear();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int round = 0; round < warmups + samples; ++round) {
    cudaError_t error = cudaSuccess;
    qw38::cuda::AttentionCache committed{device->d_ck, device->d_cv};
    qw38::cuda::AttentionCache cand{device->d_candk, device->d_candv};
    cudaEventRecord(start, nullptr);
    if (candidate) {
      error = qw38::cuda::opt108::launch_llama_vec_stack(
          *device->config, device->position, device->d_q, device->d_k,
          device->d_v, device->d_qs, device->d_ks, device->d_g, committed, cand,
          device->d_nq, device->d_nk, device->d_out, device->d_vkq,
          device->d_meta, 0, nullptr);
    } else {
      const int n_parts =
          qw38::cuda::decode_kv_parts_for_position(device->position);
      error = qw38::cuda::launch_attention_prepare_partitioned(
          *device->config, device->position, device->d_q, device->d_k,
          device->d_v, device->d_qs, device->d_ks, device->d_g, committed, cand,
          device->d_nq, device->d_nk, device->d_scores, device->d_out,
          device->d_vkq, device->d_meta, n_parts, nullptr);
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    if (error != cudaSuccess) {
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      return fail_cuda("primitive launch", error);
    }
    if (round >= warmups) {
      float elapsed = 0.0F;
      cudaEventElapsedTime(&elapsed, start, stop);
      ms->push_back(elapsed);
      std::printf(
          "QW38_OPT108_CASE={\"observation_unit\":\"independent_round\","
          "\"sample_index\":%d,\"path\":\"%s\",\"position\":%zu,\"ms\":%.9g}\n",
          round - warmups, candidate ? "llama_vec_nvidia" : "production",
          device->position, elapsed);
    }
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return 0;
}

int run_primitive_position(std::size_t position, int warmups, int samples) {
  const std::uint32_t capacity =
      static_cast<std::uint32_t>(std::max<std::size_t>(position + 16, 256));
  const qw38::cuda::AttentionConfig config{kQueryHeads, kKvHeads, kWidth, kRotary,
                                           capacity};
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(config, position, 89, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  DeviceCase device{};
  int rc = alloc_case(config, position, query, key, value, gate, qscale, kscale,
                      committed_key, committed_value, &device);
  if (rc != 0) return rc;
  LaunchResult check;
  rc = launch_candidate(&device, &check);
  if (rc != 0) {
    device.free_all();
    return rc;
  }
  std::vector<float> control_ms;
  std::vector<float> candidate_ms;
  rc = time_path(&device, false, warmups, samples, &control_ms);
  rc |= time_path(&device, true, warmups, samples, &candidate_ms);
  device.free_all();
  if (rc != 0) return rc;
  double control_mean = 0.0;
  double candidate_mean = 0.0;
  for (float item : control_ms) control_mean += item;
  for (float item : candidate_ms) candidate_mean += item;
  control_mean /= static_cast<double>(control_ms.size());
  candidate_mean /= static_cast<double>(candidate_ms.size());
  const double saving = control_mean - candidate_mean;
  std::printf(
      "primitive_position=%zu n_parts=%d occupancy=%d ntiles_kv=%d "
      "control_ms=%.9g candidate_ms=%.9g saving_ms=%.9g faster=%s "
      "prepared_q_once=%s nvidia_float2=%s\n",
      position, check.n_parts, check.occupancy, check.ntiles_kv, control_mean,
      candidate_mean, saving, saving > 0.0 ? "true" : "false",
      check.prepared_q_once ? "true" : "false",
      check.nvidia_float2 ? "true" : "false");
  return 0;
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
  if (std::strcmp(qw38::cuda::selected_decode_attention_vec128_path(),
                  "warp_query") != 0) {
    std::fprintf(stderr, "production vec128 pin must remain warp_query\n");
    return 1;
  }
  int rc = 0;
  rc |= run_attrs();
  if (std::strcmp(workload, "primitive") == 0) {
    const int warmups = options.warmups > 0 ? options.warmups : 1;
    const int samples = options.samples > 0 ? options.samples : 3;
    const std::size_t position =
        options.position > 0 ? static_cast<std::size_t>(options.position) : 2048;
    rc |= run_case("primitive_parity", position,
                   static_cast<std::uint32_t>(position + 16), true);
    rc |= run_primitive_position(position, warmups, samples);
  } else {
    rc |= run_case("tiny_p0", 0, 32, false);
    rc |= run_case("tiny_p1", 1, 32, true);
    if (std::strcmp(workload, "smoke") != 0) {
      rc |= run_case("tiny_p31", 31, 64, false);
      rc |= run_case("tiny_p32", 32, 64, true);
      rc |= run_zero_case(1);
      rc |= run_zero_case(32);
      rc |= run_repeat_eager();
    }
    if (std::strcmp(workload, "smoke") != 0 &&
        std::strcmp(workload, "correctness") != 0) {
      rc |= run_case("prod_p127", 127, 256, false);
      rc |= run_case("prod_p128", 128, 256, true);
    } else if (std::strcmp(workload, "correctness") == 0) {
      rc |= run_case("prod_p127", 127, 256, true);
    }
    if (std::strcmp(workload, "screen") == 0 ||
        std::strcmp(workload, "acceptance") == 0) {
      const int warmups =
          std::strcmp(workload, "acceptance") == 0
              ? (options.warmups > 0 ? options.warmups : 3)
              : (options.warmups > 0 ? options.warmups : 1);
      const int samples =
          std::strcmp(workload, "acceptance") == 0
              ? (options.samples > 0 ? options.samples : 10)
              : (options.samples > 0 ? options.samples : 3);
      for (int position : kPrimitivePositions) {
        rc |= run_case("primitive_parity", static_cast<std::size_t>(position),
                       static_cast<std::uint32_t>(position + 16), true);
        rc |= run_primitive_position(static_cast<std::size_t>(position),
                                     warmups, samples);
      }
    }
    if (std::strcmp(workload, "acceptance") == 0) {
      rc |= run_case("prod_p2047", 2047, 2064, false);
      rc |= run_case("prod_p2048", 2048, 2064, true);
      rc |= run_case("prod_p4096", 4096, 4112, true);
    }
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-108\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"candidate\":\"llama_vec_nvidia\",\"control\":\"production_opt107\","
      "\"nvidia_float2_v_accum\":true,\"amd_half2_v_accum\":false,"
      "\"f16_cache_migration\":false,\"prepared_q_once\":true,"
      "\"cpy_bytes\":16,\"threads\":128,"
      "\"shipping_decode_attention_vec128\":\"%s\","
      "\"crossover_threshold\":%d,\"nonfinite\":%d,\"pass\":%s,"
      "\"keep\":false}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_decode_attention_vec128_path(),
      qw38::cuda::selected_decode_attention_crossover_threshold(),
      pass ? 0 : 1, pass ? "true" : "false");
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}
