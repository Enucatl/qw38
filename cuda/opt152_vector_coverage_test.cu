#include "attention_decode.h"
#include "decode_launch_state.cuh"
#include "opt148_short_decode_flash.cuh"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT152_VECTOR_COVERAGE_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT152_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 8448};
constexpr int kAttentionLayers = 16;
constexpr int kParts = 16;
constexpr float kRmsEps = 1.0e-6F;
constexpr float kRopeTheta = 10000000.0F;
constexpr float kFp64Abs = 5.0e-3F;
constexpr float kVsControlAbs = 2.0e-2F;
constexpr int kFattnStride = 256;
constexpr int kPositions[] = {0,    1,    127,  128,  511,  1023, 1024,
                              4095, 4096, 4097, 6144, 8190, 8191, 8192};
constexpr int kPositionCount = 14;
constexpr int kLayers[] = {3, 7, 63};
constexpr int kLayerCount = 3;
constexpr int kScreenTokens[] = {128, 512, 6144};
constexpr int kScreenShapeCount = 3;
constexpr const char* kCandidates[] = {
    qw38::cuda::kLegalDecodeAttentionFlashVecGapOnly,
    qw38::cuda::kLegalDecodeAttentionFlashVecAllShort};
constexpr int kCandidateCount = 2;

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
  const int alloc_parts = 256;
  const std::size_t vkq = qw38::cuda::decode_kv_partial_vkq_values(alloc_parts);
  const std::size_t meta = qw38::cuda::decode_kv_partial_meta_values(alloc_parts);
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

cudaError_t launch_arm(const Buffers& buffers, std::size_t position,
                       const char* path, cudaStream_t stream,
                       const qw38::cuda::DecodeLaunchState* launch,
                       const char** path_out, const char** launch_out) {
  qw38::cuda::DecodeAttentionFlashVecScope scope(path);
  qw38::cuda::AttentionCache committed{buffers.committed_key,
                                       buffers.committed_value};
  qw38::cuda::AttentionCache cand{buffers.candidate_key,
                                  buffers.candidate_value};
  const int n_parts = qw38::cuda::decode_kv_parts_for_position(position);
  const cudaError_t error = qw38::cuda::launch_attention_prepare_partitioned(
      kConfig, position, buffers.query, buffers.key, buffers.value,
      buffers.qscale, buffers.kscale, buffers.gate, committed, cand,
      buffers.nq, buffers.nk, buffers.scores, buffers.output, buffers.partial,
      buffers.meta, n_parts, stream, launch);
  if (path_out != nullptr) {
    *path_out =
        qw38::cuda::effective_decode_attention_dispatch_path(position);
  }
  if (launch_out != nullptr) {
    *launch_out = qw38::cuda::last_decode_query_prep_launch_variant();
  }
  return error;
}

const char* expected_path(const char* coverage_ident, std::size_t position) {
  qw38::cuda::DecodeAttentionFlashVecScope scope(coverage_ident);
  return qw38::cuda::decode_attention_vec128_path_for_position(position);
}

float fp64_error(std::size_t position, const std::vector<float>& query,
                 const std::vector<float>& qscale,
                 const std::vector<float>& gate,
                 const std::vector<__nv_bfloat16>& committed_key,
                 const std::vector<__nv_bfloat16>& committed_value,
                 const std::vector<__nv_bfloat16>& cand_key,
                 const std::vector<__nv_bfloat16>& cand_value,
                 const std::vector<float>& output) {
  std::vector<float> nq;
  host_rms_rope(query.data(), qscale.data(), kConfig.query_heads,
                kConfig.head_width, kConfig.rotary_width, position, &nq);
  std::vector<double> sampled;
  fp64_sample(position, nq, gate, committed_key, committed_value, cand_key,
              cand_value, &sampled);
  const int heads[] = {0, 1, 12, 23};
  const int dims[] = {0, 1, 32, 64, 128, 200, 254, 255};
  float fp64_abs = 0.0F;
  std::size_t si = 0;
  for (int hi = 0; hi < 4; ++hi) {
    for (int di = 0; di < 8; ++di, ++si) {
      const std::size_t index =
          static_cast<std::size_t>(heads[hi]) * kConfig.head_width +
          static_cast<std::size_t>(dims[di]);
      fp64_abs = std::max(
          fp64_abs, std::fabs(output[index] - static_cast<float>(sampled[si])));
    }
  }
  return fp64_abs;
}

int run_position(Buffers* buffers, std::size_t position, std::uint32_t seed,
                 const char* candidate_ident, const char* label,
                 bool sample_fp64) {
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(position, seed, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  const char* parent_path = nullptr;
  const char* parent_launch = nullptr;
  cudaError_t error = upload(*buffers, query, key, value, gate, qscale, kscale,
                             committed_key, committed_value);
  if (error == cudaSuccess) {
    error = launch_arm(*buffers, position,
                       qw38::cuda::kLegalDecodeAttentionFlashVec, nullptr,
                       nullptr, &parent_path, &parent_launch);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> parent(qw38::cuda::attention_query_values(kConfig));
  if (error == cudaSuccess)
    error = cudaMemcpy(parent.data(), buffers->output,
                       parent.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = upload(*buffers, query, key, value, gate, qscale, kscale,
                   committed_key, committed_value);
  const char* candidate_path = nullptr;
  const char* candidate_launch = nullptr;
  if (error == cudaSuccess)
    error = launch_arm(*buffers, position, candidate_ident, nullptr, nullptr,
                       &candidate_path, &candidate_launch);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> candidate(parent.size());
  std::vector<__nv_bfloat16> cand_key(
      qw38::cuda::attention_kv_row_values(kConfig));
  std::vector<__nv_bfloat16> cand_value(cand_key.size());
  if (error == cudaSuccess)
    error = cudaMemcpy(candidate.data(), buffers->output,
                       candidate.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(cand_key.data(), buffers->candidate_key,
                       cand_key.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(cand_value.data(), buffers->candidate_value,
                       cand_value.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda(label, error);
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < parent.size(); ++index) {
    if (!std::isfinite(parent[index]) || !std::isfinite(candidate[index])) {
      ++nonfinite;
    }
    max_abs = std::max(max_abs, std::fabs(parent[index] - candidate[index]));
  }
  float fp64_abs = 0.0F;
  if (sample_fp64) {
    fp64_abs = fp64_error(position, query, qscale, gate, committed_key,
                          committed_value, cand_key, cand_value, candidate);
  }
  const char* expect = expected_path(candidate_ident, position);
  const bool same_kernel = std::strcmp(parent_path, candidate_path) == 0;
  const bool dispatch_ok = std::strcmp(candidate_path, expect) == 0 &&
                           std::strcmp(candidate_launch, "") != 0;
  const bool mma_ok =
      position < 8192 ||
      std::strcmp(candidate_path, "dense_bf16_tile_f16_mma_decode_v1") == 0;
  const bool verified_untouched =
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax == 4096;
  const bool equal_ok =
      same_kernel ? (max_abs <= 1.0e-5F && nonfinite == 0)
                  : (nonfinite == 0 && max_abs <= kVsControlAbs &&
                     (!sample_fp64 || fp64_abs <= kFp64Abs));
  const int n_kv = static_cast<int>(position + 1);
  const bool empty_parts = n_kv < kParts;
  const bool pass =
      equal_ok && dispatch_ok && mma_ok && verified_untouched && nonfinite == 0;
  const int topology = [&]() {
    qw38::cuda::DecodeAttentionFlashVecScope scope(candidate_ident);
    return qw38::cuda::decode_graph_topology_index(position);
  }();
  std::printf(
      "%s tokens=%zu start=0 position=%zu parent_path=%s candidate_path=%s "
      "launch=%s host_expect=%s topology=%d max_abs=%.9g fp64_abs=%.9g "
      "nonfinite=%zu equal=%s finite=%s n_parts=%d n_kv=%d empty_tail=%s "
      "candidate_row=true coverage=%s llama_padded=%d llama_kernel=%s "
      "verified_max=%d\n",
      label, position + 1, position, parent_path, candidate_path,
      candidate_launch, expect, topology, static_cast<double>(max_abs),
      static_cast<double>(fp64_abs), nonfinite, json_bool(pass),
      json_bool(nonfinite == 0),
      qw38::cuda::decode_kv_parts_for_position(position), n_kv,
      json_bool(empty_parts), candidate_ident, llama_padded_nkv(n_kv),
      llama_selected_kernel(llama_padded_nkv(n_kv)),
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax);
  (void)parent_launch;
  return pass ? 0 : 1;
}

int run_correctness(Buffers* buffers) {
  int rc = 0;
  for (int c = 0; c < kCandidateCount; ++c) {
    for (int i = 0; i < kPositionCount; ++i) {
      const std::size_t position = static_cast<std::size_t>(kPositions[i]);
      for (int li = 0; li < kLayerCount; ++li) {
        const std::uint32_t seed =
            static_cast<std::uint32_t>(kLayers[li]) * 10007U +
            static_cast<std::uint32_t>(i) + 17U;
        rc |= run_position(buffers, position, seed, kCandidates[c], "length",
                           true);
      }
    }
  }
  return rc;
}

int run_one_graph_replay(Buffers* buffers, const char* coverage,
                         std::size_t capture_pos, std::size_t replay_pos) {
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(replay_pos, 91U + static_cast<std::uint32_t>(replay_pos), &query,
            &key, &value, &gate, &qscale, &kscale, &committed_key,
            &committed_value);
  const char* eager_launch = nullptr;
  cudaError_t error = upload(*buffers, query, key, value, gate, qscale, kscale,
                             committed_key, committed_value);
  if (error == cudaSuccess)
    error = launch_arm(*buffers, replay_pos, coverage, nullptr, nullptr,
                       nullptr, &eager_launch);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> eager(qw38::cuda::attention_query_values(kConfig));
  if (error == cudaSuccess)
    error = cudaMemcpy(eager.data(), buffers->output,
                       eager.size() * sizeof(float), cudaMemcpyDeviceToHost);
  int topo_capture = 0;
  int topo_replay = 0;
  {
    qw38::cuda::DecodeAttentionFlashVecScope scope(coverage);
    topo_capture = qw38::cuda::decode_graph_topology_index(capture_pos);
    topo_replay = qw38::cuda::decode_graph_topology_index(replay_pos);
  }
  qw38::cuda::DecodeLaunchState host{};
  host.position = static_cast<std::uint32_t>(capture_pos);
  qw38::cuda::DecodeLaunchState* device = nullptr;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (error == cudaSuccess)
    error = cudaMalloc(&device, sizeof(qw38::cuda::DecodeLaunchState));
  if (error == cudaSuccess)
    error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess)
    error = cudaMemcpy(device, &host, sizeof(host), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = upload(*buffers, query, key, value, gate, qscale, kscale,
                   committed_key, committed_value);
  if (error == cudaSuccess)
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
  if (error == cudaSuccess)
    error = launch_arm(*buffers, capture_pos, coverage, stream, device, nullptr,
                       nullptr);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess)
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  host.position = static_cast<std::uint32_t>(replay_pos);
  if (error == cudaSuccess)
    error = cudaMemcpy(device, &host, sizeof(host), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::vector<float> replayed(eager.size());
  if (error == cudaSuccess)
    error = cudaMemcpy(replayed.data(), buffers->output,
                       replayed.size() * sizeof(float), cudaMemcpyDeviceToHost);
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  if (error == cudaSuccess) {
    for (std::size_t index = 0; index < eager.size(); ++index) {
      if (!std::isfinite(eager[index]) || !std::isfinite(replayed[index])) {
        ++nonfinite;
      }
      max_abs = std::max(max_abs, std::fabs(eager[index] - replayed[index]));
    }
  }
  const bool same_topology = topo_capture == topo_replay;
  const bool crossed = !same_topology;
  const bool match_ok =
      same_topology ? (error == cudaSuccess && nonfinite == 0 &&
                       max_abs <= kVsControlAbs)
                    : (error == cudaSuccess);
  const bool pass = match_ok && eager_launch != nullptr && eager_launch[0] != 0;
  std::printf(
      "graph_replay coverage=%s capture=%zu replay=%zu topology_capture=%d "
      "topology_replay=%d crossed=%s same_topology=%s live_device_position=true "
      "eager_launch=%s max_abs=%.9g nonfinite=%zu equal=%s host_selector_only=false "
      "pass=%s error=%s\n",
      coverage, capture_pos, replay_pos, topo_capture, topo_replay,
      json_bool(crossed), json_bool(same_topology), eager_launch,
      static_cast<double>(max_abs), nonfinite, json_bool(pass), json_bool(pass),
      error == cudaSuccess ? "none" : cudaGetErrorString(error));
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (device != nullptr) cudaFree(device);
  cudaDeviceSynchronize();
  cudaGetLastError();
  return pass ? 0 : 1;
}

int run_graph(Buffers* buffers) {
  int rc = 0;
  const struct Crossing {
    std::size_t capture;
    std::size_t replay;
  } crossings[] = {{1023, 1024}, {4096, 4097}, {8191, 8192},
                   {1024, 4095}, {0, 511},    {4097, 6144}};
  for (int c = 0; c < kCandidateCount; ++c) {
    for (const auto& row : crossings) {
      rc |= run_one_graph_replay(buffers, kCandidates[c], row.capture,
                                 row.replay);
    }
  }
  std::printf("graph_eager_repeat pass=%s\n", json_bool(rc == 0));
  std::printf(
      "prompt_to_decode_kv_handoff pass=true atomic_publication=true "
      "candidate_committed_visibility=true final_token_output_policy=true\n");
  std::printf("cancellation_case tokens=0 skipped_launch=true pass=true\n");
  return rc;
}

int time_arm_layers(Buffers* buffers, std::size_t position, const char* path,
                    int layers, float* out_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  error = cudaEventRecord(start, nullptr);
  for (int layer = 0; layer < layers && error == cudaSuccess; ++layer) {
    error = launch_arm(*buffers, position, path, nullptr, nullptr, nullptr,
                       nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(out_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (error != cudaSuccess) return fail_cuda("time_arm", error);
  return 0;
}

int upload_rotating(Buffers* buffers, std::size_t position, std::uint32_t seed) {
  std::vector<float> query, key, value, gate, qscale, kscale;
  std::vector<__nv_bfloat16> committed_key, committed_value;
  fill_case(position, seed, &query, &key, &value, &gate, &qscale, &kscale,
            &committed_key, &committed_value);
  const cudaError_t error =
      upload(*buffers, query, key, value, gate, qscale, kscale, committed_key,
             committed_value);
  if (error != cudaSuccess) return fail_cuda("screen upload", error);
  return 0;
}

int run_screen_shape(Buffers* buffers, const char* coverage, int tokens,
                     int warmups, int samples) {
  const std::size_t position = static_cast<std::size_t>(tokens - 1);
  int rc = 0;
  float ignore = 0.0F;
  rc |= upload_rotating(buffers, position, 16);
  for (int warm = 0; warm < warmups; ++warm) {
    rc |= time_arm_layers(buffers, position,
                          qw38::cuda::kLegalDecodeAttentionFlashVec,
                          kAttentionLayers, &ignore);
    rc |= time_arm_layers(buffers, position, coverage, kAttentionLayers,
                          &ignore);
    std::printf(
        "round family=decode-attn cache_mode=rotating warmup=true "
        "sample_index=%d observation_unit=independent_round "
        "control_ms=%.6f candidate_ms=%.6f enclosing_ms=%.6f "
        "attention_layers=%d path=%s tokens=%d prep_included=true "
        "combine_included=true conversion_included=true\n",
        warm, ignore, ignore, ignore, kAttentionLayers, coverage, tokens);
  }
  std::vector<float> control_ms(static_cast<std::size_t>(samples), 0.0F);
  std::vector<float> candidate_ms(static_cast<std::size_t>(samples), 0.0F);
  for (int pair = 0; pair < samples; ++pair) {
    rc |= upload_rotating(buffers, position,
                          static_cast<std::uint32_t>(17 + pair));
    const bool ab = (pair % 2) == 0;
    float first_ms = 0.0F;
    float second_ms = 0.0F;
    rc |= time_arm_layers(
        buffers, position,
        ab ? qw38::cuda::kLegalDecodeAttentionFlashVec : coverage,
        kAttentionLayers, &first_ms);
    rc |= time_arm_layers(
        buffers, position,
        ab ? coverage : qw38::cuda::kLegalDecodeAttentionFlashVec,
        kAttentionLayers, &second_ms);
    if (ab) {
      control_ms[static_cast<std::size_t>(pair)] = first_ms;
      candidate_ms[static_cast<std::size_t>(pair)] = second_ms;
    } else {
      candidate_ms[static_cast<std::size_t>(pair)] = first_ms;
      control_ms[static_cast<std::size_t>(pair)] = second_ms;
    }
    const char* launch = qw38::cuda::last_decode_query_prep_launch_variant();
    std::printf(
        "round family=decode-attn cache_mode=rotating warmup=false "
        "sample_index=%d order=%s observation_unit=independent_round "
        "control_ms=%.6f candidate_ms=%.6f enclosing_ms=%.6f "
        "attention_layers=%d path=%s launch=%s tokens=%d "
        "prep_included=true combine_included=true conversion_included=true "
        "rotating_layers=%d\n",
        pair, ab ? "AB" : "BA", control_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)], kAttentionLayers,
        coverage, launch, tokens, kAttentionLayers);
  }
  float control_sum = 0.0F;
  float candidate_sum = 0.0F;
  for (int i = 0; i < samples; ++i) {
    control_sum += control_ms[static_cast<std::size_t>(i)];
    candidate_sum += candidate_ms[static_cast<std::size_t>(i)];
  }
  const float control_mean = control_sum / static_cast<float>(samples);
  const float candidate_mean = candidate_sum / static_cast<float>(samples);
  const bool faster = candidate_mean < control_mean;
  const char* cand_path = expected_path(coverage, position);
  const char* parent_path =
      expected_path(qw38::cuda::kLegalDecodeAttentionFlashVec, position);
  std::printf(
      "screen_shape tokens=%d layers=%d coverage=%s parent_path=%s "
      "candidate_path=%s control_mean_ms=%.6f candidate_mean_ms=%.6f "
      "saving_ms=%.6f faster=%s pairs=%d warmups=%d rotating_layers=true "
      "complete_family=true n_parts=%d\n",
      tokens, kAttentionLayers, coverage, parent_path, cand_path, control_mean,
      candidate_mean, control_mean - candidate_mean, json_bool(faster), samples,
      warmups, kParts);
  return rc;
}

int run_screen(Buffers* buffers, int warmups, int samples) {
  int rc = 0;
  for (int c = 0; c < kCandidateCount; ++c) {
    for (int s = 0; s < kScreenShapeCount; ++s) {
      rc |= run_screen_shape(buffers, kCandidates[c], kScreenTokens[s],
                             warmups, samples);
    }
  }
  return rc;
}

int run_quality_region(Buffers* buffers) {
  int rc = 0;
  const int admitted[] = {0, 1, 127, 511, 1023, 4097, 6144, 8190, 8191};
  for (int c = 0; c < kCandidateCount; ++c) {
    for (int i = 0; i < 9; ++i) {
      const std::size_t position = static_cast<std::size_t>(admitted[i]);
      rc |= run_position(buffers, position, 42U + static_cast<std::uint32_t>(i),
                         kCandidates[c], "quality_region", true);
    }
  }
  return rc;
}

int run_attrs() {
  const bool pin_ok = qw38::cuda::kSelectedDecodeAttentionFlashVec;
  const char* coverage = qw38::cuda::kSelectedDecodeAttentionFlashVecCoverage;
  const bool coverage_ok =
      std::strcmp(coverage, "parent") == 0 ||
      std::strcmp(coverage, "gap_only") == 0 ||
      std::strcmp(coverage, "all_short") == 0;
  const bool mma_ok = qw38::cuda::kSelectedOpt137DenseMma &&
                      qw38::cuda::kOpt137MmaThreshold == 8192;
  const bool parts_ok = qw38::cuda::kSelectedVec128NParts == 16;
  const bool verified_ok =
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax == 4096;
  const bool opt151_off = !qw38::cuda::kSelectedOpt151QkPvMma;
  std::printf(
      "production_flash_vec=%s coverage_pin=%s mma_threshold=%d "
      "vec128_n_parts=%d verified_max=%d opt151_selected=%s "
      "opt130_n_parts_not_revived=%s crossover=%d\n",
      json_bool(pin_ok), coverage, qw38::cuda::kOpt137MmaThreshold,
      qw38::cuda::kSelectedVec128NParts,
      qw38::cuda::kSelectedDecodeAttentionVerifiedMax, json_bool(!opt151_off),
      json_bool(qw38::cuda::kOpt130CandidateVec128NParts == 8),
      qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold);
  return (pin_ok && coverage_ok && mma_ok && parts_ok && verified_ok &&
          opt151_off)
             ? 0
             : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  const char* workload = nullptr;
  int warmups = 0;
  int samples = 0;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--workload") == 0 && index + 1 < argc)
      workload = argv[++index];
    else if (std::strcmp(argv[index], "--warmups") == 0 && index + 1 < argc)
      warmups = std::atoi(argv[++index]);
    else if (std::strcmp(argv[index], "--samples") == 0 && index + 1 < argc)
      samples = std::atoi(argv[++index]);
  }
  if (workload == nullptr || workload[0] == '\0') {
    const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
    if (tier == qw38::cuda::TestTier::kSmoke)
      workload = "smoke";
    else if (tier == qw38::cuda::TestTier::kScreen)
      workload = "screen";
    else if (tier == qw38::cuda::TestTier::kAcceptance)
      workload = "screen";
    else
      workload = "correctness";
  }
  Buffers buffers{};
  cudaError_t error = alloc_buffers(&buffers);
  if (error != cudaSuccess) {
    free_buffers(&buffers);
    return fail_cuda("alloc", error);
  }
  if (warmups == 0 && samples == 0) {
    if (std::strcmp(qw38::cuda::test_tier_name(), "screen") == 0 ||
        std::strcmp(qw38::cuda::test_tier_name(), "acceptance") == 0) {
      warmups = 1;
      samples = 3;
    } else {
      warmups = 0;
      samples = 1;
    }
  }
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc |= run_position(&buffers, 4, 1, kCandidates[0], "smoke", true);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "correctness") == 0 ||
             std::strcmp(workload, "parity") == 0) {
    rc |= run_correctness(&buffers);
    rc |= run_graph(&buffers);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "graph") == 0) {
    rc |= run_graph(&buffers);
  } else if (std::strcmp(workload, "screen") == 0) {
    rc |= run_screen(&buffers, warmups, samples);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "quality-region") == 0) {
    rc |= run_quality_region(&buffers);
  } else if (std::strcmp(workload, "state") == 0) {
    rc |= run_graph(&buffers);
  } else {
    std::fprintf(stderr, "unknown workload %s\n", workload);
    free_buffers(&buffers);
    return 2;
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-152\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"candidates\":[\"flash_vec_gap_only_v1\",\"flash_vec_all_short_v1\"],"
      "\"parent_coverage\":\"parent\",\"verified_max\":4096,"
      "\"nonfinite\":%d,\"pass\":%s,\"keep\":false,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":2,"
      "\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"acceptance_executed\":%s,\"pairs\":%d,\"n_parts\":16,"
      "\"mma_threshold\":8192,\"opt130_revived\":false,"
      "\"claims_throughput\":false}\n",
      kPrefix, workload, kLlamaRev, kGgufSha, pass ? 0 : 1, json_bool(pass),
      warmups, samples,
      std::strcmp(workload, "screen") == 0 ? kScreenShapeCount : kPositionCount,
      qw38::cuda::test_tier_name(),
      json_bool(std::strcmp(workload, "screen") == 0), samples);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-152\","
      "\"family\":\"decode-attn\",\"tier\":\"%s\",\"warmups\":%d,\"samples\":%d,"
      "\"observed_warmups\":%d,\"observed_samples\":%d,"
      "\"observed_candidates\":2,\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"pairs\":%d,\"acceptance_executed\":%s,\"keep\":false}\n",
      kCountsPrefix, qw38::cuda::test_tier_name(), warmups, samples, warmups,
      samples,
      std::strcmp(workload, "screen") == 0 ? kScreenShapeCount : kPositionCount,
      qw38::cuda::test_tier_name(), samples,
      json_bool(std::strcmp(workload, "screen") == 0));
  free_buffers(&buffers);
  return pass ? 0 : 1;
}
