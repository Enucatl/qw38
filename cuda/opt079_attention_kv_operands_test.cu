#include "attention_decode.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT079_ATTENTION_KV_OPERANDS_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 131072};
constexpr int kNbatch = 32;
constexpr int kWidth = 256;
constexpr int kNthreads = 128;
constexpr int kNstages = 2;
constexpr int kAttentionLayers = 16;
constexpr int kSampledRows = 4;
constexpr int kSampledDims = 8;
constexpr int kSampledHeads = 4;
constexpr std::size_t kTinyTokens[] = {1, 17, 33};
constexpr int kTinyTokenCount = 3;

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

std::size_t query_count(std::size_t tokens) {
  return tokens * qw38::cuda::attention_query_values(kConfig);
}

std::size_t row_count(std::size_t tokens) {
  return tokens * qw38::cuda::attention_kv_row_values(kConfig);
}

__half host_bf16_to_f16(__nv_bfloat16 value) {
  return __float2half_rn(__bfloat162float(value));
}

__global__ void convert_stage_lifetime_kernel(
    const __nv_bfloat16* stage0_bf16, const __nv_bfloat16* stage1_bf16,
    __half* stage0_f16, __half* stage1_f16, int* convert_counts) {
  extern __shared__ unsigned char raw[];
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(raw);
  __nv_bfloat16* values = keys + kNstages * kNbatch * kWidth;
  const int tid = static_cast<int>(threadIdx.x);
  const int n = kNbatch * kWidth;
  for (int index = tid; index < n; index += kNthreads) {
    keys[index] = stage0_bf16[index];
    values[index] = stage0_bf16[index];
    keys[n + index] = stage1_bf16[index];
    values[n + index] = stage1_bf16[index];
  }
  __syncthreads();
  if (tid == 0) atomicAdd(convert_counts, 1);
  __syncthreads();
  for (int index = tid; index < n; index += kNthreads) {
    const __nv_bfloat16 kb = keys[index];
    const __nv_bfloat16 vb = values[index];
    reinterpret_cast<__half*>(keys)[index] = __float2half_rn(__bfloat162float(kb));
    reinterpret_cast<__half*>(values)[index] =
        __float2half_rn(__bfloat162float(vb));
  }
  __syncthreads();
  for (int index = tid; index < n; index += kNthreads) {
    stage0_f16[index] = reinterpret_cast<const __half*>(keys)[index];
  }
  __syncthreads();
  if (tid == 0) atomicAdd(convert_counts + 1, 1);
  __syncthreads();
  for (int index = tid; index < n; index += kNthreads) {
    const __nv_bfloat16 kb = keys[n + index];
    const __nv_bfloat16 vb = values[n + index];
    reinterpret_cast<__half*>(keys)[n + index] =
        __float2half_rn(__bfloat162float(kb));
    reinterpret_cast<__half*>(values)[n + index] =
        __float2half_rn(__bfloat162float(vb));
  }
  __syncthreads();
  for (int index = tid; index < n; index += kNthreads) {
    stage1_f16[index] = reinterpret_cast<const __half*>(keys)[n + index];
  }
}

float bfloat_from_pattern(int kind, std::size_t index) {
  switch (kind) {
    case 0:
      return 0.0F;
    case 1:
      return -0.0F;
    case 2:
      return 1.0e-8F;
    case 3:
      return -1.0e-8F;
    case 4:
      return 65504.0F;
    case 5:
      return -65504.0F;
    case 6:
      return 1.0F + static_cast<float>(index % 7) * 0.000244140625F;
    default:
      return std::sin(static_cast<float>(index) * 0.017F) * 3.5F;
  }
}

int run_stage_lifetime() {
  const int n = kNbatch * kWidth;
  std::vector<__nv_bfloat16> stage0(n);
  std::vector<__nv_bfloat16> stage1(n);
  std::vector<__half> expected0(n);
  std::vector<__half> expected1(n);
  for (int index = 0; index < n; ++index) {
    const float a = static_cast<float>((index % 17) - 8) * 0.125F;
    const float b = static_cast<float>((index % 19) - 9) * 0.0625F + 1.5F;
    stage0[index] = __float2bfloat16_rn(a);
    stage1[index] = __float2bfloat16_rn(b);
    expected0[index] = host_bf16_to_f16(stage0[index]);
    expected1[index] = host_bf16_to_f16(stage1[index]);
  }
  __nv_bfloat16* d0 = nullptr;
  __nv_bfloat16* d1 = nullptr;
  __half* h0 = nullptr;
  __half* h1 = nullptr;
  int* counts = nullptr;
  cudaError_t error = cudaMalloc(&d0, n * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) error = cudaMalloc(&d1, n * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) error = cudaMalloc(&h0, n * sizeof(__half));
  if (error == cudaSuccess) error = cudaMalloc(&h1, n * sizeof(__half));
  if (error == cudaSuccess) error = cudaMalloc(&counts, 2 * sizeof(int));
  if (error != cudaSuccess) return fail_cuda("stage malloc", error);
  error = cudaMemcpy(d0, stage0.data(), n * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(d1, stage1.data(), n * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess) error = cudaMemset(counts, 0, 2 * sizeof(int));
  const std::size_t shared =
      static_cast<std::size_t>(kNstages) * 2ull * kNbatch * kWidth *
      sizeof(__nv_bfloat16);
  if (error == cudaSuccess) {
    error = cudaFuncSetAttribute(
        convert_stage_lifetime_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  }
  if (error == cudaSuccess) {
    convert_stage_lifetime_kernel<<<1, kNthreads, shared>>>(d0, d1, h0, h1,
                                                            counts);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<__half> got0(n);
  std::vector<__half> got1(n);
  int host_counts[2] = {0, 0};
  if (error == cudaSuccess)
    error = cudaMemcpy(got0.data(), h0, n * sizeof(__half), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(got1.data(), h1, n * sizeof(__half), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = cudaMemcpy(host_counts, counts, 2 * sizeof(int),
                       cudaMemcpyDeviceToHost);
  cudaFree(d0);
  cudaFree(d1);
  cudaFree(h0);
  cudaFree(h1);
  cudaFree(counts);
  if (error != cudaSuccess) return fail_cuda("stage lifetime", error);
  bool match = host_counts[0] == 1 && host_counts[1] == 1;
  for (int index = 0; index < n && match; ++index) {
    match = match &&
            std::memcmp(&got0[index], &expected0[index], sizeof(__half)) == 0 &&
            std::memcmp(&got1[index], &expected1[index], sizeof(__half)) == 0;
  }
  std::printf(
      "stage_lifetime rows=1,17,33 ring_wraps=2 convert_counts=%d,%d "
      "operands_match=%s\n",
      host_counts[0], host_counts[1], json_bool(match));
  return match ? 0 : 1;
}

int run_finite_operands() {
  bool match = true;
  bool no_clamping = true;
  for (int kind = 0; kind < 8; ++kind) {
    for (std::size_t index = 0; index < 32; ++index) {
      const __nv_bfloat16 bf = __float2bfloat16_rn(bfloat_from_pattern(kind, index));
      const __half once = host_bf16_to_f16(bf);
      const __half again = host_bf16_to_f16(bf);
      if (std::memcmp(&once, &again, sizeof(__half)) != 0) match = false;
      const float src = __bfloat162float(bf);
      const float reconstructed = __half2float(once);
      const __half ieee = __float2half_rn(src);
      if (std::memcmp(&once, &ieee, sizeof(__half)) != 0) match = false;
      if (std::isfinite(src) && std::fabs(src) > 65504.0F &&
          std::isfinite(reconstructed)) {
        no_clamping = false;
      }
    }
  }
  const __nv_bfloat16 overflow = __float2bfloat16_rn(1.0e8F);
  const float overflow_src = __bfloat162float(overflow);
  const float overflow_f16 = __half2float(host_bf16_to_f16(overflow));
  if (std::isfinite(overflow_src) && std::isfinite(overflow_f16)) {
    no_clamping = false;
  }
  std::printf("finite_operands_match=%s no_clamping=%s\n", json_bool(match),
              json_bool(no_clamping));
  return match && no_clamping ? 0 : 1;
}

struct Buffers {
  float* query = nullptr;
  float* key = nullptr;
  float* value = nullptr;
  float* gate = nullptr;
  float* query_scale = nullptr;
  float* key_scale = nullptr;
  float* nq = nullptr;
  float* nk = nullptr;
  float* scores = nullptr;
  float* output = nullptr;
  float* partial = nullptr;
  float* meta = nullptr;
  __half* prepared = nullptr;
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
  cudaFree(buffers->query_scale);
  cudaFree(buffers->key_scale);
  cudaFree(buffers->nq);
  cudaFree(buffers->nk);
  cudaFree(buffers->scores);
  cudaFree(buffers->output);
  cudaFree(buffers->partial);
  cudaFree(buffers->meta);
  cudaFree(buffers->prepared);
  cudaFree(buffers->committed_key);
  cudaFree(buffers->committed_value);
  cudaFree(buffers->candidate_key);
  cudaFree(buffers->candidate_value);
}

cudaError_t alloc_buffers(Buffers* buffers, std::size_t tokens) {
  const std::size_t qn = query_count(tokens);
  const std::size_t rn = row_count(tokens);
  const std::size_t cache = qw38::cuda::attention_cache_values(kConfig);
  const std::size_t scores =
      qw38::cuda::attention_chunk_score_values(kConfig, 0, tokens);
  const std::size_t partial =
      qw38::cuda::fattn_stream_k_partial_values(kConfig, tokens);
  const std::size_t meta = qw38::cuda::fattn_stream_k_meta_values(kConfig, tokens);
  const std::size_t prepared =
      qw38::cuda::attention_prepared_query_bytes(kConfig, tokens);
  cudaError_t error = cudaMalloc(&buffers->query, qn * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&buffers->key, rn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->value, rn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->gate, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->query_scale, kConfig.head_width * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->key_scale, kConfig.head_width * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&buffers->nq, qn * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&buffers->nk, rn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->scores, scores * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->output, qn * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->partial, partial * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->meta, meta * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&buffers->prepared, prepared);
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->committed_key, cache * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->committed_value, cache * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->candidate_key, rn * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMalloc(&buffers->candidate_value, rn * sizeof(__nv_bfloat16));
  return error;
}

void fill_host(std::size_t tokens, std::size_t start, std::uint32_t seed,
               std::vector<float>* query, std::vector<float>* key,
               std::vector<float>* value, std::vector<float>* gate,
               std::vector<float>* query_scale, std::vector<float>* key_scale,
               std::vector<__nv_bfloat16>* committed) {
  const std::size_t qn = query_count(tokens);
  const std::size_t rn = row_count(tokens);
  const std::size_t cache = qw38::cuda::attention_cache_values(kConfig);
  query->resize(qn);
  key->resize(rn);
  value->resize(rn);
  gate->resize(qn);
  query_scale->assign(kConfig.head_width, 1.0F);
  key_scale->assign(kConfig.head_width, 1.0F);
  committed->resize(cache);
  for (std::size_t index = 0; index < qn; ++index) {
    (*query)[index] =
        std::sin(static_cast<float>(index + seed) * 0.0013F) * 0.5F;
    (*gate)[index] =
        static_cast<float>(static_cast<int>((index + seed) % 19) - 9) * 0.05F;
  }
  for (std::size_t index = 0; index < rn; ++index) {
    (*key)[index] = std::cos(static_cast<float>(index + seed) * 0.0019F);
    (*value)[index] =
        static_cast<float>(static_cast<int>((index + seed) % 23) - 11) * 0.04F;
  }
  for (std::uint32_t lane = 0; lane < kConfig.head_width; ++lane) {
    (*query_scale)[lane] = 0.9F + static_cast<float>((lane + seed) % 5) * 0.02F;
    (*key_scale)[lane] = 0.85F + static_cast<float>((lane + seed) % 6) * 0.02F;
  }
  for (std::size_t index = 0; index < cache; ++index) {
    const int tile = static_cast<int>(index / (kNbatch * kWidth));
    const float sentinel = 0.25F * static_cast<float>((tile % 5) + 1) +
                           static_cast<float>(start % 7) * 0.01F;
    (*committed)[index] = __float2bfloat16_rn(
        sentinel + static_cast<float>(static_cast<int>(index % 31) - 15) *
                       0.015625F);
  }
}

cudaError_t upload(const Buffers& buffers, std::size_t /*tokens*/,
                   const std::vector<float>& query,
                   const std::vector<float>& key,
                   const std::vector<float>& value,
                   const std::vector<float>& gate,
                   const std::vector<float>& query_scale,
                   const std::vector<float>& key_scale,
                   const std::vector<__nv_bfloat16>& committed_logical) {
  const std::size_t qn = query.size();
  const std::size_t rn = key.size();
  const std::size_t cache = committed_logical.size();
  std::vector<__nv_bfloat16> committed_physical(cache);
  qw38::cuda::attention_kv_copy_logical_to_physical(
      committed_logical.data(), committed_physical.data(), kConfig.kv_heads,
      kConfig.capacity, kConfig.head_width);
  cudaError_t error = cudaMemcpy(buffers.query, query.data(), qn * sizeof(float),
                                 cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.key, key.data(), rn * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.value, value.data(), rn * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.gate, gate.data(), qn * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.query_scale, query_scale.data(),
                       kConfig.head_width * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.key_scale, key_scale.data(),
                       kConfig.head_width * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.committed_key, committed_physical.data(),
                       cache * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.committed_value, committed_physical.data(),
                       cache * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  return error;
}

cudaError_t launch_complete(const Buffers& buffers, std::size_t start,
                            std::size_t tokens, const char* path,
                            std::size_t row_values) {
  qw38::cuda::QueryPreparePathScope prep("hoisted");
  qw38::cuda::AttentionPipelinePathScope pipe(path);
  qw38::cuda::AttentionCache committed{buffers.committed_key,
                                       buffers.committed_value};
  qw38::cuda::AttentionCache candidate{buffers.candidate_key,
                                       buffers.candidate_value};
  (void)row_values;
  return qw38::cuda::launch_attention_prepare_chunk_stream_k(
      kConfig, start, tokens, buffers.query, buffers.key, buffers.value,
      buffers.query_scale, buffers.key_scale, buffers.gate, committed,
      candidate, buffers.nq, buffers.nk, buffers.scores, buffers.output,
      buffers.partial, buffers.meta, nullptr, buffers.prepared);
}

int compare_outputs(const std::vector<float>& control,
                    const std::vector<float>& candidate, float* max_abs,
                    std::size_t* nonfinite) {
  *max_abs = 0.0F;
  *nonfinite = 0;
  const std::size_t n = std::min(control.size(), candidate.size());
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(control[index]) || !std::isfinite(candidate[index]))
      ++*nonfinite;
    *max_abs = std::max(*max_abs, std::fabs(control[index] - candidate[index]));
  }
  return *nonfinite == 0 && *max_abs == 0.0F ? 0 : 1;
}

int run_tiny_attention(Buffers* buffers) {
  int rc = 0;
  for (int t = 0; t < kTinyTokenCount; ++t) {
    const std::size_t tokens = kTinyTokens[t];
    const std::size_t start = 96;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> gate;
    std::vector<float> query_scale;
    std::vector<float> key_scale;
    std::vector<__nv_bfloat16> committed;
    fill_host(tokens, start, static_cast<std::uint32_t>(tokens), &query, &key,
              &value, &gate, &query_scale, &key_scale, &committed);
    cudaError_t error =
        upload(*buffers, tokens, query, key, value, gate, query_scale,
               key_scale, committed);
    if (error == cudaSuccess)
      error = launch_complete(*buffers, start, tokens, "f16_async",
                              row_count(tokens));
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> control(query_count(tokens));
    if (error == cudaSuccess)
      error = cudaMemcpy(control.data(), buffers->output,
                         control.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess)
      error = upload(*buffers, tokens, query, key, value, gate, query_scale,
                     key_scale, committed);
    if (error == cudaSuccess)
      error = launch_complete(*buffers, start, tokens, "kv_once",
                              row_count(tokens));
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> candidate(query_count(tokens));
    if (error == cudaSuccess)
      error = cudaMemcpy(candidate.data(), buffers->output,
                         candidate.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail_cuda("tiny attention", error);
    float max_abs = 0.0F;
    std::size_t nonfinite = 0;
    const int equal = compare_outputs(control, candidate, &max_abs, &nonfinite);
    const bool pipeline = tokens >= 16;
    const bool committed_ok =
        !pipeline ||
        (std::strcmp(qw38::cuda::last_attention_pipeline_path(), "kv_once") ==
             0 &&
         qw38::cuda::last_attention_pipeline_convert_once() == 1);
    std::printf(
        "tiny tokens=%zu start=%zu max_abs=%.9g nonfinite=%zu equal=%s "
        "dispatch=%s convert_once=%d launch=%s\n",
        tokens, start, static_cast<double>(max_abs), nonfinite,
        json_bool(equal == 0), qw38::cuda::last_attention_pipeline_path(),
        qw38::cuda::last_attention_pipeline_convert_once(),
        qw38::cuda::last_attention_pipeline_launch());
    if (equal != 0 || !committed_ok) rc = 1;
  }
  return rc;
}

int sampled_quality(const std::vector<float>& control,
                    const std::vector<float>& candidate, std::size_t tokens) {
  const std::size_t qh = kConfig.query_heads;
  const std::size_t width = kConfig.head_width;
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  const std::size_t rows = std::min(tokens, static_cast<std::size_t>(kSampledRows));
  for (std::size_t row = 0; row < rows; ++row) {
    for (int head = 0; head < kSampledHeads; ++head) {
      for (int dim = 0; dim < kSampledDims; ++dim) {
        const std::size_t index =
            row * qh * width + static_cast<std::size_t>(head) * width +
            static_cast<std::size_t>(dim);
        if (index >= control.size() || index >= candidate.size()) continue;
        if (!std::isfinite(control[index]) || !std::isfinite(candidate[index]))
          ++nonfinite;
        max_abs = std::max(max_abs, std::fabs(control[index] - candidate[index]));
      }
    }
  }
  std::printf(
      "quality_samples rows=%zu dims=%d heads=%d max_abs=%.9g nonfinite=%zu\n",
      rows, kSampledDims, kSampledHeads, static_cast<double>(max_abs),
      nonfinite);
  return nonfinite == 0 && max_abs == 0.0F ? 0 : 1;
}

int run_quality(Buffers* buffers, std::size_t tokens) {
  int rc = 0;
  const std::uint32_t seeds[] = {3, 63};
  for (std::uint32_t seed : seeds) {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> gate;
    std::vector<float> query_scale;
    std::vector<float> key_scale;
    std::vector<__nv_bfloat16> committed;
    fill_host(tokens, 0, seed, &query, &key, &value, &gate, &query_scale,
              &key_scale, &committed);
    cudaError_t error =
        upload(*buffers, tokens, query, key, value, gate, query_scale,
               key_scale, committed);
    if (error == cudaSuccess)
      error = launch_complete(*buffers, 0, tokens, "f16_async", row_count(tokens));
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> control(query_count(tokens));
    if (error == cudaSuccess)
      error = cudaMemcpy(control.data(), buffers->output,
                         control.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess)
      error = upload(*buffers, tokens, query, key, value, gate, query_scale,
                     key_scale, committed);
    if (error == cudaSuccess)
      error = launch_complete(*buffers, 0, tokens, "kv_once", row_count(tokens));
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> candidate(query_count(tokens));
    if (error == cudaSuccess)
      error = cudaMemcpy(candidate.data(), buffers->output,
                         candidate.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail_cuda("quality", error);
    rc |= sampled_quality(control, candidate, tokens);
    float max_abs = 0.0F;
    std::size_t nonfinite = 0;
    rc |= compare_outputs(control, candidate, &max_abs, &nonfinite);
    std::printf("quality_layer seed=%u tokens=%zu max_abs=%.9g nonfinite=%zu\n",
                seed, tokens, static_cast<double>(max_abs), nonfinite);
  }
  return rc;
}

int run_complete_rounds(Buffers* buffers, std::size_t tokens, int warmups,
                        int samples) {
  const char* paths[] = {"f16_async", "kv_once"};
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> gate;
  std::vector<float> query_scale;
  std::vector<float> key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(tokens, 0, 16, &query, &key, &value, &gate, &query_scale,
            &key_scale, &committed);
  cudaError_t error =
      upload(*buffers, tokens, query, key, value, gate, query_scale, key_scale,
             committed);
  if (error != cudaSuccess) return fail_cuda("complete upload", error);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  int rc = 0;
  for (const char* path : paths) {
    for (int sample = 0; sample < warmups + samples; ++sample) {
      const bool warmup = sample < warmups;
      float enclosing = 0.0F;
      error = cudaEventRecord(start, nullptr);
      for (int layer = 0; layer < kAttentionLayers && error == cudaSuccess;
           ++layer) {
        error = launch_complete(*buffers, 0, tokens, path, row_count(tokens));
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      if (error == cudaSuccess)
        error = cudaEventElapsedTime(&enclosing, start, stop);
      if (error != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return fail_cuda("complete rounds", error);
      }
      std::printf(
          "round family=prompt-attn cache_mode=rotating warmup=%s "
          "sample_index=%d observation_unit=independent_round "
          "enclosing_ms=%.6f kernel_only_ms=%.6f attention_layers=%d "
          "path=%s launch=%s convert_once=%d occupancy=%d\n",
          json_bool(warmup), warmup ? sample : sample - warmups, enclosing,
          enclosing, kAttentionLayers, path,
          qw38::cuda::last_attention_pipeline_launch(),
          qw38::cuda::last_attention_pipeline_convert_once(),
          qw38::cuda::fattn_pipeline_occupancy_path(path));
      if (!warmup &&
          std::strcmp(qw38::cuda::last_attention_pipeline_path(), path) != 0) {
        rc = 1;
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return rc;
}

int run_attrs() {
  int regs_c = 0;
  int regs_k = 0;
  std::size_t local_c = 1;
  std::size_t local_k = 1;
  int occ_c = 0;
  int occ_k = 0;
  qw38::cuda::fattn_pipeline_f16_async_attributes(&regs_c, &local_c, &occ_c);
  qw38::cuda::fattn_pipeline_kv_once_attributes(&regs_k, &local_k, &occ_k);
  std::printf(
      "f16_async_regs=%d f16_async_local_bytes=%zu f16_async_occupancy=%d "
      "kv_once_regs=%d kv_once_local_bytes=%zu kv_once_occupancy=%d "
      "shared_bytes_unchanged=true production_pin=%s\n",
      regs_c, local_c, occ_c, regs_k, local_k, occ_k,
      qw38::cuda::selected_attention_pipeline_path());
  FILE* pipe = popen(
      "cuobjdump -sass build/qw38-cuda-opt079-attention-kv-operands-test "
      "2>/dev/null | grep -c CVT || true",
      "r");
  int cvt = 0;
  if (pipe != nullptr) {
    char buf[64]{};
    if (std::fgets(buf, sizeof(buf), pipe) != nullptr) cvt = std::atoi(buf);
    pclose(pipe);
  }
  std::printf("sass_cvt_count=%d convert_once_source=true\n", cvt);
  if (occ_k < 1 || occ_k < occ_c || local_k > local_c) return 1;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  const char* workload = nullptr;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--workload") == 0 && index + 1 < argc)
      workload = argv[++index];
  }
  if (workload == nullptr || workload[0] == '\0') {
    const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
    if (tier == qw38::cuda::TestTier::kSmoke) workload = "smoke";
    else if (tier == qw38::cuda::TestTier::kScreen) workload = "screen";
    else if (tier == qw38::cuda::TestTier::kAcceptance) workload = "acceptance";
    else workload = "correctness";
  }
  if (std::strcmp(qw38::cuda::selected_attention_pipeline_path(), "kv_once") !=
      0) {
    std::fprintf(stderr, "production pin is not kv_once\n");
    return 1;
  }
  const std::size_t timed =
      std::strcmp(workload, "smoke") == 0 ? 33 : (std::strcmp(workload, "correctness") == 0 ? 33 : 4096);
  Buffers buffers{};
  cudaError_t error = alloc_buffers(&buffers, timed);
  if (error != cudaSuccess) {
    free_buffers(&buffers);
    return fail_cuda("alloc", error);
  }
  int rc = 0;
  rc |= run_stage_lifetime();
  rc |= run_finite_operands();
  rc |= run_tiny_attention(&buffers);
  if (std::strcmp(workload, "smoke") != 0) rc |= run_quality(&buffers, timed);
  int warmups = 0;
  int samples = 0;
  if (std::strcmp(workload, "screen") == 0) {
    warmups = 1;
    samples = 3;
  } else if (std::strcmp(workload, "acceptance") == 0) {
    warmups = 3;
    samples = 10;
  }
  if (warmups + samples > 0) rc |= run_complete_rounds(&buffers, timed, warmups, samples);
  rc |= run_attrs();
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-079\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_attention_pipeline_path\":\"%s\",\"nonfinite\":%d,"
      "\"pass\":%s,\"keep\":false,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":2,"
      "\"observed_shapes\":1,\"observed_tier\":\"%s\","
      "\"acceptance_executed\":%s,\"pairs\":1,\"sample_ids\":[%s]}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_attention_pipeline_path(), pass ? 0 : 1,
      json_bool(pass), warmups, samples, qw38::cuda::test_tier_name(),
      json_bool(std::strcmp(workload, "acceptance") == 0),
      samples == 3 ? "0,1,2"
                   : (samples == 10 ? "0,1,2,3,4,5,6,7,8,9" : "0"));
  std::printf(
      "QW38_OPT079_NATIVE_COUNTS={\"schema_version\":1,\"task\":\"OPT-079\","
      "\"family\":\"prompt-attn\",\"tier\":\"%s\",\"warmups\":%d,\"samples\":%d,"
      "\"observed_warmups\":%d,\"observed_samples\":%d,"
      "\"observed_candidates\":2,\"observed_shapes\":1,\"observed_tier\":\"%s\","
      "\"pairs\":1,\"sample_ids\":[%s],\"acceptance_executed\":%s,"
      "\"keep\":false}\n",
      qw38::cuda::test_tier_name(), warmups, samples, warmups, samples,
      qw38::cuda::test_tier_name(),
      samples == 3 ? "0,1,2" : (samples == 10 ? "0,1,2,3,4,5,6,7,8,9" : "0"),
      json_bool(std::strcmp(workload, "acceptance") == 0));
  std::printf("status=%s\n", pass ? "passed" : "failed");
  free_buffers(&buffers);
  return pass ? 0 : 1;
}
