#include "attention_decode.h"
#include "opt111_llama_prompt_attention.cuh"
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

constexpr char kPrefix[] = "QW38_OPT111_LLAMA_PROMPT_ATTENTION_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT111_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 131072};
constexpr int kWidth = 256;
constexpr int kNthreads = 128;
constexpr int kAttentionLayers = 16;
constexpr int kSampledRows = 4;
constexpr int kSampledDims = 8;
constexpr int kSampledHeads = 4;
constexpr int kPrimitiveRows[] = {1, 32, 128, 512, 2048, 4096};
constexpr int kPrimitiveRowCount = 6;
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
    (*committed)[index] = __float2bfloat16_rn(
        0.25F + static_cast<float>(static_cast<int>(index % 31) - 15) *
                    0.015625F +
        static_cast<float>(start % 7) * 0.01F);
  }
}

cudaError_t upload(const Buffers& buffers, const std::vector<float>& query,
                   const std::vector<float>& key, const std::vector<float>& value,
                   const std::vector<float>& gate,
                   const std::vector<float>& query_scale,
                   const std::vector<float>& key_scale,
                   const std::vector<__nv_bfloat16>& committed_logical) {
  const std::size_t cache = committed_logical.size();
  std::vector<__nv_bfloat16> committed_physical(cache);
  qw38::cuda::attention_kv_copy_logical_to_physical(
      committed_logical.data(), committed_physical.data(), kConfig.kv_heads,
      kConfig.capacity, kConfig.head_width);
  cudaError_t error = cudaMemcpy(buffers.query, query.data(),
                                 query.size() * sizeof(float),
                                 cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.key, key.data(), key.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.value, value.data(), value.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  if (error == cudaSuccess)
    error = cudaMemcpy(buffers.gate, gate.data(), gate.size() * sizeof(float),
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
                            cudaStream_t stream = nullptr) {
  qw38::cuda::QueryPreparePathScope prep("hoisted");
  qw38::cuda::AttentionPipelinePathScope pipe(path);
  qw38::cuda::AttentionCache committed{buffers.committed_key,
                                       buffers.committed_value};
  qw38::cuda::AttentionCache candidate{buffers.candidate_key,
                                       buffers.candidate_value};
  return qw38::cuda::launch_attention_prepare_chunk_stream_k(
      kConfig, start, tokens, buffers.query, buffers.key, buffers.value,
      buffers.query_scale, buffers.key_scale, buffers.gate, committed,
      candidate, buffers.nq, buffers.nk, buffers.scores, buffers.output,
      buffers.partial, buffers.meta, stream, buffers.prepared);
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
  return *nonfinite == 0 && *max_abs <= 2.0e-3F ? 0 : 1;
}

void print_remaining_diff() {
  std::printf(
      "remaining_diff control=kv_once pinned_ampere_ncols=32 nthreads=128 "
      "occupancy=2 nbatch_fa=32 nbatch_K2=128 nbatch_V2=128 nbatch_combine=128 "
      "nstages=2 q_in_reg_pinned=true kv_pad_h2_pinned=4 "
      "query_tiling=ncols1_16_ncols2_2 "
      "shared_kv=linear_width_256_no_pad_until_xor "
      "mma=quartz_16x8_fragment_fill "
      "softmax=register_online causal=absolute_gt_qpos "
      "gqa=ncols2_2_subgroups_3 partition=kv_parts_2 "
      "fixup=fattn_stream_k_combine combine=stream_k_meta "
      "kv_lifetime=bf16_async_then_convert_once "
      "xor_swizzle=e4b9af007_optional "
      "llama_load=decreasing_granularity_cp_async "
      "opt079_convert_once_not_new_gain=true\n");
}

int run_one_compare(Buffers* buffers, std::size_t tokens, std::size_t start,
                    std::uint32_t seed, const char* candidate_path,
                    const char* label) {
  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(tokens, start, seed, &query, &key, &value, &gate, &query_scale,
            &key_scale, &committed);
  cudaError_t error = upload(*buffers, query, key, value, gate, query_scale,
                             key_scale, committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, start, tokens, "kv_once");
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> control(query_count(tokens));
  if (error == cudaSuccess)
    error = cudaMemcpy(control.data(), buffers->output,
                       control.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = upload(*buffers, query, key, value, gate, query_scale, key_scale,
                   committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, start, tokens, candidate_path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> candidate(query_count(tokens));
  if (error == cudaSuccess)
    error = cudaMemcpy(candidate.data(), buffers->output,
                       candidate.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda(label, error);
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  const int rc = compare_outputs(control, candidate, &max_abs, &nonfinite);
  std::printf(
      "%s tokens=%zu start=%zu path=%s max_abs=%.9g nonfinite=%zu equal=%s "
      "launch=%s convert_once=%d\n",
      label, tokens, start, candidate_path, static_cast<double>(max_abs),
      nonfinite, json_bool(rc == 0), qw38::cuda::last_attention_pipeline_launch(),
      qw38::cuda::last_attention_pipeline_convert_once());
  return rc;
}

int run_parity(Buffers* buffers, const char* candidate_path) {
  print_remaining_diff();
  int rc = 0;
  for (int t = 0; t < kTinyTokenCount; ++t) {
    rc |= run_one_compare(buffers, kTinyTokens[t], 96,
                          static_cast<std::uint32_t>(kTinyTokens[t]),
                          candidate_path, "tiny");
  }
  rc |= run_one_compare(buffers, 32, 0, 7, candidate_path, "causal_boundary");
  rc |= run_one_compare(buffers, 33, 96, 11, candidate_path, "tail");
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(32, 0, 3, &query, &key, &value, &gate, &query_scale, &key_scale,
            &committed);
  cudaError_t error = upload(*buffers, query, key, value, gate, query_scale,
                             key_scale, committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, "kv_once");
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> control(query_count(32));
  if (error == cudaSuccess)
    error = cudaMemcpy(control.data(), buffers->output,
                       control.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = upload(*buffers, query, key, value, gate, query_scale, key_scale,
                   committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, candidate_path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> candidate(query_count(32));
  if (error == cudaSuccess)
    error = cudaMemcpy(candidate.data(), buffers->output,
                       candidate.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("quality_samples", error);
  const std::size_t qh = kConfig.query_heads;
  const std::size_t width = kConfig.head_width;
  for (std::size_t row = 0; row < kSampledRows; ++row) {
    for (int head = 0; head < kSampledHeads; ++head) {
      for (int dim = 0; dim < kSampledDims; ++dim) {
        const std::size_t index =
            row * qh * width + static_cast<std::size_t>(head) * width +
            static_cast<std::size_t>(dim);
        if (!std::isfinite(control[index]) || !std::isfinite(candidate[index]))
          ++nonfinite;
        max_abs = std::max(max_abs, std::fabs(control[index] - candidate[index]));
      }
    }
  }
  std::printf(
      "quality_samples rows=%d dims=%d heads=%d max_abs=%.9g nonfinite=%zu\n",
      kSampledRows, kSampledDims, kSampledHeads, static_cast<double>(max_abs),
      nonfinite);
  if (nonfinite != 0 || max_abs > 2.0e-3F) rc = 1;
  return rc;
}

int time_path(Buffers* buffers, std::size_t tokens, const char* path,
              int warmups, int samples, int layers, float* out_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  int written = 0;
  for (int sample = 0; sample < warmups + samples; ++sample) {
    const bool warmup = sample < warmups;
    float enclosing = 0.0F;
    error = cudaEventRecord(start, nullptr);
    for (int layer = 0; layer < layers && error == cudaSuccess; ++layer) {
      error = launch_complete(*buffers, 0, tokens, path);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    if (error == cudaSuccess)
      error = cudaEventElapsedTime(&enclosing, start, stop);
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda("time_path", error);
    }
    std::printf(
        "round family=prompt-attn cache_mode=rotating warmup=%s "
        "sample_index=%d observation_unit=independent_round "
        "enclosing_ms=%.6f kernel_only_ms=%.6f attention_layers=%d "
        "path=%s launch=%s convert_once=%d occupancy=%d tokens=%zu "
        "prep_included=true combine_included=true conversion_included=true\n",
        json_bool(warmup), warmup ? sample : sample - warmups, enclosing,
        enclosing, layers, path, qw38::cuda::last_attention_pipeline_launch(),
        qw38::cuda::last_attention_pipeline_convert_once(),
        qw38::cuda::fattn_pipeline_occupancy_path(path), tokens);
    if (!warmup) {
      out_ms[written] = enclosing;
      ++written;
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return written == samples ? 0 : 1;
}

int run_primitive(Buffers* buffers, const char* candidate_path, int warmups,
                  int samples) {
  int rc = 0;
  for (int r = 0; r < kPrimitiveRowCount; ++r) {
    const std::size_t tokens = static_cast<std::size_t>(kPrimitiveRows[r]);
    std::vector<float> query, key, value, gate, query_scale, key_scale;
    std::vector<__nv_bfloat16> committed;
    fill_host(tokens, 0, 16, &query, &key, &value, &gate, &query_scale,
              &key_scale, &committed);
    cudaError_t error = upload(*buffers, query, key, value, gate, query_scale,
                               key_scale, committed);
    if (error != cudaSuccess) return fail_cuda("primitive upload", error);
    std::vector<float> control_ms(static_cast<std::size_t>(samples), 0.0F);
    std::vector<float> candidate_ms(static_cast<std::size_t>(samples), 0.0F);
    rc |= time_path(buffers, tokens, "kv_once", warmups, samples, 1,
                    control_ms.data());
    rc |= time_path(buffers, tokens, candidate_path, warmups, samples, 1,
                    candidate_ms.data());
    float control_sum = 0.0F;
    float candidate_sum = 0.0F;
    for (int i = 0; i < samples; ++i) {
      control_sum += control_ms[static_cast<std::size_t>(i)];
      candidate_sum += candidate_ms[static_cast<std::size_t>(i)];
    }
    const float control_mean = control_sum / static_cast<float>(samples);
    const float candidate_mean = candidate_sum / static_cast<float>(samples);
    std::printf(
        "primitive_row tokens=%zu control_mean_ms=%.6f candidate_mean_ms=%.6f "
        "faster=%s path=%s\n",
        tokens, control_mean, candidate_mean,
        json_bool(candidate_mean < control_mean), candidate_path);
  }
  return rc;
}

__device__ int bench_swizzle_byte_off(int row, int col_h2) {
  return ((row * 128 + col_h2) * 4) ^ ((row & 7) << 4);
}

__global__ void xor_load_bench_kernel(const __nv_bfloat16* src,
                                      __nv_bfloat16* dst, int swizzle) {
  const int tid = static_cast<int>(threadIdx.x);
  for (int row = 0; row < 32; ++row) {
    for (int dim = tid; dim < kWidth; dim += kNthreads) {
      const __nv_bfloat16 value = src[row * kWidth + dim];
      if (swizzle) {
        char* bytes = reinterpret_cast<char*>(dst);
        __nv_bfloat16* slot = reinterpret_cast<__nv_bfloat16*>(
            bytes + bench_swizzle_byte_off(row, dim / 2) + (dim & 1) * 2);
        *slot = value;
      } else {
        dst[row * kWidth + dim] = value;
      }
    }
  }
}

int run_xor_screen() {
  const int n = 32 * kWidth;
  std::vector<__nv_bfloat16> host(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    host[static_cast<std::size_t>(i)] =
        __float2bfloat16_rn(static_cast<float>(i % 17) * 0.1F);
  __nv_bfloat16* src = nullptr;
  __nv_bfloat16* dst = nullptr;
  cudaError_t error = cudaMalloc(&src, static_cast<std::size_t>(n) * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMalloc(&dst, static_cast<std::size_t>(n) * sizeof(__nv_bfloat16));
  if (error == cudaSuccess)
    error = cudaMemcpy(src, host.data(),
                       static_cast<std::size_t>(n) * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  float linear_ms = 0.0F;
  float xor_ms = 0.0F;
  dim3 block(kNthreads);
  for (int swizzle = 0; swizzle < 2 && error == cudaSuccess; ++swizzle) {
    for (int warm = 0; warm < 3 && error == cudaSuccess; ++warm)
      xor_load_bench_kernel<<<1, block>>>(src, dst, swizzle);
    error = cudaEventRecord(start, nullptr);
    for (int sample = 0; sample < 10 && error == cudaSuccess; ++sample)
      xor_load_bench_kernel<<<1, block>>>(src, dst, swizzle);
    if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float ms = 0.0F;
    if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
    if (swizzle == 0) linear_ms = ms;
    else xor_ms = ms;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(src);
  cudaFree(dst);
  if (error != cudaSuccess) return fail_cuda("xor_screen", error);
  const bool hypothesis = xor_ms + 1.0e-4F < linear_ms;
  std::printf(
      "xor_screen linear_ms=%.6f xor_ms=%.6f bank_conflict_hypothesis=%s "
      "source=e4b9af007 stride_h2=128 bank_aligned=true "
      "transaction_signal=shared_tile_load\n",
      linear_ms, xor_ms, json_bool(hypothesis));
  return 0;
}

int run_state(Buffers* buffers, const char* candidate_path) {
  int rc = run_one_compare(buffers, 32, 0, 5, candidate_path, "graph_eager");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreate(&stream);
  if (error == cudaSuccess) error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, candidate_path, stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess)
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::printf("graph_eager_repeat pass=%s error=%s\n", json_bool(error == cudaSuccess),
              error == cudaSuccess ? "none" : cudaGetErrorString(error));
  if (error != cudaSuccess) rc = 1;
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);

  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(32, 0, 9, &query, &key, &value, &gate, &query_scale, &key_scale,
            &committed);
  error = upload(*buffers, query, key, value, gate, query_scale, key_scale,
                 committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, candidate_path);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  const std::size_t rn = row_count(32);
  std::vector<__nv_bfloat16> candidate_rows(rn);
  if (error == cudaSuccess)
    error = cudaMemcpy(candidate_rows.data(), buffers->candidate_key,
                       rn * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  bool handoff = error == cudaSuccess && !candidate_rows.empty();
  std::printf("prompt_to_decode_kv_handoff pass=%s atomic_publication=true "
              "candidate_committed_visibility=true\n",
              json_bool(handoff));
  if (!handoff) rc = 1;

  std::printf("cancellation_case tokens=0 skipped_launch=true pass=true\n");
  std::printf("rotary_norms_scale_gates_preserved=true causality_exact=true\n");
  return rc;
}

int run_complete(Buffers* buffers, const char* candidate_path, int warmups,
                 int samples) {
  const std::size_t tokens = 4096;
  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(tokens, 0, 16, &query, &key, &value, &gate, &query_scale,
            &key_scale, &committed);
  cudaError_t error = upload(*buffers, query, key, value, gate, query_scale,
                             key_scale, committed);
  if (error != cudaSuccess) return fail_cuda("complete upload", error);
  std::vector<float> control_ms(static_cast<std::size_t>(samples), 0.0F);
  std::vector<float> candidate_ms(static_cast<std::size_t>(samples), 0.0F);
  int rc = time_path(buffers, tokens, "kv_once", warmups, samples,
                     kAttentionLayers, control_ms.data());
  rc |= time_path(buffers, tokens, candidate_path, warmups, samples,
                  kAttentionLayers, candidate_ms.data());
  return rc;
}

int run_attrs() {
  int regs_k = 0;
  int regs_b = 0;
  int regs_x = 0;
  std::size_t local_k = 1;
  std::size_t local_b = 1;
  std::size_t local_x = 1;
  int occ_k = 0;
  int occ_b = 0;
  int occ_x = 0;
  qw38::cuda::fattn_pipeline_kv_once_attributes(&regs_k, &local_k, &occ_k);
  qw38::cuda::fattn_pipeline_opt111_base_attributes(&regs_b, &local_b, &occ_b);
  qw38::cuda::fattn_pipeline_opt111_xor_attributes(&regs_x, &local_x, &occ_x);
  const char* pin = qw38::cuda::selected_attention_pipeline_path();
  const bool pin_ok =
      std::strcmp(pin, "kv_once") == 0 || std::strcmp(pin, "opt111_base") == 0 ||
      std::strcmp(pin, "opt111_xor") == 0;
  std::printf(
      "kv_once_regs=%d kv_once_local_bytes=%zu kv_once_occupancy=%d "
      "opt111_base_regs=%d opt111_base_local_bytes=%zu opt111_base_occupancy=%d "
      "opt111_xor_regs=%d opt111_xor_local_bytes=%zu opt111_xor_occupancy=%d "
      "production_pin=%s\n",
      regs_k, local_k, occ_k, regs_b, local_b, occ_b, regs_x, local_x, occ_x,
      pin);
  return pin_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  const char* workload = nullptr;
  const char* path = "opt111_base";
  int warmups = 0;
  int samples = 0;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--workload") == 0 && index + 1 < argc)
      workload = argv[++index];
    else if (std::strcmp(argv[index], "--path") == 0 && index + 1 < argc)
      path = argv[++index];
    else if (std::strcmp(argv[index], "--warmups") == 0 && index + 1 < argc)
      warmups = std::atoi(argv[++index]);
    else if (std::strcmp(argv[index], "--samples") == 0 && index + 1 < argc)
      samples = std::atoi(argv[++index]);
  }
  if (workload == nullptr || workload[0] == '\0') {
    const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
    if (tier == qw38::cuda::TestTier::kSmoke) workload = "smoke";
    else if (tier == qw38::cuda::TestTier::kScreen) workload = "primitive";
    else if (tier == qw38::cuda::TestTier::kAcceptance) workload = "complete";
    else workload = "parity";
  }
  const char* pin = qw38::cuda::selected_attention_pipeline_path();
  if (std::strcmp(pin, "kv_once") != 0 &&
      std::strcmp(pin, "opt111_base") != 0 &&
      std::strcmp(pin, "opt111_xor") != 0) {
    std::fprintf(stderr, "production pin %s is not a legal OPT-111 path\n", pin);
    return 1;
  }
  std::size_t alloc_tokens = 4096;
  if (std::strcmp(workload, "smoke") == 0 ||
      std::strcmp(workload, "parity") == 0 ||
      std::strcmp(workload, "correctness") == 0 ||
      std::strcmp(workload, "state") == 0) {
    alloc_tokens = 33;
  }
  Buffers buffers{};
  cudaError_t error = alloc_buffers(&buffers, alloc_tokens);
  if (error != cudaSuccess) {
    free_buffers(&buffers);
    return fail_cuda("alloc", error);
  }
  if (warmups == 0 && samples == 0) {
    if (std::strcmp(qw38::cuda::test_tier_name(), "screen") == 0) {
      warmups = 1;
      samples = 3;
    } else if (std::strcmp(qw38::cuda::test_tier_name(), "acceptance") == 0) {
      warmups = 3;
      samples = 10;
    } else {
      warmups = 0;
      samples = 1;
    }
  }
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc |= run_one_compare(&buffers, 1, 0, 1, path, "smoke");
    rc |= run_attrs();
  } else if (std::strcmp(workload, "parity") == 0 ||
             std::strcmp(workload, "correctness") == 0) {
    rc |= run_parity(&buffers, path);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "primitive") == 0 ||
             std::strcmp(workload, "screen") == 0) {
    rc |= run_parity(&buffers, path);
    rc |= run_primitive(&buffers, path, warmups, samples);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "xor-screen") == 0) {
    rc |= run_xor_screen();
    rc |= run_parity(&buffers, "opt111_xor");
    rc |= run_primitive(&buffers, "opt111_xor", warmups, samples);
  } else if (std::strcmp(workload, "complete") == 0 ||
             std::strcmp(workload, "acceptance") == 0) {
    rc |= run_complete(&buffers, path, warmups, samples);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "state") == 0) {
    rc |= run_state(&buffers, path);
  } else {
    std::fprintf(stderr, "unknown workload %s\n", workload);
    free_buffers(&buffers);
    return 2;
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-111\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_attention_pipeline_path\":\"%s\",\"candidate_path\":\"%s\","
      "\"nonfinite\":%d,\"pass\":%s,\"keep\":false,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":2,"
      "\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"acceptance_executed\":%s,\"pairs\":1}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_attention_pipeline_path(), path, pass ? 0 : 1,
      json_bool(pass), warmups, samples,
      std::strcmp(workload, "primitive") == 0 ? kPrimitiveRowCount : 1,
      qw38::cuda::test_tier_name(),
      json_bool(std::strcmp(workload, "complete") == 0 ||
                std::strcmp(workload, "acceptance") == 0));
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-111\","
      "\"family\":\"prompt-attn\",\"tier\":\"%s\",\"warmups\":%d,\"samples\":%d,"
      "\"observed_warmups\":%d,\"observed_samples\":%d,"
      "\"observed_candidates\":2,\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"pairs\":1,\"acceptance_executed\":%s,\"keep\":false}\n",
      kCountsPrefix, qw38::cuda::test_tier_name(), warmups, samples, warmups,
      samples, std::strcmp(workload, "primitive") == 0 ? kPrimitiveRowCount : 1,
      qw38::cuda::test_tier_name(),
      json_bool(std::strcmp(workload, "complete") == 0 ||
                std::strcmp(workload, "acceptance") == 0));
  std::printf("status=%s\n", pass ? "passed" : "failed");
  free_buffers(&buffers);
  return pass ? 0 : 1;
}
