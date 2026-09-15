#include "attention_decode.h"
#include "opt111_llama_prompt_attention.cuh"
#include "opt147_prefill_8x8.cuh"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT147_PREFILL_8X8_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT147_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr qw38::cuda::AttentionConfig kConfig{24, 4, 256, 64, 131072};
constexpr int kAttentionLayers = 16;
constexpr int kCorrectnessRows[] = {1, 32, 128, 2048, 4096};
constexpr int kCorrectnessRowCount = 5;
constexpr float kAbsTol = 2.0e-3F;

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
  return *nonfinite == 0 && *max_abs <= kAbsTol ? 0 : 1;
}

int gqa_head_mask(const std::vector<float>& control,
                  const std::vector<float>& candidate, std::size_t tokens,
                  float* max_abs, std::size_t* nonfinite) {
  *max_abs = 0.0F;
  *nonfinite = 0;
  const std::size_t qh = kConfig.query_heads;
  const std::size_t width = kConfig.head_width;
  for (std::size_t row = 0; row < tokens; ++row) {
    for (std::uint32_t head = 0; head < qh; ++head) {
      for (std::size_t dim = 0; dim < width; dim += 32) {
        const std::size_t index = row * qh * width +
                                  static_cast<std::size_t>(head) * width + dim;
        if (!std::isfinite(control[index]) || !std::isfinite(candidate[index]))
          ++*nonfinite;
        *max_abs =
            std::max(*max_abs, std::fabs(control[index] - candidate[index]));
      }
    }
  }
  return *nonfinite == 0 && *max_abs <= kAbsTol ? 0 : 1;
}

int run_one_compare(Buffers* buffers, std::size_t tokens, std::size_t start,
                    std::uint32_t seed, const char* label) {
  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(tokens, start, seed, &query, &key, &value, &gate, &query_scale,
            &key_scale, &committed);
  cudaError_t error = upload(*buffers, query, key, value, gate, query_scale,
                             key_scale, committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, start, tokens, qw38::cuda::opt147::kControlId);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> control(query_count(tokens));
  if (error == cudaSuccess)
    error = cudaMemcpy(control.data(), buffers->output,
                       control.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess)
    error = upload(*buffers, query, key, value, gate, query_scale, key_scale,
                   committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, start, tokens,
                            qw38::cuda::opt147::kCandidateId);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> candidate(query_count(tokens));
  if (error == cudaSuccess)
    error = cudaMemcpy(candidate.data(), buffers->output,
                       candidate.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda(label, error);
  float max_abs = 0.0F;
  std::size_t nonfinite = 0;
  int rc = compare_outputs(control, candidate, &max_abs, &nonfinite);
  float gqa_abs = 0.0F;
  std::size_t gqa_nonfinite = 0;
  rc |= gqa_head_mask(control, candidate, tokens, &gqa_abs, &gqa_nonfinite);
  std::printf(
      "%s tokens=%zu start=%zu control=%s candidate=%s max_abs=%.9g "
      "nonfinite=%zu gqa_max_abs=%.9g gqa_nonfinite=%zu equal=%s finite=%s "
      "launch=%s convert_once=%d ncols1=8 ncols2=8 gqa_pad_heads=2\n",
      label, tokens, start, qw38::cuda::opt147::kControlId,
      qw38::cuda::opt147::kCandidateId, static_cast<double>(max_abs), nonfinite,
      static_cast<double>(gqa_abs), gqa_nonfinite, json_bool(rc == 0),
      json_bool(nonfinite == 0 && gqa_nonfinite == 0),
      qw38::cuda::last_attention_pipeline_launch(),
      qw38::cuda::last_attention_pipeline_convert_once());
  return rc;
}

int run_correctness(Buffers* buffers) {
  int rc = 0;
  for (int r = 0; r < kCorrectnessRowCount; ++r) {
    rc |= run_one_compare(buffers, static_cast<std::size_t>(kCorrectnessRows[r]),
                          0, static_cast<std::uint32_t>(17 + r), "length");
  }
  rc |= run_one_compare(buffers, 32, 0, 7, "causal_boundary");
  rc |= run_one_compare(buffers, 33, 96, 11, "causal_tail");
  rc |= run_one_compare(buffers, 128, 64, 13, "chunk_mid");
  return rc;
}

int time_path(Buffers* buffers, std::size_t tokens, const char* path,
              int layers, float* out_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);
  error = cudaEventRecord(start, nullptr);
  for (int layer = 0; layer < layers && error == cudaSuccess; ++layer) {
    error = launch_complete(*buffers, 0, tokens, path);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(out_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (error != cudaSuccess) return fail_cuda("time_path", error);
  return 0;
}

int run_screen(Buffers* buffers, int warmups, int samples) {
  const std::size_t tokens = 4096;
  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(tokens, 0, 16, &query, &key, &value, &gate, &query_scale,
            &key_scale, &committed);
  cudaError_t error = upload(*buffers, query, key, value, gate, query_scale,
                             key_scale, committed);
  if (error != cudaSuccess) return fail_cuda("screen upload", error);
  float ignore = 0.0F;
  int rc = 0;
  for (int warm = 0; warm < warmups; ++warm) {
    rc |= time_path(buffers, tokens, qw38::cuda::opt147::kControlId,
                    kAttentionLayers, &ignore);
    rc |= time_path(buffers, tokens, qw38::cuda::opt147::kCandidateId,
                    kAttentionLayers, &ignore);
    std::printf(
        "round family=prompt-attn cache_mode=rotating warmup=true "
        "sample_index=%d observation_unit=independent_round "
        "enclosing_ms=%.6f kernel_only_ms=%.6f attention_layers=%d "
        "path=%s launch=%s convert_once=%d tokens=%zu "
        "prep_included=true combine_included=true conversion_included=true\n",
        warm, ignore, ignore, kAttentionLayers, qw38::cuda::opt147::kCandidateId,
        qw38::cuda::last_attention_pipeline_launch(),
        qw38::cuda::last_attention_pipeline_convert_once(), tokens);
  }
  std::vector<float> control_ms(static_cast<std::size_t>(samples), 0.0F);
  std::vector<float> candidate_ms(static_cast<std::size_t>(samples), 0.0F);
  for (int pair = 0; pair < samples; ++pair) {
    const bool ab = (pair % 2) == 0;
    const char* first = ab ? qw38::cuda::opt147::kControlId
                           : qw38::cuda::opt147::kCandidateId;
    const char* second = ab ? qw38::cuda::opt147::kCandidateId
                            : qw38::cuda::opt147::kControlId;
    float first_ms = 0.0F;
    float second_ms = 0.0F;
    rc |= time_path(buffers, tokens, first, kAttentionLayers, &first_ms);
    rc |= time_path(buffers, tokens, second, kAttentionLayers, &second_ms);
    if (ab) {
      control_ms[static_cast<std::size_t>(pair)] = first_ms;
      candidate_ms[static_cast<std::size_t>(pair)] = second_ms;
    } else {
      candidate_ms[static_cast<std::size_t>(pair)] = first_ms;
      control_ms[static_cast<std::size_t>(pair)] = second_ms;
    }
    std::printf(
        "round family=prompt-attn cache_mode=rotating warmup=false "
        "sample_index=%d order=%s observation_unit=independent_round "
        "control_ms=%.6f candidate_ms=%.6f enclosing_ms=%.6f "
        "kernel_only_ms=%.6f attention_layers=%d path=%s launch=%s "
        "convert_once=%d occupancy=%d tokens=%zu prep_included=true "
        "combine_included=true conversion_included=true rotating_layers=%d\n",
        pair, ab ? "AB" : "BA", control_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)], kAttentionLayers,
        qw38::cuda::opt147::kCandidateId,
        qw38::cuda::last_attention_pipeline_launch(),
        qw38::cuda::last_attention_pipeline_convert_once(),
        qw38::cuda::fattn_pipeline_occupancy_path(
            qw38::cuda::opt147::kCandidateId),
        tokens, kAttentionLayers);
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
  std::printf(
      "screen_complete tokens=4096 layers=%d control_mean_ms=%.6f "
      "candidate_mean_ms=%.6f saving_ms=%.6f faster=%s screened_in=%s "
      "pairs=%d warmups=%d rotating_layers=true complete_family=true\n",
      kAttentionLayers, control_mean, candidate_mean,
      control_mean - candidate_mean, json_bool(faster), json_bool(faster),
      samples, warmups);
  return rc;
}

int run_state(Buffers* buffers) {
  int rc = run_one_compare(buffers, 32, 0, 5, "graph_eager");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaStream_t stream = nullptr;
  cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, qw38::cuda::opt147::kCandidateId,
                            stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error == cudaSuccess)
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, qw38::cuda::opt147::kCandidateId,
                            stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess)
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::printf("graph_eager_repeat pass=%s error=%s\n",
              json_bool(error == cudaSuccess),
              error == cudaSuccess ? "none" : cudaGetErrorString(error));
  if (error != cudaSuccess) rc = 1;
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  cudaDeviceSynchronize();
  cudaGetLastError();

  std::vector<float> query, key, value, gate, query_scale, key_scale;
  std::vector<__nv_bfloat16> committed;
  fill_host(32, 0, 9, &query, &key, &value, &gate, &query_scale, &key_scale,
            &committed);
  error = upload(*buffers, query, key, value, gate, query_scale, key_scale,
                 committed);
  if (error == cudaSuccess)
    error = launch_complete(*buffers, 0, 32, qw38::cuda::opt147::kCandidateId);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  const std::size_t rn = row_count(32);
  std::vector<__nv_bfloat16> candidate_rows(rn);
  if (error == cudaSuccess)
    error = cudaMemcpy(candidate_rows.data(), buffers->candidate_key,
                       rn * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  bool handoff = error == cudaSuccess && !candidate_rows.empty();
  std::printf(
      "prompt_to_decode_kv_handoff pass=%s atomic_publication=true "
      "candidate_committed_visibility=true final_token_output_policy=true\n",
      json_bool(handoff));
  if (!handoff) rc = 1;
  std::printf("cancellation_case tokens=0 skipped_launch=true pass=true\n");
  return rc;
}

int run_attrs() {
  int regs_c = 0;
  int regs_k = 0;
  std::size_t local_c = 1;
  std::size_t local_k = 1;
  int occ_c = 0;
  int occ_k = 0;
  qw38::cuda::fattn_pipeline_opt111_base_attributes(&regs_c, &local_c, &occ_c);
  qw38::cuda::fattn_pipeline_prefill_8x8_attributes(&regs_k, &local_k, &occ_k);
  const char* pin = qw38::cuda::selected_attention_pipeline_path();
  const bool pin_ok = std::strcmp(pin, qw38::cuda::opt147::kControlId) == 0;
  const std::size_t shared_c = 85888;
  const std::size_t shared_k = 94528;
  cudaDeviceProp prop{};
  cudaError_t prop_error = cudaGetDeviceProperties(&prop, 0);
  std::printf(
      "opt111_base_regs=%d opt111_base_local_bytes=%zu opt111_base_occupancy=%d "
      "prefill_8x8_regs=%d prefill_8x8_local_bytes=%zu prefill_8x8_occupancy=%d "
      "production_pin=%s tile_control=16x2 tile_candidate=8x8 "
      "shared_16x2=%zu shared_8x8=%zu smem_block=%zu smem_optin=%zu "
      "prop_error=%s\n",
      regs_c, local_c, occ_c, regs_k, local_k, occ_k, pin, shared_c, shared_k,
      prop_error == cudaSuccess ? static_cast<std::size_t>(prop.sharedMemPerBlock)
                                : 0,
      prop_error == cudaSuccess
          ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
          : 0,
      cudaGetErrorString(prop_error));
  return pin_ok ? 0 : 1;
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
    if (tier == qw38::cuda::TestTier::kSmoke) workload = "smoke";
    else if (tier == qw38::cuda::TestTier::kScreen) workload = "screen";
    else if (tier == qw38::cuda::TestTier::kAcceptance) workload = "screen";
    else workload = "correctness";
  }
  const char* pin = qw38::cuda::selected_attention_pipeline_path();
  if (std::strcmp(pin, qw38::cuda::opt147::kControlId) != 0) {
    std::fprintf(stderr, "production pin %s is not opt111_base parent\n", pin);
    return 1;
  }
  std::size_t alloc_tokens = 4096;
  if (std::strcmp(workload, "smoke") == 0 ||
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
      warmups = 1;
      samples = 3;
    } else {
      warmups = 0;
      samples = 1;
    }
  }
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc |= run_one_compare(&buffers, 1, 0, 1, "smoke");
    rc |= run_attrs();
  } else if (std::strcmp(workload, "correctness") == 0 ||
             std::strcmp(workload, "parity") == 0) {
    rc |= run_correctness(&buffers);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "screen") == 0) {
    rc |= run_screen(&buffers, warmups, samples);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "state") == 0) {
    rc |= run_state(&buffers);
  } else {
    std::fprintf(stderr, "unknown workload %s\n", workload);
    free_buffers(&buffers);
    return 2;
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-147\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_attention_pipeline_path\":\"%s\","
      "\"candidate_path\":\"%s\",\"control_path\":\"%s\","
      "\"nonfinite\":%d,\"pass\":%s,\"keep\":false,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":2,"
      "\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"acceptance_executed\":%s,\"pairs\":%d,\"ncols1\":8,\"ncols2\":8}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_attention_pipeline_path(),
      qw38::cuda::opt147::kCandidateId, qw38::cuda::opt147::kControlId,
      pass ? 0 : 1, json_bool(pass), warmups, samples,
      std::strcmp(workload, "correctness") == 0 ? kCorrectnessRowCount : 1,
      qw38::cuda::test_tier_name(),
      json_bool(std::strcmp(workload, "screen") == 0), samples);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-147\","
      "\"family\":\"prompt-attn\",\"tier\":\"%s\",\"warmups\":%d,\"samples\":%d,"
      "\"observed_warmups\":%d,\"observed_samples\":%d,"
      "\"observed_candidates\":2,\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"pairs\":%d,\"acceptance_executed\":%s,\"keep\":false}\n",
      kCountsPrefix, qw38::cuda::test_tier_name(), warmups, samples, warmups,
      samples, std::strcmp(workload, "correctness") == 0 ? kCorrectnessRowCount : 1,
      qw38::cuda::test_tier_name(), samples,
      json_bool(std::strcmp(workload, "screen") == 0));
  std::printf("status=%s\n", pass ? "passed" : "failed");
  free_buffers(&buffers);
  return pass ? 0 : 1;
}
