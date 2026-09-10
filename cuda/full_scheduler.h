#ifndef QW38_CUDA_FULL_SCHEDULER_H_
#define QW38_CUDA_FULL_SCHEDULER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qw38/status.h"
#include "mixer.h"
#include "weights.h"
#include "quant_mmv.h"
#include "gdn_step.h"
#include "pdl_launch.cuh"
#ifdef QW38_DIAGNOSTIC_TRACE
#include "diagnostic_trace.h"
#endif

namespace qw38::cuda {

constexpr std::size_t kPromptChunkRows = 4096;

struct DeviceTensor final {
  const std::uint8_t* data = nullptr;
  std::size_t columns = 0;
  std::size_t rows = 0;
  QuantKind kind = QuantKind::kQ4K;
};

struct DeviceCommonLayer final {
  const float* input_norm = nullptr;
  DeviceTensor ffn_gate;
  DeviceTensor ffn_up;
  DeviceTensor ffn_down;
  const float* ffn_norm = nullptr;
};

struct DeviceGdnLayer final {
  DeviceTensor packed_qkv;
  DeviceTensor value_gate;
  DeviceTensor alpha;
  DeviceTensor beta;
  const float* convolution = nullptr;
  const float* folded_a = nullptr;
  const float* dt_bias = nullptr;
  const float* norm = nullptr;
  DeviceTensor output;
};

struct DeviceAttentionLayer final {
  DeviceTensor query_gate;
  DeviceTensor key;
  DeviceTensor value;
  const float* query_norm = nullptr;
  const float* key_norm = nullptr;
  DeviceTensor output;
};

struct DeviceLayer final {
  internal::LayerKind kind = internal::LayerKind::kGdn;
  DeviceCommonLayer common;
  DeviceGdnLayer gdn;
  DeviceAttentionLayer attention;
};

struct SyncResult final {
  std::size_t common_prefix = 0;
  std::size_t reused_tokens = 0;
  std::size_t evaluated_tokens = 0;
  bool reset_and_replayed = false;
};

using EvalPoll = Status (*)(void*) noexcept;

struct EvalControl final {
  EvalPoll poll = nullptr;
  void* context = nullptr;
};

struct SamplerState final {
  float temperature = 1.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0;
  std::uint64_t rng_state = 0;
};

struct TimingValue final {
  float milliseconds = 0.0F;
  bool measured = false;
};

// Exclusive GPU/host leaves used by OPT-043. Enclosing OPT-038 categories are
// reconstructed as sums of these leaves. Overlapping host waits are recorded
// separately and must not enter the exclusive reconstruction set.
struct LeafTimings final {
  TimingValue embedding;
  TimingValue input_norm;
  TimingValue residual_mixer;
  TimingValue activation_staging_mixer;
  TimingValue proj_packed_qkv;
  TimingValue proj_value_gate;
  TimingValue proj_alpha;
  TimingValue proj_beta;
  TimingValue proj_gdn_output;
  TimingValue proj_query_gate;
  TimingValue proj_key;
  TimingValue proj_value;
  TimingValue proj_attn_output;
  TimingValue gdn_gate_prep;
  TimingValue gdn_conv_qk_norm_recurrence;
  TimingValue gdn_output_norm;
  TimingValue attn_query_split;
  TimingValue attn_qk_prep_softmax_pv_merge;
  TimingValue attn_output_cast;
  TimingValue ffn_norm;
  TimingValue activation_staging_ffn;
  TimingValue proj_ffn_gate;
  TimingValue proj_ffn_up;
  TimingValue swiglu;
  TimingValue proj_ffn_down;
  TimingValue residual_ffn;
  TimingValue logits_norm;
  TimingValue logits_projection;
  TimingValue d2h;
  TimingValue state_copies;
  TimingValue host_graph_submit;
  TimingValue host_submission_waits;
  bool gdn_conv_fused_with_recurrence = false;
  bool gdn_output_norm_fused_with_gate = false;
  bool attention_prep_fused_with_core = false;
  bool ffn_graph_fused = false;
  bool ffn_staging_fused_into_mmv = false;
  bool swiglu_fused_with_down_stage = false;
  bool d2h_overlaps_state_copies = false;
};

struct ActivationCaptureSlot final {
  std::size_t layer = 0;
  const char* layer_kind = "";
  bool mixer_captured = false;
  bool ffn_captured = false;
  std::array<float, internal::kResidualWidth> mixer_preprojection{};
  std::array<float, internal::kResidualWidth> ffn_preprojection{};
  std::array<float, 64> mixer_prefix{};
  std::array<float, 64> ffn_prefix{};
  char mixer_sha256[65]{};
  char ffn_sha256[65]{};
  char mixer_dtype[16]{};
  char ffn_dtype[16]{};
  std::array<std::size_t, 2> mixer_shape{};
  std::array<std::size_t, 2> ffn_shape{};
};

struct ActivationCapture final {
  const char* stage = "";
  std::size_t position = 0;
  std::array<std::size_t, 6> layers{0, 3, 31, 32, 62, 63};
  std::array<ActivationCaptureSlot, 6> slots{};
  bool final_norm_captured = false;
  std::array<float, internal::kResidualWidth> final_norm{};
  std::array<float, 64> final_norm_prefix{};
  char final_norm_sha256[65]{};
  bool output_captured = false;
  std::array<float, 64> output_prefix{};
  char output_sha256[65]{};
  std::size_t output_count = 0;
};

// One request-level attribution record. A false `measured` flag means that the
// runtime boundary does not exist yet; it must not be interpreted as zero work.
struct RuntimeTimings final {
  TimingValue loading;
  TimingValue embedding;
  TimingValue gdn;
  TimingValue attention;
  TimingValue ffn;
  TimingValue logits;
  TimingValue sampling;
  TimingValue graph_launch;
  TimingValue queueing;
  TimingValue persistence;
  TimingValue idle_gaps;
  TimingValue state_commit;
  TimingValue token_total;
};

struct PrefillAttribution final {
  TimingValue embedding;
  TimingValue mixer_mmq;
  TimingValue gdn_core;
  TimingValue attention_core;
  TimingValue ffn_mmq;
  TimingValue logits;
  TimingValue commit_sync;
  TimingValue graph;
  TimingValue other_idle;
  TimingValue wall;
  std::size_t prompt_tokens = 0;
  std::size_t evaluated_tokens = 0;
  std::size_t chunk_count = 0;
  std::uint32_t prompt_graph_launches = 0;
  bool record_leaves = false;
  LeafTimings leaves{};
  ActivationCapture* capture = nullptr;
};

struct DecodeAttribution final {
  TimingValue embedding;
  TimingValue mixer_mmv;
  TimingValue gdn_core;
  TimingValue attention_core;
  TimingValue ffn_mmv;
  TimingValue logits;
  TimingValue state_commit;
  TimingValue graph;
  TimingValue other_idle;
  TimingValue wall;
  bool record_leaves = false;
  LeafTimings leaves{};
  ActivationCapture* capture = nullptr;
};

enum class PointwisePath : std::uint8_t {
  kFused = 0,
  kUnfused = 1,
};

enum class PromptPipelinePath : std::uint8_t {
  kFusedOverlapped = 0,
  kUnfusedSerial = 1,
};

struct PromptPipelineCounters final {
  std::uint32_t embedding_kernel_launches = 0;
  std::uint32_t widen_kernel_launches = 0;
  std::uint32_t residual_add_kernel_launches = 0;
  std::uint32_t rms_norm_kernel_launches = 0;
  std::uint32_t fused_residual_norm_kernel_launches = 0;
  std::uint32_t scatter_kernel_launches = 0;
  std::uint32_t blocking_d2h_copies = 0;
  std::uint32_t async_d2h_copies = 0;
  std::uint32_t device_synchronizes = 0;
  std::uint32_t stream_synchronizes = 0;
  std::uint32_t layer_polls = 0;
  std::uint32_t prompt_graph_launches = 0;
};

class SchedulerWorkspace;
class SchedulerGraphs;

class ResidentModel final {
 public:
  ResidentModel() noexcept;
  ~ResidentModel();
  ResidentModel(ResidentModel&& other) noexcept;
  ResidentModel& operator=(ResidentModel&& other) noexcept;
  ResidentModel(const ResidentModel&) = delete;
  ResidentModel& operator=(const ResidentModel&) = delete;

  Status upload(const internal::ModelWeights& weights,
                const std::uint8_t* mapped_base,
                std::size_t mapped_bytes) noexcept;
  std::size_t resident_bytes() const noexcept;
  float upload_milliseconds() const noexcept;
  std::size_t layer_count() const noexcept { return layers_.size(); }
  const DeviceLayer& layer(std::size_t index) const noexcept {
    return layers_[index];
  }
  const DeviceTensor& embedding() const noexcept { return embedding_; }
#ifdef QW38_DIAGNOSTIC_TRACE
  const DeviceCommonLayer& common_layer(std::size_t layer_index) const noexcept {
    return layers_[layer_index].common;
  }
  const float* next_input_norm(std::size_t layer_index) const noexcept {
    return layer_index + 1 < layers_.size()
               ? layers_[layer_index + 1].common.input_norm
               : nullptr;
  }
#endif

 private:
  void release() noexcept;
  std::uint8_t* blob_ = nullptr;
  std::size_t blob_bytes_ = 0;
  float upload_ms_ = 0.0F;
  DeviceTensor embedding_;
  const float* output_norm_ = nullptr;
  DeviceTensor output_;
  std::array<DeviceLayer, internal::kModelLayerCount> layers_{};

  friend Status execute_token(const ResidentModel&, std::size_t,
                              class SchedulerSession*, class SchedulerWorkspace*,
                              float*, std::size_t, float*, std::size_t,
                              float*, const EvalControl*,
                              RuntimeTimings*, PointwisePath,
                              SchedulerGraphs*, DecodeAttribution*) noexcept;
  friend Status sync_tokens(const ResidentModel&, const std::size_t*,
                            std::size_t, class SchedulerSession*,
                            class SchedulerWorkspace*, float*, std::size_t,
                            float*, std::size_t, SyncResult*,
                            const EvalControl*, SchedulerGraphs*,
                            GdnScanPath, PrefillAttribution*) noexcept;
  friend Status execute_prompt_chunk(
      const ResidentModel&, const std::size_t*, std::size_t,
      class SchedulerSession*, class SchedulerWorkspace*, float*,
      std::size_t, float*, std::size_t, const EvalControl*,
      PromptPipelinePath, PromptPipelineCounters*,
      SchedulerGraphs*, GdnScanPath, PrefillAttribution*) noexcept;
  friend class SchedulerGraphs;
};

class SchedulerSession final {
 public:
  SchedulerSession() noexcept;
  ~SchedulerSession();
  SchedulerSession(SchedulerSession&& other) noexcept;
  SchedulerSession& operator=(SchedulerSession&& other) noexcept;
  SchedulerSession(const SchedulerSession&) = delete;
  SchedulerSession& operator=(const SchedulerSession&) = delete;

  Status create(std::size_t capacity) noexcept;
  Status reset() noexcept;
  Status state_equals(const SchedulerSession& other, bool* equal) const noexcept;
  Status set_sampler_state(const SamplerState& state) noexcept;
  SamplerState sampler_state() const noexcept;
  Status save_checkpoint(const std::string& path,
                         RuntimeTimings* timings = nullptr) const noexcept;
  Status restore_checkpoint(const std::string& path,
                            SchedulerWorkspace* workspace,
                            RuntimeTimings* timings = nullptr) noexcept;
  Status copy_last_outputs(float* logits, std::size_t logits_count,
                           float* hidden,
                           std::size_t hidden_count) const noexcept;
  Status copy_tokens(std::size_t* output,
                     std::size_t output_count) const noexcept;
  std::size_t capacity() const noexcept;
  std::size_t frontier() const noexcept;
  std::size_t token_count() const noexcept;
  std::size_t allocated_bytes() const noexcept;

 private:
  void release() noexcept;
  float* gdn_convolution_ = nullptr;
  float* gdn_recurrent_ = nullptr;
  __nv_bfloat16* attention_key_ = nullptr;
  __nv_bfloat16* attention_value_ = nullptr;
  std::size_t* tokens_ = nullptr;
  float* last_logits_ = nullptr;
  float* last_hidden_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t frontier_ = 0;
  std::size_t allocated_bytes_ = 0;
  SamplerState sampler_state_{};

  friend Status execute_token(const ResidentModel&, std::size_t,
                              SchedulerSession*, class SchedulerWorkspace*,
                              float*, std::size_t, float*, std::size_t,
                              float*, const EvalControl*,
                              RuntimeTimings*, PointwisePath,
                              SchedulerGraphs*, DecodeAttribution*) noexcept;
  friend Status sync_tokens(const ResidentModel&, const std::size_t*,
                            std::size_t, SchedulerSession*,
                            class SchedulerWorkspace*, float*, std::size_t,
                            float*, std::size_t, SyncResult*,
                            const EvalControl*, SchedulerGraphs*,
                            GdnScanPath, PrefillAttribution*) noexcept;
  friend Status execute_prompt_chunk(
      const ResidentModel&, const std::size_t*, std::size_t,
      SchedulerSession*, class SchedulerWorkspace*, float*, std::size_t,
      float*, std::size_t, const EvalControl*, PromptPipelinePath,
      PromptPipelineCounters*, SchedulerGraphs*, GdnScanPath,
      PrefillAttribution*) noexcept;
  friend Status greedy_sample(const SchedulerSession&, std::size_t*,
                              RuntimeTimings*) noexcept;
};

class SchedulerWorkspace final {
 public:
  SchedulerWorkspace() noexcept;
  ~SchedulerWorkspace();
  SchedulerWorkspace(SchedulerWorkspace&& other) noexcept;
  SchedulerWorkspace& operator=(SchedulerWorkspace&& other) noexcept;
  SchedulerWorkspace(const SchedulerWorkspace&) = delete;
  SchedulerWorkspace& operator=(const SchedulerWorkspace&) = delete;

  Status create(std::size_t capacity) noexcept;
  std::size_t allocated_bytes() const noexcept;
#ifdef QW38_DIAGNOSTIC_TRACE
  Status copy_trace_taps(float* output, std::size_t count) const noexcept;
#endif

 public:
  void release() noexcept;
  float* residual_a_ = nullptr;
  float* residual_b_ = nullptr;
  __nv_bfloat16* normalized_ = nullptr;
  __nv_bfloat16* projected_bf16_ = nullptr;
  __nv_bfloat16* ffn_activated_ = nullptr;
  Q8Block* q8_ = nullptr;
  float* projection_a_ = nullptr;
  float* projection_b_ = nullptr;
  float* projection_c_ = nullptr;
  float* projection_d_ = nullptr;
  float* mixer_output_ = nullptr;
  float* gdn_decay_ = nullptr;
  float* gdn_update_ = nullptr;
  float* gdn_convolved_ = nullptr;
  float* gdn_recurrent_output_ = nullptr;
  float* gdn_candidate_convolution_ = nullptr;
  float* gdn_candidate_recurrent_ = nullptr;
  __nv_bfloat16* attention_candidate_key_ = nullptr;
  __nv_bfloat16* attention_candidate_value_ = nullptr;
  float* attention_normalized_query_ = nullptr;
  float* attention_normalized_key_ = nullptr;
  float* attention_scores_ = nullptr;
  float* logits_ = nullptr;
#ifdef QW38_DIAGNOSTIC_TRACE
  float* trace_taps_ = nullptr;
#endif
  float* candidate_logits_host_ = nullptr;
  float* candidate_hidden_host_ = nullptr;
  float* prompt_residual_a_ = nullptr;
  float* prompt_residual_b_ = nullptr;
  __nv_bfloat16* prompt_normalized_ = nullptr;
  __nv_bfloat16* prompt_projected_bf16_ = nullptr;
  Q8Block* prompt_q8_ = nullptr;
  float* prompt_projection_a_ = nullptr;
  float* prompt_projection_b_ = nullptr;
  float* prompt_projection_c_ = nullptr;
  float* prompt_projection_d_ = nullptr;
  float* prompt_mixer_output_ = nullptr;
  float* prompt_gdn_decay_ = nullptr;
  float* prompt_gdn_update_ = nullptr;
  float* prompt_gdn_convolved_ = nullptr;
  float* prompt_gdn_recurrent_output_ = nullptr;
  __nv_bfloat16* prompt_attention_candidate_key_ = nullptr;
  __nv_bfloat16* prompt_attention_candidate_value_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t prompt_chunk_rows_ = 0;
  std::size_t allocated_bytes_ = 0;
  cudaStream_t prompt_compute_stream_ = nullptr;
  cudaStream_t prompt_copy_stream_ = nullptr;
  cudaEvent_t prompt_compute_done_ = nullptr;

  friend Status execute_token(const ResidentModel&, std::size_t,
                              SchedulerSession*, SchedulerWorkspace*, float*,
                              std::size_t, float*, std::size_t,
                              float*, const EvalControl*,
                              RuntimeTimings*, PointwisePath,
                              SchedulerGraphs*, DecodeAttribution*) noexcept;
  friend Status sync_tokens(const ResidentModel&, const std::size_t*,
                            std::size_t, SchedulerSession*, SchedulerWorkspace*,
                            float*, std::size_t, float*, std::size_t,
                            SyncResult*, const EvalControl*,
                            SchedulerGraphs*, GdnScanPath,
                            PrefillAttribution*) noexcept;
  friend Status execute_prompt_chunk(
      const ResidentModel&, const std::size_t*, std::size_t,
      SchedulerSession*, SchedulerWorkspace*, float*, std::size_t, float*,
      std::size_t, const EvalControl*, PromptPipelinePath,
      PromptPipelineCounters*, SchedulerGraphs*, GdnScanPath,
      PrefillAttribution*) noexcept;
  friend class SchedulerGraphs;
};

class SchedulerGraphs final {
 public:
  SchedulerGraphs() noexcept;
  ~SchedulerGraphs();
  SchedulerGraphs(SchedulerGraphs&& other) noexcept;
  SchedulerGraphs& operator=(SchedulerGraphs&& other) noexcept;
  SchedulerGraphs(const SchedulerGraphs&) = delete;
  SchedulerGraphs& operator=(const SchedulerGraphs&) = delete;

  Status create(const ResidentModel& model,
                SchedulerWorkspace* workspace) noexcept;
  std::size_t graph_count() const noexcept;
  std::size_t decode_graph_count() const noexcept;
  std::size_t prompt_graph_count() const noexcept;
  std::size_t prompt_graph_rows() const noexcept;
  std::size_t allocated_bytes() const noexcept;

 private:
  void release() noexcept;
  bool matches(const ResidentModel& model,
               const SchedulerWorkspace* workspace) const noexcept;
  std::array<cudaGraph_t, internal::kModelLayerCount> graphs_{};
  std::array<cudaGraphExec_t, internal::kModelLayerCount> executions_{};
  std::array<cudaGraph_t, internal::kModelLayerCount> prompt_graphs_{};
  std::array<cudaGraphExec_t, internal::kModelLayerCount> prompt_executions_{};
  const ResidentModel* model_ = nullptr;
  const SchedulerWorkspace* workspace_ = nullptr;
  std::size_t decode_graph_count_ = 0;
  std::size_t prompt_graph_count_ = 0;
  std::size_t prompt_rows_ = 0;
  std::size_t allocated_bytes_ = 0;

  friend Status execute_token(const ResidentModel&, std::size_t,
                              SchedulerSession*, SchedulerWorkspace*, float*,
                              std::size_t, float*, std::size_t, float*,
                              const EvalControl*, RuntimeTimings*,
                              PointwisePath, SchedulerGraphs*,
                              DecodeAttribution*) noexcept;
  friend Status execute_prompt_chunk(
      const ResidentModel&, const std::size_t*, std::size_t,
      SchedulerSession*, SchedulerWorkspace*, float*, std::size_t, float*,
      std::size_t, const EvalControl*, PromptPipelinePath,
      PromptPipelineCounters*, SchedulerGraphs*, GdnScanPath,
      PrefillAttribution*) noexcept;
};

Status execute_token(const ResidentModel& model, std::size_t token,
                     SchedulerSession* session, SchedulerWorkspace* workspace,
                     float* host_logits, std::size_t logits_count,
                     float* host_hidden, std::size_t hidden_count,
                     float* elapsed_milliseconds,
                     const EvalControl* control = nullptr,
                     RuntimeTimings* timings = nullptr,
                     PointwisePath pointwise_path =
                         PointwisePath::kFused,
                     SchedulerGraphs* graphs = nullptr,
                     DecodeAttribution* decode_attribution = nullptr) noexcept;

#ifdef QW38_DIAGNOSTIC_TRACE
Status execute_token_traced(const ResidentModel& model, std::size_t token,
                            SchedulerSession* session,
                            SchedulerWorkspace* workspace, float* host_logits,
                            std::size_t logits_count, float* host_hidden,
                            std::size_t hidden_count,
                            float* elapsed_milliseconds,
                            const internal::TraceFilter& filter,
                            internal::TraceSink sink, void* context) noexcept;
#endif

cudaError_t launch_residual_add_fp32(const float* residual,
                                     const float* correction, std::size_t count,
                                     float* output,
                                     cudaStream_t stream) noexcept;

cudaError_t launch_prepare_gdn_gate_rows(
    const float* alpha, const float* beta, const float* folded_a,
    const float* dt_bias, std::size_t key_heads, std::size_t replicas,
    std::size_t token_count, float* log_decay, float* update,
    cudaStream_t stream) noexcept;

cudaError_t launch_split_attention_rows_prompt(
    const float* packed, std::size_t query_values, std::size_t head_width,
    std::size_t token_count, float* query, float* gate,
    cudaStream_t stream) noexcept;

cudaError_t launch_rms_norm_rows_fp32_to_bf16(const float* input,
                                              const float* scale,
                                              std::size_t width,
                                              std::size_t token_count,
                                              __nv_bfloat16* output,
                                              cudaStream_t stream) noexcept;

cudaError_t launch_residual_add_norm_rows_fp32_to_bf16(
    const float* residual, const float* correction, const float* scale,
    std::size_t width, std::size_t token_count, float* output,
    __nv_bfloat16* normalized, cudaStream_t stream) noexcept;

cudaError_t execute_prompt_ffn(
    const DeviceCommonLayer& layer, const float* residual,
    SchedulerWorkspace* workspace, float* after_mixer, float* output,
    const float* next_input_norm, std::size_t token_count,
    cudaStream_t stream) noexcept;

Status execute_prompt_chunk(
    const ResidentModel& model, const std::size_t* tokens,
    std::size_t token_count, SchedulerSession* session,
    SchedulerWorkspace* workspace, float* host_logits,
    std::size_t logits_count, float* host_hidden, std::size_t hidden_count,
    const EvalControl* control = nullptr,
    PromptPipelinePath path = PromptPipelinePath::kFusedOverlapped,
    PromptPipelineCounters* counters = nullptr,
    SchedulerGraphs* graphs = nullptr,
    GdnScanPath gdn_scan = GdnScanPath::kFusedTokenLoop,
    PrefillAttribution* attribution = nullptr) noexcept;

Status greedy_sample(const SchedulerSession& session,
                     std::size_t* token,
                     RuntimeTimings* timings = nullptr) noexcept;

Status sync_tokens(const ResidentModel& model, const std::size_t* tokens,
                   std::size_t token_count, SchedulerSession* session,
                   SchedulerWorkspace* workspace, float* host_logits,
                   std::size_t logits_count, float* host_hidden,
                   std::size_t hidden_count, SyncResult* result,
                   const EvalControl* control = nullptr,
                   SchedulerGraphs* graphs = nullptr,
                   GdnScanPath gdn_scan = GdnScanPath::kFusedTokenLoop,
                   PrefillAttribution* attribution = nullptr) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_FULL_SCHEDULER_H_
