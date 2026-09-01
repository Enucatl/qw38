#ifndef QW38_CUDA_FULL_SCHEDULER_H_
#define QW38_CUDA_FULL_SCHEDULER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qw38/status.h"
#include "weights.h"
#include "quant_mmv.h"

namespace qw38::cuda {

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

enum class PointwisePath : std::uint8_t {
  kFused = 0,
  kUnfused = 1,
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
                              SchedulerGraphs*) noexcept;
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
                              SchedulerGraphs*) noexcept;
  friend Status sync_tokens(const ResidentModel&, const std::size_t*,
                            std::size_t, SchedulerSession*,
                            class SchedulerWorkspace*, float*, std::size_t,
                            float*, std::size_t, SyncResult*,
                            const EvalControl*) noexcept;
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
  Status copy_trace_taps(float* output, std::size_t count) const noexcept;

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
  float* trace_taps_ = nullptr;
  float* candidate_logits_host_ = nullptr;
  float* candidate_hidden_host_ = nullptr;
  std::size_t capacity_ = 0;
  std::size_t allocated_bytes_ = 0;

  friend Status execute_token(const ResidentModel&, std::size_t,
                              SchedulerSession*, SchedulerWorkspace*, float*,
                              std::size_t, float*, std::size_t,
                              float*, const EvalControl*,
                              RuntimeTimings*, PointwisePath,
                              SchedulerGraphs*) noexcept;
  friend Status sync_tokens(const ResidentModel&, const std::size_t*,
                            std::size_t, SchedulerSession*, SchedulerWorkspace*,
                            float*, std::size_t, float*, std::size_t,
                            SyncResult*, const EvalControl*) noexcept;
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
  std::size_t allocated_bytes() const noexcept;

 private:
  void release() noexcept;
  bool matches(const ResidentModel& model,
               const SchedulerWorkspace* workspace) const noexcept;
  std::array<cudaGraph_t, internal::kModelLayerCount> graphs_{};
  std::array<cudaGraphExec_t, internal::kModelLayerCount> executions_{};
  const ResidentModel* model_ = nullptr;
  const SchedulerWorkspace* workspace_ = nullptr;
  std::size_t graph_count_ = 0;
  std::size_t allocated_bytes_ = 0;

  friend Status execute_token(const ResidentModel&, std::size_t,
                              SchedulerSession*, SchedulerWorkspace*, float*,
                              std::size_t, float*, std::size_t, float*,
                              const EvalControl*, RuntimeTimings*,
                              PointwisePath, SchedulerGraphs*) noexcept;
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
                     SchedulerGraphs* graphs = nullptr) noexcept;

Status greedy_sample(const SchedulerSession& session,
                     std::size_t* token,
                     RuntimeTimings* timings = nullptr) noexcept;

Status sync_tokens(const ResidentModel& model, const std::size_t* tokens,
                   std::size_t token_count, SchedulerSession* session,
                   SchedulerWorkspace* workspace, float* host_logits,
                   std::size_t logits_count, float* host_hidden,
                   std::size_t hidden_count, SyncResult* result,
                   const EvalControl* control = nullptr) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_FULL_SCHEDULER_H_
