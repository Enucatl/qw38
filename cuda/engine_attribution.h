#ifndef QW38_CUDA_ENGINE_ATTRIBUTION_H_
#define QW38_CUDA_ENGINE_ATTRIBUTION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

namespace qw38::cuda {

constexpr std::size_t kEngineEventPoolDefault = 1536;
constexpr std::size_t kEngineOpNameBytes = 96;
constexpr std::size_t kEngineFusedMemberBytes = 192;

// Shared Quartz/llama family record. Member counts are informational; fused
// complete-work time is charged once on the enclosing interval.
struct EngineOpRecord final {
  char engine[16]{};
  char phase[16]{};
  int sequence_position = -1;
  int token_position = -1;
  int layer = -1;
  char role[64]{};
  char tensor_name[kEngineOpNameBytes]{};
  char tensor_type[24]{};
  int m = 0;
  int n = 0;
  int k = 0;
  std::int64_t strides[4]{};
  unsigned long long stream = 0;
  int stream_index = 0;
  char launch_family[32]{};
  char fused_member_ids[kEngineFusedMemberBytes]{};
  int fused_member_count = 1;
  int start_event_id = -1;
  int end_event_id = -1;
  int scope_id = -1;
  int parent_scope_id = -1;
  char graph_mode[32]{};
  char attribution_role[16]{};
  float start_ms = 0.0F;
  float end_ms = 0.0F;
  float complete_work_ms = 0.0F;
  bool attributed = false;
  bool pool_overflow = false;
};

struct EngineAttribution final {
  EngineOpRecord* records = nullptr;
  std::size_t capacity = 0;
  std::size_t count = 0;
  bool pool_overflow = false;
  bool record = false;
  int sequence_position = 0;
  int token_position = 0;
  int layer = -1;
  char engine[16]{};
  char phase[16]{};
  char graph_mode[32]{};
  float uninstrumented_wall_ms = 0.0F;
  float instrumented_wall_ms = 0.0F;
  float profiling_overhead_ms = 0.0F;
  float shipping_graph_wall_ms = 0.0F;
  float eager_diagnostic_work_ms = 0.0F;
  float uncovered_wall_ms = 0.0F;
  bool uninstrumented_measured = false;
  bool instrumented_measured = false;
  cudaEvent_t sequence_epoch = nullptr;
  bool sequence_epoch_ready = false;
  int next_scope_id = 1;
  int token_scope_id = 0;
};

struct EngineEventSlot final {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaStream_t stream = nullptr;
  EngineOpRecord meta{};
  bool occupied = false;
};

// Preallocated CUDA-event pool. Resolve elapsed times once after the run.
template <std::size_t kCapacity>
class EngineEventPool final {
 public:
  EngineEventPool() noexcept = default;
  ~EngineEventPool() { destroy(); }
  EngineEventPool(const EngineEventPool&) = delete;
  EngineEventPool& operator=(const EngineEventPool&) = delete;

  cudaError_t ensure_epoch() noexcept {
    if (epoch_ != nullptr) return cudaSuccess;
    return cudaEventCreate(&epoch_);
  }

  cudaError_t record_epoch(cudaStream_t stream) noexcept {
    cudaError_t error = ensure_epoch();
    if (error == cudaSuccess) error = cudaEventRecord(epoch_, stream);
    return error;
  }

  cudaError_t begin(const EngineOpRecord& meta, cudaStream_t stream) noexcept {
    if (active_ || overflow_) return cudaErrorInvalidValue;
    if (count_ >= kCapacity) {
      overflow_ = true;
      return cudaErrorInvalidValue;
    }
    EngineEventSlot& slot = slots_[count_];
    cudaError_t error = cudaSuccess;
    if (slot.start == nullptr) error = cudaEventCreate(&slot.start);
    if (error == cudaSuccess && slot.stop == nullptr) {
      error = cudaEventCreate(&slot.stop);
    }
    if (error == cudaSuccess) error = cudaEventRecord(slot.start, stream);
    if (error != cudaSuccess) return error;
    slot.stream = stream;
    slot.meta = meta;
    slot.meta.stream = reinterpret_cast<unsigned long long>(stream);
    slot.meta.start_event_id = static_cast<int>(count_) * 2;
    slot.meta.end_event_id = slot.meta.start_event_id + 1;
    slot.occupied = true;
    active_ = true;
    return cudaSuccess;
  }

  cudaError_t end() noexcept {
    if (!active_ || count_ >= kCapacity) return cudaErrorInvalidValue;
    EngineEventSlot& slot = slots_[count_];
    cudaError_t error = cudaEventRecord(slot.stop, slot.stream);
    if (error == cudaSuccess) {
      ++count_;
      active_ = false;
    }
    return error;
  }

  cudaError_t collect(EngineAttribution* dest) noexcept {
    if (active_) return cudaErrorInvalidValue;
    if (dest != nullptr && overflow_) dest->pool_overflow = true;
    for (std::size_t index = 0; index < count_; ++index) {
      EngineEventSlot& slot = slots_[index];
      cudaError_t error = cudaEventSynchronize(slot.stop);
      float elapsed = 0.0F;
      float start_ms = 0.0F;
      float end_ms = 0.0F;
      if (error == cudaSuccess) {
        error = cudaEventElapsedTime(&elapsed, slot.start, slot.stop);
      }
      if (error == cudaSuccess && epoch_ != nullptr) {
        error = cudaEventElapsedTime(&start_ms, epoch_, slot.start);
        if (error == cudaSuccess) {
          error = cudaEventElapsedTime(&end_ms, epoch_, slot.stop);
        }
      }
      if (error != cudaSuccess) return error;
      slot.meta.start_ms = start_ms;
      slot.meta.end_ms = end_ms;
      slot.meta.complete_work_ms = elapsed;
      slot.meta.attributed = true;
      slot.meta.pool_overflow = overflow_;
      if (dest != nullptr && dest->records != nullptr) {
        if (dest->count >= dest->capacity) {
          dest->pool_overflow = true;
          overflow_ = true;
          return cudaErrorInvalidValue;
        }
        dest->records[dest->count] = slot.meta;
        ++dest->count;
      }
    }
    return cudaSuccess;
  }

  cudaError_t reset() noexcept {
    if (active_) return cudaErrorInvalidValue;
    for (std::size_t index = 0; index < count_; ++index) {
      slots_[index].occupied = false;
      slots_[index].stream = nullptr;
      slots_[index].meta = {};
    }
    count_ = 0;
    overflow_ = false;
    active_ = false;
    return cudaSuccess;
  }

  bool overflow() const noexcept { return overflow_; }
  std::size_t size() const noexcept { return count_; }
  static constexpr std::size_t capacity() noexcept { return kCapacity; }

 private:
  void destroy() noexcept {
    for (std::size_t index = 0; index < kCapacity; ++index) {
      if (slots_[index].stop != nullptr) cudaEventDestroy(slots_[index].stop);
      if (slots_[index].start != nullptr) {
        cudaEventDestroy(slots_[index].start);
      }
    }
    if (epoch_ != nullptr) cudaEventDestroy(epoch_);
  }

  std::array<EngineEventSlot, kCapacity> slots_{};
  cudaEvent_t epoch_ = nullptr;
  std::size_t count_ = 0;
  bool active_ = false;
  bool overflow_ = false;
};

inline cudaError_t ensure_sequence_epoch(EngineAttribution* dest) noexcept {
  if (dest == nullptr) return cudaErrorInvalidValue;
  if (dest->sequence_epoch != nullptr) return cudaSuccess;
  return cudaEventCreate(&dest->sequence_epoch);
}

inline cudaError_t record_sequence_epoch(EngineAttribution* dest,
                                         cudaStream_t stream) noexcept {
  cudaError_t error = ensure_sequence_epoch(dest);
  if (error == cudaSuccess) error = cudaEventRecord(dest->sequence_epoch, stream);
  if (error == cudaSuccess) dest->sequence_epoch_ready = true;
  return error;
}

inline void destroy_sequence_epoch(EngineAttribution* dest) noexcept {
  if (dest == nullptr || dest->sequence_epoch == nullptr) return;
  cudaEventDestroy(dest->sequence_epoch);
  dest->sequence_epoch = nullptr;
  dest->sequence_epoch_ready = false;
}

inline void copy_cstr(char* dest, std::size_t bytes, const char* text) noexcept {
  if (dest == nullptr || bytes == 0) return;
  if (text == nullptr) {
    dest[0] = '\0';
    return;
  }
  std::snprintf(dest, bytes, "%s", text);
}

inline void write_engine_record_json(FILE* out, const EngineOpRecord& rec,
                                     bool last) noexcept {
  if (out == nullptr) return;
  std::fprintf(
      out,
      "{\"engine\":\"%s\",\"phase\":\"%s\",\"sequence_position\":%d,"
      "\"token_position\":%d,\"layer\":%d,\"role\":\"%s\","
      "\"tensor_name\":\"%s\",\"tensor_type\":\"%s\",\"m\":%d,\"n\":%d,"
      "\"k\":%d,\"strides\":[%lld,%lld,%lld,%lld],\"stream\":%llu,"
      "\"stream_index\":%d,\"launch_family\":\"%s\","
      "\"fused_member_ids\":\"%s\",\"fused_member_count\":%d,"
      "\"start_event_id\":%d,\"end_event_id\":%d,\"scope_id\":%d,"
      "\"parent_scope_id\":%d,\"graph_mode\":\"%s\","
      "\"attribution_role\":\"%s\",\"start_ms\":%.9g,\"end_ms\":%.9g,"
      "\"complete_work_ms\":%.9g,\"attributed\":%s,\"pool_overflow\":%s}%s",
      rec.engine, rec.phase, rec.sequence_position, rec.token_position,
      rec.layer, rec.role, rec.tensor_name, rec.tensor_type, rec.m, rec.n,
      rec.k, static_cast<long long>(rec.strides[0]),
      static_cast<long long>(rec.strides[1]),
      static_cast<long long>(rec.strides[2]),
      static_cast<long long>(rec.strides[3]), rec.stream, rec.stream_index,
      rec.launch_family, rec.fused_member_ids, rec.fused_member_count,
      rec.start_event_id, rec.end_event_id, rec.scope_id, rec.parent_scope_id,
      rec.graph_mode, rec.attribution_role, rec.start_ms, rec.end_ms,
      rec.complete_work_ms, rec.attributed ? "true" : "false",
      rec.pool_overflow ? "true" : "false", last ? "" : ",");
}

}  // namespace qw38::cuda

#endif  // QW38_CUDA_ENGINE_ATTRIBUTION_H_
