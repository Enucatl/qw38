#pragma once

#include "cuda/buffer.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <expected>
#include <span>

namespace qw38::cuda {

// The candidate keeps BF16 projection activations. There is no activation
// scale buffer or packing step; a normalized BF16 row can feed every compatible
// projection without another conversion.
inline constexpr std::uint32_t kPrefillMaxTokens = 1024;
inline constexpr std::uint32_t kPrefillDefaultTokens = 1024;
inline constexpr std::uint32_t kPrefillDefaultWeightRows = 512;
inline constexpr std::uint32_t kPrefillMaxK = 17408;
inline constexpr std::uint32_t kPrefillHeadRows = 8;

// Validate an externally borrowed device span before a multi-stage layer
// begins mutating persistent state or output buffers.
[[nodiscard]] std::expected<void, Error> validate_prefill_device_span(
    void const* pointer, std::uint64_t bytes, std::uint32_t alignment,
    int device);

struct PrefillWeight {
  void const* codes{};
  void const* scales{};
  std::uint16_t layout{};
  std::uint16_t quantizer{};
  std::uint32_t n{};
  std::uint32_t k{};
  std::uint32_t padded_n{};
  std::uint32_t padded_k{};
  std::uint64_t codes_bytes{};
  std::uint64_t scales_bytes{};
};

enum class PrefillEpilogue : std::uint8_t {
  StoreBf16 = 1,
  StoreFp32 = 2,
  ResidualAddFp32 = 3,
};

enum class PrefillDispatch : std::uint8_t {
  BoundedUnpackBf16Cublas = 1,
};

// All arrays are device resident and token-major. output has physical stride
// weight.n, and only valid_tokens rows are written. first_position describes
// the caller's absolute token range; projection arithmetic is position free.
struct PrefillProjection {
  PrefillWeight weight{};
  std::uint16_t const* input{};
  void* output{};
  float const* residual{};
  std::uint32_t valid_tokens{};
  std::uint64_t first_position{};
  PrefillDispatch dispatch{PrefillDispatch::BoundedUnpackBf16Cublas};
  PrefillEpilogue epilogue{PrefillEpilogue::StoreBf16};
};

class PrefillEngine {
 public:
  // The Stream object and every borrowed device buffer must outlive the engine
  // and all enqueued work. Keep the Stream object at a stable address; do not
  // close or replace it while this engine exists. Synchronize before reusing or
  // releasing operands after an operation succeeds.
  PrefillEngine() = default;
  PrefillEngine(PrefillEngine&& other) noexcept;
  PrefillEngine& operator=(PrefillEngine&& other) noexcept;
  ~PrefillEngine();
  PrefillEngine(PrefillEngine const&) = delete;
  PrefillEngine& operator=(PrefillEngine const&) = delete;

  [[nodiscard]] static std::expected<PrefillEngine, Error> create(
      Stream const& stream, std::uint32_t token_capacity = kPrefillDefaultTokens,
      std::uint32_t weight_rows = kPrefillDefaultWeightRows);

  [[nodiscard]] std::expected<void, Error> project(
      PrefillProjection const& projection);
  [[nodiscard]] std::expected<void, Error> paired_swiglu(
      PrefillWeight const& gate, PrefillWeight const& up,
      std::uint16_t const* normalized, std::uint16_t* swiglu,
      std::uint32_t valid_tokens, std::uint64_t first_position);
  [[nodiscard]] std::expected<void, Error> mlp(
      PrefillWeight const& gate, PrefillWeight const& up,
      PrefillWeight const& down, float const* h_mid,
      std::uint16_t const* gamma, float eps, float* next_h,
      std::uint32_t valid_tokens, std::uint64_t first_position);

  // Generation reads one final row. Evaluation reads only requested rows;
  // each requested row is a bounded one-row tile in caller order. Callers
  // submit larger evaluation requests in groups of at most eight rows.
  [[nodiscard]] std::expected<void, Error> head_generation(
      PrefillWeight const& head, std::uint16_t const* normalized,
      std::uint32_t valid_tokens, std::uint64_t first_position, float* logits);
  [[nodiscard]] std::expected<void, Error> head_evaluation(
      PrefillWeight const& head, std::uint16_t const* normalized,
      std::uint32_t valid_tokens, std::uint64_t first_position,
      std::span<std::uint32_t const> requested_rows, float* logits);

  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept {
    return workspace_.bytes();
  }
  [[nodiscard]] std::uint32_t token_capacity() const noexcept {
    return token_capacity_;
  }
  [[nodiscard]] std::uint32_t weight_rows() const noexcept { return weight_rows_; }
  [[nodiscard]] Stream const* stream() const noexcept { return stream_; }

 private:
  [[nodiscard]] std::expected<void, Error> contract(
      PrefillWeight const& weight, std::uint32_t valid_tokens) const;
  [[nodiscard]] std::expected<void, Error> gemm_tile(
      PrefillWeight const& weight, std::uint16_t const* input,
      std::uint32_t valid_tokens, std::uint32_t row_start, float* accum);
  [[nodiscard]] std::expected<void, Error> head_contract(
      PrefillWeight const& head, std::uint16_t const* normalized,
      std::uint32_t valid_tokens, float* logits, std::uint32_t output_rows);
  void release() noexcept;

  Stream const* stream_{};
  cublasHandle_t handle_{};
  DeviceBuffer workspace_{};
  std::uint16_t* weight_tile_{};
  float* accum_a_{};
  float* accum_b_{};
  std::uint16_t* normalized_{};
  std::uint16_t* swiglu_{};
  std::byte* library_workspace_{};
  std::uint32_t token_capacity_{};
  std::uint32_t weight_rows_{};
  cudaStream_t native_stream_{};
  int device_{-1};
};

}  // namespace qw38::cuda
