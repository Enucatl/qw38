#pragma once

#include "runtime/error.hpp"
#include "runtime/sizes.hpp"

#include "format/constants.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace qw38::runtime {

// Ordered kernel stages on the single eager stream. Mixer is GDN xor attention.
enum class KernelStage : int {
  Embed = 0,
  LayerNorm = 1,
  Mixer = 2,
  MixerOutput = 3,
  MlpNorm = 4,
  MlpSwiglu = 5,
  MlpDown = 6,
  FinalNorm = 7,
  LmHead = 8,
};

struct LiveInterval {
  KernelStage first{KernelStage::Embed};
  KernelStage last{KernelStage::Embed};

  friend bool operator==(LiveInterval const&, LiveInterval const&) = default;
};

struct ScratchRequest {
  qw38::format::ScratchKind kind{qw38::format::ScratchKind::NormalizedHidden};
  qw38::format::ArithmeticDtype dtype{qw38::format::ArithmeticDtype::Bf16};
  std::uint64_t bytes{};
  std::vector<LiveInterval> live;
  // Nonzero: mutually exclusive with other requests in the same group even if
  // live intervals overlap (GDN vs attention mixer on one stream).
  int exclusive_group{0};
};

struct ScratchPlacement {
  qw38::format::ScratchKind kind{};
  qw38::format::ArithmeticDtype dtype{};
  std::uint64_t bytes{};
  std::uint64_t offset{};

  friend bool operator==(ScratchPlacement const&,
                         ScratchPlacement const&) = default;
};

struct ArenaPlan {
  std::uint64_t total_bytes{};
  std::uint64_t alignment{qw38::format::kSpanAlignment};
  std::vector<ScratchPlacement> placements;
};

[[nodiscard]] bool intervals_overlap(LiveInterval a, LiveInterval b) noexcept;

[[nodiscard]] std::expected<ArenaPlan, Error> plan_scratch_arena(
    std::span<ScratchRequest const> requests,
    std::uint64_t alignment = qw38::format::kSpanAlignment);

[[nodiscard]] std::expected<std::vector<ScratchRequest>, Error>
v0_scratch_requests(std::uint64_t token_capacity);
[[nodiscard]] std::expected<std::vector<ScratchRequest>, Error>
v0_scratch_requests(std::uint64_t token_capacity,
                    std::uint64_t attention_capacity);

[[nodiscard]] std::expected<ArenaPlan, Error> plan_v0_arena(
    std::uint64_t token_capacity = kArenaTokenCapacity);
[[nodiscard]] std::expected<ArenaPlan, Error> plan_v0_arena(
    std::uint64_t token_capacity, std::uint64_t attention_capacity);

[[nodiscard]] ScratchPlacement const* find_placement(
    ArenaPlan const& plan, qw38::format::ScratchKind kind) noexcept;

}  // namespace qw38::runtime
