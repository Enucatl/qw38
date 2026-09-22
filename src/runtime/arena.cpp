#include "runtime/arena.hpp"

#include "format/layout.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace qw38::runtime {
namespace {

using qw38::format::align_up;
using qw38::format::checked_add;
using qw38::format::checked_mul;

Error map_format(qw38::format::FormatError const& error) {
  if (error.code == qw38::format::FormatErrorCode::Overflow) {
    return make_error(ErrorCode::Overflow, error.field, error.detail,
                      error.offset);
  }
  return from_format(error);
}

bool ranges_overlap(std::uint64_t a0, std::uint64_t a1, std::uint64_t b0,
                    std::uint64_t b1) noexcept {
  return a0 < b1 && b0 < a1;
}

struct Slot {
  std::uint64_t bytes{};
  std::vector<LiveInterval> live;
  std::vector<std::size_t> request_indices;
};

bool live_conflict(std::span<LiveInterval const> a,
                   std::span<LiveInterval const> b) noexcept {
  for (auto const& ia : a) {
    for (auto const& ib : b) {
      if (intervals_overlap(ia, ib)) {
        return true;
      }
    }
  }
  return false;
}

std::expected<std::uint64_t, Error> aligned_bytes(std::uint64_t bytes,
                                                  std::uint64_t alignment) {
  auto r = align_up(bytes, alignment, 0, "arena.bytes");
  if (!r) {
    return std::unexpected(map_format(r.error()));
  }
  return *r;
}

}  // namespace

bool intervals_overlap(LiveInterval a, LiveInterval b) noexcept {
  return !(a.last < b.first || b.last < a.first);
}

ScratchPlacement const* find_placement(ArenaPlan const& plan,
                                       qw38::format::ScratchKind kind) noexcept {
  for (auto const& p : plan.placements) {
    if (p.kind == kind) {
      return &p;
    }
  }
  return nullptr;
}

std::expected<ArenaPlan, Error> plan_scratch_arena(
    std::span<ScratchRequest const> requests, std::uint64_t alignment) {
  try {
  if (alignment == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "arena.align",
                                      "alignment must be nonzero"));
  }
  for (auto const& req : requests) {
    if (req.bytes == 0) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "arena.bytes",
                                        "scratch request must be nonzero"));
    }
    if (req.live.empty()) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "arena.live",
                                        "scratch request needs a live range"));
    }
    for (auto const& iv : req.live) {
      if (iv.last < iv.first) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, "arena.live",
                                          "live interval last < first"));
      }
    }
  }

  std::vector<Slot> slots;
  std::vector<int> assigned(requests.size(), -1);
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (assigned[i] >= 0) {
      continue;
    }
    Slot slot;
    slot.bytes = requests[i].bytes;
    slot.live = requests[i].live;
    slot.request_indices.push_back(i);
    assigned[i] = static_cast<int>(slots.size());
    if (requests[i].exclusive_group != 0) {
      for (std::size_t j = i + 1; j < requests.size(); ++j) {
        if (requests[j].exclusive_group != requests[i].exclusive_group) {
          continue;
        }
        slot.bytes = std::max(slot.bytes, requests[j].bytes);
        slot.live.insert(slot.live.end(), requests[j].live.begin(),
                         requests[j].live.end());
        slot.request_indices.push_back(j);
        assigned[j] = static_cast<int>(slots.size());
      }
    }
    slots.push_back(std::move(slot));
  }

  std::vector<std::size_t> order(slots.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return slots[a].bytes > slots[b].bytes;
  });

  struct Placed {
    std::uint64_t offset{};
    std::uint64_t bytes{};
    std::span<LiveInterval const> live;
  };
  std::vector<Placed> placed;
  std::vector<std::uint64_t> slot_offset(slots.size(), 0);
  std::uint64_t total = 0;

  for (std::size_t idx : order) {
    auto const sized = aligned_bytes(slots[idx].bytes, alignment);
    if (!sized) {
      return std::unexpected(sized.error());
    }
    std::uint64_t offset = 0;
    bool found = false;
    while (!found) {
      found = true;
      for (auto const& p : placed) {
        if (!live_conflict(slots[idx].live, p.live)) {
          continue;
        }
        if (!ranges_overlap(offset, offset + *sized, p.offset,
                            p.offset + p.bytes)) {
          continue;
        }
        auto next = checked_add(p.offset, p.bytes, 0, "arena.offset");
        if (!next) {
          return std::unexpected(map_format(next.error()));
        }
        auto aligned = aligned_bytes(*next, alignment);
        if (!aligned) {
          return std::unexpected(aligned.error());
        }
        offset = *aligned;
        found = false;
        break;
      }
      if (offset > std::numeric_limits<std::uint64_t>::max() - *sized) {
        return std::unexpected(make_error(ErrorCode::Overflow, "arena.offset",
                                          "scratch arena does not fit"));
      }
    }
    auto end = checked_add(offset, *sized, 0, "arena.total");
    if (!end) {
      return std::unexpected(map_format(end.error()));
    }
    total = std::max(total, *end);
    slot_offset[idx] = offset;
    placed.push_back(Placed{.offset = offset,
                            .bytes = *sized,
                            .live = slots[idx].live});
  }

  ArenaPlan plan;
  plan.total_bytes = total;
  plan.alignment = alignment;
  plan.placements.reserve(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto const slot = static_cast<std::size_t>(assigned[i]);
    plan.placements.push_back(ScratchPlacement{
        .kind = requests[i].kind,
        .dtype = requests[i].dtype,
        .bytes = requests[i].bytes,
        .offset = slot_offset[slot],
    });
  }
  return plan;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "arena.plan",
                                      "host allocation failed"));
  } catch (std::length_error const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "arena.plan",
                                      "host allocation size is invalid"));
  }
}

std::expected<std::vector<ScratchRequest>, Error> v0_scratch_requests(
    std::uint64_t token_capacity) {
  return v0_scratch_requests(token_capacity, token_capacity);
}

std::expected<std::vector<ScratchRequest>, Error> v0_scratch_requests(
    std::uint64_t token_capacity, std::uint64_t attention_capacity) {
  try {
  if (token_capacity == 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "arena.token_capacity",
                                      "arena token capacity must be > 0"));
  }
  if (auto st = validate_kv_capacity(attention_capacity); !st) {
    return std::unexpected(st.error());
  }
  auto scale = [&](std::uint64_t per_token,
                   std::string_view field) -> std::expected<std::uint64_t, Error> {
    auto r = checked_mul(per_token, token_capacity, 0, field);
    if (!r) {
      return std::unexpected(map_format(r.error()));
    }
    return *r;
  };

  using qw38::format::ArithmeticDtype;
  using qw38::format::ScratchKind;
  std::vector<ScratchRequest> out;
  auto add = [&](ScratchKind kind, ArithmeticDtype dtype, std::uint64_t per_token,
                 std::vector<LiveInterval> live, int group,
                 std::string_view field) -> std::expected<void, Error> {
    auto bytes = scale(per_token, field);
    if (!bytes) {
      return std::unexpected(bytes.error());
    }
    out.push_back(ScratchRequest{.kind = kind,
                                 .dtype = dtype,
                                 .bytes = *bytes,
                                 .live = std::move(live),
                                 .exclusive_group = group});
    return {};
  };

  if (auto st = add(ScratchKind::NormalizedHidden, ArithmeticDtype::Bf16,
                    kNormalizedBytesPerToken,
                    {{KernelStage::LayerNorm, KernelStage::Mixer},
                     {KernelStage::MlpNorm, KernelStage::MlpSwiglu},
                     {KernelStage::FinalNorm, KernelStage::LmHead}},
                    0, "scratch.normalized");
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = add(ScratchKind::GdnWorkspace, ArithmeticDtype::Fp32,
                    kGdnWorkspaceBytesPerToken,
                    {{KernelStage::Mixer, KernelStage::MixerOutput}}, 1,
                    "scratch.gdn");
      !st) {
    return std::unexpected(st.error());
  }
  auto attention_bytes = attn_workspace_bytes_for_capacity(attention_capacity);
  if (!attention_bytes) {
    return std::unexpected(attention_bytes.error());
  }
  out.push_back(ScratchRequest{
      .kind = ScratchKind::AttentionWorkspace,
      .dtype = ArithmeticDtype::Fp32,
      .bytes = *attention_bytes,
      .live = {{KernelStage::Mixer, KernelStage::MixerOutput}},
      .exclusive_group = 1,
  });
  if (auto st = add(ScratchKind::MlpSwiglu, ArithmeticDtype::Bf16,
                    kSwigluBytesPerToken,
                    {{KernelStage::MlpSwiglu, KernelStage::MlpDown}}, 0,
                    "scratch.swiglu");
      !st) {
    return std::unexpected(st.error());
  }
  if (auto st = add(ScratchKind::Logits, ArithmeticDtype::Fp32,
                    kLogitsBytesPerToken,
                    {{KernelStage::LmHead, KernelStage::LmHead}}, 0,
                    "scratch.logits");
      !st) {
    return std::unexpected(st.error());
  }
  return out;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "arena.requests",
                                      "host allocation failed"));
  } catch (std::length_error const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "arena.requests",
                                      "host allocation size is invalid"));
  }
}

std::expected<ArenaPlan, Error> plan_v0_arena(std::uint64_t token_capacity) {
  return plan_v0_arena(token_capacity, token_capacity);
}

std::expected<ArenaPlan, Error> plan_v0_arena(
    std::uint64_t token_capacity, std::uint64_t attention_capacity) {
  auto reqs = v0_scratch_requests(token_capacity, attention_capacity);
  if (!reqs) {
    return std::unexpected(reqs.error());
  }
  return plan_scratch_arena(*reqs);
}

}  // namespace qw38::runtime
