#include "runtime/sizes.hpp"

#include "format/layout.hpp"

namespace qw38::runtime {
namespace {

using qw38::format::checked_add;
using qw38::format::checked_mul;
using qw38::format::FormatError;

Error overflow_error(FormatError const& error) {
  return make_error(ErrorCode::Overflow, error.field, error.detail, error.offset);
}

std::expected<std::uint64_t, Error> mul(std::uint64_t a, std::uint64_t b,
                                        std::string_view field) {
  auto r = checked_mul(a, b, 0, field);
  if (!r) {
    return std::unexpected(overflow_error(r.error()));
  }
  return *r;
}

std::expected<std::uint64_t, Error> add(std::uint64_t a, std::uint64_t b,
                                        std::string_view field) {
  auto r = checked_add(a, b, 0, field);
  if (!r) {
    return std::unexpected(overflow_error(r.error()));
  }
  return *r;
}

qw38::format::TensorShape shape3(std::uint64_t a, std::uint64_t b,
                                 std::uint64_t c) {
  qw38::format::TensorShape s{};
  s.rank = 3;
  s.logical[0] = a;
  s.logical[1] = b;
  s.logical[2] = c;
  s.padded[0] = a;
  s.padded[1] = b;
  s.padded[2] = c;
  return s;
}

qw38::format::TensorShape shape2(std::uint64_t a, std::uint64_t b) {
  qw38::format::TensorShape s{};
  s.rank = 2;
  s.logical[0] = a;
  s.logical[1] = b;
  s.padded[0] = a;
  s.padded[1] = b;
  return s;
}

}  // namespace

std::expected<std::uint64_t, Error> kv_cache_bytes(std::uint64_t capacity) {
  return mul(kKvBytesPerToken, capacity, "kv.capacity");
}

std::expected<std::uint64_t, Error> persistent_state_bytes(
    std::uint64_t capacity) {
  auto kv = kv_cache_bytes(capacity);
  if (!kv) {
    return kv;
  }
  return add(kFixedPersistentBytes, *kv, "state.total");
}

std::expected<std::uint64_t, Error> residual_bytes(
    std::uint64_t token_capacity) {
  return mul(kResidualBytesPerToken, token_capacity, "residual.bytes");
}

std::expected<std::uint64_t, Error> gdn_s_byte_offset(std::uint32_t layer,
                                                      std::uint32_t value_head,
                                                      std::uint32_t value,
                                                      std::uint32_t key) {
  if (layer >= kGdnLayers || value_head >= kGdnValueHeads ||
      value >= kGdnValueDim || key >= kGdnKeyDim) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn_s.index",
                                      "coordinate out of range"));
  }
  // Physical [layer, value_head, value, key] FP32, key contiguous.
  std::uint64_t off = layer;
  auto a = mul(off, kGdnValueHeads, "gdn_s.layer");
  if (!a) {
    return a;
  }
  off = *a + value_head;
  a = mul(off, kGdnValueDim, "gdn_s.head");
  if (!a) {
    return a;
  }
  off = *a + value;
  a = mul(off, kGdnKeyDim, "gdn_s.value");
  if (!a) {
    return a;
  }
  off = *a + key;
  return mul(off, qw38::format::kFp32Size, "gdn_s.elem");
}

std::expected<std::uint64_t, Error> conv_history_byte_offset(
    std::uint32_t layer, std::uint32_t tap, std::uint32_t channel) {
  if (layer >= kConvLayers || tap >= kConvTaps || channel >= kConvChannels) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "conv.index",
                                      "coordinate out of range"));
  }
  // Physical [layer, tap, channel] BF16, channel contiguous.
  std::uint64_t off = layer;
  auto a = mul(off, kConvTaps, "conv.layer");
  if (!a) {
    return a;
  }
  off = *a + tap;
  a = mul(off, kConvChannels, "conv.tap");
  if (!a) {
    return a;
  }
  off = *a + channel;
  return mul(off, qw38::format::kBf16Size, "conv.elem");
}

std::expected<std::uint64_t, Error> kv_byte_offset(
    std::uint32_t layer, std::uint32_t component, std::uint32_t head,
    std::uint64_t token, std::uint32_t dim, std::uint64_t capacity) {
  if (layer >= kAttnLayers || component >= kKvComponents || head >= kKvHeads ||
      token >= capacity || dim >= kHeadDim) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "kv.index",
                                      "coordinate out of range"));
  }
  // Physical [layer, component, kv_head, capacity, 256] BF16, head dim contiguous.
  std::uint64_t off = layer;
  auto a = mul(off, kKvComponents, "kv.layer");
  if (!a) {
    return a;
  }
  off = *a + component;
  a = mul(off, kKvHeads, "kv.component");
  if (!a) {
    return a;
  }
  off = *a + head;
  a = mul(off, capacity, "kv.head");
  if (!a) {
    return a;
  }
  off = *a + token;
  a = mul(off, kHeadDim, "kv.token");
  if (!a) {
    return a;
  }
  off = *a + dim;
  return mul(off, qw38::format::kBf16Size, "kv.elem");
}

std::array<qw38::format::StateAllocation, 3> language_persistent_schema() {
  using qw38::format::ArithmeticDtype;
  using qw38::format::PhysicalLayoutId;
  using qw38::format::StateKind;

  qw38::format::StateAllocation gdn{};
  gdn.kind = StateKind::GdnS;
  gdn.dtype = ArithmeticDtype::Fp32;
  gdn.layout = PhysicalLayoutId::CudaFp32GdnSHvKV0;
  gdn.shape_per_layer = shape3(kGdnValueHeads, kGdnValueDim, kGdnKeyDim);
  gdn.layer_count = kGdnLayers;
  gdn.component_count = 1;
  gdn.bytes_per_layer = 3145728;
  gdn.total_bytes = kGdnSBytes;

  qw38::format::StateAllocation conv{};
  conv.kind = StateKind::ConvolutionHistory;
  conv.dtype = ArithmeticDtype::Bf16;
  conv.layout = PhysicalLayoutId::CudaBf16ConvHistoryV0;
  conv.shape_per_layer = shape2(kConvTaps, kConvChannels);
  conv.layer_count = kConvLayers;
  conv.component_count = 1;
  conv.bytes_per_layer = 61440;
  conv.total_bytes = kConvHistoryBytes;

  qw38::format::StateAllocation kv{};
  kv.kind = StateKind::KvCache;
  kv.dtype = ArithmeticDtype::Bf16;
  kv.layout = PhysicalLayoutId::CudaBf16KvCacheV0;
  kv.shape_per_layer = shape2(kKvHeads, kHeadDim);
  kv.layer_count = kAttnLayers;
  kv.component_count = kKvComponents;
  kv.declared_capacity = 0;
  kv.bytes_per_token = kKvBytesPerToken;
  kv.populated_length_distinct_from_capacity = true;

  return {gdn, conv, kv};
}

std::expected<void, Error> require_language_state(
    std::span<qw38::format::StateAllocation const> state) {
  auto const want = language_persistent_schema();
  if (state.size() != want.size()) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact, "state",
                                      "language-only state must have GDN/conv/KV"));
  }
  for (std::size_t i = 0; i < want.size(); ++i) {
    auto const& got = state[i];
    auto const& w = want[i];
    if (got.kind != w.kind || got.dtype != w.dtype || got.layout != w.layout ||
        got.layer_count != w.layer_count ||
        got.component_count != w.component_count ||
        got.shape_per_layer != w.shape_per_layer ||
        got.bytes_per_token != w.bytes_per_token) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "state",
                                        "state schema does not match V0 language"));
    }
    if (got.kind != qw38::format::StateKind::KvCache &&
        (got.bytes_per_layer != w.bytes_per_layer ||
         got.total_bytes != w.total_bytes)) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "state",
                                        "fixed state byte counts mismatch"));
    }
  }
  return {};
}

std::expected<void, Error> require_language_scratch(
    std::span<qw38::format::ScratchAllocation const> scratch) {
  using qw38::format::ArithmeticDtype;
  using qw38::format::ScratchKind;
  struct Req {
    ScratchKind kind;
    ArithmeticDtype dtype;
    std::uint64_t bytes;
  };
  Req const reqs[] = {
      {ScratchKind::ResidualH, ArithmeticDtype::Fp32, kResidualBytesPerToken},
      {ScratchKind::ResidualHMid, ArithmeticDtype::Fp32, kResidualBytesPerToken},
      {ScratchKind::NormalizedHidden, ArithmeticDtype::Bf16,
       kNormalizedBytesPerToken},
      {ScratchKind::GdnWorkspace, ArithmeticDtype::Fp32,
       kGdnWorkspaceBytesPerToken},
      {ScratchKind::AttentionWorkspace, ArithmeticDtype::Fp32,
       kAttentionWorkspaceBytesPerToken},
      {ScratchKind::MlpSwiglu, ArithmeticDtype::Bf16, kSwigluBytesPerToken},
      {ScratchKind::Logits, ArithmeticDtype::Fp32, kLogitsBytesPerToken},
  };
  for (auto const& req : reqs) {
    bool found = false;
    for (auto const& s : scratch) {
      if (s.kind != req.kind) {
        continue;
      }
      found = true;
      if (s.dtype != req.dtype || s.bytes != req.bytes) {
        return std::unexpected(make_error(
            ErrorCode::MalformedArtifact, "scratch",
            "scratch record does not match decode-sized V0 language schema"));
      }
    }
    if (!found) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "scratch",
                                        "missing language scratch kind"));
    }
  }
  return {};
}

}  // namespace qw38::runtime
