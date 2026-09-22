#include "runtime/session.hpp"

#include "cuda/copy.hpp"
#include "cuda/stream.hpp"
#include "runtime/model.hpp"

#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace qw38::runtime {
namespace {

TensorView view_from_buffer(qw38::cuda::DeviceBuffer const& buf,
                            qw38::format::ArithmeticDtype dtype,
                            qw38::format::PhysicalLayoutId layout,
                            qw38::format::StorageClass storage,
                            std::uint8_t rank,
                            std::array<std::uint64_t, qw38::format::kMaxRank> extent) {
  TensorView v{};
  v.pointer = const_cast<void*>(buf.data());
  v.dtype = dtype;
  v.layout = layout;
  v.storage = storage;
  v.space = MemorySpace::Device;
  v.rank = rank;
  v.extent = extent;
  return v;
}

std::array<std::uint64_t, qw38::format::kMaxRank> extent4(
    std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
  std::array<std::uint64_t, qw38::format::kMaxRank> e{};
  e[0] = a;
  e[1] = b;
  e[2] = c;
  e[3] = d;
  return e;
}

std::array<std::uint64_t, qw38::format::kMaxRank> extent3(std::uint64_t a,
                                                          std::uint64_t b,
                                                          std::uint64_t c) {
  std::array<std::uint64_t, qw38::format::kMaxRank> e{};
  e[0] = a;
  e[1] = b;
  e[2] = c;
  return e;
}

std::array<std::uint64_t, qw38::format::kMaxRank> extent2(std::uint64_t a,
                                                          std::uint64_t b) {
  std::array<std::uint64_t, qw38::format::kMaxRank> e{};
  e[0] = a;
  e[1] = b;
  return e;
}

std::array<std::uint64_t, qw38::format::kMaxRank> extent5(
    std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d,
    std::uint64_t e5) {
  std::array<std::uint64_t, qw38::format::kMaxRank> e{};
  e[0] = a;
  e[1] = b;
  e[2] = c;
  e[3] = d;
  e[4] = e5;
  return e;
}

}  // namespace

std::expected<KvPopulatedSlot, Error> KvPopulatedSlot::bind(
    std::uint64_t* value, std::uint64_t capacity) {
  if (value == nullptr) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, "populated",
        "host populated-length pointer is required"));
  }
  if (auto st = validate_kv_capacity(capacity); !st) {
    return std::unexpected(st.error());
  }
  if (*value > capacity) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                      "kv.populated",
                                      "populated length exceeds capacity"));
  }
  KvPopulatedSlot slot;
  slot.value_ = value;
  slot.capacity_ = capacity;
  return slot;
}

std::expected<std::uint64_t, Error> KvPopulatedSlot::value() const {
  if (value_ == nullptr) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, "populated",
        "host populated-length pointer is required"));
  }
  if (*value_ > capacity_) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                      "populated",
                                      "populated length exceeds capacity"));
  }
  return *value_;
}

std::expected<void, Error> KvPopulatedSlot::commit_append(
    std::uint64_t position) const {
  auto populated = value();
  if (!populated) {
    return std::unexpected(populated.error());
  }
  if (position >= capacity_) {
    return std::unexpected(make_error(ErrorCode::InvalidCapacity, "position",
                                      "position exceeds KV capacity"));
  }
  if (*populated != position) {
    return std::unexpected(make_error(
        ErrorCode::InvalidPopulatedLength, "position",
        "position must equal the append/populated contract"));
  }
  *value_ = position + 1u;
  return {};
}

std::expected<ConvCursorSlot, Error> ConvCursorSlot::bind(
    std::uint32_t* value) {
  if (value == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "cursor",
                                      "host cursor pointer is required"));
  }
  if (*value >= kConvTaps) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "cursor",
                                      "convolution cursor must be < 3"));
  }
  ConvCursorSlot slot;
  slot.value_ = value;
  return slot;
}

std::expected<std::uint32_t, Error> ConvCursorSlot::value() const {
  if (value_ == nullptr || *value_ >= kConvTaps) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "cursor",
                                      "invalid host cursor"));
  }
  return *value_;
}

std::expected<void, Error> ConvCursorSlot::commit_advance(
    std::uint32_t cursor) const {
  auto current = value();
  if (!current) {
    return std::unexpected(current.error());
  }
  if (*current != cursor) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "cursor",
                                      "cursor changed before commit"));
  }
  *value_ = (cursor + 1u) % kConvTaps;
  return {};
}

std::expected<GdnPositionSlot, Error> GdnPositionSlot::bind(
    std::uint64_t* value) {
  if (value == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "position",
                                      "host GDN position pointer is required"));
  }
  GdnPositionSlot slot;
  slot.value_ = value;
  return slot;
}

std::expected<std::uint64_t, Error> GdnPositionSlot::value() const {
  if (value_ == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "position",
                                      "host GDN position pointer is required"));
  }
  return *value_;
}

std::expected<void, Error> GdnPositionSlot::validate(
    std::uint64_t position) const {
  auto expected = value();
  if (!expected) {
    return std::unexpected(expected.error());
  }
  if (position != *expected) {
    return std::unexpected(make_error(
        ErrorCode::InvalidPopulatedLength, "position",
        position < *expected ? "GDN token position was already executed"
                             : "GDN token position skips the session sequence"));
  }
  if (position == std::numeric_limits<std::uint64_t>::max()) {
    return std::unexpected(make_error(ErrorCode::Overflow, "position",
                                      "GDN token position cannot advance"));
  }
  return {};
}

std::expected<void, Error> GdnPositionSlot::commit(
    std::uint64_t position) const {
  if (auto st = validate(position); !st) {
    return st;
  }
  *value_ = position + 1u;
  return {};
}

std::expected<Session, Error> Session::create(
    Model const& model, std::shared_ptr<qw38::cuda::Stream> stream,
    std::uint64_t kv_capacity) {
  try {
  if (auto st = validate_kv_capacity(kv_capacity); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_language_state(model.state()); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_language_scratch(model.scratch()); !st) {
    return std::unexpected(st.error());
  }

  auto kv_n = kv_cache_bytes(kv_capacity);
  if (!kv_n) {
    return std::unexpected(kv_n.error());
  }
  auto persist = persistent_state_bytes(kv_capacity);
  if (!persist) {
    return std::unexpected(persist.error());
  }
  auto res_n = residual_bytes(kArenaTokenCapacity);
  if (!res_n) {
    return std::unexpected(res_n.error());
  }
  auto arena = plan_v0_arena(kArenaTokenCapacity, kv_capacity);
  if (!arena) {
    return std::unexpected(arena.error());
  }

  Session s;
  s.stream_ = std::move(stream);
  s.kv_capacity_ = kv_capacity;
  s.gdn_position_.fill(0);
  s.kv_populated_.fill(0);
  s.persistent_bytes_ = *persist;
  s.arena_ = std::move(*arena);

  auto const device = s.stream_->device();
  auto gdn = qw38::cuda::DeviceBuffer::allocate(kGdnSBytes, device);
  if (!gdn) {
    return std::unexpected(from_cuda(gdn.error()));
  }
  auto conv = qw38::cuda::DeviceBuffer::allocate(kConvHistoryBytes, device);
  if (!conv) {
    return std::unexpected(from_cuda(conv.error()));
  }
  s.gdn_s_ = std::move(*gdn);
  s.conv_history_ = std::move(*conv);
  if (*kv_n != 0) {
    auto kv = qw38::cuda::DeviceBuffer::allocate(*kv_n, device);
    if (!kv) {
      return std::unexpected(from_cuda(kv.error()));
    }
    s.kv_ = std::move(*kv);
  }
  auto h = qw38::cuda::DeviceBuffer::allocate(*res_n, device);
  if (!h) {
    return std::unexpected(from_cuda(h.error()));
  }
  auto h_mid = qw38::cuda::DeviceBuffer::allocate(*res_n, device);
  if (!h_mid) {
    return std::unexpected(from_cuda(h_mid.error()));
  }
  s.residual_h_ = std::move(*h);
  s.residual_h_mid_ = std::move(*h_mid);
  auto scratch =
      qw38::cuda::DeviceBuffer::allocate(s.arena_.total_bytes, device);
  if (!scratch) {
    return std::unexpected(from_cuda(scratch.error()));
  }
  s.scratch_ = std::move(*scratch);

  if (auto st = qw38::cuda::zero(s.gdn_s_, *s.stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(s.conv_history_, *s.stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(s.kv_, *s.stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(s.residual_h_, *s.stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(s.residual_h_mid_, *s.stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(s.scratch_, *s.stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  s.conv_cursor_.fill(0);
  if (auto st = s.stream_->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return s;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "session.create",
                                      "host allocation failed"));
  } catch (std::length_error const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "session.create",
                                      "host allocation size is invalid"));
  }
}

std::expected<void, Error> Session::zero_persistent() {
  if (!stream_) {
    return std::unexpected(
        make_error(ErrorCode::Internal, "session", "missing stream"));
  }
  if (auto st = qw38::cuda::zero(gdn_s_, *stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(conv_history_, *stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st = qw38::cuda::zero(kv_, *stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<void, Error> Session::reset() {
  if (auto st = zero_persistent(); !st) {
    return st;
  }
  if (auto st = stream_->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  conv_cursor_.fill(0);
  gdn_position_.fill(0);
  kv_populated_.fill(0);
  return {};
}

std::expected<void, Error> Session::validate_metadata() const {
  for (auto populated : kv_populated_) {
    if (populated > kv_capacity_) {
      return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                        "session.populated",
                                        "populated length exceeds capacity"));
    }
  }
  for (auto cursor : conv_cursor_) {
    if (cursor >= kConvTaps) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                        "session.cursor",
                                        "convolution cursor must be < 3"));
    }
  }
  return {};
}

std::expected<SessionSnapshot, Error> Session::save() const {
  try {
  if (!stream_) {
    return std::unexpected(
        make_error(ErrorCode::Internal, "session.save", "missing stream"));
  }
  if (auto st = validate_metadata(); !st) {
    return std::unexpected(st.error());
  }
  SessionSnapshot snap;
  snap.gdn_s.resize(static_cast<std::size_t>(gdn_s_.bytes()));
  snap.conv_history.resize(static_cast<std::size_t>(conv_history_.bytes()));
  snap.kv.resize(static_cast<std::size_t>(kv_.bytes()));
  snap.conv_cursor = conv_cursor_;
  snap.gdn_position = gdn_position_;
  snap.kv_populated = kv_populated_;
  if (auto st = qw38::cuda::copy_d2h(snap.gdn_s, gdn_s_.data(), *stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st =
          qw38::cuda::copy_d2h(snap.conv_history, conv_history_.data(), *stream_);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (!kv_.empty()) {
    if (auto st = qw38::cuda::copy_d2h(snap.kv, kv_.data(), *stream_); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
  }
  if (auto st = stream_->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return snap;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "session.save",
                                      "host allocation failed"));
  } catch (std::length_error const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "session.save",
                                      "host allocation size is invalid"));
  }
}

std::expected<void, Error> Session::restore(SessionSnapshot const& snap) {
  if (!stream_) {
    return std::unexpected(
        make_error(ErrorCode::Internal, "session.restore", "missing stream"));
  }
  if (snap.gdn_s.size() != gdn_s_.bytes() ||
      snap.conv_history.size() != conv_history_.bytes() ||
      snap.kv.size() != kv_.bytes()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "snapshot",
                                      "snapshot size does not match session"));
  }
  for (auto populated : snap.kv_populated) {
    if (populated > kv_capacity_) {
      return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                        "snapshot.populated",
                                        "populated length exceeds capacity"));
    }
  }
  for (auto c : snap.conv_cursor) {
    if (c >= kConvTaps) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                        "snapshot.cursor",
                                        "convolution cursor must be < 3"));
    }
  }
  if (auto st = qw38::cuda::copy_h2d(gdn_s_.data(), snap.gdn_s, *stream_); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (auto st =
          qw38::cuda::copy_h2d(conv_history_.data(), snap.conv_history, *stream_);
      !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  if (!kv_.empty()) {
    if (auto st = qw38::cuda::copy_h2d(kv_.data(), snap.kv, *stream_); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
  }
  if (auto st = stream_->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  conv_cursor_ = snap.conv_cursor;
  gdn_position_ = snap.gdn_position;
  kv_populated_ = snap.kv_populated;
  return {};
}

TensorView Session::gdn_s() const noexcept {
  return view_from_buffer(gdn_s_, qw38::format::ArithmeticDtype::Fp32,
                          qw38::format::PhysicalLayoutId::CudaFp32GdnSHvKV0,
                          qw38::format::StorageClass::Fp32, 4,
                          extent4(kGdnLayers, kGdnValueHeads, kGdnValueDim,
                                  kGdnKeyDim));
}

TensorView Session::conv_history() const noexcept {
  return view_from_buffer(conv_history_, qw38::format::ArithmeticDtype::Bf16,
                          qw38::format::PhysicalLayoutId::CudaBf16ConvHistoryV0,
                          qw38::format::StorageClass::Bf16, 3,
                          extent3(kConvLayers, kConvTaps, kConvChannels));
}

TensorView Session::kv() const noexcept {
  return view_from_buffer(
      kv_, qw38::format::ArithmeticDtype::Bf16,
      qw38::format::PhysicalLayoutId::CudaBf16KvCacheV0,
      qw38::format::StorageClass::Bf16, 5,
      extent5(kAttnLayers, kKvComponents, kKvHeads, kv_capacity_, kHeadDim));
}

TensorView Session::residual_h() const noexcept {
  return view_from_buffer(residual_h_, qw38::format::ArithmeticDtype::Fp32,
                          qw38::format::PhysicalLayoutId::CudaFp32VectorV0,
                          qw38::format::StorageClass::Fp32, 2,
                          extent2(kArenaTokenCapacity, kHidden));
}

TensorView Session::residual_h_mid() const noexcept {
  return view_from_buffer(residual_h_mid_, qw38::format::ArithmeticDtype::Fp32,
                          qw38::format::PhysicalLayoutId::CudaFp32VectorV0,
                          qw38::format::StorageClass::Fp32, 2,
                          extent2(kArenaTokenCapacity, kHidden));
}

std::expected<WorkspaceView, Error> Session::scratch(
    qw38::format::ScratchKind kind) const {
  auto const* p = find_placement(arena_, kind);
  if (p == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "scratch.kind",
                   "kind is not in the liveness-planned arena"));
  }
  if (p->offset + p->bytes > scratch_.bytes()) {
    return std::unexpected(
        make_error(ErrorCode::Internal, "scratch", "placement exceeds arena"));
  }
  WorkspaceView workspace{};
  workspace.pointer =
      const_cast<std::byte*>(scratch_.as_bytes()) + p->offset;
  workspace.bytes = p->bytes;
  workspace.kind = kind;

  auto add_region = [&](std::uint64_t offset, std::uint64_t bytes,
                        qw38::format::ArithmeticDtype dtype,
                        std::uint8_t rank, std::uint64_t e0,
                        std::uint64_t e1 = 0) {
    auto& region = workspace.region[workspace.region_count++];
    region.offset = offset;
    region.bytes = bytes;
    region.stride_bytes = bytes;
    if (kind == qw38::format::ScratchKind::GdnWorkspace) {
      region.stride_bytes = kGdnWorkspaceBytesPerToken;
      region.repetitions = p->bytes / kGdnWorkspaceBytesPerToken;
    }
    auto& v = region.tensor;
    v.pointer = workspace.pointer + offset;
    v.dtype = dtype;
    v.layout = dtype == qw38::format::ArithmeticDtype::Fp32
                   ? qw38::format::PhysicalLayoutId::CudaFp32VectorV0
                   : qw38::format::PhysicalLayoutId::CudaBf16RowMajorV0;
    v.storage = dtype == qw38::format::ArithmeticDtype::Fp32
                    ? qw38::format::StorageClass::Fp32
                    : qw38::format::StorageClass::Bf16;
    v.rank = rank;
    v.extent[0] = e0;
    v.extent[1] = e1;
  };

  if (kind == qw38::format::ScratchKind::GdnWorkspace) {
    add_region(kGdnOffQkv, kGdnBytesQkv,
               qw38::format::ArithmeticDtype::Bf16, 1, kConvChannels);
    add_region(kGdnOffZ, kGdnBytesZ, qw38::format::ArithmeticDtype::Bf16, 2,
               kGdnValueHeads, kGdnValueDim);
    add_region(kGdnOffConvolved, kGdnBytesConvolved,
               qw38::format::ArithmeticDtype::Bf16, 1, kConvChannels);
    add_region(kGdnOffQHat, kGdnBytesQHat,
               qw38::format::ArithmeticDtype::Fp32, 2, kGdnKeyHeads,
               kGdnKeyDim);
    add_region(kGdnOffKHat, kGdnBytesKHat,
               qw38::format::ArithmeticDtype::Fp32, 2, kGdnKeyHeads,
               kGdnKeyDim);
    add_region(kGdnOffA, kGdnBytesGate, qw38::format::ArithmeticDtype::Fp32,
               1, kGdnValueHeads);
    add_region(kGdnOffB, kGdnBytesGate, qw38::format::ArithmeticDtype::Fp32,
               1, kGdnValueHeads);
    add_region(kGdnOffAlpha, kGdnBytesGate,
               qw38::format::ArithmeticDtype::Fp32, 1, kGdnValueHeads);
    add_region(kGdnOffBeta, kGdnBytesGate,
               qw38::format::ArithmeticDtype::Fp32, 1, kGdnValueHeads);
    add_region(kGdnOffO, kGdnBytesO, qw38::format::ArithmeticDtype::Fp32, 2,
               kGdnValueHeads, kGdnValueDim);
    add_region(kGdnOffU, kGdnBytesU, qw38::format::ArithmeticDtype::Bf16, 2,
               kGdnValueHeads, kGdnValueDim);
    return workspace;
  }
  if (kind == qw38::format::ScratchKind::AttentionWorkspace) {
    add_region(kAttnOffQg, kAttnBytesQg,
               qw38::format::ArithmeticDtype::Bf16, 2, kQueryHeads,
               2u * kHeadDim);
    add_region(kAttnOffK, kAttnBytesK, qw38::format::ArithmeticDtype::Bf16, 2,
               kKvHeads, kHeadDim);
    add_region(kAttnOffV, kAttnBytesV, qw38::format::ArithmeticDtype::Bf16, 2,
               kKvHeads, kHeadDim);
    add_region(kAttnOffQ, kAttnBytesQ, qw38::format::ArithmeticDtype::Bf16, 2,
               kQueryHeads, kHeadDim);
    add_region(kAttnOffG, kAttnBytesG, qw38::format::ArithmeticDtype::Bf16, 2,
               kQueryHeads, kHeadDim);
    add_region(kAttnOffPartials, p->bytes - kAttnOffPartials,
               qw38::format::ArithmeticDtype::Fp32, 1,
               (p->bytes - kAttnOffPartials) / qw38::format::kFp32Size);
    return workspace;
  }

  if (!p->dtype) {
    return std::unexpected(make_error(
        ErrorCode::Internal, "scratch", "composite workspace metadata missing"));
  }
  add_region(0, p->bytes, *p->dtype, 1,
             p->bytes / qw38::format::element_size(*p->dtype));
  return workspace;
}

std::expected<void, Error> Session::set_populated_length(
    std::uint32_t attention_layer, std::uint64_t populated) {
  if (attention_layer >= kAttnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "kv.layer",
                                      "attention layer index must be < 16"));
  }
  if (populated > kv_capacity_) {
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                      "kv.populated",
                                      "populated length exceeds capacity"));
  }
  kv_populated_[attention_layer] = populated;
  return {};
}

std::expected<KvPopulatedSlot, Error> Session::kv_populated_slot(
    std::uint32_t attention_layer) {
  if (attention_layer >= kAttnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "kv.layer",
                                      "attention layer index must be < 16"));
  }
  return KvPopulatedSlot::bind(&kv_populated_[attention_layer], kv_capacity_);
}

std::expected<ConvCursorSlot, Error> Session::conv_cursor_slot(
    std::uint32_t gdn_layer) {
  if (gdn_layer >= kConvLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "conv.layer",
                                      "GDN layer index must be < 48"));
  }
  return ConvCursorSlot::bind(&conv_cursor_[gdn_layer]);
}

std::expected<GdnPositionSlot, Error> Session::gdn_position_slot(
    std::uint32_t gdn_layer) {
  if (gdn_layer >= kGdnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "gdn.layer",
                                      "GDN layer index must be < 48"));
  }
  return GdnPositionSlot::bind(&gdn_position_[gdn_layer]);
}

std::expected<void, Error> Session::set_conv_cursor(
    std::array<std::uint32_t, kConvLayers> cursor) {
  for (auto c : cursor) {
    if (c >= kConvTaps) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "conv.cursor",
                                        "convolution cursor must be < 3"));
    }
  }
  conv_cursor_ = cursor;
  return {};
}

qw38::cuda::Stream const* detail::SessionPlanAccess::stream(
    Session const& session) noexcept {
  return session.stream_.get();
}

std::expected<KvPopulatedSlot, Error> detail::SessionPlanAccess::kv_populated(
    Session& session, std::uint32_t attention_layer) {
  return session.kv_populated_slot(attention_layer);
}

std::expected<ConvCursorSlot, Error> detail::SessionPlanAccess::conv_cursor(
    Session& session, std::uint32_t gdn_layer) {
  return session.conv_cursor_slot(gdn_layer);
}

std::expected<GdnPositionSlot, Error> detail::SessionPlanAccess::gdn_position(
    Session& session, std::uint32_t gdn_layer) {
  return session.gdn_position_slot(gdn_layer);
}

}  // namespace qw38::runtime
