#include "runtime/session.hpp"

#include "cuda/copy.hpp"
#include "cuda/stream.hpp"
#include "runtime/model.hpp"

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
  v.writable = true;
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

std::expected<Session, Error> Session::create(
    Model const& model, std::shared_ptr<qw38::cuda::Stream> stream,
    std::uint64_t kv_capacity) {
  try {
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
  auto arena = plan_v0_arena(kArenaTokenCapacity);
  if (!arena) {
    return std::unexpected(arena.error());
  }

  Session s;
  s.stream_ = std::move(stream);
  s.kv_capacity_ = kv_capacity;
  s.kv_populated_.fill(0);
  s.persistent_bytes_ = *persist;
  s.arena_ = std::move(*arena);

  auto gdn = qw38::cuda::DeviceBuffer::allocate(kGdnSBytes);
  if (!gdn) {
    return std::unexpected(from_cuda(gdn.error()));
  }
  auto conv = qw38::cuda::DeviceBuffer::allocate(kConvHistoryBytes);
  if (!conv) {
    return std::unexpected(from_cuda(conv.error()));
  }
  s.gdn_s_ = std::move(*gdn);
  s.conv_history_ = std::move(*conv);
  if (*kv_n != 0) {
    auto kv = qw38::cuda::DeviceBuffer::allocate(*kv_n);
    if (!kv) {
      return std::unexpected(from_cuda(kv.error()));
    }
    s.kv_ = std::move(*kv);
  }
  auto h = qw38::cuda::DeviceBuffer::allocate(*res_n);
  if (!h) {
    return std::unexpected(from_cuda(h.error()));
  }
  auto h_mid = qw38::cuda::DeviceBuffer::allocate(*res_n);
  if (!h_mid) {
    return std::unexpected(from_cuda(h_mid.error()));
  }
  s.residual_h_ = std::move(*h);
  s.residual_h_mid_ = std::move(*h_mid);
  auto scratch = qw38::cuda::DeviceBuffer::allocate(s.arena_.total_bytes);
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
  conv_cursor_.fill(0);
  kv_populated_.fill(0);
  return {};
}

std::expected<void, Error> Session::reset() {
  if (auto st = zero_persistent(); !st) {
    return st;
  }
  if (auto st = stream_->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return {};
}

std::expected<SessionSnapshot, Error> Session::save() const {
  try {
  if (!stream_) {
    return std::unexpected(
        make_error(ErrorCode::Internal, "session.save", "missing stream"));
  }
  SessionSnapshot snap;
  snap.gdn_s.resize(static_cast<std::size_t>(gdn_s_.bytes()));
  snap.conv_history.resize(static_cast<std::size_t>(conv_history_.bytes()));
  snap.kv.resize(static_cast<std::size_t>(kv_.bytes()));
  snap.conv_cursor = conv_cursor_;
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
  conv_cursor_ = snap.conv_cursor;
  kv_populated_ = snap.kv_populated;
  if (auto st = stream_->sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
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

std::expected<TensorView, Error> Session::scratch(
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
  TensorView v{};
  v.pointer = const_cast<std::byte*>(scratch_.as_bytes()) + p->offset;
  v.dtype = p->dtype;
  v.layout = qw38::format::PhysicalLayoutId::CudaBf16RowMajorV0;
  if (p->dtype == qw38::format::ArithmeticDtype::Fp32) {
    v.layout = qw38::format::PhysicalLayoutId::CudaFp32VectorV0;
    v.storage = qw38::format::StorageClass::Fp32;
  } else {
    v.storage = qw38::format::StorageClass::Bf16;
  }
  v.space = MemorySpace::Device;
  v.writable = true;
  v.rank = 1;
  auto const elem = qw38::format::element_size(p->dtype);
  v.extent[0] = elem == 0 ? p->bytes : p->bytes / elem;
  return v;
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

std::expected<std::uint64_t*, Error> Session::kv_populated_slot(
    std::uint32_t attention_layer) {
  if (attention_layer >= kAttnLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "kv.layer",
                                      "attention layer index must be < 16"));
  }
  return &kv_populated_[attention_layer];
}

std::expected<std::uint32_t*, Error> Session::conv_cursor_slot(
    std::uint32_t gdn_layer) {
  if (gdn_layer >= kConvLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "conv.layer",
                                      "GDN layer index must be < 48"));
  }
  return &conv_cursor_[gdn_layer];
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

}  // namespace qw38::runtime
