#include "runtime/prefill.hpp"

#include "cuda/activation.hpp"
#include "cuda/gdn.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace qw38::runtime {
namespace {

constexpr std::uint64_t kWorkspaceBytesPerToken =
    2u * kHidden + 2u * kConvChannels + 2u * kGdnZWidth +
    2u * 4u * kGdnValueHeads + 2u * kConvChannels +
    2u * 4u * kGdnKeyHeads * kGdnKeyDim +
    2u * 4u * kGdnValueHeads +
    4u * kGdnZWidth + 2u * kGdnZWidth;
static_assert(kWorkspaceBytesPerToken == 117504u);

struct Region { std::uintptr_t begin, end; };

Region region(void const* ptr, std::uint64_t bytes) {
  auto const begin = reinterpret_cast<std::uintptr_t>(ptr);
  return {begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(Region a, Region b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

std::expected<void, Error> reject_overlap(Region writable,
                                           void const* ptr,
                                           std::uint64_t bytes) {
  if (ptr && bytes && overlaps(writable, region(ptr, bytes)))
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.gdn.alias", "writable buffer overlaps live operand"));
  return {};
}

}  // namespace

std::expected<PrefillGdnLayerPlan, Error> bind_prefill_gdn_layer(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream) {
  if (!is_gdn_language_layer(layer))
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "layer",
                                      "prefill GDN requires a GDN layer"));
  auto gdn = bind_gdn_plan(model, session, layer, stream);
  if (!gdn) return std::unexpected(gdn.error());
  auto projections = bind_prefill_layer_projections(model, layer, stream);
  if (!projections) return std::unexpected(projections.error());
  return PrefillGdnLayerPlan{*gdn, *projections};
}

std::expected<PrefillGdnWorkspace, Error> PrefillGdnWorkspace::create(
    std::uint32_t token_capacity, int device) {
  if (token_capacity == 0 || token_capacity > qw38::cuda::kPrefillMaxTokens)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.gdn.capacity", "invalid token capacity"));
  auto allocation = qw38::cuda::DeviceBuffer::allocate(
      kWorkspaceBytesPerToken * token_capacity, device);
  if (!allocation) return std::unexpected(from_cuda(allocation.error()));
  PrefillGdnWorkspace result;
  result.storage_ = std::move(*allocation);
  result.capacity_ = token_capacity;
  return result;
}

PrefillGdnSlices PrefillGdnWorkspace::slices() noexcept {
  std::byte* cursor = storage_.as_bytes();
  auto next = [&](std::uint64_t row_bytes) {
    std::byte* result = cursor;
    cursor += row_bytes * capacity_;
    return result;
  };
  PrefillGdnSlices v;
  v.normalized = reinterpret_cast<std::uint16_t*>(next(2u * kHidden));
  v.qkv = reinterpret_cast<std::uint16_t*>(next(2u * kConvChannels));
  v.z = reinterpret_cast<std::uint16_t*>(next(2u * kGdnZWidth));
  v.a = reinterpret_cast<float*>(next(4u * kGdnValueHeads));
  v.b = reinterpret_cast<float*>(next(4u * kGdnValueHeads));
  v.convolved = reinterpret_cast<std::uint16_t*>(next(2u * kConvChannels));
  v.q_hat = reinterpret_cast<float*>(next(4u * kGdnKeyHeads * kGdnKeyDim));
  v.k_hat = reinterpret_cast<float*>(next(4u * kGdnKeyHeads * kGdnKeyDim));
  v.alpha = reinterpret_cast<float*>(next(4u * kGdnValueHeads));
  v.beta = reinterpret_cast<float*>(next(4u * kGdnValueHeads));
  v.o = reinterpret_cast<float*>(next(4u * kGdnZWidth));
  v.u = reinterpret_cast<std::uint16_t*>(next(2u * kGdnZWidth));
  return v;
}

std::expected<void, Error> execute_prefill_gdn_layer(
    PrefillGdnLayerPlan const& plan, PrefillGdnWorkspace& workspace,
    qw38::cuda::PrefillEngine& engine, float const* residual,
    float* h_mid, float* next_h, std::uint32_t valid_tokens,
    std::uint64_t first_position, std::uint32_t recurrence_interval) {
  auto const& gdn = plan.gdn_;
  auto const& p = plan.projections_;
  auto const* stream = gdn.front.stream;
  if (!stream || stream->empty() || engine.stream() != stream ||
      gdn.session_state == nullptr || gdn.session_state->is_poisoned() ||
      !p.gdn || p.layer != gdn.front.language_layer ||
      valid_tokens == 0 || valid_tokens > workspace.token_capacity() ||
      valid_tokens > engine.token_capacity() ||
      recurrence_interval == 0 ||
      recurrence_interval > qw38::cuda::kPrefillMaxTokens ||
      workspace.device() != stream->device() ||
      first_position > std::numeric_limits<std::uint64_t>::max() - valid_tokens)
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.gdn",
                                      "plan, stream, interval or chunk mismatch"));
  if (auto st = gdn.position.validate(first_position); !st)
    return std::unexpected(st.error());
  auto cursor = gdn.front.cursor.value();
  if (!cursor) return std::unexpected(cursor.error());

  std::uint64_t const residual_bytes =
      static_cast<std::uint64_t>(valid_tokens) * kHidden * 4u;
  for (void const* ptr : {static_cast<void const*>(residual),
                          static_cast<void const*>(h_mid),
                          static_cast<void const*>(next_h)}) {
    auto st = qw38::cuda::validate_prefill_device_span(
        ptr, residual_bytes, 4u, stream->device());
    if (!st) return std::unexpected(from_cuda(st.error()));
  }
  auto const r = region(residual, residual_bytes);
  auto const mid = region(h_mid, residual_bytes);
  auto const out = region(next_h, residual_bytes);
  auto const scratch = region(workspace.data(), workspace.bytes());
  if (overlaps(r, mid) || overlaps(r, out) || overlaps(mid, out) ||
      overlaps(r, scratch) || overlaps(mid, scratch) || overlaps(out, scratch))
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.gdn.alias", "activation buffers overlap"));
  std::array<Region, 2> writable{mid, out};
  auto const& f = gdn.front;
  std::array<std::pair<void const*, std::uint64_t>, 9> persistent{{
      {f.history.pointer, kConvTaps * kConvChannels * 2u},
      {gdn.s.pointer, kGdnSBytes},
      {f.gamma.pointer, kHidden * 2u},
      {f.taps.pointer, kConvKernel * kConvChannels * 2u},
      {f.a_log.pointer, kGdnValueHeads * 2u},
      {f.dt_bias.pointer, kGdnValueHeads * 2u},
      {gdn.gated_gamma.pointer, kGdnValueDim * 2u},
      {p.mlp_gamma, kHidden * 2u},
      {workspace.data(), workspace.bytes()}}};
  for (Region target : writable)
    for (auto const& [ptr, bytes] : persistent)
      if (auto st = reject_overlap(target, ptr, bytes); !st) return st;
  for (auto const& [ptr, bytes] : persistent)
    if (auto st = reject_overlap(r, ptr, bytes); !st) return st;
  for (auto const& weight : {p.first, p.second, p.third, p.fourth,
                             p.mixer_out, p.mlp_gate, p.mlp_up, p.mlp_down})
    for (Region target : writable) {
      if (auto st = reject_overlap(target, weight.codes, weight.codes_bytes); !st)
        return st;
      if (auto st = reject_overlap(target, weight.scales, weight.scales_bytes); !st)
        return st;
    }

  auto s = workspace.slices();
  auto fail = [&](Error error) -> std::expected<void, Error> {
    gdn.session_state->poison();
    // A failed launch can leave earlier work queued on this stream. Keep the
    // borrowed input, output and workspace lifetime safe on error return.
    (void)stream->sync();
    return std::unexpected(std::move(error));
  };
  auto cuda_step = [&](std::expected<void, qw38::cuda::Error> status)
      -> std::expected<void, Error> {
    if (!status) return fail(from_cuda(status.error()));
    return {};
  };
  auto project = [&](qw38::cuda::PrefillWeight const& weight,
                     std::uint16_t const* input, void* output,
                     qw38::cuda::PrefillEpilogue epilogue,
                     float const* add = nullptr) -> std::expected<void, Error> {
    return cuda_step(engine.project(qw38::cuda::PrefillProjection{
        .weight = weight, .input = input, .output = output, .residual = add,
        .valid_tokens = valid_tokens, .first_position = first_position,
        .epilogue = epilogue}));
  };

  if (auto st = cuda_step(qw38::cuda::launch_hidden_rms(
          residual, static_cast<std::uint16_t const*>(f.gamma.pointer),
          f.eps, valid_tokens, s.normalized, *stream)); !st) return st;
  using qw38::cuda::PrefillEpilogue;
  if (auto st = project(p.first, s.normalized, s.qkv, PrefillEpilogue::StoreBf16);
      !st) return st;
  if (auto st = project(p.second, s.normalized, s.z, PrefillEpilogue::StoreBf16);
      !st) return st;
  if (auto st = project(p.third, s.normalized, s.a, PrefillEpilogue::StoreFp32);
      !st) return st;
  if (auto st = project(p.fourth, s.normalized, s.b, PrefillEpilogue::StoreFp32);
      !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_gdn_prefill_conv(
          s.qkv, static_cast<std::uint16_t const*>(f.taps.pointer),
          static_cast<std::uint16_t const*>(f.history.pointer), *cursor,
          s.convolved, valid_tokens, *stream)); !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_gdn_prefill_history_commit(
          s.qkv, static_cast<std::uint16_t*>(f.history.pointer), *cursor,
          valid_tokens, *stream)); !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_gdn_prefill_prepare(
          s.convolved, s.a, s.b,
          static_cast<std::uint16_t const*>(f.a_log.pointer),
          static_cast<std::uint16_t const*>(f.dt_bias.pointer), f.eps,
          s.q_hat, s.k_hat, s.alpha, s.beta, valid_tokens, *stream)); !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_gdn_prefill_recurrence(
          s.q_hat, s.k_hat, s.alpha, s.beta, s.convolved,
          static_cast<float*>(gdn.s.pointer), gdn.s_layer, s.o,
          valid_tokens, recurrence_interval, *stream)); !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_gdn_gated_rms(
          s.o, s.z, static_cast<std::uint16_t const*>(gdn.gated_gamma.pointer),
          f.eps, valid_tokens * kGdnValueHeads, s.u, *stream)); !st) return st;
  if (auto st = project(p.mixer_out, s.u, h_mid,
                        PrefillEpilogue::ResidualAddFp32, residual); !st) return st;
  if (auto st = execute_prefill_mlp(p, engine, h_mid, next_h,
                                     valid_tokens, first_position, f.eps); !st)
    return fail(st.error());
  if (auto st = stream->sync(); !st) return fail(from_cuda(st.error()));
  if (auto st = f.cursor.commit_chunk(*cursor, valid_tokens); !st)
    return fail(st.error());
  if (auto st = gdn.position.commit_chunk(first_position, valid_tokens); !st)
    return fail(st.error());
  return {};
}

}  // namespace qw38::runtime
