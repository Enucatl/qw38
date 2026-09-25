#include "runtime/prefill.hpp"

#include "cuda/activation.hpp"
#include "cuda/attention.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace qw38::runtime {
namespace {

constexpr std::uint64_t kBytesPerToken = 2u *
    (kHidden + kQgWidth + 2u * kAttnKvWidth +
     3u * kAttnOutWidth);
static_assert(kBytesPerToken == 75776u);

struct Region { std::uintptr_t begin, end; };
Region region(void const* ptr, std::uint64_t bytes) {
  auto const begin = reinterpret_cast<std::uintptr_t>(ptr);
  return {begin, begin + static_cast<std::uintptr_t>(bytes)};
}
bool overlaps(Region a, Region b) noexcept {
  return a.begin < b.end && b.begin < a.end;
}

}  // namespace

std::expected<PrefillAttentionLayerPlan, Error> bind_prefill_attention_layer(
    Model const& model, Session& session, std::uint32_t layer,
    qw38::cuda::Stream const& stream) {
  if (!is_attention_language_layer(layer))
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "layer",
                                      "prefill attention requires an attention layer"));
  auto prep = bind_attention_prep_plan(model, session, layer, stream);
  if (!prep) return std::unexpected(prep.error());
  auto projections = bind_prefill_layer_projections(model, layer, stream);
  if (!projections) return std::unexpected(projections.error());
  return PrefillAttentionLayerPlan{*prep, *projections};
}

std::expected<PrefillAttentionWorkspace, Error>
PrefillAttentionWorkspace::create(std::uint32_t token_capacity, int device) {
  if (token_capacity == 0 || token_capacity > qw38::cuda::kPrefillMaxTokens)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.attention.capacity", "invalid token capacity"));
  auto allocation = qw38::cuda::DeviceBuffer::allocate(
      kBytesPerToken * token_capacity, device);
  if (!allocation) return std::unexpected(from_cuda(allocation.error()));
  PrefillAttentionWorkspace result;
  result.storage_ = std::move(*allocation);
  result.capacity_ = token_capacity;
  return result;
}

PrefillAttentionSlices PrefillAttentionWorkspace::slices() noexcept {
  std::byte* cursor = storage_.as_bytes();
  auto next = [&](std::uint64_t row_bytes) {
    std::byte* result = cursor;
    cursor += row_bytes * capacity_;
    return reinterpret_cast<std::uint16_t*>(result);
  };
  return {next(2u * kHidden), next(2u * kQgWidth),
          next(2u * kAttnKvWidth), next(2u * kAttnKvWidth),
          next(2u * kAttnOutWidth), next(2u * kAttnOutWidth),
          next(2u * kAttnOutWidth)};
}

std::expected<void, Error> execute_prefill_attention_layer(
    PrefillAttentionLayerPlan const& plan, PrefillAttentionWorkspace& workspace,
    qw38::cuda::PrefillEngine& engine, float const* residual,
    float* h_mid, float* next_h, std::uint32_t valid_tokens,
    std::uint64_t first_position) {
  auto const& prep = plan.prep_;
  auto const& p = plan.projections_;
  auto const* stream = prep.stream;
  if (!stream || stream->empty() || engine.stream() != stream ||
      !prep.session_state || prep.session_state->is_poisoned() ||
      p.gdn || p.layer != prep.language_layer ||
      valid_tokens == 0 || valid_tokens > workspace.token_capacity() ||
      valid_tokens > engine.token_capacity() ||
      workspace.device() != stream->device() ||
      first_position > std::numeric_limits<std::int32_t>::max() ||
      valid_tokens - 1u >
          static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) -
              first_position ||
      first_position >= prep.kv_capacity ||
      valid_tokens > prep.kv_capacity - first_position)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.attention", "plan, stream or chunk mismatch"));
  auto populated = prep.populated.value();
  if (!populated) return std::unexpected(populated.error());
  if (*populated != first_position)
    return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                      "prefill.attention", "chunk must append at populated length"));

  std::uint64_t const residual_bytes =
      static_cast<std::uint64_t>(valid_tokens) * kHidden * 4u;
  for (void const* ptr : {static_cast<void const*>(residual),
                          static_cast<void const*>(h_mid),
                          static_cast<void const*>(next_h)}) {
    auto st = qw38::cuda::validate_prefill_device_span(
        ptr, residual_bytes, 4u, stream->device());
    if (!st) return std::unexpected(from_cuda(st.error()));
  }
  std::array<Region, 4> active{region(residual, residual_bytes),
      region(h_mid, residual_bytes), region(next_h, residual_bytes),
      region(workspace.data(), workspace.bytes())};
  for (std::size_t i = 0; i < active.size(); ++i)
    for (std::size_t j = i + 1; j < active.size(); ++j)
      if (overlaps(active[i], active[j]))
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "prefill.attention.alias", "activation buffers overlap"));
  std::array<std::pair<void const*, std::uint64_t>, 6> resident{{
      {prep.kv.pointer, 16u * 2u * kKvHeads * prep.kv_capacity *
                            kHeadDim * 2u},
      {prep.gamma.pointer, kHidden * 2u},
      {prep.gamma_q.pointer, kHeadDim * 2u},
      {prep.gamma_k.pointer, kHeadDim * 2u},
      {prep.inv_freq.pointer, 32u * 4u},
      {p.mlp_gamma, kHidden * 2u}}};
  for (auto const& [ptr, bytes] : resident)
    for (Region a : active)
      if (overlaps(a, region(ptr, bytes)))
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "prefill.attention.alias", "buffer overlaps resident model or state"));
  for (auto const& weight : {p.first, p.second, p.third, p.mixer_out,
                             p.mlp_gate, p.mlp_up, p.mlp_down})
    for (Region a : active)
      if (overlaps(a, region(weight.codes, weight.codes_bytes)) ||
          (weight.scales_bytes && overlaps(a, region(weight.scales, weight.scales_bytes))))
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "prefill.attention.alias", "buffer overlaps weight"));

  auto s = workspace.slices();
  auto fail = [&](Error error) -> std::expected<void, Error> {
    prep.session_state->poison();
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
  using qw38::cuda::PrefillEpilogue;
  if (auto st = cuda_step(qw38::cuda::launch_hidden_rms(
          residual, static_cast<std::uint16_t const*>(prep.gamma.pointer),
          prep.eps, valid_tokens, s.normalized, *stream)); !st) return st;
  if (auto st = project(p.first, s.normalized, s.qg, PrefillEpilogue::StoreBf16);
      !st) return st;
  if (auto st = project(p.second, s.normalized, s.k, PrefillEpilogue::StoreBf16);
      !st) return st;
  if (auto st = project(p.third, s.normalized, s.v, PrefillEpilogue::StoreBf16);
      !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_attention_prepare_chunk(
          s.qg, s.k, s.v,
          static_cast<std::uint16_t const*>(prep.gamma_q.pointer),
          static_cast<std::uint16_t const*>(prep.gamma_k.pointer),
          static_cast<float const*>(prep.inv_freq.pointer), prep.eps,
          first_position, valid_tokens, s.q, s.g,
          static_cast<std::uint16_t*>(prep.kv.pointer), prep.attn_layer,
          prep.kv_capacity, *stream)); !st) return st;
  if (auto st = cuda_step(qw38::cuda::launch_attention_prefill_scan(
          s.q, s.g, static_cast<std::uint16_t const*>(prep.kv.pointer),
          prep.attn_layer, prep.kv_capacity, first_position,
          valid_tokens, s.y, *stream)); !st) return st;
  if (auto st = project(p.mixer_out, s.y, h_mid,
                        PrefillEpilogue::ResidualAddFp32, residual); !st) return st;
  if (auto st = execute_prefill_mlp(p, engine, h_mid, next_h,
                                    valid_tokens, first_position, prep.eps); !st)
    return fail(st.error());
  if (auto st = stream->sync(); !st) return fail(from_cuda(st.error()));
  if (auto st = prep.populated.commit_chunk(first_position, valid_tokens); !st)
    return fail(st.error());
  return {};
}

}  // namespace qw38::runtime
