#include "runtime/mlp.hpp"
#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"
#include "runtime/sizes.hpp"

#include "cuda/alloc.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/prefill.hpp"
#include "cuda/upload.hpp"
#include "compiler/quantization/reference.hpp"
#include "format/floatcvt.hpp"
#include "format/unpack.hpp"
#include "reference/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

template <class T>
std::span<std::byte const> bytes(std::vector<T> const& values) {
  return {reinterpret_cast<std::byte const*>(values.data()),
          values.size() * sizeof(T)};
}

template <class T>
std::vector<T> download(void const* device, std::size_t count,
                        qw38::cuda::Stream const& stream) {
  std::vector<T> host(count);
  auto st = qw38::cuda::copy_d2h(host.data(), device, count * sizeof(T), stream);
  if (!st || !stream.sync()) return {};
  return host;
}

bool finite_bf16(std::vector<std::uint16_t> const& values) {
  return std::ranges::all_of(values, [](auto bits) {
    return std::isfinite(qw38::format::bf16_to_fp32(bits));
  });
}

std::vector<std::uint16_t> reconstructed_prefix(
    qw38::cuda::PrefillWeight const& weight, std::uint32_t rows,
    qw38::cuda::Stream const& stream) {
  auto const padded_n = qw38::cuda::decode_pad_n(rows);
  auto const code_bytes = qw38::cuda::decode_code_bytes(
      weight.layout, padded_n, weight.padded_k);
  auto const scale_bytes = qw38::cuda::decode_scale_bytes(
      weight.layout, padded_n, weight.padded_k);
  std::vector<std::byte> codes(code_bytes), scales(scale_bytes);
  if (!qw38::cuda::copy_d2h(codes.data(), weight.codes, code_bytes, stream) ||
      (scale_bytes && !qw38::cuda::copy_d2h(
          scales.data(), weight.scales, scale_bytes, stream)) ||
      !stream.sync()) return {};
  auto const layout = static_cast<qw38::format::PhysicalLayoutId>(weight.layout);
  if (layout == qw38::format::PhysicalLayoutId::CudaBf16DenseTileV0) {
    auto dense = qw38::compiler::decode_bf16_payload(layout, codes, rows, weight.k);
    return dense ? std::move(*dense) : std::vector<std::uint16_t>{};
  }
  auto logical = qw38::format::unpack_cuda_v0(
      static_cast<qw38::format::LogicalQuantizerId>(weight.quantizer),
      layout, rows, weight.k, codes, scales);
  if (!logical) return {};
  auto dense = qw38::compiler::dequantize_to_bf16(*logical);
  return dense ? std::move(*dense) : std::vector<std::uint16_t>{};
}

bool close_enough(float got, float expected, float& maximum,
                  float absolute = 0.02f, float relative = 0.002f) {
  if (!std::isfinite(got) || !std::isfinite(expected)) return false;
  float const difference = std::fabs(got - expected);
  maximum = std::max(maximum, difference);
  return difference <= absolute + relative * std::fabs(expected);
}

int run(char const* artifact) {
  using namespace qw38::runtime;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto model = runtime->load(artifact);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  auto const& stream = runtime->stream();
  std::vector<PrefillLayerProjectionPlan> plans;
  plans.reserve(kLanguageLayers);
  for (std::uint32_t layer = 0; layer < kLanguageLayers; ++layer) {
    auto p = bind_prefill_layer_projections(*model, layer, stream);
    if (!p) { std::cerr << error_message(p.error()) << '\n'; return 1; }
    plans.push_back(*p);
  }
  auto head = bind_prefill_head(*model, stream);
  if (!head) { std::cerr << error_message(head.error()) << '\n'; return 1; }
  auto engine = qw38::cuda::PrefillEngine::create(stream, 128);
  if (!engine) { std::cerr << qw38::cuda::error_message(engine.error()) << '\n'; return 1; }

  constexpr std::uint32_t m = 3;
  auto embedding = model->payload("model.language_model.embed_tokens.weight");
  if (!embedding) return 1;
  std::vector<std::uint16_t> input_bf16(m * kHidden);
  std::array<std::uint32_t, m> ids{1000, 2000, 3000};
  auto const* table = static_cast<std::uint16_t const*>(embedding->pointer);
  for (std::uint32_t i = 0; i < m; ++i) {
    auto st = qw38::cuda::copy_d2h(input_bf16.data() + i * kHidden,
                                   table + static_cast<std::uint64_t>(ids[i]) * kHidden,
                                   kHidden * 2u, stream);
    if (!st) return 1;
  }
  if (!stream.sync()) return 1;
  std::vector<float> hmid(m * kHidden);
  for (std::size_t i = 0; i < hmid.size(); ++i)
    hmid[i] = qw38::format::bf16_to_fp32(input_bf16[i]);
  auto din = qw38::cuda::upload(bytes(input_bf16), stream);
  auto dhmid = qw38::cuda::upload(bytes(hmid), stream);
  auto dnext = qw38::cuda::DeviceBuffer::allocate(hmid.size() * 4u);
  if (!din || !dhmid || !dnext) return 1;

  // Every selected mixer projection receives the same BF16 input without a
  // per-consumer activation conversion. The output projection consumes a
  // separate K-width operand in TASK-024/025; this checks its full shape.
  auto const projection = [&](qw38::cuda::PrefillWeight const& weight,
                              qw38::cuda::PrefillEpilogue epi,
                              std::uint16_t const* x,
                              float const* residual = nullptr) {
    std::uint64_t const bytes = static_cast<std::uint64_t>(m) * weight.n *
                                (epi == qw38::cuda::PrefillEpilogue::StoreBf16 ? 2u : 4u);
    auto out = qw38::cuda::DeviceBuffer::allocate(bytes);
    if (!out) return false;
    auto result = engine->project(qw38::cuda::PrefillProjection{
        .weight = weight, .input = x, .output = out->data(),
        .residual = residual, .valid_tokens = m, .first_position = 512,
        .epilogue = epi});
    if (!result) { std::cerr << qw38::cuda::error_message(result.error()) << '\n'; return false; }
    if (epi == qw38::cuda::PrefillEpilogue::StoreBf16) {
      auto got = download<std::uint16_t>(out->data(), m * weight.n, stream);
      return got.size() == m * weight.n && finite_bf16(got);
    }
    auto got = download<float>(out->data(), m * weight.n, stream);
    return got.size() == m * weight.n &&
           std::ranges::all_of(got, [](float value) { return std::isfinite(value); });
  };
  auto const* x = static_cast<std::uint16_t const*>(din->data());
  float projection_max_abs = 0.0f;
  auto prefix_output = qw38::cuda::DeviceBuffer::allocate(kQgWidth * 2u);
  if (!prefix_output) return 1;
  for (auto const& plan : plans) {
    auto const& weight = plan.first;
    auto launched = engine->project(qw38::cuda::PrefillProjection{
        .weight = weight, .input = x, .output = prefix_output->data(),
        .valid_tokens = 1, .first_position = 512,
        .epilogue = qw38::cuda::PrefillEpilogue::StoreBf16});
    if (!launched) return 1;
    auto got = download<std::uint16_t>(prefix_output->data(), 8, stream);
    auto dense = reconstructed_prefix(weight, 8, stream);
    if (got.size() != 8 || dense.size() != 8u * weight.k) return 1;
    auto ref = qw38::compiler::reference_gemv_bf16(
        dense, std::span<std::uint16_t const>{input_bf16.data(), weight.k},
        8, weight.k);
    if (!ref) return 1;
    for (std::uint32_t j = 0; j < 8; ++j) {
      float const value = qw38::format::bf16_to_fp32(got[j]);
      if (!close_enough(value, (*ref)[j], projection_max_abs)) {
        std::cerr << "layer " << plan.layer << " projection prefix mismatch\n";
        return 1;
      }
    }
  }
  for (auto index : {0u, 3u}) {
    auto const& p = plans[index];
    if (!projection(p.first, qw38::cuda::PrefillEpilogue::StoreBf16, x) ||
        !projection(p.second, qw38::cuda::PrefillEpilogue::StoreBf16, x) ||
        !projection(p.third,
                    p.gdn ? qw38::cuda::PrefillEpilogue::StoreFp32
                          : qw38::cuda::PrefillEpilogue::StoreBf16, x) ||
        (p.gdn && !projection(p.fourth, qw38::cuda::PrefillEpilogue::StoreFp32, x)))
      return 1;
    std::vector<std::uint16_t> mixer_input(m * p.mixer_out.k);
    for (std::size_t i = 0; i < mixer_input.size(); ++i)
      mixer_input[i] = input_bf16[i % input_bf16.size()];
    auto dmixer = qw38::cuda::upload(bytes(mixer_input), stream);
    auto dresidual = qw38::cuda::upload(bytes(hmid), stream);
    if (!dmixer || !dresidual ||
        !projection(p.mixer_out, qw38::cuda::PrefillEpilogue::ResidualAddFp32,
                    static_cast<std::uint16_t const*>(dmixer->data()),
                    static_cast<float const*>(dresidual->data()))) return 1;
  }

  auto const alloc_before = qw38::cuda::malloc_count();
  auto mlp = execute_prefill_mlp(plans[0], *engine,
                                static_cast<float const*>(dhmid->data()),
                                static_cast<float*>(dnext->data()), m, 512);
  if (!mlp) { std::cerr << error_message(mlp.error()) << '\n'; return 1; }
  if (qw38::cuda::malloc_count() != alloc_before) return 1;
  auto got_mlp = download<float>(dnext->data(), hmid.size(), stream);
  if (got_mlp.size() != hmid.size()) return 1;
  if (!std::ranges::all_of(got_mlp, [](float value) { return std::isfinite(value); }))
    return 1;
  auto gate_before = download<std::uint8_t>(plans[0].mlp_gate.codes, 64, stream);
  if (gate_before.size() != 64) return 1;
  auto reject_overlap = execute_prefill_mlp(plans[0], *engine,
      static_cast<float const*>(dhmid->data()),
      static_cast<float*>(const_cast<void*>(plans[0].mlp_gate.codes)), m, 512);
  auto gate_after = download<std::uint8_t>(plans[0].mlp_gate.codes, 64, stream);
  if (reject_overlap || gate_after != gate_before) {
    std::cerr << "MLP output/weight overlap was not rejected before work\n";
    return 1;
  }
  auto session = runtime->create_session(*model, 1);
  if (!session) return 1;
  auto decode = bind_mlp_plan(*model, *session, 0, stream);
  if (!decode) { std::cerr << error_message(decode.error()) << '\n'; return 1; }
  float max_diff = 0.0f;
  for (std::uint32_t i = 0; i < m; ++i) {
    auto dst = session->residual_h_mid();
    if (!qw38::cuda::copy_h2d(dst.pointer, hmid.data() + i * kHidden,
                              kHidden * 4u, stream)) return 1;
    auto st = execute_decode_mlp(*decode);
    if (!st) { std::cerr << error_message(st.error()) << '\n'; return 1; }
    auto row = download<float>(session->residual_h().pointer, kHidden, stream);
    if (row.size() != kHidden) return 1;
    for (std::uint32_t j = 0; j < kHidden; ++j)
      if (!std::isfinite(row[j])) return 1;
      else max_diff = std::max(max_diff,
          std::fabs(row[j] - got_mlp[i * kHidden + j]));
  }
  if (max_diff > 0.15f) {
    std::cerr << "MLP prefill/decode max difference " << max_diff << '\n';
    return 1;
  }
  auto host_gate = reconstructed_prefix(plans[0].mlp_gate, kFfnWidth, stream);
  auto host_up = reconstructed_prefix(plans[0].mlp_up, kFfnWidth, stream);
  auto host_down = reconstructed_prefix(plans[0].mlp_down, kHidden, stream);
  auto host_gamma = download<std::uint16_t>(plans[0].mlp_gamma, kHidden, stream);
  if (host_gate.size() != static_cast<std::size_t>(kFfnWidth) * kHidden ||
      host_up.size() != static_cast<std::size_t>(kFfnWidth) * kHidden ||
      host_down.size() != static_cast<std::size_t>(kHidden) * kFfnWidth ||
      host_gamma.size() != kHidden) return 1;
  auto mlp_ref = qw38::reference::decode_mlp_reference(
      std::span<float const>{hmid.data(), kHidden}, host_gamma, 1.0e-6f,
      host_gate, host_up, host_down);
  if (!mlp_ref) return 1;
  float mlp_ref_max_abs = 0.0f;
  for (std::uint32_t j = 0; j < kHidden; ++j) {
    if (!close_enough(got_mlp[j], mlp_ref->residual[j], mlp_ref_max_abs,
                      qw38::reference::tol::kMlpResidualAbs,
                      qw38::reference::tol::kMlpResidualRel)) {
      std::cerr << "independent MLP reference mismatch at " << j << '\n';
      return 1;
    }
  }

  auto dgen = qw38::cuda::DeviceBuffer::allocate(kVocab * 4u);
  auto deval = qw38::cuda::DeviceBuffer::allocate(2u * kVocab * 4u);
  if (!dgen || !deval) return 1;
  auto gen = engine->head_generation(*head, x, m, 512,
                                     static_cast<float*>(dgen->data()));
  std::array<std::uint32_t, 2> rows{2, 0};
  auto eval = engine->head_evaluation(*head, x, m, 512, rows,
                                      static_cast<float*>(deval->data()));
  if (!gen || !eval) return 1;
  auto g = download<float>(dgen->data(), kVocab, stream);
  auto e = download<float>(deval->data(), 2u * kVocab, stream);
  if (g.size() != kVocab || e.size() != 2u * kVocab) return 1;
  for (std::uint32_t j = 0; j < kVocab; ++j) {
    if (!std::isfinite(g[j]) || g[j] != e[j] ||
        !std::isfinite(e[kVocab + j])) {
      std::cerr << "head mode mismatch at vocabulary row " << j << '\n';
      return 1;
    }
  }
  auto head_prefix = reconstructed_prefix(*head, 8, stream);
  if (head_prefix.size() != 8u * kHidden) return 1;
  float head_max_abs = 0.0f;
  for (std::uint32_t row : {2u, 0u}) {
    auto ref = qw38::compiler::reference_gemv_bf16(
        head_prefix,
        std::span<std::uint16_t const>{input_bf16.data() + row * kHidden,
                                       kHidden}, 8, kHidden);
    if (!ref) return 1;
    auto const* actual = row == 2 ? g.data() : e.data() + kVocab;
    for (std::uint32_t j = 0; j < 8; ++j)
      if (!close_enough(actual[j], (*ref)[j], head_max_abs)) return 1;
  }
  std::array<std::uint32_t, 2> duplicate{0, 0};
  if (engine->head_evaluation(*head, x, m, 512, duplicate,
                              static_cast<float*>(deval->data()))) return 1;
  std::array<std::uint32_t, 9> excessive{};
  if (engine->head_evaluation(*head, x, m, 512, excessive,
                              static_cast<float*>(deval->data()))) return 1;
  std::array<std::uint32_t, 8> order{7, 0, 2, 5, 1, 6, 3, 4};
  std::vector<std::uint16_t> eight_inputs(8u * kHidden);
  for (std::uint32_t i = 0; i < 8; ++i)
    std::copy_n(input_bf16.data() + (i % m) * kHidden, kHidden,
                eight_inputs.data() + i * kHidden);
  auto deight_input = qw38::cuda::upload(bytes(eight_inputs), stream);
  auto deight_logits = qw38::cuda::DeviceBuffer::allocate(
      8u * static_cast<std::uint64_t>(kVocab) * 4u);
  if (!deight_input || !deight_logits) return 1;
  if (!engine->head_evaluation(*head,
      static_cast<std::uint16_t const*>(deight_input->data()), 8, 512,
      order, static_cast<float*>(deight_logits->data()))) return 1;
  for (std::uint32_t i = 0; i < 8; ++i) {
    auto actual = download<float>(
        static_cast<float const*>(deight_logits->data()) +
            static_cast<std::uint64_t>(i) * kVocab, 8, stream);
    auto ref = qw38::compiler::reference_gemv_bf16(
        head_prefix,
        std::span<std::uint16_t const>{
            eight_inputs.data() + static_cast<std::uint64_t>(order[i]) * kHidden,
            kHidden}, 8, kHidden);
    if (actual.size() != 8 || !ref) return 1;
    for (std::uint32_t j = 0; j < 8; ++j)
      if (!close_enough(actual[j], (*ref)[j], head_max_abs)) return 1;
  }
  std::array<std::uint32_t, 1> out_of_range{m};
  if (engine->head_evaluation(*head, x, m, 512, out_of_range,
                              static_cast<float*>(deval->data()))) return 1;
  auto shared = qw38::cuda::DeviceBuffer::allocate(
      2u * kVocab * 4u + m * kHidden * 2u);
  if (!shared || !qw38::cuda::copy_h2d(shared->data(), input_bf16.data(),
                                       input_bf16.size() * 2u, stream)) return 1;
  auto const* shared_rows = static_cast<std::uint16_t const*>(shared->data());
  auto* overlap_logits = reinterpret_cast<float*>(
      static_cast<std::uint16_t*>(shared->data()) + kHidden);
  auto before_overlap = download<std::uint16_t>(shared->data(),
                                                input_bf16.size(), stream);
  auto reject_head = engine->head_evaluation(*head, shared_rows, m, 512,
                                             rows, overlap_logits);
  auto after_overlap = download<std::uint16_t>(shared->data(),
                                               input_bf16.size(), stream);
  if (reject_head || before_overlap != after_overlap) {
    std::cerr << "head output/input overlap was not rejected before work\n";
    return 1;
  }
  std::cout << "bound_layers=64 mixer_sets=2 mlp_rows=3 head_rows=2"
            << " mlp_decode_max_abs=" << max_diff
            << " mlp_reference_max_abs=" << mlp_ref_max_abs
            << " projection_prefix_max_abs=" << projection_max_abs
            << " head_prefix_max_abs=" << head_max_abs
            << " workspace_bytes=" << engine->workspace_bytes() << '\n';
  return 0;
}

}  // namespace

int main() {
  auto const* path = std::getenv("QW38_AUTHORITY_ARTIFACT");
  if (!path) { std::cout << "prefill artifact integration skipped\n"; return 0; }
  return run(path);
}
