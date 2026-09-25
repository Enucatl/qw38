#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"
#include "runtime/mlp.hpp"

#include "cuda/copy.hpp"
#include "cuda/alloc.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {
// BF16 projection stores can differ between GEMM and GEMV, while the
// residual remains FP32. These bounds exceed the observed 0.0024 maximum
// output difference yet reject a 0.25 output perturbation below.
constexpr float kLayerAbs = 0.02f, kLayerRel = 0.002f;
constexpr float kPartitionAbs = 0.005f;
template <typename T>
std::span<std::byte const> bytes(std::vector<T> const& v) {
  return {reinterpret_cast<std::byte const*>(v.data()), v.size() * sizeof(T)};
}
template <typename T>
std::vector<T> download(void const* ptr, std::size_t count,
                        qw38::cuda::Stream const& stream) {
  std::vector<T> result(count);
  if (!qw38::cuda::copy_d2h(result.data(), ptr, count * sizeof(T), stream) ||
      !stream.sync()) return {};
  return result;
}
bool close(std::vector<float> const& a, std::vector<float> const& b,
           float abs, float rel, float& max_diff) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    float const diff = std::fabs(a[i] - b[i]);
    max_diff = std::max(max_diff, diff);
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
        diff > abs + rel * std::fabs(b[i])) return false;
  }
  return true;
}
bool close_bf16(std::vector<std::uint16_t> const& a,
                std::vector<std::uint16_t> const& b,
                float abs, float rel, float& max_diff) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    float const x = qw38::format::bf16_to_fp32(a[i]);
    float const y = qw38::format::bf16_to_fp32(b[i]);
    float const diff = std::fabs(x - y);
    max_diff = std::max(max_diff, diff);
    if (!std::isfinite(x) || !std::isfinite(y) ||
        diff > abs + rel * std::fabs(y)) return false;
  }
  return true;
}

int run(char const* artifact) {
  using namespace qw38::runtime;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto model = runtime->load(artifact);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  auto session = runtime->create_session(*model, 9);
  if (!session) { std::cerr << error_message(session.error()) << '\n'; return 1; }
  auto const& stream = runtime->stream();
  auto plan = bind_prefill_attention_layer(*model, *session, 3, stream);
  auto decode = bind_attention_mixer_plan(*model, *session, 3, stream);
  auto mlp = bind_mlp_plan(*model, *session, 3, stream);
  auto engine = qw38::cuda::PrefillEngine::create(stream, 8);
  auto workspace = PrefillAttentionWorkspace::create(8, stream.device());
  if (!plan || !decode || !mlp || !engine || !workspace) {
    std::cerr << "attention prefill bind failed\n"; return 1;
  }
  auto embedding = model->payload("model.language_model.embed_tokens.weight");
  if (!embedding) return 1;
  constexpr std::uint32_t count = 5;
  std::vector<std::uint16_t> input_bf16((count + 1u) * kHidden);
  auto const* table = static_cast<std::uint16_t const*>(embedding->pointer);
  for (std::uint32_t t = 0; t <= count; ++t)
    if (!qw38::cuda::copy_d2h(input_bf16.data() + t * kHidden,
        table + static_cast<std::uint64_t>(2000u + t * 29u) * kHidden,
        kHidden * 2u, stream)) return 1;
  if (!stream.sync()) return 1;
  std::vector<float> input(input_bf16.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = qw38::format::bf16_to_fp32(input_bf16[i]);
  auto din = qw38::cuda::upload(bytes(input), stream);
  std::vector<float> sentinel(input.size(), 77.0f);
  auto dmid = qw38::cuda::upload(bytes(sentinel), stream);
  auto dout = qw38::cuda::upload(bytes(sentinel), stream);
  if (!din || !dmid || !dout) return 1;
  auto const* in = static_cast<float const*>(din->data());
  auto* mid = static_cast<float*>(dmid->data());
  auto* out = static_cast<float*>(dout->data());
  auto execute = [&](std::uint32_t start, std::uint32_t n) {
    auto st = execute_prefill_attention_layer(*plan, *workspace, *engine,
        in + static_cast<std::size_t>(start) * kHidden,
        mid + static_cast<std::size_t>(start) * kHidden,
        out + static_cast<std::size_t>(start) * kHidden, n, start);
    if (!st) std::cerr << error_message(st.error()) << '\n';
    return bool(st);
  };
  auto const allocations = qw38::cuda::malloc_count();
  if (!execute(0, count) || qw38::cuda::malloc_count() != allocations) return 1;
  auto full_mid = download<float>(mid, count * kHidden, stream);
  auto full_out = download<float>(out, (count + 1u) * kHidden, stream);
  if (full_out.size() != input.size()) return 1;
  auto perturbed = std::vector<float>(full_out.begin(),
                                      full_out.begin() + count * kHidden);
  perturbed[0] += 0.25f;
  float negative_diff = 0.0f;
  if (close(perturbed,
            {full_out.begin(), full_out.begin() + count * kHidden},
            kLayerAbs, kLayerRel, negative_diff)) {
    std::cerr << "numerical negative control failed\n"; return 1;
  }
  auto full_kv = download<std::uint16_t>(session->kv().pointer,
                                         2u * kKvHeads * 9u * kHeadDim, stream);
  auto full_meta = session->kv_populated(0);
  if (full_kv.size() != 2u * kKvHeads * 9u * kHeadDim) return 1;
  bool tail_untouched = true;
  for (std::uint32_t component = 0; component < 2u; ++component)
    for (std::uint32_t head = 0; head < kKvHeads; ++head)
      for (std::uint32_t token = count; token < 9u; ++token)
        for (std::uint32_t d = 0; d < kHeadDim; ++d)
          tail_untouched &= full_kv[
              ((component * kKvHeads + head) * 9u + token) * kHeadDim + d] == 0u;
  if (full_mid.size() != count * kHidden || full_out.size() != input.size() ||
      full_kv.empty() || !full_meta || *full_meta != count || !tail_untouched ||
      !std::all_of(full_out.begin() + count * kHidden, full_out.end(),
                   [](float v) { return v == 77.0f; })) return 1;
  if (!session->reset() || PrefillAttentionWorkspace::create(0, stream.device()) ||
      PrefillAttentionWorkspace::create(qw38::cuda::kPrefillMaxTokens + 1u,
                                         stream.device()) ||
      execute_prefill_attention_layer(*plan, *workspace, *engine,
                                      in, mid, out, 1, 1) ||
      execute_prefill_attention_layer(*plan, *workspace, *engine,
                                      in, mid, out, 9, 0) ||
      execute_prefill_attention_layer(*plan, *workspace, *engine,
                                      in, mid, out, 1, 9) ||
      execute_prefill_attention_layer(*plan, *workspace, *engine,
                                      in, mid, out, 2, 8) ||
      execute_prefill_attention_layer(*plan, *workspace, *engine,
                                      in, const_cast<float*>(in), out, 1, 0)) {
    std::cerr << "prefill boundary check failed\n"; return 1;
  }
  if (!execute(0, 2) || !execute(2, 3)) return 1;
  auto split_mid = download<float>(mid, count * kHidden, stream);
  auto split_out = download<float>(out, count * kHidden, stream);
  auto split_kv = download<std::uint16_t>(session->kv().pointer, full_kv.size(), stream);
  float mid_diff = 0.0f, out_diff = 0.0f;
  if (!close(split_mid, full_mid, kPartitionAbs, 0.0f, mid_diff) ||
      !close(split_out, {full_out.begin(), full_out.begin() + count * kHidden},
             kPartitionAbs, 0.0f, out_diff) || split_kv != full_kv ||
      session->kv_populated(0) != count) {
    std::cerr << "chunk partition mismatch " << mid_diff << ',' << out_diff << '\n';
    return 1;
  }
  if (!session->reset() || !execute(0, 2)) return 1;
  auto snapshot = session->save();
  if (!snapshot || !execute(2, 3)) return 1;
  auto suffix = download<float>(out + 2u * kHidden, 3u * kHidden, stream);
  auto suffix_kv = download<std::uint16_t>(session->kv().pointer,
                                           full_kv.size(), stream);
  if (!session->restore(*snapshot) || !execute(2, 3) ||
      suffix != download<float>(out + 2u * kHidden, 3u * kHidden, stream) ||
      suffix_kv != download<std::uint16_t>(session->kv().pointer,
                                           full_kv.size(), stream) ||
      session->kv_populated(0) != count) {
    std::cerr << "snapshot replay mismatch\n"; return 1;
  }
  if (!session->reset() || !execute(0, count) ||
      !qw38::cuda::copy_h2d(session->residual_h().pointer,
                             input.data() + count * kHidden,
                             kHidden * 4u, stream) ||
      !execute_decode_attention(*decode, count) || !execute_decode_mlp(*mlp))
    return 1;
  auto continuation = download<float>(session->residual_h().pointer, kHidden, stream);
  auto continuation_mid = download<float>(session->residual_h_mid().pointer,
                                          kHidden, stream);
  auto continuation_kv = download<std::uint16_t>(session->kv().pointer,
                                                 full_kv.size(), stream);
  if (!session->reset()) return 1;
  std::vector<float> decode_out(count * kHidden), decode_mid(count * kHidden);
  for (std::uint32_t t = 0; t <= count; ++t) {
    if (!qw38::cuda::copy_h2d(session->residual_h().pointer,
                              input.data() + t * kHidden,
                              kHidden * 4u, stream) ||
        !execute_decode_attention(*decode, t) || !execute_decode_mlp(*mlp))
      return 1;
    auto row = download<float>(session->residual_h().pointer, kHidden, stream);
    auto mixer_row = download<float>(session->residual_h_mid().pointer,
                                     kHidden, stream);
    if (row.size() != kHidden || mixer_row.size() != kHidden) return 1;
    if (t < count) {
      std::copy(row.begin(), row.end(), decode_out.begin() + t * kHidden);
      std::copy(mixer_row.begin(), mixer_row.end(),
                decode_mid.begin() + t * kHidden);
    } else {
      float diff = 0.0f, mid_diff = 0.0f;
      if (!close(row, continuation, kLayerAbs, kLayerRel, diff) ||
          !close(mixer_row, continuation_mid,
                 kLayerAbs, kLayerRel, mid_diff)) {
        std::cerr << "prefill-to-decode continuation mismatch "
                  << diff << ',' << mid_diff << '\n';
        return 1;
      }
    }
  }
  auto decode_kv = download<std::uint16_t>(session->kv().pointer,
                                           full_kv.size(), stream);
  float decode_diff = 0.0f, decode_mid_diff = 0.0f, kv_diff = 0.0f;
  bool const output_ok = close(decode_out,
             {full_out.begin(), full_out.begin() + count * kHidden},
             kLayerAbs, kLayerRel, decode_diff);
  bool const mixer_ok = close(decode_mid, full_mid,
                              kLayerAbs, kLayerRel, decode_mid_diff);
  bool const kv_ok = close_bf16(decode_kv, continuation_kv,
                                0.005f, 0.002f, kv_diff);
  if (!output_ok || !mixer_ok || !kv_ok ||
      session->kv_populated(0) != count + 1u) {
    std::cerr << "decode comparison mismatch output=" << output_ok
              << " mixer=" << mixer_ok << " kv=" << kv_ok
              << " max=" << decode_diff << ',' << decode_mid_diff << ','
              << kv_diff
              << '\n'; return 1;
  }
  if (!session->reset()) return 1;
  qw38::cuda::testing::fail_next_stream_sync();
  if (execute(0, 1) || session->kv_populated(0) != 0u ||
      session->save() || execute(0, 1)) {
    std::cerr << "failed attention prefill did not poison and recover\n";
    return 1;
  }
  auto failed_kv = download<std::uint16_t>(session->kv().pointer,
                                           full_kv.size(), stream);
  if (failed_kv.size() != full_kv.size() ||
      std::all_of(failed_kv.begin(), failed_kv.end(),
                  [](std::uint16_t x) { return x == 0u; }) ||
      !session->reset()) {
    std::cerr << "failed attention prefill device mutation or reset mismatch\n";
    return 1;
  }
  std::cout << "rows=5 partition_mid_max=" << mid_diff
            << " partition_out_max=" << out_diff
            << " decode_out_max=" << decode_diff
            << " decode_mid_max=" << decode_mid_diff
            << " decode_kv_max=" << kv_diff
            << " workspace_bytes=" << workspace->bytes() + engine->workspace_bytes()
            << '\n';
  return 0;
}
}  // namespace

int main() {
  auto const* artifact = std::getenv("QW38_AUTHORITY_ARTIFACT");
  if (!artifact) { std::cerr << "QW38_AUTHORITY_ARTIFACT is required\n"; return 1; }
  return run(artifact);
}
