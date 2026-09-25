#include "runtime/mlp.hpp"
#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"

#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "cuda/stream.hpp"
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

template <class T>
std::span<std::byte const> bytes(std::vector<T> const& v) {
  return {reinterpret_cast<std::byte const*>(v.data()), v.size() * sizeof(T)};
}

template <class T>
std::vector<T> download(void const* ptr, std::size_t n,
                        qw38::cuda::Stream const& stream) {
  std::vector<T> result(n);
  if (!qw38::cuda::copy_d2h(result.data(), ptr, n * sizeof(T), stream) ||
      !stream.sync()) { std::cerr << "artifact download failed\n"; return {}; }
  return result;
}

bool close(std::vector<float> const& x, std::vector<float> const& y,
           float abs, float rel, float& max_diff) {
  if (x.size() != y.size()) return false;
  for (std::size_t i = 0; i < x.size(); ++i) {
    float const diff = std::fabs(x[i] - y[i]);
    max_diff = std::max(max_diff, diff);
    if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
        diff > abs + rel * std::fabs(y[i])) return false;
  }
  return true;
}

bool close_bf16(std::vector<std::uint16_t> const& x,
                std::vector<std::uint16_t> const& y,
                float abs, float rel, float& max_diff) {
  if (x.size() != y.size()) return false;
  for (std::size_t i = 0; i < x.size(); ++i) {
    float const a = qw38::format::bf16_to_fp32(x[i]);
    float const b = qw38::format::bf16_to_fp32(y[i]);
    float const diff = std::fabs(a - b);
    max_diff = std::max(max_diff, diff);
    if (!std::isfinite(a) || !std::isfinite(b) ||
        diff > abs + rel * std::fabs(b)) return false;
  }
  return true;
}

int run(char const* artifact) {
  using namespace qw38::runtime;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto model = runtime->load(artifact);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  auto session = runtime->create_session(*model, 8);
  if (!session) { std::cerr << error_message(session.error()) << '\n'; return 1; }
  auto const& stream = runtime->stream();
  constexpr std::uint32_t layer = 4, gdn_index = 3;
  auto plan = bind_prefill_gdn_layer(*model, *session, layer, stream);
  auto decode = bind_gdn_plan(*model, *session, layer, stream);
  auto mlp = bind_mlp_plan(*model, *session, layer, stream);
  auto engine = qw38::cuda::PrefillEngine::create(stream, 8);
  auto workspace = PrefillGdnWorkspace::create(8, stream.device());
  if (!plan || !decode || !mlp || !engine || !workspace) {
    std::cerr << "GDN prefill/decode bind or workspace failed\n"; return 1;
  }
  constexpr std::uint32_t count = 4;
  auto embedding = model->payload("model.language_model.embed_tokens.weight");
  if (!embedding) return 1;
  std::vector<std::uint16_t> input_bf16((count + 1u) * kHidden);
  auto const* table = static_cast<std::uint16_t const*>(embedding->pointer);
  for (std::uint32_t t = 0; t < count + 1u; ++t)
    if (!qw38::cuda::copy_d2h(input_bf16.data() + t * kHidden,
        table + static_cast<std::uint64_t>(1000u + t * 17u) * kHidden,
        kHidden * 2u, stream)) return 1;
  if (!stream.sync()) return 1;
  std::vector<float> input(input_bf16.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = qw38::format::bf16_to_fp32(input_bf16[i]);
  auto din = qw38::cuda::upload(bytes(input), stream);
  std::vector<float> sentinel((count + 1u) * kHidden, 77.0f);
  auto dmid = qw38::cuda::upload(bytes(sentinel), stream);
  auto dout = qw38::cuda::upload(bytes(sentinel), stream);
  if (!din || !dmid || !dout) return 1;
  auto* mid = static_cast<float*>(dmid->data());
  auto* out = static_cast<float*>(dout->data());
  auto const* in = static_cast<float const*>(din->data());
  auto execute = [&](std::uint32_t start, std::uint32_t n,
                     std::uint32_t interval) {
    auto st = execute_prefill_gdn_layer(*plan, *workspace, *engine,
        in + static_cast<std::size_t>(start) * kHidden,
        mid + static_cast<std::size_t>(start) * kHidden,
        out + static_cast<std::size_t>(start) * kHidden,
        n, start, interval);
    if (!st) std::cerr << error_message(st.error()) << '\n';
    return bool(st);
  };
  auto const allocations = qw38::cuda::malloc_count();
  if (!execute(0, count, 3) || qw38::cuda::malloc_count() != allocations) return 1;
  auto full_mid = download<float>(mid, count * kHidden, stream);
  auto full_mid_guard = download<float>(mid + count * kHidden, kHidden, stream);
  auto full_out = download<float>(out, (count + 1u) * kHidden, stream);
  auto full_state = download<float>(static_cast<float const*>(session->gdn_s().pointer) +
                                   gdn_index * kGdnSElemsPerLayer, kGdnSElemsPerLayer, stream);
  auto full_history = download<std::uint16_t>(static_cast<std::uint16_t const*>(session->conv_history().pointer) +
                                              gdn_index * kConvTaps * kConvChannels,
                                              kConvTaps * kConvChannels, stream);
  if (full_mid.size() != count * kHidden ||
      full_mid_guard.size() != kHidden ||
      !std::all_of(full_mid_guard.begin(), full_mid_guard.end(),
                   [](float x) { return x == 77.0f; }) ||
      full_out.size() != (count + 1u) * kHidden ||
      full_state.size() != kGdnSElemsPerLayer ||
      full_history.size() != kConvTaps * kConvChannels ||
      !std::all_of(full_out.begin() + count * kHidden, full_out.end(),
                   [](float x) { return x == 77.0f; }) ||
      session->conv_cursor()[gdn_index] != 1u) return 1;
  if (!session->reset()) return 1;
  if (PrefillGdnWorkspace::create(0, stream.device()) ||
      PrefillGdnWorkspace::create(qw38::cuda::kPrefillMaxTokens + 1u,
                                  stream.device())) return 1;
  if (execute_prefill_gdn_layer(*plan, *workspace, *engine,
        in, mid, out, 9, 0, 1) || session->conv_cursor()[gdn_index] != 0u) {
    std::cerr << "workspace capacity was not enforced before mutation\n"; return 1;
  }
  if (execute_prefill_gdn_layer(*plan, *workspace, *engine,
        static_cast<float const*>(session->gdn_s().pointer), mid, out,
        1, 0, 1) || session->conv_cursor()[gdn_index] != 0u) {
    std::cerr << "state/input overlap was not rejected before mutation\n"; return 1;
  }
  if (execute_prefill_gdn_layer(*plan, *workspace, *engine,
       in, const_cast<float*>(in), out, 1, 0, 1)) {
    std::cerr << "overlapping output was not rejected\n"; return 1;
  }
  if (!execute_prefill_gdn_layer(*plan, *workspace, *engine,
                                 in, mid, out, 1, 0) ||
      session->conv_cursor()[gdn_index] != 1u || !session->reset()) {
    std::cerr << "default recurrence interval rejected short tail\n"; return 1;
  }
  if (!execute(0, 1, 1) || !execute(1, 3, 2)) return 1;
  auto split_mid = download<float>(mid, count * kHidden, stream);
  auto split_out = download<float>(out, count * kHidden, stream);
  auto split_state = download<float>(static_cast<float const*>(session->gdn_s().pointer) +
                                    gdn_index * kGdnSElemsPerLayer, kGdnSElemsPerLayer, stream);
  auto split_history = download<std::uint16_t>(static_cast<std::uint16_t const*>(session->conv_history().pointer) +
                                               gdn_index * kConvTaps * kConvChannels,
                                               kConvTaps * kConvChannels, stream);
  float max_mid = 0.0f, max_out = 0.0f, max_state = 0.0f;
  if (!close(split_mid, full_mid, 0.15f, 0.004f, max_mid) ||
      !close(split_out, {full_out.begin(), full_out.begin() + count * kHidden},
             0.3f, 0.004f, max_out) ||
      !close(split_state, full_state, 7e-4f, 2e-4f, max_state) ||
      split_history != full_history || session->conv_cursor()[gdn_index] != 1u) {
    std::cerr << "chunk partition mismatch\n"; return 1;
  }
  // A same-schedule suffix must replay exactly after snapshot restore.
  if (!session->reset() || !execute(0, 2, 2)) return 1;
  auto snapshot = session->save();
  if (!snapshot || !execute(2, 2, 2)) return 1;
  auto replay_mid = download<float>(mid + 2u * kHidden, 2u * kHidden, stream);
  auto replay_out = download<float>(out + 2u * kHidden, 2u * kHidden, stream);
  auto replay_state = download<float>(static_cast<float const*>(session->gdn_s().pointer) +
                                     gdn_index * kGdnSElemsPerLayer, kGdnSElemsPerLayer, stream);
  auto replay_history = download<std::uint16_t>(static_cast<std::uint16_t const*>(session->conv_history().pointer) +
                                                gdn_index * kConvTaps * kConvChannels,
                                                 kConvTaps * kConvChannels, stream);
  if (!session->restore(*snapshot) || !execute(2, 2, 2)) return 1;
  if (replay_mid.size() != 2u * kHidden || replay_out.size() != 2u * kHidden ||
      replay_state.size() != kGdnSElemsPerLayer ||
      replay_history.size() != kConvTaps * kConvChannels ||
      replay_mid != download<float>(mid + 2u * kHidden, 2u * kHidden, stream) ||
      replay_out != download<float>(out + 2u * kHidden, 2u * kHidden, stream) ||
      replay_state != download<float>(static_cast<float const*>(session->gdn_s().pointer) +
                                      gdn_index * kGdnSElemsPerLayer,
                                      kGdnSElemsPerLayer, stream) ||
      replay_history != download<std::uint16_t>(static_cast<std::uint16_t const*>(session->conv_history().pointer) +
                                                gdn_index * kConvTaps * kConvChannels,
                                                 kConvTaps * kConvChannels, stream)) {
    std::cerr << "snapshot suffix replay mismatch\n"; return 1;
  }
  // The sync fault fires after GPU work completes. Host metadata stays at the
  // old position and the session rejects reuse until reset or restore.
  if (!session->reset()) return 1;
  auto position_slot = detail::SessionPlanAccess::gdn_position(*session, gdn_index);
  if (!position_slot) return 1;
  qw38::cuda::testing::fail_next_stream_sync();
  if (execute(0, 1, 1) || session->conv_cursor()[gdn_index] != 0u ||
      position_slot->value() != 0u ||
      session->save() || execute(0, 1, 1)) {
    std::cerr << "failed prefill did not poison or preserve host metadata\n"; return 1;
  }
  auto failed_history = download<std::uint16_t>(
      static_cast<std::uint16_t const*>(session->conv_history().pointer) +
          gdn_index * kConvTaps * kConvChannels,
      kConvTaps * kConvChannels, stream);
  if (failed_history.size() != kConvTaps * kConvChannels ||
      std::all_of(failed_history.begin(), failed_history.end(),
                  [](std::uint16_t x) { return x == 0; }) ||
      !session->reset()) {
    std::cerr << "failed prefill mutation or recovery mismatch\n"; return 1;
  }
  // Compare a four-token prefill followed by one decode with five decode
  // steps on the same nonzero GDN layer, including persistent state.
  if (!execute(0, count, 3)) return 1;
  if (!qw38::cuda::copy_h2d(session->residual_h().pointer,
                            input.data() + count * kHidden, kHidden * 4u, stream)) return 1;
  auto continuation = execute_decode_gdn(*decode, count);
  if (!continuation || !execute_decode_mlp(*mlp)) return 1;
  auto prefill_decode_mid = download<float>(session->residual_h_mid().pointer, kHidden, stream);
  auto prefill_decode_out = download<float>(session->residual_h().pointer, kHidden, stream);
  auto prefill_decode_state = download<float>(
      static_cast<float const*>(session->gdn_s().pointer) + gdn_index * kGdnSElemsPerLayer,
      kGdnSElemsPerLayer, stream);
  auto prefill_decode_history = download<std::uint16_t>(
      static_cast<std::uint16_t const*>(session->conv_history().pointer) +
          gdn_index * kConvTaps * kConvChannels,
      kConvTaps * kConvChannels, stream);
  auto prefill_decode_snapshot = session->save();
  if (prefill_decode_mid.size() != kHidden || prefill_decode_out.size() != kHidden ||
      prefill_decode_state.size() != kGdnSElemsPerLayer ||
      prefill_decode_history.size() != kConvTaps * kConvChannels ||
      !prefill_decode_snapshot ||
      prefill_decode_snapshot->conv_cursor[gdn_index] != 2u ||
      prefill_decode_snapshot->gdn_position[gdn_index] != 5u) return 1;
  if (!session->reset()) return 1;
  std::vector<float> decode_mid(count * kHidden), decode_out(count * kHidden);
  std::vector<float> decode_fifth_mid, decode_fifth_out;
  for (std::uint32_t t = 0; t < count + 1u; ++t) {
    if (!qw38::cuda::copy_h2d(session->residual_h().pointer,
                              input.data() + t * kHidden, kHidden * 4u, stream)) return 1;
    auto gdn = execute_decode_gdn(*decode, t);
    if (!gdn || !execute_decode_mlp(*mlp)) return 1;
    auto one_mid = download<float>(session->residual_h_mid().pointer, kHidden, stream);
    auto one_out = download<float>(session->residual_h().pointer, kHidden, stream);
    if (one_mid.size() != kHidden || one_out.size() != kHidden) return 1;
    if (t < count) {
      std::copy(one_mid.begin(), one_mid.end(), decode_mid.begin() + t * kHidden);
      std::copy(one_out.begin(), one_out.end(), decode_out.begin() + t * kHidden);
    } else {
      decode_fifth_mid = std::move(one_mid);
      decode_fifth_out = std::move(one_out);
    }
  }
  auto decode_state = download<float>(
      static_cast<float const*>(session->gdn_s().pointer) + gdn_index * kGdnSElemsPerLayer,
      kGdnSElemsPerLayer, stream);
  auto decode_history = download<std::uint16_t>(
      static_cast<std::uint16_t const*>(session->conv_history().pointer) +
          gdn_index * kConvTaps * kConvChannels,
      kConvTaps * kConvChannels, stream);
  auto decode_snapshot = session->save();
  float decode_mid_diff = 0.0f, decode_out_diff = 0.0f;
  float fifth_mid_diff = 0.0f, fifth_out_diff = 0.0f, continuation_state_diff = 0.0f;
  bool const prefix_mid_ok = close(full_mid, decode_mid, 0.2f, 0.005f, decode_mid_diff);
  bool const prefix_out_ok = close({full_out.begin(), full_out.begin() + count * kHidden},
                                   decode_out, 0.35f, 0.005f, decode_out_diff);
  bool const fifth_mid_ok = close(prefill_decode_mid, decode_fifth_mid,
                                   0.2f, 0.005f, fifth_mid_diff);
  bool const fifth_out_ok = close(prefill_decode_out, decode_fifth_out,
                                   0.35f, 0.005f, fifth_out_diff);
  bool const state_ok = close(prefill_decode_state, decode_state,
                              8e-4f, 2e-4f, continuation_state_diff);
  float history_diff = 0.0f;
  bool const history_ok = close_bf16(prefill_decode_history, decode_history,
                                      0.02f, 0.006f, history_diff);
  bool const metadata_ok = decode_snapshot &&
      decode_snapshot->conv_cursor == prefill_decode_snapshot->conv_cursor &&
      decode_snapshot->gdn_position == prefill_decode_snapshot->gdn_position;
  if (!prefix_mid_ok || !prefix_out_ok || !fifth_mid_ok || !fifth_out_ok ||
      !state_ok || !history_ok || !metadata_ok) {
    std::cerr << "prefill/decode layer mismatch: prefix_mid=" << prefix_mid_ok
              << " prefix_out=" << prefix_out_ok << " fifth_mid=" << fifth_mid_ok
              << " fifth_out=" << fifth_out_ok << " state=" << state_ok
              << " history=" << history_ok << " metadata=" << metadata_ok
              << " max=" << decode_mid_diff << ',' << decode_out_diff << ','
              << fifth_mid_diff << ',' << fifth_out_diff << ','
              << continuation_state_diff << ',' << history_diff << '\n'; return 1;
  }
  std::cout << "rows=4 partition_mid_max=" << max_mid
            << " partition_out_max=" << max_out
            << " partition_state_max=" << max_state
            << " decode_mid_max=" << decode_mid_diff
            << " decode_out_max=" << decode_out_diff
            << " continuation_mid_max=" << fifth_mid_diff
            << " continuation_out_max=" << fifth_out_diff
            << " continuation_state_max=" << continuation_state_diff
            << " continuation_history_max=" << history_diff
            << " workspace_bytes=" << workspace->bytes() << '\n';
  return 0;
}

}  // namespace

int main() {
  auto const* artifact = std::getenv("QW38_AUTHORITY_ARTIFACT");
  if (!artifact) { std::cout << "GDN prefill artifact integration skipped\n"; return 0; }
  return run(artifact);
}
