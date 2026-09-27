#include "runtime/prefill.hpp"
#include "runtime/attention.hpp"
#include "runtime/runtime.hpp"
#include "cuda/alloc.hpp"
#include "cuda/attention.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "format/floatcvt.hpp"

#include <cuda_runtime.h>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {
using namespace qw38;
template<class T, class E> T take(std::expected<T, E>&& value) {
  if (!value) { std::cerr << error_message(value.error()) << '\n'; std::exit(1); }
  return std::move(*value);
}
template<class E> void check(std::expected<void, E> value) {
  if (!value) { std::cerr << error_message(value.error()) << '\n'; std::exit(1); }
}
unsigned number(char const* arg) {
  std::string_view text(arg); unsigned n = 0;
  auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), n);
  if (ec != std::errc{} || end != text.end()) std::exit(2);
  return n;
}

// Reproduce: qw38_bench_attention_prefill --decode ARTIFACT 4096 output.f32
// Starts at projected q/g,k,v, matching the prefill diagnostic boundary.
// Includes production QK normalization/RoPE, cache append, scan/merge/gate,
// resident artifact output projection and residual epilogue; D2H is separate.
int decode_boundary(char const* path, unsigned visible, char const* saved_path) {
  if (visible == 0 || visible > 32768) return 2;
  auto const setup_start = std::chrono::steady_clock::now();
  auto rt = take(runtime::Runtime::create());
  auto artifact = take(format::Artifact::open(path));
  auto model = take(rt.upload_diagnostic(artifact, runtime::DiagnosticWeights::Layer, 3));
  auto session = take(rt.create_session(model, visible));
  auto const& stream = rt.stream();
  auto plan = take(runtime::bind_attention_mixer_plan(model, session, 3, stream));
  auto data = [](std::size_t count, unsigned seed) {
    std::vector<std::uint16_t> values(count);
    for (std::size_t i = 0; i < count; ++i)
      values[i] = format::fp32_to_bf16_rne(.01f * (int((i * seed + i / 256) % 101) - 50));
    return values;
  };
  auto upload = [&](void* target, std::size_t count, unsigned seed) {
    auto values = data(count, seed);
    check(cuda::copy_h2d(target, std::as_bytes(std::span(values)), stream));
    check(stream.sync());
  };
  upload(plan.prep.scratch.qg.pointer, 12288, 7);
  upload(plan.prep.scratch.k.pointer, 1024, 11);
  upload(plan.prep.scratch.v.pointer, 1024, 13);
  upload(plan.core.kv.pointer, 2ull * 1024 * visible, 17);
  std::vector<float> residual(5120), result(5120);
  for (unsigned i = 0; i < residual.size(); ++i)
    residual[i] = (int(i * 19 % 89) - 44) * .03125f;
  check(cuda::copy_h2d(plan.core.residual.pointer, std::as_bytes(std::span(residual)), stream));
  check(session.set_populated_length(0, visible - 1));
  auto begin = take(cuda::Event::create_timing()), end = take(cuda::Event::create_timing());
  check(stream.sync());
  double const setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_start).count();
  auto const allocations = cuda::malloc_count();
  auto const start = std::chrono::steady_clock::now();
  check(begin.record(stream));
  check(cuda::launch_attention_prepare_chunk(
      static_cast<std::uint16_t const*>(plan.prep.scratch.qg.pointer),
      static_cast<std::uint16_t const*>(plan.prep.scratch.k.pointer),
      static_cast<std::uint16_t const*>(plan.prep.scratch.v.pointer),
      static_cast<std::uint16_t const*>(plan.prep.gamma_q.pointer),
      static_cast<std::uint16_t const*>(plan.prep.gamma_k.pointer),
      static_cast<float const*>(plan.prep.inv_freq.pointer), plan.prep.eps,
      visible - 1, 1, static_cast<std::uint16_t*>(plan.core.q.pointer),
      static_cast<std::uint16_t*>(plan.core.g.pointer),
      static_cast<std::uint16_t*>(plan.core.kv.pointer), 0, visible, stream));
  check(plan.prep.populated.commit_append(visible - 1));
  auto output = take(runtime::execute_attention_core(plan.core));
  check(end.record(stream));
  check(end.sync());
  double const host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  auto const hot_allocations = cuda::malloc_count() - allocations;
  auto const readout_start = std::chrono::steady_clock::now();
  check(cuda::copy_d2h(result.data(), output.pointer, result.size() * 4, stream));
  check(stream.sync());
  double const readout_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - readout_start).count();
  for (float x : result) if (!std::isfinite(x)) return 1;
  std::ofstream saved(saved_path, std::ios::binary);
  saved.write(reinterpret_cast<char const*>(result.data()), result.size() * 4);
  if (!saved) return 1;
  std::cout << std::setprecision(12) << "phase=decode rows=1 visible=" << visible
      << " policy=" << unsigned(model.schema().precision.id)
      << " setup_ms=" << setup_ms << " host_ms=" << host_ms
      << " gpu_event_interval_ms=" << take(cuda::elapsed_ms(begin, end))
      << " diagnostic_d2h_host_ms=" << readout_ms
      << " diagnostic_d2h_bytes=" << result.size() * 4
      << " hot_allocations=" << hot_allocations
      << " timing_schema=2 boundary=prepare_append_scan_gate_output_projection_residual"
      << " first_use=true readout_included=false runs=1 warmups=0 finite=true\n";
  return hot_allocations ? 1 : 0;
}
}

// FAST-03: identical prepared inputs and initial KV for two M256 calls versus
// one M512 call. Includes prepare/append, scan/gate/direct FP8 packing, real
// output projection, residual add and completion, including first use.
int main(int argc, char** argv) {
  if (argc == 5 && std::string_view(argv[1]) == "--decode")
    return decode_boundary(argv[2], number(argv[3]), argv[4]);
  if (argc != 6 && argc != 7) {
    std::cerr << "usage: attention_prefill_bench ARTIFACT CHUNK PREFIX ROWS OUTPUT_F32 [small|large]\n"
              << "       attention_prefill_bench --decode ARTIFACT VISIBLE OUTPUT_F32\n";
    return 2;
  }
  unsigned const chunk = number(argv[2]), prefix = number(argv[3]), rows = number(argv[4]);
  if (argc == 7 && std::string_view(argv[6]) != "small" && std::string_view(argv[6]) != "large") return 2;
  auto const regime = argc == 7 && std::string_view(argv[6]) == "small"
      ? cuda::ContextRegime::Small : cuda::ContextRegime::Large;
  auto const schedule = cuda::prefill_attention_schedule(regime, chunk, prefix);
  auto const heads = schedule.heads;
  if ((chunk != 32 && chunk != 256 && chunk != 512) ||
      (rows != 32 && rows != 512) || chunk > rows || prefix > 32256) return 2;
  auto setup_start = std::chrono::steady_clock::now();
  auto rt = take(runtime::Runtime::create());
  auto artifact = take(format::Artifact::open(argv[1]));
  auto model = take(rt.upload_diagnostic(artifact, runtime::DiagnosticWeights::Layer, 3));
  auto session = take(rt.create_session(model, prefix + rows));
  auto const& stream = rt.stream();
  auto plan = take(runtime::bind_prefill_layer_projections(model, 3, stream, &session));
  auto engine = take(cuda::PrefillEngine::create(stream, chunk,
      cuda::kPrefillDefaultWeightRows, cuda::PrefillDispatch::BoundedUnpackBf16Cublas, true));
  auto workspace = take(runtime::PrefillAttentionWorkspace::create(chunk, stream.device()));
  auto s = workspace.slices();
  auto upload = [&](std::vector<std::uint16_t> const& host) {
    auto device = take(cuda::DeviceBuffer::allocate(host.size() * 2));
    check(cuda::copy_h2d(device.data(), std::as_bytes(std::span(host)), stream));
    check(stream.sync());
    return device;
  };
  auto data = [](std::size_t count, unsigned seed) {
    std::vector<std::uint16_t> values(count);
    for (std::size_t i = 0; i < count; ++i)
      values[i] = format::fp32_to_bf16_rne(.01f * (int((i * seed + i / 256) % 101) - 50));
    return values;
  };
  auto qg = upload(data(rows * 12288ull, 7));
  auto k = upload(data(rows * 1024ull, 11));
  auto v = upload(data(rows * 1024ull, 13));
  auto gamma = upload(std::vector<std::uint16_t>(512, format::fp32_to_bf16_rne(1.f)));
  auto frequencies = take(cuda::DeviceBuffer::allocate(32 * 4));
  check(cuda::zero(frequencies, stream));
  auto kv = static_cast<std::uint16_t*>(session.kv().pointer);
  auto initial_kv = data(2ull * 4 * (prefix + rows) * 256, 17);
  check(cuda::copy_h2d(kv, std::as_bytes(std::span(initial_kv)), stream));
  auto residual = take(cuda::DeviceBuffer::allocate(rows * 5120ull * 4));
  auto output = take(cuda::DeviceBuffer::allocate(rows * 5120ull * 4));
  check(cuda::zero(residual, stream));
  auto begin = take(cuda::Event::create_timing()), end = take(cuda::Event::create_timing());
  check(stream.sync());
  double const setup_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - setup_start).count();
  auto const allocations = cuda::malloc_count();
  check(begin.record(stream));
  auto start = std::chrono::steady_clock::now();
  for (unsigned offset = 0; offset < rows; offset += chunk) {
    check(cuda::launch_attention_prepare_chunk(
        static_cast<std::uint16_t const*>(qg.data()) + offset * 12288ull,
        static_cast<std::uint16_t const*>(k.data()) + offset * 1024ull,
        static_cast<std::uint16_t const*>(v.data()) + offset * 1024ull,
        static_cast<std::uint16_t const*>(gamma.data()) + 256,
        static_cast<std::uint16_t const*>(gamma.data()),
        static_cast<float const*>(frequencies.data()), 1e-6f, prefix + offset,
        chunk, s.q, s.g, kv, 0, prefix + rows, stream));
    auto packed = engine.fp8_operand(chunk, 6144);
    check(cuda::launch_attention_prefill_scan(s.q, s.g, kv, 0, prefix + rows,
        prefix + offset, chunk, s.y, stream, cuda::kAttnPrefillQueryTile,
        const_cast<std::uint8_t*>(packed.codes.data()), const_cast<float*>(packed.scales.data()), regime));
    check(engine.project({.weight = plan.mixer_out, .input = s.y,
        .output = static_cast<float*>(output.data()) + offset * 5120ull,
        .residual = static_cast<float const*>(residual.data()) + offset * 5120ull,
        .valid_tokens = chunk, .first_position = prefix + offset,
        .epilogue = cuda::PrefillEpilogue::ResidualAddFp32, .packed = packed}));
    check(stream.sync());
  }
  check(end.record(stream)); check(end.sync());
  double const host_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  auto const hot_allocations = cuda::malloc_count() - allocations;
  auto resources = take(cuda::attention_prefill_resources(cuda::kAttnPrefillQueryTile, chunk, regime, prefix));
  std::vector<float> result(rows * 5120ull);
  auto const readout_start = std::chrono::steady_clock::now();
  check(cuda::copy_d2h(result.data(), output.data(), result.size() * 4, stream));
  check(stream.sync());
  double const readout_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - readout_start).count();
  for (float x : result) if (!std::isfinite(x)) return 1;
  std::ofstream saved(argv[5], std::ios::binary);
  saved.write(reinterpret_cast<char const*>(result.data()), result.size() * 4);
  if (!saved) return 1;
  std::cout << std::setprecision(12) << "chunk=" << chunk << " prefix=" << prefix
      << " rows=" << rows << " setup_ms=" << setup_ms << " host_ms=" << host_ms
      << " gpu_event_interval_ms=" << take(cuda::elapsed_ms(begin, end))
      << " workspace_bytes=" << workspace.bytes() + engine.workspace_bytes()
      << " heads_per_cta=" << heads
      << " key_tile=" << schedule.key_tile
      << " blocks=" << (24 / heads) * ((chunk + 127) / 128 * 4)
      << " threads=" << heads * 128
      << " registers=" << resources.registers << " shared_bytes=" << resources.shared_bytes
      << " local_bytes=" << resources.local_bytes << " blocks_per_sm=" << resources.occupancy_blocks_per_sm
      << " hot_allocations=" << hot_allocations
      << " diagnostic_d2h_host_ms=" << readout_ms << " diagnostic_d2h_bytes=" << result.size() * 4
      << " timing_schema=2 boundary=prepare_append_scan_gate_pack_output_projection_residual"
      << " first_use=true readout_included=false per_chunk_sync=true runs=1 warmups=0 finite=true\n";
  return hot_allocations ? 1 : 0;
}
