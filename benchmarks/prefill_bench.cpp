#include "runtime/language_model.hpp"
#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"
#include "runtime/sizes.hpp"

#include "cuda/alloc.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/event.hpp"
#include "cuda/prefill.hpp"
#include "cuda/upload.hpp"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

template <class T>
std::span<std::byte const> bytes(std::vector<T> const& values) {
  return {reinterpret_cast<std::byte const*>(values.data()),
          values.size() * sizeof(T)};
}

bool parse_u32(char const* text, std::uint32_t& out) {
  std::string_view const view(text);
  auto [end, error] = std::from_chars(view.data(), view.data() + view.size(), out);
  return error == std::errc{} && end == view.data() + view.size();
}

int run(char const* artifact, std::uint32_t tile_rows,
        std::string_view selected_family, std::uint32_t selected_m,
        std::uint32_t repetitions) {
  using namespace qw38;
  using namespace qw38::runtime;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto model = runtime->load(artifact);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  auto const& stream = runtime->stream();
  auto gdn = bind_prefill_layer_projections(*model, 0, stream);
  auto attn = bind_prefill_layer_projections(*model, 3, stream);
  auto head = bind_prefill_head(*model, stream);
  std::uint32_t const capacity = std::max(selected_m, 128u);
  auto engine = cuda::PrefillEngine::create(stream, capacity, tile_rows);
  if (!gdn || !attn || !head || !engine) {
    std::cerr << "prefill plan binding failed\n";
    return 1;
  }
  auto session = runtime->create_session(*model, 8);
  if (!session) return 1;
  auto decode = LanguageModelPlan::bind(*model, *session, stream);
  if (!decode) { std::cerr << error_message(decode.error()) << '\n'; return 1; }
  auto normalized = session->scratch(format::ScratchKind::NormalizedHidden);
  auto swiglu = session->scratch(format::ScratchKind::MlpSwiglu);
  auto gdn_scratch = session->scratch(format::ScratchKind::GdnWorkspace);
  auto attn_scratch = session->scratch(format::ScratchKind::AttentionWorkspace);
  if (!normalized || !swiglu || !gdn_scratch || !attn_scratch) return 1;
  constexpr std::uint32_t captured = 8;
  std::vector<std::uint16_t> hidden(captured * kHidden);
  std::vector<std::uint16_t> gdn_out(captured * kGdnZWidth);
  std::vector<std::uint16_t> attn_out(captured * kAttnOutWidth);
  std::vector<std::uint16_t> mlp_down(captured * kFfnWidth);
  std::vector<float> residual(captured * kHidden);
  auto const capture_start = std::chrono::steady_clock::now();
  for (std::uint32_t i = 0; i < captured; ++i) {
    auto result = decode->decode_token(1000u + i * 17u, i);
    if (!result) { std::cerr << error_message(result.error()) << '\n'; return 1; }
    auto copy = [&](void* dst, void const* src, std::uint64_t count) {
      return cuda::copy_d2h(dst, src, count, stream).has_value();
    };
    if (!copy(hidden.data() + i * kHidden,
              normalized->region[0].tensor.pointer, kHidden * 2u) ||
        !copy(gdn_out.data() + i * kGdnZWidth,
              gdn_scratch->pointer + kGdnOffU, kGdnZWidth * 2u) ||
        !copy(attn_out.data() + i * kAttnOutWidth,
              attn_scratch->pointer + kAttnOffQ, kAttnOutWidth * 2u) ||
        !copy(mlp_down.data() + i * kFfnWidth,
              swiglu->region[0].tensor.pointer, kFfnWidth * 2u) ||
        !copy(residual.data() + i * kHidden,
              session->residual_h().pointer, kHidden * 4u) ||
        !stream.sync()) return 1;
  }
  auto const capture_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - capture_start).count();

  auto repeat_bf16 = [&](std::vector<std::uint16_t> const& source,
                         std::uint32_t width) {
    std::vector<std::uint16_t> out(static_cast<std::uint64_t>(capacity) * width);
    for (std::uint32_t i = 0; i < capacity; ++i)
      std::copy_n(source.data() + (i % captured) * width, width,
                  out.data() + i * width);
    return cuda::upload(bytes(out), stream);
  };
  auto x_hidden = repeat_bf16(hidden, kHidden);
  auto x_gdn_out = repeat_bf16(gdn_out, kGdnZWidth);
  auto x_attn_out = repeat_bf16(attn_out, kAttnOutWidth);
  auto x_mlp_down = repeat_bf16(mlp_down, kFfnWidth);
  std::vector<float> hmid(static_cast<std::uint64_t>(capacity) * kHidden);
  for (std::uint32_t i = 0; i < capacity; ++i)
    std::copy_n(residual.data() + (i % captured) * kHidden, kHidden,
                hmid.data() + i * kHidden);
  auto x_residual = cuda::upload(bytes(hmid), stream);
  auto output = cuda::DeviceBuffer::allocate(
      static_cast<std::uint64_t>(capacity) * kFfnWidth * 4u);
  auto logits = cuda::DeviceBuffer::allocate(2u * kVocab * 4u);
  if (!x_hidden || !x_gdn_out || !x_attn_out || !x_mlp_down ||
      !x_residual || !output || !logits) return 1;
  std::size_t free_bytes = 0, total_bytes = 0;
  if (!cuda::check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) return 1;
  std::cout << "context,artifact_bytes," << std::filesystem::file_size(artifact)
            << ",model_device_bytes," << model->device_bytes()
            << ",workspace_bytes," << engine->workspace_bytes()
            << ",free_bytes," << free_bytes << ",total_bytes," << total_bytes
            << ",capture_ms," << capture_ms
            << ",activation_source,8_populated_candidate_decode_rows_repeated"
            << ",tile_rows," << tile_rows << '\n';
  std::cout << "sample,family,m,n,k,tile_rows,repetition,gpu_ms,host_ms\n";
  auto start = cuda::Event::create_timing();
  auto end = cuda::Event::create_timing();
  if (!start || !end) return 1;
  auto const* hidden_ptr = static_cast<std::uint16_t const*>(x_hidden->data());
  auto const* gdn_ptr = static_cast<std::uint16_t const*>(x_gdn_out->data());
  auto const* attn_ptr = static_cast<std::uint16_t const*>(x_attn_out->data());
  auto const* down_ptr = static_cast<std::uint16_t const*>(x_mlp_down->data());
  auto const* residual_ptr = static_cast<float const*>(x_residual->data());

  struct Case {
    std::string_view name;
    cuda::PrefillWeight weight;
    std::uint16_t const* input;
    cuda::PrefillEpilogue epilogue;
    float const* residual;
  };
  std::array<Case, 12> cases{{
      {"gdn_qkv", gdn->first, hidden_ptr, cuda::PrefillEpilogue::StoreBf16, nullptr},
      {"gdn_z", gdn->second, hidden_ptr, cuda::PrefillEpilogue::StoreBf16, nullptr},
      {"gdn_a", gdn->third, hidden_ptr, cuda::PrefillEpilogue::StoreFp32, nullptr},
      {"gdn_b", gdn->fourth, hidden_ptr, cuda::PrefillEpilogue::StoreFp32, nullptr},
      {"gdn_out", gdn->mixer_out, gdn_ptr, cuda::PrefillEpilogue::ResidualAddFp32, residual_ptr},
      {"attn_qg", attn->first, hidden_ptr, cuda::PrefillEpilogue::StoreBf16, nullptr},
      {"attn_k", attn->second, hidden_ptr, cuda::PrefillEpilogue::StoreBf16, nullptr},
      {"attn_v", attn->third, hidden_ptr, cuda::PrefillEpilogue::StoreBf16, nullptr},
      {"attn_out", attn->mixer_out, attn_ptr, cuda::PrefillEpilogue::ResidualAddFp32, residual_ptr},
      {"mlp_down", gdn->mlp_down, down_ptr, cuda::PrefillEpilogue::ResidualAddFp32, residual_ptr},
      {"mlp_gate", gdn->mlp_gate, hidden_ptr, cuda::PrefillEpilogue::StoreFp32, nullptr},
      {"mlp_up", gdn->mlp_up, hidden_ptr, cuda::PrefillEpilogue::StoreFp32, nullptr},
  }};
  std::array<std::uint32_t, 10> const sizes{
      1, 2, 8, 32, 64, 127, 128, 256, 512, 1024};
  bool const profile = std::getenv("QW38_NSYS_CAPTURE") != nullptr;
  if (profile && !cuda::check(cudaProfilerStart(), "cudaProfilerStart")) return 1;
  for (auto m : sizes) {
    if (m > capacity) continue;
    if (selected_m && selected_m != m) continue;
    auto bench = [&](std::string_view name, std::uint32_t n, std::uint32_t k,
                     auto&& launch) -> bool {
      if (selected_family != "all" && selected_family != name) return true;
      if (profile) {
        auto const label = std::string(name) + ":m=" + std::to_string(m);
        (void)nvtxRangePushA(label.c_str());
      }
      for (std::uint32_t rep = 0; rep < repetitions + 5u; ++rep) {
        if (!stream.sync()) return false;
        auto host_start = std::chrono::steady_clock::now();
        if (!start->record(stream)) return false;
        if (!launch()) return false;
        if (!end->record(stream) || !end->sync()) return false;
        auto host_end = std::chrono::steady_clock::now();
        auto ms = cuda::elapsed_ms(*start, *end);
        if (!ms) return false;
        if (rep >= 5u) {
          auto host_ms = std::chrono::duration<double, std::milli>(
              host_end - host_start).count();
          std::cout << "sample," << name << ',' << m << ',' << n << ',' << k
                    << ',' << tile_rows << ',' << rep - 5u << ',' << *ms
                    << ',' << host_ms << '\n';
        }
      }
      if (profile) (void)nvtxRangePop();
      return true;
    };
    for (auto const& c : cases) {
      auto launch = [&] {
        auto st = engine->project(cuda::PrefillProjection{
            .weight = c.weight, .input = c.input, .output = output->data(),
            .residual = c.residual, .valid_tokens = m, .first_position = 512,
            .epilogue = c.epilogue});
        if (!st) std::cerr << cuda::error_message(st.error()) << '\n';
        return bool(st);
      };
      if (!bench(c.name, c.weight.n, c.weight.k, launch)) return 1;
    }
    auto mlp = [&] {
      auto st = engine->mlp(gdn->mlp_gate, gdn->mlp_up, gdn->mlp_down,
                            residual_ptr, gdn->mlp_gamma, 1.0e-6f,
                            static_cast<float*>(output->data()), m, 512);
      if (!st) std::cerr << cuda::error_message(st.error()) << '\n';
      return bool(st);
    };
    if (!bench("mlp_complete", kHidden, kHidden, mlp)) return 1;
    auto head_gen = [&] {
      auto st = engine->head_generation(*head, hidden_ptr, m, 512,
                                        static_cast<float*>(logits->data()));
      if (!st) std::cerr << cuda::error_message(st.error()) << '\n';
      return bool(st);
    };
    if (!bench("head_generation", kVocab, kHidden, head_gen)) return 1;
    std::array<std::uint32_t, 2> requested{0, m - 1u};
    auto head_eval = [&] {
      auto rows = m == 1 ? std::span<std::uint32_t const>{requested.data(), 1}
                         : std::span<std::uint32_t const>{requested.data(), 2};
      auto st = engine->head_evaluation(*head, hidden_ptr, m, 512, rows,
                                        static_cast<float*>(logits->data()));
      if (!st) std::cerr << cuda::error_message(st.error()) << '\n';
      return bool(st);
    };
    if (!bench("head_evaluation", kVocab, kHidden, head_eval)) return 1;
  }
  if (profile && !cuda::check(cudaProfilerStop(), "cudaProfilerStop")) return 1;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: qw38_bench_prefill ARTIFACT TILE_ROWS [FAMILY|all [M|0 [REPS]]]\n";
    return 2;
  }
  std::uint32_t tile_rows = 0, m = 0, reps = 20;
  if (!parse_u32(argv[2], tile_rows) ||
      (argc >= 5 && !parse_u32(argv[4], m)) ||
      (argc == 6 && !parse_u32(argv[5], reps)) || reps == 0) return 2;
  return run(argv[1], tile_rows, argc >= 4 ? argv[3] : "all", m, reps);
}
