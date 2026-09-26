#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <utility>
#include <vector>

namespace {

std::vector<std::uint32_t> read_ids(std::filesystem::path const& path) {
  std::ifstream input(path, std::ios::binary);
  std::vector<unsigned char> bytes(std::istreambuf_iterator<char>{input}, {});
  if (bytes.empty() || bytes.size() % 4u) return {};
  std::vector<std::uint32_t> ids(bytes.size() / 4u);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    auto const at = i * 4u;
    ids[i] = std::uint32_t(bytes[at]) |
             (std::uint32_t(bytes[at + 1]) << 8u) |
             (std::uint32_t(bytes[at + 2]) << 16u) |
             (std::uint32_t(bytes[at + 3]) << 24u);
  }
  return ids;
}

std::size_t free_memory() {
  std::size_t free = 0, total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) return 0;
  return free;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: qw38_bench_language_prefill ARTIFACT PROMPT_U32LE\n";
    return 2;
  }
  using namespace qw38::runtime;
  auto source = read_ids(argv[2]);
  if (source.empty()) return 2;
  auto runtime = Runtime::create();
  if (!runtime) { std::cerr << error_message(runtime.error()) << '\n'; return 1; }
  auto const initial_free = free_memory();
  auto model = runtime->load(argv[1]);
  if (!model) { std::cerr << error_message(model.error()) << '\n'; return 1; }
  auto const after_model = free_memory();
  auto session = runtime->create_session(*model, 32768u + 32u);
  if (!session) { std::cerr << error_message(session.error()) << '\n'; return 1; }
  auto plan = LanguageModelPlan::bind(*model, *session, runtime->stream());
  if (!plan) { std::cerr << error_message(plan.error()) << '\n'; return 1; }
  auto const after_session = free_memory();
  if (!initial_free || !after_model || !after_session) return 1;
  std::vector<std::uint32_t> prompt(4096);
  for (std::size_t i = 0; i < prompt.size(); ++i) prompt[i] = source[i % source.size()];
  using Clock = std::chrono::steady_clock;
  auto timed = [&](auto&& call) {
    auto const start = Clock::now();
    auto result = call();
    auto const ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (!result) std::cerr << error_message(result.error()) << '\n';
    return std::pair{std::move(result), ms};
  };
  auto [generation, generation_ms] = timed([&] {
    return plan->prefill_tokens(std::span<std::uint32_t const>{prompt.data(), 256});
  });
  if (!generation) return 1;
  auto const after_prefill = free_memory();
  auto [handoff, handoff_ms] = timed([&] {
    return plan->decode_token(generation->argmax, 256);
  });
  if (!handoff || !session->reset()) return 1;
  constexpr std::array<std::uint64_t, 4> rows{0, 63, 64, 255};
  std::uint32_t delivered = 0;
  auto [evaluation, evaluation_ms] = timed([&] {
    return plan->prefill_tokens(
        std::span<std::uint32_t const>{prompt.data(), 256}, rows,
        [&](std::uint64_t, std::span<float const> logits)
            -> std::expected<void, Error> {
          if (logits.size() != kVocab)
            return std::unexpected(make_error(ErrorCode::Internal, "bench.head",
                                              "requested row has wrong width"));
          ++delivered;
          return {};
        });
  });
  if (!evaluation || delivered != rows.size() || !session->reset()) return 1;
  auto const after_evaluation = free_memory();
  auto [long_generation, long_ms] = timed([&] { return plan->prefill_tokens(prompt); });
  if (!long_generation || !session->reset()) return 1;
  std::vector<std::uint32_t> context(32768);
  for (std::size_t i = 0; i < context.size(); ++i)
    context[i] = source[i % source.size()];
  auto [context_generation, context_ms] = timed([&] {
    return plan->prefill_tokens(context);
  });
  if (!context_generation) return 1;
  auto const after_long = free_memory();
  auto [context_handoff, context_handoff_ms] = timed([&] {
    return plan->decode_token(context_generation->argmax, 32768);
  });
  if (!context_handoff) return 1;
  auto const minimum_free = std::min({after_model, after_session, after_prefill,
                                     after_evaluation, after_long});
  std::cout << "model_device_bytes=" << model->device_bytes()
            << " session_persistent_bytes=" << session->persistent_bytes()
            << " session_scratch_bytes=" << session->arena_plan().total_bytes
            << " free_initial=" << initial_free
            << " free_after_model=" << after_model
            << " free_after_session=" << after_session
            << " free_after_prefill=" << after_prefill
            << " free_after_evaluation=" << after_evaluation
            << " free_after_long=" << after_long
            << " measured_peak_bytes=" << initial_free - minimum_free
            << " reserve_bytes=2147483648\n";
  std::cout << "final_row_256_ms=" << generation_ms
            << " requested_rows_256_ms=" << evaluation_ms
            << " requested_rows=" << delivered
            << " handoff_decode_ms=" << handoff_ms
            << " final_row_4096_ms=" << long_ms
            << " final_row_32768_ms=" << context_ms
            << " handoff_decode_32768_ms=" << context_handoff_ms << '\n';
  return minimum_free >= 2147483648ull ? 0 : 1;
}
