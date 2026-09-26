// PERF-01 public-API adapters. Compile once per engine; no inference code is shared.
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <span>
#include <string>
#include <vector>

#include "runtime/profiling.hpp"

#ifdef QW38_LLAMA_BENCH
#include "llama.h"
#include "ggml-backend.h"
#else
#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"
#endif

namespace {
using Clock = std::chrono::steady_clock;
[[noreturn]] void fail(std::string const& message) {
  std::cerr << message << '\n';
  std::exit(1);
}
void check(bool value, std::string const& message) { if (!value) fail(message); }
void synchronize_gpu() { check(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed"); }
std::size_t free_bytes() {
  std::size_t free = 0, total = 0;
  check(cudaMemGetInfo(&free, &total) == cudaSuccess, "CUDA memory query failed");
  return free;
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
unsigned number(char const* text) {
  unsigned value = 0;
  std::string input(text);
  auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  check(error == std::errc{} && end == input.data() + input.size(), "invalid integer");
  return value;
}
std::vector<std::uint32_t> read_ids(char const* path, unsigned count) {
  std::ifstream file(path, std::ios::binary);
  std::vector<std::uint32_t> result(count);
  for (auto& id : result) {
    std::array<unsigned char, 4> b{};
    check(bool(file.read(reinterpret_cast<char*>(b.data()), 4)), "short/missing token input");
    id = std::uint32_t(b[0]) | std::uint32_t(b[1]) << 8 |
         std::uint32_t(b[2]) << 16 | std::uint32_t(b[3]) << 24;
    check(id < 248320, "invalid token ID");
  }
  check(file.get() == EOF, "unexpected trailing input tokens");
  return result;
}
template<class T> void array(std::ostream& out, std::span<T const> values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

#ifdef QW38_LLAMA_BENCH
std::string hex(std::string_view bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (unsigned char c : bytes) { result += digits[c >> 4]; result += digits[c & 15]; }
  return result;
}
int vocabulary(char const* path, char const* text_path, char const* prefix) {
  auto params = llama_model_default_params();
  params.vocab_only = true;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      llama_model_load_from_file(path, params), llama_model_free);
  check(bool(model), "vocabulary load failed");
  auto* vocab = llama_model_get_vocab(model.get());
  std::ifstream input(text_path, std::ios::binary);
  std::string text{std::istreambuf_iterator<char>{input}, {}};
  check(!text.empty(), "missing performance prompt");
  std::vector<llama_token> ids(text.size() + 1);
  auto const count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()),
                                   ids.data(), static_cast<int>(ids.size()), false, false);
  check(count > 0, "tokenization failed");
  ids.resize(count);
  std::ofstream encoded(std::string(prefix) + ".u32le", std::ios::binary);
  for (auto id : ids) for (unsigned shift = 0; shift < 32; shift += 8) encoded.put(char((id >> shift) & 255));
  std::ofstream mapping(std::string(prefix) + ".tsv");
  for (int id = 0; id < llama_vocab_n_tokens(vocab); ++id) {
    std::vector<char> piece(256);
    auto size = llama_token_to_piece(vocab, id, piece.data(), piece.size(), 0, true);
    if (size < 0) {
      piece.resize(-size);
      size = llama_token_to_piece(vocab, id, piece.data(), piece.size(), 0, true);
    }
    check(size >= 0, "piece decode failed");
    mapping << id << '\t' << hex(llama_vocab_get_text(vocab, id)) << '\t'
            << hex(std::string_view(piece.data(), size)) << '\t' << llama_vocab_get_attr(vocab, id) << '\n';
  }
  std::ofstream metadata(std::string(prefix) + ".metadata.tsv");
  for (int i = 0; i < llama_model_meta_count(model.get()); ++i) {
    std::array<char, 256> key{};
    std::array<char, 16384> value{};
    auto k = llama_model_meta_key_by_index(model.get(), i, key.data(), key.size());
    if (std::string_view(key.data()) == "tokenizer.ggml.tokens" ||
        std::string_view(key.data()) == "tokenizer.ggml.scores" ||
        std::string_view(key.data()) == "tokenizer.ggml.token_type" ||
        std::string_view(key.data()) == "tokenizer.ggml.merges") continue;
    auto v = llama_model_meta_val_str_by_index(model.get(), i, value.data(), value.size());
    check(k >= 0 && k < int(key.size()) && v >= 0 && v < int(value.size()), "metadata truncated");
    metadata << key.data() << '\t' << hex(value.data()) << '\n';
  }
  check(bool(encoded) && bool(mapping) && bool(metadata), "vocabulary evidence write failed");
  return 0;
}
struct Engine {
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
  std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
  struct Batch {
    llama_batch value;
    explicit Batch(int size) : value(llama_batch_init(size, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;
  } batch{2048};
  std::span<float const> logits;
  unsigned position = 0;
  explicit Engine(char const* path, unsigned capacity, std::ostream& out) {
    ggml_backend_load_all_from_path("/app");
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = -1;
    auto* gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    check(gpu != nullptr, "missing llama GPU backend");
    llama_model_tensor_buft_override overrides[] = {
        {"token_embd.weight", ggml_backend_dev_buffer_type(gpu)}, {nullptr, nullptr}};
    params.tensor_buft_overrides = overrides;
    auto start = Clock::now();
    model.reset(llama_model_load_from_file(path, params));
    check(bool(model), "llama model load failed");
    synchronize_gpu();
    out << "\"load_upload_ms\":" << elapsed(start) << ',';
    check(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 248320,
          "wrong llama vocabulary");
    auto cp = llama_context_default_params();
    cp.n_ctx = capacity;
    cp.n_seq_max = 1;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.no_perf = true;
    start = Clock::now();
    context.reset(llama_init_from_model(model.get(), cp));
    check(bool(context), "llama context creation failed");
    synchronize_gpu();
    out << "\"session_bind_ms\":" << elapsed(start)
        << ",\"capacity_requested\":" << capacity
        << ",\"capacity_allocated\":" << llama_n_ctx(context.get())
        << ",\"batch\":" << llama_n_batch(context.get())
        << ",\"microbatch\":" << llama_n_ubatch(context.get())
        << ",\"host_threads\":" << llama_n_threads(context.get())
        << ",\"kv_k_type\":\"" << ggml_type_name(cp.type_k)
        << "\",\"kv_v_type\":\"" << ggml_type_name(cp.type_v)
        << "\",\"flash_attention_requested\":\"" << llama_flash_attn_type_name(cp.flash_attn_type)
        << "\",\"graph_mode\":\"production-default\"";
  }
  unsigned prefill(std::span<std::uint32_t const> ids) {
    for (std::size_t begin = 0; begin < ids.size();) {
      auto const count = std::min<std::size_t>(llama_n_batch(context.get()), ids.size() - begin);
      auto& b = batch.value;
      b.n_tokens = static_cast<int>(count);
      for (std::size_t i = 0; i < count; ++i) {
        b.token[i] = static_cast<llama_token>(ids[begin + i]);
        b.pos[i] = static_cast<llama_pos>(position + i);
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = begin + i + 1 == ids.size();
      }
      check(llama_decode(context.get(), b) == 0, "llama decode failed");
      position += static_cast<unsigned>(count);
      begin += count;
    }
    auto* values = llama_get_logits_ith(context.get(), -1);
    check(values != nullptr, "missing llama logits");
    logits = {values, 248320};
    return static_cast<unsigned>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  unsigned decode(unsigned token) { return prefill(std::span{&token, 1}); }

};
#else
using namespace qw38::runtime;
template<class T, class E> T take(std::expected<T, E>&& value) {
  if (!value) fail(error_message(value.error()));
  return std::move(*value);
}
struct Engine {
  Runtime runtime;
  Model model;
  Session session;
  LanguageModelPlan plan;
  std::span<float const> logits;
  unsigned position = 0;
  // Bind only after owners have reached their final addresses.
  explicit Engine(char const* path, unsigned capacity, std::ostream& out)
      : runtime(take(Runtime::create())),
        model([&] {
          auto start = Clock::now();
          auto value = take(runtime.load(path));
          synchronize_gpu();
          out << "\"load_upload_ms\":" << elapsed(start) << ',';
          return value;
        }()),
        session(take(runtime.create_session(model, capacity))),
        plan(take(LanguageModelPlan::bind(model, session, runtime.stream()))) {
    std::ostringstream digest;
    for (auto byte : model.schema().integrity.back().digest.bytes)
      digest << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    check(digest.str() == "41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43",
          "artifact differs from TASK-026 accepted identity");
    out << "\"manifest_digest\":\"" << digest.str() << "\",\"capacity_requested\":" << capacity
        << ",\"capacity_allocated\":" << session.kv_capacity()
        << ",\"model_device_bytes\":" << model.device_bytes()
        << ",\"persistent_bytes\":" << session.persistent_bytes()
        << ",\"scratch_bytes\":" << session.arena_plan().total_bytes
        << ",\"chunk\":256,\"kv_type\":\"BF16\",\"state_type\":\"FP32\""
        << ",\"activation_type\":\"BF16\",\"graph_mode\":\"none\"";
  }
  unsigned prefill(std::span<std::uint32_t const> ids) {
    auto result = take(plan.prefill_tokens(ids));
    position += static_cast<unsigned>(ids.size());
    logits = result.logits;
    return result.argmax;
  }
  unsigned decode(unsigned token) {
    auto result = take(plan.decode_token(token, position));
    ++position;
    logits = result.logits;
    return result.argmax;
  }

};
#endif
}  // namespace

int main(int argc, char** argv) {
#ifdef QW38_LLAMA_BENCH
  if (argc == 5 && std::string_view(argv[1]) == "--vocabulary")
    return vocabulary(argv[2], argv[3], argv[4]);
#endif
  if (argc != 6) {
    std::cerr << "usage: request_bench MODEL INPUT_U32LE prefill|decode|request T OUTPUT_JSONL\n";
    return 2;
  }
  std::string const mode(argv[3]);
  auto const depth = number(argv[4]);
  check(mode == "prefill" || mode == "decode" || mode == "request", "invalid workload");
  check(depth > 0 && depth <= 32768, "invalid depth");
  auto tokens = read_ids(argv[2], depth + 128);
  auto prompt = std::span<std::uint32_t const>{tokens}.first(depth);
  std::ofstream out(argv[5]);
  check(bool(out), "cannot open output");
  out << std::setprecision(17);
  auto start = Clock::now();
  synchronize_gpu();
  auto const cuda_init_ms = elapsed(start);
  auto const initial_free = free_bytes();
  start = Clock::now();
  out << "{\"kind\":\"setup\",";
  Engine engine(argv[1], depth + 128, out);
  synchronize_gpu();
  out << ",\"cold_setup_ms\":" << elapsed(start) << ",\"cuda_init_ms\":" << cuda_init_ms
      << ",\"initial_free_bytes\":" << initial_free
      << ",\"after_bind_free_bytes\":" << free_bytes() << "}\n";
  // A fresh session is consumed exactly once. Only populated-decode setup
  // needs a prompt outside the timer; requests include first-use setup costs.
  double populated_setup_ms = 0;
  if (mode == "decode") {
    start = Clock::now();
    engine.prefill(prompt);
    synchronize_gpu();
    populated_setup_ms = elapsed(start);
    check(engine.position == depth, "incorrect incoming decode population");
  }
  qw38::runtime::profiling::enabled = std::getenv("QW38_PROFILE") != nullptr;
  std::vector<unsigned> outputs(mode == "prefill" ? 1u : 128u);
  std::vector<double> steps(mode == "decode" ? 128u : mode == "request" ? 127u : 0u);
  synchronize_gpu();
  qw38::runtime::profiling::ScopedRange measured_range("perf01_measured");
  start = Clock::now();
  double ttft_ms = 0;
  if (mode != "decode") {
    outputs[0] = engine.prefill(prompt);
    synchronize_gpu();
    ttft_ms = elapsed(start);
  }
  auto const tail_start = Clock::now();
  for (unsigned i = 0; i < steps.size(); ++i) {
    auto const step_start = Clock::now();
    outputs[mode == "decode" ? i : i + 1] = engine.decode(
        mode == "decode" ? tokens[depth + i] : outputs[i]);
    synchronize_gpu();
    steps[i] = elapsed(step_start);
  }
  synchronize_gpu();
  auto const end = Clock::now();
  auto const total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  auto const tail_ms = steps.empty() ? 0 : std::chrono::duration<double, std::milli>(end - tail_start).count();
  measured_range.close();
  check(engine.position == depth + steps.size(), "incorrect final population");
  check(std::all_of(engine.logits.begin(), engine.logits.end(), [](float x) { return std::isfinite(x); }),
        "nonfinite final logits");
  out << "{\"kind\":\"sample\",\"row\":\"" << mode << '-' << depth
      << "\",\"total_ms\":" << total_ms << ",\"ttft_ms\":" << ttft_ms
      << ",\"tail_ms\":" << tail_ms << ",\"populated_setup_ms\":" << populated_setup_ms
      << ",\"free_bytes\":" << free_bytes() << ",\"steps_ms\":";
  array(out, std::span<double const>{steps});
  out << ",\"output_ids\":";
  array(out, std::span<unsigned const>{outputs});
  out << ",\"final_position\":" << engine.position << ",\"finite_logits\":true}\n";
  out << "{\"kind\":\"complete\",\"runs\":1,\"warmups\":0}\n";
  out.flush();
  return out ? 0 : 1;
}
