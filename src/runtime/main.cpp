#include "runtime/language_model.hpp"
#include "runtime/profiling.hpp"
#include "runtime/runtime.hpp"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <vector>

namespace {

bool parse_u32(std::string_view text, std::uint32_t& value) {
  auto const* begin = text.data();
  auto const* end = begin + text.size();
  auto [ptr, ec] = std::from_chars(begin, end, value);
  return ec == std::errc{} && ptr == end;
}

bool parse_tokens(std::string_view text, std::vector<std::uint32_t>& out) {
  if (text.empty()) return false;
  while (!text.empty()) {
    auto const comma = text.find(',');
    auto const item = text.substr(0, comma);
    std::uint32_t id{};
    if (!parse_u32(item, id)) return false;
    out.push_back(id);
    if (comma == std::string_view::npos) break;
    text.remove_prefix(comma + 1);
    if (text.empty()) return false;
  }
  return !out.empty();
}

bool read_tokens(std::filesystem::path const& path,
                 std::vector<std::uint32_t>& out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
  if (bytes.empty() || bytes.size() % 4 != 0) return false;
  out.resize(bytes.size() / 4);
  for (std::size_t i = 0; i < out.size(); ++i) {
    auto const at = 4 * i;
    out[i] = static_cast<std::uint32_t>(bytes[at]) |
             (static_cast<std::uint32_t>(bytes[at + 1]) << 8u) |
             (static_cast<std::uint32_t>(bytes[at + 2]) << 16u) |
             (static_cast<std::uint32_t>(bytes[at + 3]) << 24u);
  }
  return true;
}

void usage() {
  std::cerr << "usage: qw38-decode --artifact FILE "
               "(--tokens ID[,ID...] | --tokens-file U32LE) "
               "[--generate N] [--capacity N] [--profile]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path artifact;
  std::filesystem::path tokens_file;
  std::vector<std::uint32_t> tokens;
  std::uint32_t generate = 0;
  std::uint32_t capacity = 0;
  bool profile = false;
  for (int i = 1; i < argc;) {
    if (std::string_view(argv[i]) == "--profile") {
      profile = true;
      ++i;
      continue;
    }
    if (i + 1 >= argc) { usage(); return 2; }
    std::string_view const key = argv[i];
    std::string_view const value = argv[i + 1];
    if (key == "--artifact") artifact = value;
    else if (key == "--tokens") {
      if (!parse_tokens(value, tokens)) { usage(); return 2; }
    } else if (key == "--tokens-file") {
      tokens_file = value;
    } else if (key == "--generate") {
      if (!parse_u32(value, generate)) { usage(); return 2; }
    } else if (key == "--capacity") {
      if (!parse_u32(value, capacity)) { usage(); return 2; }
    } else { usage(); return 2; }
    i += 2;
  }
  if (!tokens_file.empty() && !tokens.empty()) { usage(); return 2; }
  if (!tokens_file.empty() && !read_tokens(tokens_file, tokens)) {
    std::cerr << "failed to read U32LE tokens from " << tokens_file << '\n';
    return 2;
  }
  if (artifact.empty() || tokens.empty() ||
      (profile && generate < 2) ||
      (capacity != 0 && capacity < tokens.size() +
                                    static_cast<std::uint64_t>(generate))) {
    usage();
    return 2;
  }
  if (capacity == 0) capacity = static_cast<std::uint32_t>(
      tokens.size() + static_cast<std::uint64_t>(generate));

  std::cout << "primary-language-only; bounded prefill; MTP disabled\n";
  using Clock = std::chrono::steady_clock;
  auto const elapsed_ms = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  qw38::runtime::profiling::enabled = profile;
  auto const load_start = Clock::now();
  qw38::runtime::profiling::ScopedRange load_range("model_load");
  auto runtime = qw38::runtime::Runtime::create();
  if (!runtime) {
    std::cerr << qw38::runtime::error_message(runtime.error()) << '\n';
    return 1;
  }
  auto model = runtime->load(artifact);
  if (!model) {
    std::cerr << qw38::runtime::error_message(model.error()) << '\n';
    return 1;
  }
  auto const model_load_ms = elapsed_ms(load_start);
  load_range.close();
  auto const session_bind_start = Clock::now();
  qw38::runtime::profiling::ScopedRange setup_range("session_bind");
  auto session = runtime->create_session(*model, capacity);
  if (!session) {
    std::cerr << qw38::runtime::error_message(session.error()) << '\n';
    return 1;
  }
  auto plan = qw38::runtime::LanguageModelPlan::bind(
      *model, *session, runtime->stream());
  if (!plan) {
    std::cerr << qw38::runtime::error_message(plan.error()) << '\n';
    return 1;
  }
  auto const session_bind_ms = elapsed_ms(session_bind_start);
  setup_range.close();
  std::uint32_t next = 0;
  std::uint64_t position = tokens.size();
  auto const prompt_start = Clock::now();
  qw38::runtime::profiling::ScopedRange prompt_range("prompt_ingestion");
  auto prompt_result = plan->prefill_tokens(tokens);
  if (!prompt_result) {
    std::cerr << qw38::runtime::error_message(prompt_result.error()) << '\n';
    return 1;
  }
  next = prompt_result->argmax;
  auto const prompt_ms = elapsed_ms(prompt_start);
  prompt_range.close();
  std::cout << "prompt_tokens=" << tokens.size() << " next_argmax=" << next
            << " logits=248320 FP32\n";
  auto const decode_start = Clock::now();
  qw38::runtime::profiling::ScopedRange decode_range("populated_decode");
  for (std::uint32_t i = 0; i < generate; ++i) {
    std::cout << "generated_token=" << next << '\n';
    if (i + 1 == generate) break;
    auto decoded = plan->decode_token(next, position++);
    if (!decoded) {
      std::cerr << qw38::runtime::error_message(decoded.error()) << '\n';
      return 1;
    }
    next = decoded->argmax;
  }
  auto const decode_ms = elapsed_ms(decode_start);
  decode_range.close();
  if (profile) {
    std::cerr << "profile_ms model_load=" << model_load_ms
              << " session_bind=" << session_bind_ms
              << " prompt_ingestion=" << prompt_ms
              << " prompt_tokens=" << tokens.size()
              << " populated_decode=" << decode_ms
              << " decode_steps=" << (generate - 1)
              << " decode_ms_per_step=" << decode_ms / (generate - 1)
              << " populated_start=" << tokens.size() << '\n';
  }
  return 0;
}
