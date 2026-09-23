#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iostream>
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

void usage() {
  std::cerr << "usage: qw38-decode --artifact FILE --tokens ID[,ID...] "
               "[--generate N] [--capacity N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path artifact;
  std::vector<std::uint32_t> tokens;
  std::uint32_t generate = 0;
  std::uint32_t capacity = 0;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) { usage(); return 2; }
    std::string_view const key = argv[i];
    std::string_view const value = argv[i + 1];
    if (key == "--artifact") artifact = value;
    else if (key == "--tokens") {
      if (!parse_tokens(value, tokens)) { usage(); return 2; }
    } else if (key == "--generate") {
      if (!parse_u32(value, generate)) { usage(); return 2; }
    } else if (key == "--capacity") {
      if (!parse_u32(value, capacity)) { usage(); return 2; }
    } else { usage(); return 2; }
  }
  if (artifact.empty() || tokens.empty() ||
      (capacity != 0 && capacity < tokens.size() +
                                    static_cast<std::uint64_t>(generate))) {
    usage();
    return 2;
  }
  if (capacity == 0) capacity = static_cast<std::uint32_t>(
      tokens.size() + static_cast<std::uint64_t>(generate));

  std::cout << "primary-language-only; prompt setup uses slow repeated decode; "
               "MTP disabled\n";
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
  std::uint32_t next = 0;
  std::uint64_t position = 0;
  for (auto token : tokens) {
    auto decoded = plan->decode_token(token, position++);
    if (!decoded) {
      std::cerr << qw38::runtime::error_message(decoded.error()) << '\n';
      return 1;
    }
    next = decoded->argmax;
  }
  std::cout << "prompt_tokens=" << tokens.size() << " next_argmax=" << next
            << " logits=248320 FP32\n";
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
  return 0;
}
