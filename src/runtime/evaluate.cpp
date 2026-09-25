#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::filesystem::path artifact;
  std::filesystem::path cases;
  std::filesystem::path output;
  std::vector<std::uint32_t> eos_ids;
};

struct Case {
  std::string id;
  std::filesystem::path prompt_file;
  std::optional<std::filesystem::path> target_file;
  std::optional<std::filesystem::path> mask_file;
  std::uint32_t generation_cap{};
};

bool parse_u32(std::string_view text, std::uint32_t& value) {
  if (text.empty()) return false;
  auto const [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

bool parse_list(std::string_view text, std::vector<std::uint32_t>& values) {
  while (!text.empty()) {
    auto const comma = text.find(',');
    auto const item = text.substr(0, comma);
    std::uint32_t value{};
    if (!parse_u32(item, value)) return false;
    values.push_back(value);
    if (comma == std::string_view::npos) return true;
    text.remove_prefix(comma + 1);
  }
  return false;
}

bool parse_options(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    std::string_view const key = argv[i];
    if (key == "--help") {
      std::cout << "usage: qw38-evaluate --artifact FILE --cases TSV --output DIR "
                   "--eos-ids ID[,ID...]\n";
      return false;
    }
    if (i + 1 >= argc) return false;
    std::string_view const value = argv[++i];
    if (key == "--artifact") options.artifact = value;
    else if (key == "--cases") options.cases = value;
    else if (key == "--output") options.output = value;
    else if (key == "--eos-ids") {
      if (!parse_list(value, options.eos_ids)) return false;
    } else {
      return false;
    }
  }
  return !options.artifact.empty() && !options.cases.empty() &&
         !options.output.empty() && !options.eos_ids.empty();
}

bool read_u32le(std::filesystem::path const& path,
                std::vector<std::uint32_t>& values) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
  if (bytes.empty() || bytes.size() % 4 != 0) return false;
  values.resize(bytes.size() / 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto const at = 4 * i;
    values[i] = static_cast<std::uint32_t>(bytes[at]) |
                (static_cast<std::uint32_t>(bytes[at + 1]) << 8u) |
                (static_cast<std::uint32_t>(bytes[at + 2]) << 16u) |
                (static_cast<std::uint32_t>(bytes[at + 3]) << 24u);
  }
  return true;
}

bool read_bytes(std::filesystem::path const& path,
                std::vector<unsigned char>& bytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  bytes.assign(std::istreambuf_iterator<char>(input), {});
  return true;
}

bool write_u32le(std::filesystem::path const& path,
                 std::vector<std::uint32_t> const& values) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  for (std::uint32_t id : values) {
    unsigned char const bytes[] = {
        static_cast<unsigned char>(id & 0xffu),
        static_cast<unsigned char>((id >> 8u) & 0xffu),
        static_cast<unsigned char>((id >> 16u) & 0xffu),
        static_cast<unsigned char>((id >> 24u) & 0xffu),
    };
    output.write(reinterpret_cast<char const*>(bytes), 4);
  }
  return output.good();
}

bool write_logits(std::ofstream& output, std::span<float const> logits) {
  if (!output || logits.size() != qw38::runtime::kVocab) return false;
  for (float value : logits) {
    if (!std::isfinite(value)) return false;
    auto const bits = std::bit_cast<std::uint32_t>(value);
    unsigned char const bytes[] = {
        static_cast<unsigned char>(bits & 0xffu),
        static_cast<unsigned char>((bits >> 8u) & 0xffu),
        static_cast<unsigned char>((bits >> 16u) & 0xffu),
        static_cast<unsigned char>((bits >> 24u) & 0xffu),
    };
    output.write(reinterpret_cast<char const*>(bytes), 4);
  }
  return output.good();
}

std::string hex(qw38::format::Hash256 const& digest) {
  std::ostringstream out;
  for (auto byte : digest.bytes) {
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned int>(byte);
  }
  return out.str();
}

bool parse_case(std::string const& row, Case& record) {
  std::vector<std::string_view> fields;
  std::string_view remaining = row;
  while (true) {
    auto const tab = remaining.find('\t');
    fields.push_back(remaining.substr(0, tab));
    if (tab == std::string_view::npos) break;
    remaining.remove_prefix(tab + 1);
  }
  if (fields.size() != 5 || fields[0].empty() || fields[1].empty()) return false;
  record.id = fields[0];
  record.prompt_file = fields[1];
  if (fields[2] != "-") record.target_file = std::filesystem::path(fields[2]);
  if (fields[3] != "-") record.mask_file = std::filesystem::path(fields[3]);
  return parse_u32(fields[4], record.generation_cap) &&
         record.generation_cap > 0 &&
         record.target_file.has_value() == record.mask_file.has_value();
}

bool is_eos(std::uint32_t token, std::vector<std::uint32_t> const& eos_ids) {
  return std::ranges::find(eos_ids, token) != eos_ids.end();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) return 2;
  using namespace qw38;
  using namespace qw38::runtime;
  auto fail = [](Error const& error) {
    std::cerr << error_message(error) << '\n';
    return 1;
  };
  std::filesystem::create_directories(options.output);
  auto runtime = Runtime::create();
  if (!runtime) return fail(runtime.error());
  auto model = runtime->load(options.artifact);
  if (!model) return fail(model.error());

  auto const& schema = model->schema();
  if (schema.integrity.empty()) {
    std::cerr << "candidate artifact has no integrity manifest\n";
    return 1;
  }
  auto identity = std::ofstream(options.output / "candidate_identity.json", std::ios::trunc);
  if (!identity) return 1;
  std::array<std::uint64_t, 4> storage_counts{};
  std::array<std::uint64_t, 6> quantizer_counts{};
  for (auto const& tensor : schema.tensors) {
    auto const storage = static_cast<std::uint16_t>(tensor.storage);
    auto const quantizer = static_cast<std::uint16_t>(tensor.quantizer);
    if (storage >= 1 && storage <= storage_counts.size()) ++storage_counts[storage - 1];
    if (quantizer >= 0x0100 && quantizer <= 0x0105) ++quantizer_counts[quantizer - 0x0100];
  }
  identity << "{\"manifest_digest\":\"" << hex(schema.integrity.back().digest)
           << "\",\"compiler\":{\"ident\":\"" << schema.compiler.ident
           << "\",\"major\":" << schema.compiler.major
           << ",\"minor\":" << schema.compiler.minor
           << ",\"patch\":" << schema.compiler.patch
           << "},\"source_hash\":\"" << hex(schema.source_hash)
           << "\",\"config_hash\":\"" << hex(schema.config_hash)
           << "\",\"tokenizer_hash\":\"" << hex(schema.tokenizer_hash)
           << "\",\"precision_policy_id\":"
           << static_cast<unsigned int>(schema.precision.id)
           << ",\"tensor_count\":" << schema.tensors.size()
           << ",\"storage_counts\":{\"int4_grouped\":" << storage_counts[0]
           << ",\"int8_grouped\":" << storage_counts[1]
           << ",\"bf16\":" << storage_counts[2]
           << ",\"fp32\":" << storage_counts[3]
           << "},\"logical_quantizer_counts\":{\"none\":" << quantizer_counts[0]
           << ",\"q4_g64_v0\":" << quantizer_counts[1]
           << ",\"q8_g32_v0\":" << quantizer_counts[2]
           << ",\"q4_g64_candidate_v1\":" << quantizer_counts[3]
           << ",\"q8_g32_candidate_v1\":" << quantizer_counts[4]
           << ",\"q4_k_candidate_v2\":" << quantizer_counts[5]
           << "},\"decode_dispatch\":{\"activation_policy\":\"bf16\","
              "\"quantized_kernel\":\"grouped_gemv\","
              "\"fallback\":null}}\n";
  identity.close();
  if (!identity) return 1;

  std::ifstream cases(options.cases);
  std::ofstream results(options.output / "cases.jsonl", std::ios::trunc);
  if (!cases || !results) return 1;
  std::string row;
  std::uint32_t case_count = 0;
  while (std::getline(cases, row)) {
    Case record;
    if (!parse_case(row, record)) {
      std::cerr << "malformed case TSV row\n";
      return 2;
    }
    std::vector<std::uint32_t> prompt;
    if (!read_u32le(record.prompt_file, prompt)) {
      std::cerr << "invalid prompt tokens for " << record.id << '\n';
      return 1;
    }
    std::vector<std::uint32_t> targets;
    std::vector<unsigned char> mask;
    if (record.target_file &&
        (!read_u32le(*record.target_file, targets) ||
         !read_bytes(*record.mask_file, mask) || targets.size() != mask.size() ||
         std::ranges::any_of(mask, [](unsigned char bit) { return bit != 1; }))) {
      std::cerr << "invalid teacher target or loss mask for " << record.id << '\n';
      return 1;
    }
    for (auto token : prompt) {
      if (token >= kVocab) {
        std::cerr << "prompt token outside vocabulary for " << record.id << '\n';
        return 1;
      }
    }
    for (auto token : targets) {
      if (token >= kVocab) {
        std::cerr << "teacher token outside vocabulary for " << record.id << '\n';
        return 1;
      }
    }
    auto const continuation_count =
        std::max<std::size_t>(targets.size(), record.generation_cap);
    if (prompt.size() + continuation_count >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      std::cerr << "case capacity exceeds runtime position range: " << record.id << '\n';
      return 1;
    }
    auto session = runtime->create_session(
        *model, prompt.size() + continuation_count);
    if (!session) return fail(session.error());
    auto plan = LanguageModelPlan::bind(*model, *session, runtime->stream());
    if (!plan) return fail(plan.error());

    auto prompt_result = plan->setup_prompt_slow(prompt);
    if (!prompt_result) return fail(prompt_result.error());
    std::vector<std::uint32_t> generated;
    std::string stop_reason = "cap";
    std::uint32_t next = prompt_result->argmax;
    std::uint64_t position = prompt.size();
    for (std::uint32_t step = 0; step < record.generation_cap; ++step) {
      generated.push_back(next);
      if (is_eos(next, options.eos_ids)) {
        stop_reason = "eos";
        break;
      }
      if (step + 1 < record.generation_cap) {
        auto decoded = plan->decode_token(next, position++);
        if (!decoded) return fail(decoded.error());
        next = decoded->argmax;
      }
    }
    if (!write_u32le(options.output / (record.id + ".generated.u32le"), generated)) {
      std::cerr << "failed to write generated IDs for " << record.id << '\n';
      return 1;
    }
    if (auto reset = session->reset(); !reset) return fail(reset.error());

    if (!targets.empty()) {
      auto teacher_logits = std::ofstream(
          options.output / (record.id + ".candidate-logits.f32le"),
          std::ios::binary | std::ios::trunc);
      if (!teacher_logits) return 1;
      auto score = plan->setup_prompt_slow(prompt);
      if (!score) return fail(score.error());
      for (std::size_t i = 0; i < targets.size(); ++i) {
        if (!write_logits(teacher_logits, score->logits)) {
          std::cerr << "failed to write candidate logits for " << record.id << '\n';
          return 1;
        }
        if (i + 1 < targets.size()) {
          auto decoded = plan->decode_token(targets[i], prompt.size() + i);
          if (!decoded) return fail(decoded.error());
          score = std::move(decoded);
        }
      }
      teacher_logits.close();
    }
    results << "{\"id\":\"" << record.id << "\",\"status\":\"complete\""
            << ",\"prompt_tokens\":" << prompt.size()
            << ",\"target_tokens\":" << targets.size()
            << ",\"generation_tokens\":" << generated.size()
            << ",\"generation_stop\":\"" << stop_reason << "\"}\n";
    if (!results) return 1;
    ++case_count;
    std::cout << "evaluated_cases=" << case_count << " last_id=" << record.id
              << '\n' << std::flush;
  }
  std::cout << "evaluated_cases=" << case_count
            << " vocab_size=" << kVocab
            << " outputs=FP32 full-vocabulary,language-only\n";
  return 0;
}
