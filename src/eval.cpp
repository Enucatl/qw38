#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "qw38/engine.h"
#include "model.h"
#include "quant.h"
#include "sha256.h"
#include "template.h"
#include "tokenizer.h"

namespace {
constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr const char* kModelRevision =
    "0669b98607d47046c7c2b3f801011d54a08cfccf";
constexpr const char* kModelSha256 =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";

int write_inventory(const char* model_path, const char* output_path) {
  std::error_code error;
  if (std::filesystem::exists(output_path, error) || error) {
    std::cerr << "inventory output already exists or cannot be inspected\n";
    return 1;
  }
  qw38::Engine engine;
  qw38::Status status = qw38::Engine::open(model_path, &engine);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  qw38::internal::ModelInfo info;
  status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  const std::string temporary =
      std::string(output_path) + ".tmp." + std::to_string(getpid());
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "cannot create inventory output\n";
    return 1;
  }
  output << "{\n  \"schema_version\": 1,\n";
  output << "  \"model_sha256\": \"" << kModelSha256 << "\",\n";
  output << "  \"gguf_version\": " << info.gguf_version << ",\n";
  output << "  \"data_offset\": " << info.data_offset << ",\n";
  output << "  \"tensor_count\": " << info.tensors.size() << ",\n";
  output << "  \"tensors\": [\n";
  for (std::size_t index = 0; index < info.tensors.size(); ++index) {
    const qw38::internal::TensorInfo& tensor = info.tensors[index];
    const std::uint64_t absolute = info.data_offset + tensor.offset;
    std::string digest;
    status = qw38::internal::sha256_bytes(mapping.data() + absolute,
                                          tensor.storage_bytes, &digest);
    if (!status.is_ok()) {
      output.close();
      std::remove(temporary.c_str());
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    output << "    {\"name\": \"" << tensor.name << "\", \"role\": \""
           << tensor.semantic_role << "\", \"shape\": [";
    for (std::size_t dimension = 0; dimension < tensor.dimensions.size();
         ++dimension) {
      if (dimension != 0) output << ", ";
      output << tensor.dimensions[dimension];
    }
    output << "], \"dtype\": \""
           << qw38::internal::ggml_type_name(tensor.type)
           << "\", \"offset\": " << tensor.offset
           << ", \"absolute_offset\": " << absolute
           << ", \"storage_bytes\": " << tensor.storage_bytes
           << ", \"padded_span_bytes\": " << tensor.padded_span_bytes
           << ", \"sha256\": \"" << digest << "\"}";
    output << (index + 1 == info.tensors.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  output.close();
  if (!output || std::rename(temporary.c_str(), output_path) != 0) {
    std::remove(temporary.c_str());
    std::cerr << "cannot commit inventory output\n";
    return 1;
  }
  std::cout << "inventory=" << output_path << '\n';
  return 0;
}

int hex_value(char character) {
  if (character >= '0' && character <= '9') return character - '0';
  if (character >= 'a' && character <= 'f') return character - 'a' + 10;
  if (character >= 'A' && character <= 'F') return character - 'A' + 10;
  return -1;
}

bool parse_hex(const std::string& hex, std::vector<std::uint8_t>* bytes) {
  if (hex.size() % 2 != 0) return false;
  bytes->clear();
  bytes->reserve(hex.size() / 2);
  for (std::size_t index = 0; index < hex.size(); index += 2) {
    const int high = hex_value(hex[index]);
    const int low = hex_value(hex[index + 1]);
    if (high < 0 || low < 0) return false;
    bytes->push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return true;
}

void write_float_hex(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr char kHex[] = "0123456789abcdef";
  for (unsigned int byte = 0; byte < 4; ++byte) {
    const unsigned int value_byte = (bits >> (byte * 8U)) & 0xFFU;
    std::cout << kHex[value_byte >> 4U] << kHex[value_byte & 15U];
  }
}

int check_quant(const std::string& kind, const std::string& hex) {
  std::vector<std::uint8_t> block;
  if (!parse_hex(hex, &block)) {
    std::cerr << "invalid_argument: quant block is not even-length hexadecimal\n";
    return 1;
  }
  std::array<float, qw38::internal::kQuantBlockValues> decoded{};
  std::array<float, qw38::internal::kQuantBlockValues> activation{};
  for (std::size_t index = 0; index < activation.size(); ++index) {
    const int numerator = static_cast<int>((index * 37) % 101) - 50;
    activation[index] = static_cast<float>(numerator) / 32.0F;
  }
  float dot = 0.0F;
  qw38::Status status;
  if (kind == "q4_k") {
    status = qw38::internal::decode_q4_k(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q4_k(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else if (kind == "q6_k") {
    status = qw38::internal::decode_q6_k(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q6_k(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else {
    std::cerr << "invalid_argument: quant kind must be q4_k or q6_k\n";
    return 1;
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "decoded_f32_le_hex=";
  for (float value : decoded) write_float_hex(value);
  std::cout << "\ndot_f32_le_hex=";
  write_float_hex(dot);
  std::cout << '\n';
  return 0;
}

int tokenize_hex(const char* model_path, const std::string& hex) {
  if (hex.size() % 2 != 0) {
    std::cerr << "invalid_argument: input hex has odd length\n";
    return 1;
  }
  std::string input;
  input.reserve(hex.size() / 2);
  for (std::size_t index = 0; index < hex.size(); index += 2) {
    const int high = hex_value(hex[index]);
    const int low = hex_value(hex[index + 1]);
    if (high < 0 || low < 0) {
      std::cerr << "invalid_argument: input is not hexadecimal\n";
      return 1;
    }
    input.push_back(static_cast<char>((high << 4) | low));
  }
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::Tokenizer tokenizer;
  if (status.is_ok()) status = tokenizer.build(info);
  std::vector<std::uint32_t> ids;
  if (status.is_ok()) status = tokenizer.encode(input, &ids);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << ids[index];
  }
  std::cout << '\n';
  return 0;
}

int render_template_case(const std::string& name) {
  using qw38::internal::Message;
  using qw38::internal::MessageRole;
  qw38::internal::TemplateInput input;
  if (name == "user_no_thinking") {
    input.messages = {{MessageRole::kUser, "Hello"}};
    input.options.enable_thinking = false;
  } else if (name == "user_default_xhigh") {
    input.messages = {{MessageRole::kUser, "Explain quartz."}};
  } else if (name == "system_low") {
    input.messages = {{MessageRole::kSystem, "Be concise."},
                      {MessageRole::kUser, "Why a watch?"}};
    input.options.reasoning_effort = "low";
  } else if (name == "assistant_history") {
    Message assistant{MessageRole::kAssistant, "4"};
    assistant.reasoning_content = "Two plus two is four.";
    input.messages = {{MessageRole::kUser, "2+2?"}, assistant,
                      {MessageRole::kUser, "And plus 3?"}};
    input.options.enable_thinking = false;
  } else if (name == "tools_and_result") {
    Message assistant{MessageRole::kAssistant, ""};
    assistant.reasoning_content = "I should check.";
    assistant.tool_calls = {
        {"weather", {{"city", "Bern"}, {"unit", "C"}}}};
    input.messages = {{MessageRole::kUser, "Weather in Bern?"}, assistant,
                      {MessageRole::kTool, "{\"temperature\":18}"}};
    input.canonical_tool_json = {
        R"({"function": {"description": "Get current weather", "name": "weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"})"};
  } else if (name == "no_messages") {
  } else if (name == "late_system") {
    input.messages = {{MessageRole::kUser, "Hi"},
                      {MessageRole::kSystem, "Too late"}};
  } else if (name == "invalid_reasoning_effort") {
    input.messages = {{MessageRole::kUser, "Hi"}};
    input.options.reasoning_effort = "maximum";
  } else if (name == "image_rejected_in_v1") {
    Message message{MessageRole::kUser, ""};
    message.has_unsupported_content = true;
    input.messages = {message};
  } else if (name == "leading_developer_mapped_to_system") {
    input.messages = {{MessageRole::kDeveloper, "Follow API policy."},
                      {MessageRole::kUser, "Hi"}};
    input.options.enable_thinking = false;
  } else {
    std::cerr << "unknown template fixture\n";
    return 2;
  }
  std::string rendered;
  const qw38::Status status = qw38::internal::render_chat(input, &rendered);
  if (!status.is_ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  std::cout << rendered;
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--build-info") {
    std::cout << "brand=" << kBrand << '\n';
    std::cout << "cxx=17\n";
    std::cout << "cuda_target=sm_120\n";
    std::cout << "model_revision=" << kModelRevision << '\n';
    std::cout << "model_sha256=" << kModelSha256 << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--inspect-gguf") {
    qw38::internal::ModelInfo info;
    const qw38::Status status = qw38::internal::inspect_gguf(argv[2], &info);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "version=" << info.gguf_version << '\n';
    std::cout << "architecture=" << info.architecture << '\n';
    std::cout << "name=" << info.name << '\n';
    std::cout << "metadata=" << info.metadata_count << '\n';
    std::cout << "tensors=" << info.tensors.size() << '\n';
    std::cout << "data_offset=" << info.data_offset << '\n';
    std::cout << "block_count=" << info.block_count << '\n';
    std::cout << "context_length=" << info.context_length << '\n';
    std::cout << "embedding_length=" << info.embedding_length << '\n';
    std::cout << "query_heads=" << info.query_heads << '\n';
    std::cout << "kv_heads=" << info.kv_heads << '\n';
    std::cout << "rope_dimensions=" << info.rope_dimensions << '\n';
    std::cout << "tokenizer_model=" << info.tokenizer_model << '\n';
    std::cout << "tokenizer_tokens=" << info.tokenizer_tokens.size() << '\n';
    std::cout << "tokenizer_merges=" << info.tokenizer_merges.size() << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--sha256") {
    std::string digest;
    const qw38::Status status = qw38::internal::sha256_file(argv[2], &digest);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << digest << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--verify-model") {
    qw38::Engine engine;
    const qw38::Status status = qw38::Engine::open(argv[2], &engine);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "verified=pinned-qwen3.8-27b-q4_k_m\n";
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--check-contract") {
    qw38::internal::ModelInfo info;
    qw38::Status status = qw38::internal::inspect_gguf(argv[2], &info);
    if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "contract=qwen3.8-27b-q4_k_m\n";
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "--inventory-gguf") {
    return write_inventory(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--tokenize-hex") {
    return tokenize_hex(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--render-template-case") {
    return render_template_case(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-quant") {
    return check_quant(argv[2], argv[3]);
  }
  std::cout << kBrand << '\n';
  std::cerr << "qw38-eval: use --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "--verify-model PATH, --check-contract PATH, or "
               "--inventory-gguf PATH OUTPUT, --tokenize-hex PATH HEX, or "
               "--render-template-case NAME, or --check-quant KIND HEX\n";
  return 2;
}
