#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "qw38/engine.h"
#include "attention.h"
#include "gdn.h"
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
  const std::size_t value_count = kind == "q8_0"
                                      ? qw38::internal::kQ80BlockValues
                                      : qw38::internal::kQuantBlockValues;
  std::vector<float> decoded(value_count);
  std::vector<float> activation(value_count);
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
  } else if (kind == "q8_0") {
    status = qw38::internal::decode_q8_0(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q8_0(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else {
    std::cerr << "invalid_argument: quant kind must be q4_k, q6_k, or q8_0\n";
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

void write_float_vector(const char* name, const std::vector<float>& values) {
  std::cout << name << '=';
  for (float value : values) write_float_hex(value);
  std::cout << '\n';
}

float fixture_query(std::size_t token, std::size_t head, std::size_t lane) {
  const int numerator =
      static_cast<int>((token * 11 + head * 7 + lane * 3) % 19) - 9;
  return static_cast<float>(numerator) / 8.0F;
}

float fixture_key(std::size_t token, std::size_t head, std::size_t lane) {
  const int numerator =
      static_cast<int>((token * 13 + head * 5 + lane * 7) % 23) - 11;
  return static_cast<float>(numerator) / 8.0F;
}

float fixture_value(std::size_t token, std::size_t head, std::size_t lane) {
  const int numerator =
      static_cast<int>((token * 17 + head * 3 + lane * 5) % 29) - 14;
  return static_cast<float>(numerator) / 8.0F;
}

int check_gdn_recurrent(const std::string& chunking) {
  constexpr qw38::internal::GdnShape kShape{2, 6, 3, 2};
  constexpr std::size_t kTokens = 5;
  std::vector<std::size_t> chunks;
  if (chunking == "whole") {
    chunks = {5};
  } else if (chunking == "mixed") {
    chunks = {2, 1, 2};
  } else if (chunking == "token") {
    chunks = {1, 1, 1, 1, 1};
  } else {
    std::cerr << "invalid_argument: unknown GDN chunking\n";
    return 1;
  }
  std::vector<float> state(kShape.value_heads * kShape.key_width *
                           kShape.value_width);
  for (std::size_t head = 0; head < kShape.value_heads; ++head) {
    for (std::size_t key_lane = 0; key_lane < kShape.key_width; ++key_lane) {
      for (std::size_t value_lane = 0; value_lane < kShape.value_width;
           ++value_lane) {
        const int numerator = static_cast<int>(
                                  (head * 17 + key_lane * 5 + value_lane * 3) %
                                  13) -
                              6;
        state[head * kShape.key_width * kShape.value_width +
              key_lane * kShape.value_width + value_lane] =
            static_cast<float>(numerator) / 32.0F;
      }
    }
  }
  std::vector<float> all_output;
  std::vector<float> all_log_decay;
  std::vector<float> all_beta;
  std::size_t token = 0;
  for (std::size_t chunk : chunks) {
    for (std::size_t offset = 0; offset < chunk; ++offset, ++token) {
      std::array<float, 6> query{};
      std::array<float, 6> key{};
      std::array<float, 12> value{};
      std::array<float, 6> a{};
      std::array<float, 6> b{};
      std::array<float, 6> a_log{};
      std::array<float, 6> dt_bias{};
      std::array<float, 12> output{};
      std::array<float, 6> log_decay{};
      std::array<float, 6> beta{};
      for (std::size_t head = 0; head < kShape.key_heads; ++head) {
        for (std::size_t lane = 0; lane < kShape.key_width; ++lane) {
          query[head * kShape.key_width + lane] =
              fixture_query(token, head, lane);
          key[head * kShape.key_width + lane] =
              fixture_key(token, head, lane);
        }
      }
      for (std::size_t head = 0; head < kShape.value_heads; ++head) {
        for (std::size_t lane = 0; lane < kShape.value_width; ++lane) {
          value[head * kShape.value_width + lane] =
              fixture_value(token, head, lane);
        }
        a[head] = static_cast<float>(
                      static_cast<int>((token * 5 + head * 3) % 13) - 6) /
                  8.0F;
        b[head] = static_cast<float>(
                      static_cast<int>((token * 7 + head * 2) % 11) - 5) /
                  8.0F;
        a_log[head] = std::log(0.25F * static_cast<float>(head + 1));
        dt_bias[head] =
            static_cast<float>(static_cast<int>(head) - 2) / 8.0F;
      }
      const qw38::Status status = qw38::internal::gdn_recurrent_step(
          kShape, query.data(), query.size(), key.data(), key.size(),
          value.data(), value.size(), a.data(), b.data(), a_log.data(),
          dt_bias.data(), a.size(), state.data(), state.size(), output.data(),
          output.size(), log_decay.data(), beta.data());
      if (!status.is_ok()) {
        std::cerr << qw38::status_code_name(status.code()) << ": "
                  << status.message() << '\n';
        return 1;
      }
      all_output.insert(all_output.end(), output.begin(), output.end());
      all_log_decay.insert(all_log_decay.end(), log_decay.begin(),
                           log_decay.end());
      all_beta.insert(all_beta.end(), beta.begin(), beta.end());
    }
  }
  if (token != kTokens) {
    std::cerr << "internal: GDN fixture chunking has the wrong token count\n";
    return 1;
  }
  write_float_vector("output_f32_le_hex", all_output);
  write_float_vector("final_state_f32_le_hex", state);
  write_float_vector("log_decay_f32_le_hex", all_log_decay);
  write_float_vector("beta_f32_le_hex", all_beta);
  return 0;
}

int check_gdn_convolution(const std::string& chunking) {
  constexpr std::size_t kChannels = 3;
  constexpr std::size_t kWidth = 4;
  std::vector<std::size_t> chunks;
  if (chunking == "whole") {
    chunks = {6};
  } else if (chunking == "mixed") {
    chunks = {1, 2, 3};
  } else if (chunking == "token") {
    chunks = {1, 1, 1, 1, 1, 1};
  } else {
    std::cerr << "invalid_argument: unknown GDN chunking\n";
    return 1;
  }
  std::array<float, kChannels * kWidth> weights{};
  std::array<float, kChannels * kWidth> state{};
  for (std::size_t channel = 0; channel < kChannels; ++channel) {
    for (std::size_t index = 0; index < kWidth; ++index) {
      const int numerator =
          static_cast<int>((channel * 11 + index * 3) % 13) - 6;
      weights[channel * kWidth + index] =
          static_cast<float>(numerator) / 8.0F;
    }
  }
  std::vector<float> all_output;
  std::size_t token = 0;
  for (std::size_t chunk : chunks) {
    for (std::size_t offset = 0; offset < chunk; ++offset, ++token) {
      std::array<float, kChannels> input{};
      std::array<float, kChannels> output{};
      for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const int numerator =
            static_cast<int>((token * 7 + channel * 5) % 17) - 8;
        input[channel] = static_cast<float>(numerator) / 8.0F;
      }
      const qw38::Status status = qw38::internal::causal_depthwise_conv_step(
          kChannels, kWidth, input.data(), input.size(), weights.data(),
          weights.size(), state.data(), state.size(), output.data(),
          output.size());
      if (!status.is_ok()) {
        std::cerr << qw38::status_code_name(status.code()) << ": "
                  << status.message() << '\n';
        return 1;
      }
      all_output.insert(all_output.end(), output.begin(), output.end());
    }
  }
  write_float_vector("output_f32_le_hex", all_output);
  write_float_vector("final_state_f32_le_hex", {state.begin(), state.end()});
  return 0;
}

int check_gdn(const std::string& component, const std::string& chunking) {
  if (component == "recurrent") return check_gdn_recurrent(chunking);
  if (component == "convolution") return check_gdn_convolution(chunking);
  if (component == "invalid_shape") {
    float value = 0.0F;
    const qw38::internal::GdnShape shape{2, 5, 3, 2};
    const qw38::Status status = qw38::internal::gdn_recurrent_step(
        shape, &value, 1, &value, 1, &value, 1, &value, &value, &value,
        &value, 1, &value, 1, &value, 1, &value, &value);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  if (component == "invalid_buffers") {
    float value = 0.0F;
    const qw38::internal::GdnShape shape{1, 1, 1, 1};
    const qw38::Status status = qw38::internal::gdn_recurrent_step(
        shape, &value, 0, &value, 1, &value, 1, &value, &value, &value,
        &value, 1, &value, 1, &value, 1, &value, &value);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  std::cerr << "invalid_argument: GDN component must be recurrent or convolution\n";
  return 1;
}

float attention_fixture_value(int layer, std::size_t token, std::size_t head,
                              std::size_t lane, int salt) {
  const int value =
      (layer * 3 + static_cast<int>(token * 11 + head * 7 + lane * 5) + salt) %
          31 -
      15;
  return static_cast<float>(value) / 8.0F;
}

bool supported_attention_layer(int layer) {
  return layer == 3 || layer == 7 || layer == 63;
}

int check_attention(int layer) {
  if (!supported_attention_layer(layer)) {
    std::cerr << "invalid_argument: attention fixture layer must be 3, 7, or 63\n";
    return 1;
  }
  constexpr qw38::internal::AttentionShape kShape{6, 2, 8, 4, 4};
  constexpr std::size_t kQueryValues = kShape.query_heads * kShape.head_width;
  constexpr std::size_t kKvValues = kShape.kv_heads * kShape.head_width;
  constexpr std::size_t kCacheValues = kShape.capacity * kKvValues;
  std::array<float, kShape.head_width> query_weight{};
  std::array<float, kShape.head_width> key_weight{};
  for (std::size_t lane = 0; lane < kShape.head_width; ++lane) {
    query_weight[lane] =
        static_cast<float>(static_cast<int>(lane) - 3) / 64.0F;
    key_weight[lane] =
        static_cast<float>(3 - static_cast<int>(lane)) / 64.0F;
  }
  std::array<float, kCacheValues> key_cache{};
  std::array<float, kCacheValues> value_cache{};
  for (std::size_t index = 0; index < kCacheValues; ++index) {
    key_cache[index] = 1000.0F + static_cast<float>(index);
    value_cache[index] = -1000.0F - static_cast<float>(index);
  }
  std::vector<float> all_output;
  for (std::size_t token = 0; token < kShape.capacity; ++token) {
    std::array<float, kQueryValues> query{};
    std::array<float, kQueryValues> gate{};
    std::array<float, kKvValues> key{};
    std::array<float, kKvValues> value{};
    std::array<float, kShape.capacity> scores{};
    std::array<float, kQueryValues> output{};
    for (std::size_t head = 0; head < kShape.query_heads; ++head) {
      for (std::size_t lane = 0; lane < kShape.head_width; ++lane) {
        const std::size_t index = head * kShape.head_width + lane;
        query[index] = attention_fixture_value(layer, token, head, lane, 1);
        gate[index] = attention_fixture_value(layer, token, head, lane, 9);
      }
    }
    for (std::size_t head = 0; head < kShape.kv_heads; ++head) {
      for (std::size_t lane = 0; lane < kShape.head_width; ++lane) {
        const std::size_t index = head * kShape.head_width + lane;
        key[index] = attention_fixture_value(layer, token, head, lane, 4);
        value[index] = attention_fixture_value(layer, token, head, lane, 13);
      }
    }
    const qw38::Status status = qw38::internal::attention_decode_step(
        kShape, token, query.data(), query.size(), key.data(), key.size(),
        value.data(), value.size(), query_weight.data(), key_weight.data(),
        gate.data(), gate.size(), key_cache.data(), key_cache.size(),
        value_cache.data(), value_cache.size(), scores.data(), scores.size(),
        output.data(), output.size());
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    all_output.insert(all_output.end(), output.begin(), output.end());
  }
  write_float_vector("output_f32_le_hex", all_output);
  write_float_vector("key_cache_f32_le_hex",
                     {key_cache.begin(), key_cache.end()});
  write_float_vector("value_cache_f32_le_hex",
                     {value_cache.begin(), value_cache.end()});
  return 0;
}

int check_ffn(int layer) {
  if (!supported_attention_layer(layer)) {
    std::cerr << "invalid_argument: FFN fixture layer must be 3, 7, or 63\n";
    return 1;
  }
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kIntermediate = 6;
  std::array<float, kHidden> input{};
  std::array<float, kIntermediate * kHidden> gate_weights{};
  std::array<float, kIntermediate * kHidden> up_weights{};
  std::array<float, kHidden * kIntermediate> down_weights{};
  for (std::size_t lane = 0; lane < kHidden; ++lane) {
    input[lane] = attention_fixture_value(layer, 0, 0, lane, 2);
  }
  for (std::size_t row = 0; row < kIntermediate; ++row) {
    for (std::size_t column = 0; column < kHidden; ++column) {
      gate_weights[row * kHidden + column] =
          attention_fixture_value(layer, row, 0, column, 3) / 4.0F;
      up_weights[row * kHidden + column] =
          attention_fixture_value(layer, row, 0, column, 7) / 4.0F;
    }
  }
  for (std::size_t row = 0; row < kHidden; ++row) {
    for (std::size_t column = 0; column < kIntermediate; ++column) {
      down_weights[row * kIntermediate + column] =
          attention_fixture_value(layer, row, 0, column, 11) / 4.0F;
    }
  }
  std::array<float, kIntermediate> gate{};
  std::array<float, kIntermediate> up{};
  std::array<float, kIntermediate> activated{};
  std::array<float, kHidden> output{};
  const qw38::Status status = qw38::internal::swiglu_ffn(
      input.data(), input.size(), gate_weights.data(), up_weights.data(),
      kIntermediate, down_weights.data(), gate.data(), up.data(),
      activated.data(), output.data());
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  write_float_vector("gate_f32_le_hex", {gate.begin(), gate.end()});
  write_float_vector("up_f32_le_hex", {up.begin(), up.end()});
  write_float_vector("activated_f32_le_hex",
                     {activated.begin(), activated.end()});
  write_float_vector("output_f32_le_hex", {output.begin(), output.end()});
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
  if (argc == 4 && std::string(argv[1]) == "--check-gdn") {
    return check_gdn(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-attention") {
    return check_attention(std::atoi(argv[2]));
  }
  if (argc == 3 && std::string(argv[1]) == "--check-ffn") {
    return check_ffn(std::atoi(argv[2]));
  }
  std::cout << kBrand << '\n';
  std::cerr << "qw38-eval: use --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "--verify-model PATH, --check-contract PATH, or "
               "--inventory-gguf PATH OUTPUT, --tokenize-hex PATH HEX, or "
               "--render-template-case NAME, --check-quant KIND HEX, or "
               "--check-gdn COMPONENT CHUNKING, --check-attention LAYER, or "
               "--check-ffn LAYER\n";
  return 2;
}
