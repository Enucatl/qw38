#include <algorithm>
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
#include "conversion.h"
#include "gdn.h"
#include "mixer.h"
#include "model.h"
#include "projection.h"
#include "quant.h"
#include "scheduler.h"
#include "scalar_runtime.h"
#include "sha256.h"
#include "template.h"
#include "tensor.h"
#include "tokenizer.h"
#include "weights.h"

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

bool parse_size(const char* text, std::size_t* value) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (*end != '\0' || parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  *value = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_float_hex(const std::string& hex, std::vector<float>* values) {
  std::vector<std::uint8_t> bytes;
  if (!parse_hex(hex, &bytes) || bytes.size() % 4 != 0) return false;
  values->resize(bytes.size() / 4);
  for (std::size_t index = 0; index < values->size(); ++index) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[index * 4]) |
        (static_cast<std::uint32_t>(bytes[index * 4 + 1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[index * 4 + 2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[index * 4 + 3]) << 24U);
    std::memcpy(values->data() + index, &bits, sizeof(bits));
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

template <std::size_t Size>
void write_float_array(const char* name, const std::array<float, Size>& values) {
  std::cout << name << '=';
  for (float value : values) write_float_hex(value);
  std::cout << '\n';
}

int check_conversion(const std::string& component) {
  if (component == "permutation") {
    constexpr std::array<float, 12> kGrouped{
        0.0F, 1.0F, 10.0F, 11.0F, 20.0F, 21.0F,
        100.0F, 101.0F, 110.0F, 111.0F, 120.0F, 121.0F};
    std::array<float, kGrouped.size()> tiled{};
    std::array<float, kGrouped.size()> roundtrip{};
    qw38::Status status = qw38::internal::gdn_grouped_to_tiled(
        kGrouped.data(), 2, 3, 2, tiled.data(), tiled.size());
    if (status.is_ok()) {
      status = qw38::internal::gdn_tiled_to_grouped(
          tiled.data(), 2, 3, 2, roundtrip.data(), roundtrip.size());
    }
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("grouped_f32_le_hex", kGrouped);
    write_float_array("tiled_f32_le_hex", tiled);
    write_float_array("roundtrip_f32_le_hex", roundtrip);
    return 0;
  }
  if (component == "gates") {
    constexpr std::array<float, 4> kA{-0.75F, -0.125F, 0.5F, 1.25F};
    constexpr std::array<float, 4> kB{-1.0F, -0.25F, 0.25F, 1.0F};
    constexpr std::array<float, 4> kALog{-1.5F, -0.5F, 0.0F, 0.75F};
    constexpr std::array<float, 4> kDt{-0.25F, 0.0F, 0.25F, 0.5F};
    std::array<float, 4> folded_a{};
    std::array<float, 4> source_decay{};
    std::array<float, 4> source_beta{};
    std::array<float, 4> gguf_decay{};
    std::array<float, 4> gguf_beta{};
    for (std::size_t index = 0; index < folded_a.size(); ++index) {
      folded_a[index] = -std::exp(kALog[index]);
    }
    qw38::Status status = qw38::internal::gdn_gates_from_source(
        kA.data(), kB.data(), kALog.data(), kDt.data(), kA.size(),
        source_decay.data(), source_beta.data());
    if (status.is_ok()) {
      status = qw38::internal::gdn_gates_from_gguf(
          kA.data(), kB.data(), folded_a.data(), kDt.data(), kA.size(),
          gguf_decay.data(), gguf_beta.data());
    }
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("folded_a_f32_le_hex", folded_a);
    write_float_array("source_decay_f32_le_hex", source_decay);
    write_float_array("gguf_decay_f32_le_hex", gguf_decay);
    write_float_array("source_beta_f32_le_hex", source_beta);
    write_float_array("gguf_beta_f32_le_hex", gguf_beta);
    return 0;
  }
  if (component == "norm") {
    constexpr std::array<float, 4> kInput{-2.0F, -0.5F, 1.0F, 3.0F};
    constexpr std::array<float, 4> kSourceWeight{-0.125F, 0.0F, 0.25F,
                                                 0.5F};
    std::array<float, 4> gguf_scale{};
    std::array<float, 4> source_output{};
    std::array<float, 4> gguf_output{};
    for (std::size_t index = 0; index < gguf_scale.size(); ++index) {
      gguf_scale[index] = 1.0F + kSourceWeight[index];
    }
    qw38::Status status = qw38::internal::rms_norm(
        kInput.data(), kSourceWeight.data(), kInput.size(),
        source_output.data());
    if (status.is_ok()) {
      status = qw38::internal::rms_norm_scale(
          kInput.data(), gguf_scale.data(), kInput.size(), gguf_output.data());
    }
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("gguf_scale_f32_le_hex", gguf_scale);
    write_float_array("source_output_f32_le_hex", source_output);
    write_float_array("gguf_output_f32_le_hex", gguf_output);
    return 0;
  }
  if (component == "invalid_folded_a") {
    float value = 0.0F;
    float result = 0.0F;
    const qw38::Status status = qw38::internal::gdn_gates_from_gguf(
        &value, &value, &value, &value, 1, &result, &result);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  std::cerr << "invalid_argument: unknown conversion component\n";
  return 1;
}

int check_weight_binding(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  auto find = [&info](const std::string& name) {
    return std::find_if(info.tensors.begin(), info.tensors.end(),
                        [&name](const qw38::internal::TensorInfo& tensor) {
                          return tensor.name == name;
                        });
  };
  if (mode == "missing") {
    find("blk.0.ssm_a")->name = "blk.0.missing_ssm_a";
  } else if (mode == "role") {
    find("blk.0.ssm_a")->semantic_role = "gdn_dt_bias";
  } else if (mode == "shape") {
    find("blk.0.ssm_a")->dimensions[0] = 47;
  } else if (mode == "range") {
    find("output.weight")->offset = mapping.size();
  } else if (mode != "valid") {
    std::cerr << "invalid_argument: unknown weight-binding mode\n";
    return 1;
  }
  qw38::internal::ModelWeights weights;
  status = qw38::internal::bind_model_weights(info, mapping, &weights);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::size_t gdn_layers = 0;
  std::size_t attention_layers = 0;
  for (const qw38::internal::LayerWeights& layer : weights.layers) {
    if (layer.kind == qw38::internal::LayerKind::kGdn) {
      ++gdn_layers;
    } else {
      ++attention_layers;
    }
  }
  std::cout << "bound_tensors=" << weights.bound_tensor_count << '\n';
  std::cout << "gdn_layers=" << gdn_layers << '\n';
  std::cout << "attention_layers=" << attention_layers << '\n';
  std::cout << "embedding_columns=" << weights.token_embedding.columns << '\n';
  std::cout << "embedding_rows=" << weights.token_embedding.rows << '\n';
  std::cout << "final_norm_values=" << weights.output_norm.count << '\n';
  std::cout << "logit_rows=" << weights.output.rows << '\n';
  std::vector<float> final_norm(weights.output_norm.count);
  status = qw38::internal::vector_decode(weights.output_norm, final_norm.data(),
                                         final_norm.size());
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  write_float_vector("final_norm_endpoints_f32_le_hex",
                     {final_norm.front(), final_norm.back()});
  return 0;
}

int check_projection_layout(const std::string& component) {
  if (component == "gdn") {
    constexpr std::array<float, 12> kPacked{0.0F,  1.0F,  2.0F,  3.0F,
                                            10.0F, 11.0F, 12.0F, 13.0F,
                                            20.0F, 21.0F, 22.0F, 23.0F};
    std::array<float, 4> query{};
    std::array<float, 4> key{};
    std::array<float, 4> value{};
    const qw38::Status status = qw38::internal::split_gdn_qkv(
        kPacked.data(), kPacked.size(), 2, 2, 2, 2, query.data(), query.size(),
        key.data(), key.size(), value.data(), value.size());
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("query_f32_le_hex", query);
    write_float_array("key_f32_le_hex", key);
    write_float_array("value_f32_le_hex", value);
    return 0;
  }
  if (component == "attention") {
    constexpr std::array<float, 12> kPacked{
        0.0F, 1.0F, 10.0F, 11.0F, 100.0F, 101.0F,
        110.0F, 111.0F, 200.0F, 201.0F, 210.0F, 211.0F};
    std::array<float, 6> query{};
    std::array<float, 6> gate{};
    const qw38::Status status = qw38::internal::split_attention_query_gate(
        kPacked.data(), kPacked.size(), 3, 2, query.data(), query.size(),
        gate.data(), gate.size());
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("query_f32_le_hex", query);
    write_float_array("gate_f32_le_hex", gate);
    return 0;
  }
  if (component == "invalid_alias") {
    std::array<float, 4> packed{};
    std::array<float, 2> gate{};
    const qw38::Status status = qw38::internal::split_attention_query_gate(
        packed.data(), packed.size(), 1, 2, packed.data(), 2, gate.data(),
        gate.size());
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  if (component == "invalid_count") {
    std::array<float, 4> packed{};
    std::array<float, 2> query{};
    std::array<float, 2> gate{};
    const qw38::Status status = qw38::internal::split_attention_query_gate(
        packed.data(), packed.size() - 1, 1, 2, query.data(), query.size(),
        gate.data(), gate.size());
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  std::cerr << "invalid_argument: unknown projection-layout component\n";
  return 1;
}

int check_mixer_projections(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> activation(qw38::internal::kResidualWidth);
  for (std::size_t index = 0; index < activation.size(); ++index) {
    activation[index] = static_cast<float>(
                            static_cast<int>((index * 37) % 101) - 50) /
                        32.0F;
  }
  std::vector<float> gdn_packed(qw38::internal::kGdnPackedQkvWidth, NAN);
  std::vector<float> gdn_value_gate(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> gdn_alpha(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> gdn_beta(qw38::internal::kGdnGateCount, NAN);
  qw38::internal::GdnProjectionWorkspace gdn_workspace{
      gdn_packed.data(), gdn_packed.size(), gdn_value_gate.data(),
      gdn_value_gate.size(), gdn_alpha.data(), gdn_alpha.size(),
      gdn_beta.data(), gdn_beta.size()};
  if (mode == "invalid_workspace") {
    --gdn_workspace.packed_qkv_count;
    status = qw38::internal::project_gdn_mixer(
        weights.layers[0].gdn, activation.data(), activation.size(),
        gdn_workspace);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  if (mode != "valid") {
    std::cerr << "invalid_argument: unknown mixer-projection mode\n";
    return 1;
  }
  status = qw38::internal::project_gdn_mixer(
      weights.layers[0].gdn, activation.data(), activation.size(),
      gdn_workspace);

  std::vector<float> attention_packed(
      qw38::internal::kAttentionPackedQueryGateWidth, NAN);
  std::vector<float> attention_query(qw38::internal::kAttentionQueryWidth, NAN);
  std::vector<float> attention_gate(qw38::internal::kAttentionQueryWidth, NAN);
  std::vector<float> attention_key(qw38::internal::kAttentionKvWidth, NAN);
  std::vector<float> attention_value(qw38::internal::kAttentionKvWidth, NAN);
  const qw38::internal::AttentionProjectionWorkspace attention_workspace{
      attention_packed.data(), attention_packed.size(), attention_query.data(),
      attention_query.size(), attention_gate.data(), attention_gate.size(),
      attention_key.data(), attention_key.size(), attention_value.data(),
      attention_value.size()};
  if (status.is_ok()) {
    status = qw38::internal::project_attention_mixer(
        weights.layers[3].attention, activation.data(), activation.size(),
        attention_workspace);
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (status.is_ok() &&
      (!finite(gdn_packed) || !finite(gdn_value_gate) || !finite(gdn_alpha) ||
       !finite(gdn_beta) || !finite(attention_packed) ||
       !finite(attention_query) || !finite(attention_gate) ||
       !finite(attention_key) || !finite(attention_value))) {
    status = {qw38::StatusCode::kInternal,
              "mixer projection left a nonfinite workspace value"};
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  write_float_vector("blk.0.attn_qkv.weight", {gdn_packed[0], gdn_packed[2047],
                                                gdn_packed[2048],
                                                gdn_packed[4095],
                                                gdn_packed[4096],
                                                gdn_packed[10239]});
  write_float_vector("blk.0.attn_gate.weight",
                     {gdn_value_gate.front(), gdn_value_gate.back()});
  write_float_vector("blk.0.ssm_alpha.weight",
                     {gdn_alpha.front(), gdn_alpha.back()});
  write_float_vector("blk.0.ssm_beta.weight",
                     {gdn_beta.front(), gdn_beta.back()});
  write_float_vector("blk.3.attn_q.weight",
                     {attention_packed[0], attention_packed[255],
                      attention_packed[256], attention_packed[511],
                      attention_packed[512], attention_packed[767],
                      attention_packed[12287]});
  write_float_vector("blk.3.attn_k.weight",
                     {attention_key.front(), attention_key.back()});
  write_float_vector("blk.3.attn_v.weight",
                     {attention_value.front(), attention_value.back()});
  write_float_vector("attention_query_split_f32_le_hex",
                     {attention_query[0], attention_query[255],
                      attention_query[256], attention_query[511]});
  write_float_vector("attention_gate_split_f32_le_hex",
                     {attention_gate[0], attention_gate[255],
                      attention_gate.back()});
  std::cout << "computed_values="
            << gdn_packed.size() + gdn_value_gate.size() + gdn_alpha.size() +
                   gdn_beta.size() + attention_packed.size() +
                   attention_query.size() + attention_gate.size() +
                   attention_key.size() + attention_value.size()
            << '\n';
  return 0;
}

int check_real_gdn_step(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> input_norm(qw38::internal::kResidualWidth);
  std::vector<float> convolution(qw38::internal::kGdnConvolutionValues);
  std::vector<float> folded_a(qw38::internal::kGdnGateCount);
  std::vector<float> dt_bias(qw38::internal::kGdnGateCount);
  std::vector<float> recurrent_norm(qw38::internal::kGdnHeadWidth);
  const qw38::internal::GdnScalarParameters parameters{
      input_norm.data(), input_norm.size(), convolution.data(),
      convolution.size(), folded_a.data(), folded_a.size(), dt_bias.data(),
      dt_bias.size(), recurrent_norm.data(), recurrent_norm.size()};
  std::vector<float> ffn_norm(qw38::internal::kResidualWidth);
  const qw38::internal::FfnScalarParameters ffn_parameters{ffn_norm.data(),
                                                           ffn_norm.size()};
  const bool layer_mode = mode == "layer" || mode == "layer_invalid_ffn";
  if (layer_mode) {
    const qw38::internal::GdnLayerScalarParameters layer_parameters{
        parameters, ffn_parameters};
    status = qw38::internal::prepare_gdn_layer_scalar_parameters(
        weights.layers[0], layer_parameters);
  } else {
    status = qw38::internal::prepare_gdn_scalar_parameters(weights.layers[0],
                                                           parameters);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> residual(qw38::internal::kResidualWidth);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = static_cast<float>(
                          static_cast<int>((index * 37) % 101) - 50) /
                      32.0F;
  }
  std::vector<float> conv_state(qw38::internal::kGdnConvolutionValues);
  std::vector<float> recurrent_state(
      qw38::internal::kGdnRecurrentStateValues);
  qw38::internal::GdnLayerStateView state{
      conv_state.data(), conv_state.size(), recurrent_state.data(),
      recurrent_state.size()};

  std::vector<float> normalized(qw38::internal::kResidualWidth, NAN);
  std::vector<float> packed(qw38::internal::kGdnPackedQkvWidth, NAN);
  std::vector<float> projected_gate(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> projected_alpha(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> projected_beta(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> convolved(qw38::internal::kGdnPackedQkvWidth, NAN);
  std::vector<float> query(qw38::internal::kGdnKeyWidth, NAN);
  std::vector<float> key(qw38::internal::kGdnKeyWidth, NAN);
  std::vector<float> value_tiled(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> value_grouped(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> gate_grouped(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> alpha_grouped(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> beta_grouped(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> folded_grouped(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> dt_grouped(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> log_decay(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> update_gate(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> recurrent_output(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> gated_grouped(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> gated_tiled(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> mixer_output(qw38::internal::kResidualWidth, NAN);
  std::vector<float> post_mixer(qw38::internal::kResidualWidth, NAN);
  std::vector<float> ffn_normalized(qw38::internal::kResidualWidth, NAN);
  std::vector<float> ffn_gate(qw38::internal::kFfnWidth, NAN);
  std::vector<float> ffn_up(qw38::internal::kFfnWidth, NAN);
  std::vector<float> ffn_activated(qw38::internal::kFfnWidth, NAN);
  std::vector<float> ffn_correction(qw38::internal::kResidualWidth, NAN);
  std::vector<float> output(qw38::internal::kResidualWidth, NAN);
  const qw38::internal::GdnProjectionWorkspace projection_workspace{
      packed.data(), packed.size(), projected_gate.data(), projected_gate.size(),
      projected_alpha.data(), projected_alpha.size(), projected_beta.data(),
      projected_beta.size()};
  qw38::internal::GdnStepWorkspace workspace{
      normalized.data(), normalized.size(), projection_workspace,
      convolved.data(), convolved.size(), query.data(), query.size(), key.data(),
      key.size(), value_tiled.data(), value_tiled.size(), value_grouped.data(),
      value_grouped.size(), gate_grouped.data(), gate_grouped.size(),
      alpha_grouped.data(), beta_grouped.data(), folded_grouped.data(),
      dt_grouped.data(), log_decay.data(), update_gate.data(),
      update_gate.size(), recurrent_output.data(), recurrent_output.size(),
      gated_grouped.data(), gated_grouped.size(), gated_tiled.data(),
      gated_tiled.size(), mixer_output.data(), mixer_output.size()};
  qw38::internal::FfnStepWorkspace ffn_workspace{
      ffn_normalized.data(), ffn_normalized.size(), ffn_gate.data(),
      ffn_gate.size(), ffn_up.data(), ffn_up.size(), ffn_activated.data(),
      ffn_activated.size(), ffn_correction.data(), ffn_correction.size()};
  if (mode == "invalid_workspace") --workspace.gated_tiled_count;
  if (mode == "layer_invalid_ffn") --ffn_workspace.activated_count;
  if (mode != "valid" && mode != "invalid_workspace" && !layer_mode) {
    std::cerr << "invalid_argument: unknown real-GDN-step mode\n";
    return 1;
  }
  if (layer_mode) {
    const qw38::internal::GdnLayerScalarParameters layer_parameters{
        parameters, ffn_parameters};
    const qw38::internal::GdnLayerWorkspace layer_workspace{
        workspace, ffn_workspace, post_mixer.data(), post_mixer.size()};
    status = qw38::internal::execute_gdn_layer_step(
        weights.layers[0], layer_parameters, residual.data(), residual.size(),
        state, layer_workspace, output.data(), output.size());
  } else {
    status = qw38::internal::execute_gdn_mixer_step(
        weights.layers[0], parameters, residual.data(), residual.size(), state,
        workspace, output.data(), output.size());
  }
  if (!status.is_ok()) {
    if (mode == "invalid_workspace" || mode == "layer_invalid_ffn") {
      const bool unchanged =
          std::all_of(conv_state.begin(), conv_state.end(),
                      [](float value) { return value == 0.0F; }) &&
          std::all_of(recurrent_state.begin(), recurrent_state.end(),
                      [](float value) { return value == 0.0F; });
      std::cout << "state_unchanged=" << (unchanged ? 1 : 0) << '\n';
      const bool output_untouched = std::all_of(
          output.begin(), output.end(),
          [](float value) { return std::isnan(value); });
      std::cout << "output_untouched=" << (output_untouched ? 1 : 0) << '\n';
    }
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (layer_mode &&
      (!finite(post_mixer) || !finite(ffn_normalized) || !finite(ffn_gate) ||
       !finite(ffn_up) || !finite(ffn_activated) || !finite(ffn_correction) ||
       !finite(output))) {
    std::cerr << "internal: GDN layer left a nonfinite value\n";
    return 1;
  }
  const auto taps = [](const std::vector<float>& values,
                       std::initializer_list<std::size_t> indices) {
    std::vector<float> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices) selected.push_back(values[index]);
    return selected;
  };
  write_float_vector("normalized_f32_le_hex",
                     taps(normalized, {0, 1, 5118, 5119}));
  write_float_vector("convolved_f32_le_hex",
                     taps(convolved, {0, 127, 2048, 2175, 4096, 4223, 6144,
                                      6271, 4224, 4351}));
  write_float_vector("value_grouped_f32_le_hex",
                     taps(value_grouped, {0, 127, 128, 255, 384, 511}));
  write_float_vector("gate_controls_f32_le_hex",
                     taps(alpha_grouped, {0, 1, 3, 47}));
  write_float_vector("log_decay_f32_le_hex",
                     taps(log_decay, {0, 1, 3, 47}));
  write_float_vector("update_gate_f32_le_hex",
                     taps(update_gate, {0, 1, 3, 47}));
  write_float_vector("recurrent_output_f32_le_hex",
                     taps(recurrent_output, {0, 127, 128, 255, 384, 511}));
  write_float_vector("gated_grouped_f32_le_hex",
                     taps(gated_grouped, {0, 127, 128, 255, 384, 511}));
  write_float_vector("mixer_output_f32_le_hex",
                     taps(mixer_output, {0, 1, 2559, 5119}));
  write_float_vector("residual_output_f32_le_hex",
                     taps(output, {0, 1, 2559, 5119}));
  if (layer_mode) {
    write_float_vector("post_mixer_f32_le_hex",
                       taps(post_mixer, {0, 1, 2559, 5119}));
    write_float_vector("ffn_normalized_f32_le_hex",
                       taps(ffn_normalized, {0, 1, 2559, 5119}));
    write_float_vector("ffn_gate_f32_le_hex",
                       taps(ffn_gate, {0, 1, 8703, 8704, 17407}));
    write_float_vector("ffn_up_f32_le_hex",
                       taps(ffn_up, {0, 1, 8703, 8704, 17407}));
    write_float_vector("ffn_activated_f32_le_hex",
                       taps(ffn_activated, {0, 1, 8703, 8704, 17407}));
    write_float_vector("ffn_correction_f32_le_hex",
                       taps(ffn_correction, {0, 1, 2559, 5119}));
  }
  write_float_vector("convolution_state_f32_le_hex",
                     taps(conv_state, {0, 1, 2, 3, 16384, 16385, 16386,
                                       16387}));
  write_float_vector("recurrent_state_f32_le_hex",
                     taps(recurrent_state, {0, 127, 16383, 16384, 32767,
                                            49152, 65535}));
  return 0;
}

int check_real_ffn_step(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> norm(qw38::internal::kResidualWidth);
  const qw38::internal::FfnScalarParameters parameters{norm.data(), norm.size()};
  status = qw38::internal::prepare_ffn_scalar_parameters(
      weights.layers[0].common, parameters);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::vector<float> residual(qw38::internal::kResidualWidth);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = static_cast<float>(
                          static_cast<int>((index * 37) % 101) - 50) /
                      32.0F;
  }
  std::vector<float> normalized(qw38::internal::kResidualWidth, NAN);
  std::vector<float> gate(qw38::internal::kFfnWidth, NAN);
  std::vector<float> up(qw38::internal::kFfnWidth, NAN);
  std::vector<float> activated(qw38::internal::kFfnWidth, NAN);
  std::vector<float> correction(qw38::internal::kResidualWidth, NAN);
  std::vector<float> output(qw38::internal::kResidualWidth, NAN);
  qw38::internal::FfnStepWorkspace workspace{
      normalized.data(), normalized.size(), gate.data(), gate.size(), up.data(),
      up.size(), activated.data(), activated.size(), correction.data(),
      correction.size()};
  if (mode == "invalid_workspace") --workspace.activated_count;
  if (mode != "valid" && mode != "invalid_workspace") {
    std::cerr << "invalid_argument: unknown real-FFN-step mode\n";
    return 1;
  }
  status = qw38::internal::execute_ffn_step(
      weights.layers[0].common, parameters, residual.data(), residual.size(),
      workspace, output.data(), output.size());
  if (!status.is_ok()) {
    const bool untouched =
        std::all_of(gate.begin(), gate.end(),
                    [](float value) { return std::isnan(value); }) &&
        std::all_of(output.begin(), output.end(),
                    [](float value) { return std::isnan(value); });
    std::cout << "outputs_untouched=" << (untouched ? 1 : 0) << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(normalized) || !finite(gate) || !finite(up) ||
      !finite(activated) || !finite(correction) || !finite(output)) {
    std::cerr << "internal: FFN left a nonfinite workspace value\n";
    return 1;
  }
  const auto taps = [](const std::vector<float>& values,
                       std::initializer_list<std::size_t> indices) {
    std::vector<float> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices) selected.push_back(values[index]);
    return selected;
  };
  const std::initializer_list<std::size_t> kWideTaps{0, 1, 8703, 8704, 17407};
  const std::initializer_list<std::size_t> kResidualTaps{0, 1, 2559, 5119};
  write_float_vector("normalized_f32_le_hex",
                     taps(normalized, kResidualTaps));
  write_float_vector("gate_f32_le_hex", taps(gate, kWideTaps));
  write_float_vector("up_f32_le_hex", taps(up, kWideTaps));
  write_float_vector("activated_f32_le_hex", taps(activated, kWideTaps));
  write_float_vector("correction_f32_le_hex", taps(correction, kResidualTaps));
  write_float_vector("residual_output_f32_le_hex", taps(output, kResidualTaps));
  std::cout << "workspace_values="
            << normalized.size() + gate.size() + up.size() + activated.size() +
                   correction.size()
            << '\n';
  return 0;
}

int check_real_attention_step(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  constexpr std::size_t kCapacity = 2;
  std::vector<float> input_norm(qw38::internal::kResidualWidth);
  std::vector<float> query_norm(qw38::internal::kAttentionHeadWidth);
  std::vector<float> key_norm(qw38::internal::kAttentionHeadWidth);
  const qw38::internal::AttentionScalarParameters parameters{
      input_norm.data(), input_norm.size(), query_norm.data(),
      query_norm.size(), key_norm.data(), key_norm.size()};
  std::vector<float> ffn_norm(qw38::internal::kResidualWidth);
  const qw38::internal::FfnScalarParameters ffn_parameters{ffn_norm.data(),
                                                           ffn_norm.size()};
  const bool layer_mode = mode == "layer" || mode == "layer_invalid_ffn";
  if (layer_mode) {
    const qw38::internal::AttentionLayerScalarParameters layer_parameters{
        parameters, ffn_parameters};
    status = qw38::internal::prepare_attention_layer_scalar_parameters(
        weights.layers[3], layer_parameters);
  } else {
    status = qw38::internal::prepare_attention_scalar_parameters(
        weights.layers[3], parameters);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> normalized(qw38::internal::kResidualWidth, NAN);
  std::vector<float> packed_query_gate(
      qw38::internal::kAttentionPackedQueryGateWidth, NAN);
  std::vector<float> query(qw38::internal::kAttentionQueryWidth, NAN);
  std::vector<float> gate(qw38::internal::kAttentionQueryWidth, NAN);
  std::vector<float> key(qw38::internal::kAttentionKvWidth, NAN);
  std::vector<float> value(qw38::internal::kAttentionKvWidth, NAN);
  std::vector<float> attention_output(qw38::internal::kAttentionQueryWidth,
                                      NAN);
  std::vector<float> scores(kCapacity, NAN);
  std::vector<float> mixer_output(qw38::internal::kResidualWidth, NAN);
  std::vector<float> post_mixer(qw38::internal::kResidualWidth, NAN);
  std::vector<float> ffn_normalized(qw38::internal::kResidualWidth, NAN);
  std::vector<float> ffn_gate(qw38::internal::kFfnWidth, NAN);
  std::vector<float> ffn_up(qw38::internal::kFfnWidth, NAN);
  std::vector<float> ffn_activated(qw38::internal::kFfnWidth, NAN);
  std::vector<float> ffn_correction(qw38::internal::kResidualWidth, NAN);
  std::vector<float> key_cache(kCapacity * qw38::internal::kAttentionKvWidth,
                               NAN);
  std::vector<float> value_cache(
      kCapacity * qw38::internal::kAttentionKvWidth, NAN);
  std::vector<float> output(qw38::internal::kResidualWidth, NAN);
  std::vector<float> first_output(qw38::internal::kResidualWidth, NAN);
  qw38::internal::AttentionStepWorkspace workspace{
      normalized.data(),
      normalized.size(),
      {packed_query_gate.data(), packed_query_gate.size(), query.data(),
       query.size(), gate.data(), gate.size(), key.data(), key.size(),
       value.data(), value.size()},
      attention_output.data(),
      attention_output.size(),
      scores.data(),
      scores.size(),
      mixer_output.data(),
      mixer_output.size()};
  qw38::internal::FfnStepWorkspace ffn_workspace{
      ffn_normalized.data(), ffn_normalized.size(), ffn_gate.data(),
      ffn_gate.size(), ffn_up.data(), ffn_up.size(), ffn_activated.data(),
      ffn_activated.size(), ffn_correction.data(), ffn_correction.size()};
  const qw38::internal::AttentionLayerStateView state{
      key_cache.data(), key_cache.size(), value_cache.data(),
      value_cache.size(), kCapacity};
  if (mode == "invalid_workspace") --workspace.attention_output_count;
  if (mode == "layer_invalid_ffn") --ffn_workspace.activated_count;
  if (mode != "valid" && mode != "invalid_workspace" &&
      mode != "capacity" && !layer_mode) {
    std::cerr << "invalid_argument: unknown real-attention-step mode\n";
    return 1;
  }

  const auto fill_residual = [](std::vector<float>* residual,
                                std::size_t token) {
    for (std::size_t index = 0; index < residual->size(); ++index) {
      (*residual)[index] = static_cast<float>(
                               static_cast<int>((index * 37 + token * 17) %
                                                101) -
                               50) /
                           32.0F;
    }
  };
  std::vector<float> residual(qw38::internal::kResidualWidth);
  fill_residual(&residual, 0);
  const std::size_t first_position = mode == "capacity" ? kCapacity : 0;
  const qw38::internal::AttentionLayerScalarParameters layer_parameters{
      parameters, ffn_parameters};
  const qw38::internal::AttentionLayerWorkspace layer_workspace{
      workspace, ffn_workspace, post_mixer.data(), post_mixer.size()};
  if (layer_mode) {
    status = qw38::internal::execute_attention_layer_step(
        weights.layers[3], layer_parameters, first_position, residual.data(),
        residual.size(), state, layer_workspace, output.data(), output.size());
  } else {
    status = qw38::internal::execute_attention_mixer_step(
        weights.layers[3], parameters, first_position, residual.data(),
        residual.size(), state, workspace, output.data(), output.size());
  }
  if (!status.is_ok()) {
    const bool state_unchanged =
        std::all_of(key_cache.begin(), key_cache.end(),
                    [](float item) { return std::isnan(item); }) &&
        std::all_of(value_cache.begin(), value_cache.end(),
                    [](float item) { return std::isnan(item); });
    const bool output_untouched = std::all_of(
        output.begin(), output.end(),
        [](float item) { return std::isnan(item); });
    std::cout << "state_unchanged=" << (state_unchanged ? 1 : 0) << '\n';
    std::cout << "output_untouched=" << (output_untouched ? 1 : 0) << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  first_output = output;
  fill_residual(&residual, 1);
  if (layer_mode) {
    status = qw38::internal::execute_attention_layer_step(
        weights.layers[3], layer_parameters, 1, residual.data(),
        residual.size(), state, layer_workspace, output.data(), output.size());
  } else {
    status = qw38::internal::execute_attention_mixer_step(
        weights.layers[3], parameters, 1, residual.data(), residual.size(),
        state, workspace, output.data(), output.size());
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float item) { return std::isfinite(item); });
  };
  if (!finite(normalized) || !finite(packed_query_gate) || !finite(query) ||
      !finite(gate) || !finite(key) || !finite(value) ||
      !finite(attention_output) || !finite(mixer_output) ||
      !finite(key_cache) || !finite(value_cache) || !finite(first_output) ||
      !finite(output) ||
      (layer_mode &&
       (!finite(post_mixer) || !finite(ffn_normalized) || !finite(ffn_gate) ||
        !finite(ffn_up) || !finite(ffn_activated) ||
        !finite(ffn_correction)))) {
    std::cerr << "internal: attention left a nonfinite value\n";
    return 1;
  }
  const auto taps = [](const std::vector<float>& values,
                       std::initializer_list<std::size_t> indices) {
    std::vector<float> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices) selected.push_back(values[index]);
    return selected;
  };
  const std::initializer_list<std::size_t> kResidualTaps{0, 1, 2559, 5119};
  const std::initializer_list<std::size_t> kHeadTaps{
      0, 63, 64, 255, 5 * 256, 6 * 256, 23 * 256 + 255};
  const std::initializer_list<std::size_t> kCacheTaps{
      0, 31, 32, 63, 64, 255, 256, 511,
      1024, 1024 + 31, 1024 + 32, 1024 + 63, 1024 + 64, 1024 + 255,
      1024 + 256, 1024 + 511};
  write_float_vector("normalized_f32_le_hex",
                     taps(normalized, kResidualTaps));
  write_float_vector("query_f32_le_hex", taps(query, kHeadTaps));
  write_float_vector("gate_f32_le_hex", taps(gate, kHeadTaps));
  write_float_vector("key_cache_f32_le_hex", taps(key_cache, kCacheTaps));
  write_float_vector("value_cache_f32_le_hex", taps(value_cache, kCacheTaps));
  write_float_vector("attention_output_f32_le_hex",
                     taps(attention_output, kHeadTaps));
  write_float_vector("first_output_f32_le_hex",
                     taps(first_output, kResidualTaps));
  write_float_vector("mixer_output_f32_le_hex",
                     taps(mixer_output, kResidualTaps));
  write_float_vector("residual_output_f32_le_hex",
                     taps(output, kResidualTaps));
  if (layer_mode) {
    write_float_vector("post_mixer_f32_le_hex",
                       taps(post_mixer, kResidualTaps));
    write_float_vector("ffn_normalized_f32_le_hex",
                       taps(ffn_normalized, kResidualTaps));
    write_float_vector("ffn_gate_f32_le_hex",
                       taps(ffn_gate, {0, 1, 8703, 8704, 17407}));
    write_float_vector("ffn_up_f32_le_hex",
                       taps(ffn_up, {0, 1, 8703, 8704, 17407}));
    write_float_vector("ffn_activated_f32_le_hex",
                       taps(ffn_activated, {0, 1, 8703, 8704, 17407}));
    write_float_vector("ffn_correction_f32_le_hex",
                       taps(ffn_correction, kResidualTaps));
  }
  std::cout << "kv_values=" << key_cache.size() + value_cache.size() << '\n';
  return 0;
}

int check_real_model_boundaries(const char* model_path,
                                const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  if (mode != "valid" && mode != "invalid_token" &&
      mode != "invalid_workspace") {
    std::cerr << "invalid_argument: unknown real-model-boundary mode\n";
    return 1;
  }

  std::vector<float> embedding_zero(qw38::internal::kResidualWidth, NAN);
  std::vector<float> embedding(qw38::internal::kResidualWidth, NAN);
  std::vector<float> embedding_last(qw38::internal::kResidualWidth, NAN);
  if (mode == "invalid_token") {
    status = qw38::internal::embed_token(
        weights, qw38::internal::kVocabularySize, embedding.data(),
        embedding.size());
    const bool untouched = std::all_of(
        embedding.begin(), embedding.end(),
        [](float value) { return std::isnan(value); });
    std::cout << "embedding_untouched=" << (untouched ? 1 : 0) << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  status = qw38::internal::embed_token(weights, 0, embedding_zero.data(),
                                       embedding_zero.size());
  if (status.is_ok()) {
    status = qw38::internal::embed_token(weights, 42, embedding.data(),
                                         embedding.size());
  }
  if (status.is_ok()) {
    status = qw38::internal::embed_token(
        weights, qw38::internal::kVocabularySize - 1, embedding_last.data(),
        embedding_last.size());
  }

  std::vector<float> output_norm(qw38::internal::kResidualWidth);
  const qw38::internal::OutputScalarParameters parameters{output_norm.data(),
                                                           output_norm.size()};
  if (status.is_ok()) {
    status =
        qw38::internal::prepare_output_scalar_parameters(weights, parameters);
  }
  std::vector<float> normalized(qw38::internal::kResidualWidth, NAN);
  std::vector<float> logits(qw38::internal::kVocabularySize, NAN);
  qw38::internal::OutputWorkspace workspace{normalized.data(),
                                            normalized.size()};
  if (mode == "invalid_workspace") --workspace.normalized_count;
  if (status.is_ok()) {
    status = qw38::internal::project_logits(
        weights, parameters, embedding.data(), embedding.size(), workspace,
        logits.data(), logits.size());
  }
  if (!status.is_ok()) {
    if (mode == "invalid_workspace") {
      const bool normalized_untouched = std::all_of(
          normalized.begin(), normalized.end(),
          [](float value) { return std::isnan(value); });
      const bool logits_untouched = std::all_of(
          logits.begin(), logits.end(),
          [](float value) { return std::isnan(value); });
      std::cout << "normalized_untouched=" << (normalized_untouched ? 1 : 0)
                << '\n';
      std::cout << "logits_untouched=" << (logits_untouched ? 1 : 0) << '\n';
    }
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(embedding_zero) || !finite(embedding) ||
      !finite(embedding_last) || !finite(normalized) || !finite(logits)) {
    std::cerr << "internal: model boundary left a nonfinite value\n";
    return 1;
  }
  const auto taps = [](const std::vector<float>& values,
                       std::initializer_list<std::size_t> indices) {
    std::vector<float> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices) selected.push_back(values[index]);
    return selected;
  };
  const std::initializer_list<std::size_t> kHiddenTaps{0, 1, 2559, 5119};
  const std::initializer_list<std::size_t> kLogitTaps{0, 1, 42, 1000, 248319};
  write_float_vector("embedding_0_f32_le_hex",
                     taps(embedding_zero, kHiddenTaps));
  write_float_vector("embedding_42_f32_le_hex",
                     taps(embedding, kHiddenTaps));
  write_float_vector("embedding_last_f32_le_hex",
                     taps(embedding_last, kHiddenTaps));
  write_float_vector("final_normalized_f32_le_hex",
                     taps(normalized, kHiddenTaps));
  write_float_vector("logits_f32_le_hex", taps(logits, kLogitTaps));
  const auto greedy = std::max_element(logits.begin(), logits.end());
  std::cout << "greedy_token=" << std::distance(logits.begin(), greedy) << '\n';
  std::cout << "logit_count=" << logits.size() << '\n';
  return 0;
}

int check_real_scalar_token(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  if (mode != "valid" && mode != "invalid_workspace") {
    std::cerr << "invalid_argument: unknown real-scalar-token mode\n";
    return 1;
  }

  qw38::internal::ScalarModelParameters parameters;
  status = qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  qw38::internal::ScalarSessionState state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(1, &state);
  }
  qw38::internal::ScalarWorkspace workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(1, &workspace);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize, NAN);
  if (mode == "invalid_workspace") workspace.ffn_activated.pop_back();
  status = qw38::internal::execute_scalar_token(
      weights, parameters, 42, &state, &workspace, logits.data(), logits.size());
  if (!status.is_ok()) {
    const bool state_unchanged =
        std::all_of(state.gdn_convolution.begin(),
                    state.gdn_convolution.end(),
                    [](float value) { return value == 0.0F; }) &&
        std::all_of(state.gdn_recurrent.begin(), state.gdn_recurrent.end(),
                    [](float value) { return value == 0.0F; }) &&
        std::all_of(state.attention_key.begin(), state.attention_key.end(),
                    [](float value) { return value == 0.0F; }) &&
        std::all_of(state.attention_value.begin(), state.attention_value.end(),
                    [](float value) { return value == 0.0F; });
    const bool logits_untouched = std::all_of(
        logits.begin(), logits.end(),
        [](float value) { return std::isnan(value); });
    std::cout << "state_unchanged=" << (state_unchanged ? 1 : 0) << '\n';
    std::cout << "logits_untouched=" << (logits_untouched ? 1 : 0) << '\n';
    std::cout << "frontier=" << state.frontier << '\n';
    std::cout << "layers_completed=" << workspace.layers_completed << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(workspace.activation_a) || !finite(workspace.final_normalized) ||
      !finite(logits) || !finite(state.gdn_convolution) ||
      !finite(state.gdn_recurrent) || !finite(state.attention_key) ||
      !finite(state.attention_value)) {
    std::cerr << "internal: scalar token execution left a nonfinite value\n";
    return 1;
  }
  std::size_t gdn_mutated = 0;
  std::size_t attention_mutated = 0;
  for (std::size_t layer = 0; layer < qw38::internal::kModelLayerCount;
       ++layer) {
    if (layer % 4 == 3) {
      const auto& view = state.attention[layer];
      const bool changed =
          std::any_of(view.key_cache, view.key_cache + view.key_cache_count,
                      [](float value) { return value != 0.0F; }) ||
          std::any_of(view.value_cache,
                      view.value_cache + view.value_cache_count,
                      [](float value) { return value != 0.0F; });
      if (changed) ++attention_mutated;
    } else {
      const auto& view = state.gdn[layer];
      const bool changed =
          std::any_of(view.convolution,
                      view.convolution + view.convolution_count,
                      [](float value) { return value != 0.0F; }) ||
          std::any_of(view.recurrent, view.recurrent + view.recurrent_count,
                      [](float value) { return value != 0.0F; });
      if (changed) ++gdn_mutated;
    }
  }
  const auto taps = [](const std::vector<float>& values,
                       std::initializer_list<std::size_t> indices) {
    std::vector<float> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices) selected.push_back(values[index]);
    return selected;
  };
  const std::initializer_list<std::size_t> kHiddenTaps{0, 1, 2559, 5119};
  write_float_vector("final_hidden_f32_le_hex",
                     taps(workspace.activation_a, kHiddenTaps));
  write_float_vector("final_normalized_f32_le_hex",
                     taps(workspace.final_normalized, kHiddenTaps));
  write_float_vector("logits_f32_le_hex",
                     taps(logits, {0, 1, 42, 1000, 248319}));
  write_float_vector(
      "gdn_state_f32_le_hex",
      {state.gdn[0].convolution[3], state.gdn[0].recurrent[0],
       state.gdn[1].convolution[3], state.gdn[1].recurrent[0],
       state.gdn[62].convolution[3], state.gdn[62].recurrent[0]});
  write_float_vector(
      "attention_state_f32_le_hex",
      {state.attention[3].key_cache[0], state.attention[3].value_cache[0],
       state.attention[7].key_cache[0], state.attention[7].value_cache[0],
       state.attention[63].key_cache[0], state.attention[63].value_cache[0]});
  const auto greedy = std::max_element(logits.begin(), logits.end());
  std::cout << "greedy_token=" << std::distance(logits.begin(), greedy) << '\n';
  std::cout << "gdn_slots_mutated=" << gdn_mutated << '\n';
  std::cout << "attention_slots_mutated=" << attention_mutated << '\n';
  std::cout << "frontier=" << state.frontier << '\n';
  std::cout << "layers_completed=" << workspace.layers_completed << '\n';
  std::cout << "logit_count=" << logits.size() << '\n';
  std::cout << "prepared_values="
            << parameters.input_norms.size() + parameters.ffn_norms.size() +
                   parameters.gdn_convolution.size() +
                   parameters.gdn_folded_a.size() +
                   parameters.gdn_dt_bias.size() +
                   parameters.gdn_recurrent_norm.size() +
                   parameters.attention_query_norm.size() +
                   parameters.attention_key_norm.size() +
                   parameters.output_norm.size()
            << '\n';
  std::cout << "state_values="
            << state.gdn_convolution.size() + state.gdn_recurrent.size() +
                   state.attention_key.size() + state.attention_value.size()
            << '\n';
  std::cout << "workspace_values="
            << workspace.activation_a.size() + workspace.activation_b.size() +
                   workspace.post_mixer.size() +
                   workspace.gdn_normalized.size() +
                   workspace.gdn_packed.size() +
                   workspace.gdn_projected_gate.size() +
                   workspace.gdn_projected_alpha.size() +
                   workspace.gdn_projected_beta.size() +
                   workspace.gdn_convolved.size() +
                   workspace.gdn_query.size() + workspace.gdn_key.size() +
                   workspace.gdn_value_tiled.size() +
                   workspace.gdn_value_grouped.size() +
                   workspace.gdn_gate_grouped.size() +
                   workspace.gdn_alpha_grouped.size() +
                   workspace.gdn_beta_grouped.size() +
                   workspace.gdn_folded_grouped.size() +
                   workspace.gdn_dt_grouped.size() +
                   workspace.gdn_log_decay.size() +
                   workspace.gdn_update_gate.size() +
                   workspace.gdn_recurrent_output.size() +
                   workspace.gdn_gated_grouped.size() +
                   workspace.gdn_gated_tiled.size() +
                   workspace.gdn_mixer_output.size() +
                   workspace.attention_normalized.size() +
                   workspace.attention_packed.size() +
                   workspace.attention_query.size() +
                   workspace.attention_gate.size() +
                   workspace.attention_key.size() +
                   workspace.attention_value.size() +
                   workspace.attention_output.size() +
                   workspace.attention_scores.size() +
                   workspace.attention_mixer_output.size() +
                   workspace.ffn_normalized.size() +
                   workspace.ffn_gate.size() + workspace.ffn_up.size() +
                   workspace.ffn_activated.size() +
                   workspace.ffn_correction.size() +
                   workspace.final_normalized.size()
            << '\n';
  return 0;
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

int check_matvec(const std::string& kind, const char* columns_text,
                 const char* rows_text, const std::string& payload_hex,
                 const std::string& activation_hex) {
  std::size_t columns = 0;
  std::size_t rows = 0;
  std::uint32_t type = 0;
  if (kind == "f32") {
    type = 0;
  } else if (kind == "q8_0") {
    type = 8;
  } else if (kind == "q4_k") {
    type = 12;
  } else if (kind == "q6_k") {
    type = 14;
  } else {
    std::cerr << "invalid_argument: unknown matrix tensor type\n";
    return 1;
  }
  std::vector<std::uint8_t> payload;
  std::vector<float> activation;
  if (!parse_size(columns_text, &columns) || !parse_size(rows_text, &rows) ||
      !parse_hex(payload_hex, &payload) ||
      !parse_float_hex(activation_hex, &activation)) {
    std::cerr << "invalid_argument: malformed matrix dimensions or hex\n";
    return 1;
  }
  qw38::internal::TensorView view;
  qw38::Status status = qw38::internal::make_tensor_view(
      payload.data(), payload.size(), type, columns, rows, &view);
  std::vector<float> output(rows);
  std::vector<float> decoded(columns);
  if (status.is_ok()) {
    status = qw38::internal::tensor_matvec(
        view, activation.data(), activation.size(), output.data(), output.size());
  }
  if (status.is_ok()) {
    status = qw38::internal::tensor_row_decode(
        view, 0, decoded.data(), decoded.size());
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "columns=" << view.columns << "\nrows=" << view.rows
            << "\nrow_bytes=" << view.row_bytes << '\n';
  write_float_vector("output_f32_le_hex", output);
  write_float_vector("row0_f32_le_hex", decoded);
  return 0;
}

int check_tensor_row(const char* model_path, const std::string& name,
                     const char* row_text) {
  std::size_t row = 0;
  if (!parse_size(row_text, &row)) {
    std::cerr << "invalid_argument: malformed tensor row index\n";
    return 1;
  }
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::TensorView view;
  if (status.is_ok()) {
    status = qw38::internal::bind_tensor_view(info, mapping, name, &view);
  }
  std::vector<float> activation;
  if (status.is_ok()) {
    activation.resize(view.columns);
    for (std::size_t index = 0; index < activation.size(); ++index) {
      const int numerator = static_cast<int>((index * 37) % 101) - 50;
      activation[index] = static_cast<float>(numerator) / 32.0F;
    }
  }
  float dot = 0.0F;
  if (status.is_ok()) {
    status = qw38::internal::tensor_row_dot(
        view, row, activation.data(), activation.size(), &dot);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "dtype=" << qw38::internal::ggml_type_name(view.type)
            << "\ncolumns=" << view.columns << "\nrows=" << view.rows
            << "\nrow_bytes=" << view.row_bytes << "\ndot_f32_le_hex=";
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
  if (argc == 4 && std::string(argv[1]) == "--check-gdn") {
    return check_gdn(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-conversion") {
    return check_conversion(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-weight-binding") {
    return check_weight_binding(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-projection-layout") {
    return check_projection_layout(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-mixer-projections") {
    return check_mixer_projections(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-gdn-step") {
    return check_real_gdn_step(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-ffn-step") {
    return check_real_ffn_step(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-attention-step") {
    return check_real_attention_step(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-model-boundaries") {
    return check_real_model_boundaries(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-scalar-token") {
    return check_real_scalar_token(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-attention") {
    return check_attention(std::atoi(argv[2]));
  }
  if (argc == 3 && std::string(argv[1]) == "--check-ffn") {
    return check_ffn(std::atoi(argv[2]));
  }
  if (argc == 7 && std::string(argv[1]) == "--check-matvec") {
    return check_matvec(argv[2], argv[3], argv[4], argv[5], argv[6]);
  }
  if (argc == 5 && std::string(argv[1]) == "--check-tensor-row") {
    return check_tensor_row(argv[2], argv[3], argv[4]);
  }
  std::cout << kBrand << '\n';
  std::cerr << "qw38-eval: use --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "--verify-model PATH, --check-contract PATH, or "
               "--inventory-gguf PATH OUTPUT, --tokenize-hex PATH HEX, or "
               "--render-template-case NAME, --check-quant KIND HEX, or "
               "--check-gdn COMPONENT CHUNKING, --check-attention LAYER, or "
               "--check-conversion COMPONENT, "
               "--check-weight-binding MODEL MODE, "
               "--check-projection-layout COMPONENT, "
               "--check-mixer-projections MODEL MODE, "
               "--check-real-gdn-step MODEL MODE, "
               "--check-real-ffn-step MODEL MODE, "
               "--check-real-attention-step MODEL MODE, "
               "--check-real-model-boundaries MODEL MODE, "
               "--check-real-scalar-token MODEL MODE, "
               "--check-ffn LAYER, --check-matvec KIND COLUMNS ROWS PAYLOAD "
               "ACTIVATION, or --check-tensor-row MODEL NAME ROW\n";
  return 2;
}
