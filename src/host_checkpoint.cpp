#include "host_checkpoint.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace qw38::internal {
namespace {

constexpr char kMagic[8] = {'Q', 'W', '3', '8', 'C', 'P', 'U', '1'};
constexpr std::uint32_t kVersion = 1;

void write_u32(std::ostream* output, std::uint32_t value) {
  std::array<unsigned char, 4> bytes{};
  for (std::uint32_t index = 0; index < 4; ++index) {
    bytes[index] = static_cast<unsigned char>(value >> (index * 8U));
  }
  output->write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void write_u64(std::ostream* output, std::uint64_t value) {
  std::array<unsigned char, 8> bytes{};
  for (std::uint32_t index = 0; index < 8; ++index) {
    bytes[index] = static_cast<unsigned char>(value >> (index * 8U));
  }
  output->write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

bool read_u32(std::istream* input, std::uint32_t* value) {
  std::array<unsigned char, 4> bytes{};
  input->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!*input) return false;
  *value = 0;
  for (std::uint32_t index = 0; index < 4; ++index) {
    *value |= static_cast<std::uint32_t>(bytes[index]) << (index * 8U);
  }
  return true;
}

bool read_u64(std::istream* input, std::uint64_t* value) {
  std::array<unsigned char, 8> bytes{};
  input->read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!*input) return false;
  *value = 0;
  for (std::uint32_t index = 0; index < 8; ++index) {
    *value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  }
  return true;
}

}  // namespace

Status save_host_checkpoint(const std::string& path,
                            const ModelGeometry& geometry,
                            const ScalarSessionState& state,
                            const std::vector<Token>& tokens,
                            const std::vector<float>& logits,
                            const HostSamplerState& sampler) noexcept {
  if (path.empty() || !geometry_is_valid(geometry) ||
      tokens.size() != state.frontier ||
      logits.size() != geometry.vocabulary) {
    return {StatusCode::kInvalidArgument,
            "host checkpoint path or payload is invalid"};
  }
  const std::filesystem::path destination(path);
  const std::filesystem::path staging = destination.string() + ".tmp";
  std::error_code error;
  std::filesystem::remove(staging, error);
  {
    std::ofstream output(staging, std::ios::binary | std::ios::trunc);
    if (!output) {
      return {StatusCode::kIoError, "cannot create host checkpoint staging file"};
    }
    output.write(kMagic, sizeof(kMagic));
    write_u32(&output, kVersion);
    write_u32(&output, geometry.identity);
    write_u64(&output, geometry.layer_count);
    write_u64(&output, state.capacity);
    write_u64(&output, state.frontier);
    write_u64(&output, tokens.size());
    output.write(reinterpret_cast<const char*>(tokens.data()),
                 static_cast<std::streamsize>(tokens.size() * sizeof(Token)));
    write_u64(&output, state.gdn_convolution.size());
    output.write(reinterpret_cast<const char*>(state.gdn_convolution.data()),
                 static_cast<std::streamsize>(state.gdn_convolution.size() *
                                              sizeof(float)));
    write_u64(&output, state.gdn_recurrent.size());
    output.write(reinterpret_cast<const char*>(state.gdn_recurrent.data()),
                 static_cast<std::streamsize>(state.gdn_recurrent.size() *
                                              sizeof(float)));
    write_u64(&output, state.attention_key.size());
    output.write(reinterpret_cast<const char*>(state.attention_key.data()),
                 static_cast<std::streamsize>(state.attention_key.size() *
                                              sizeof(float)));
    write_u64(&output, state.attention_value.size());
    output.write(reinterpret_cast<const char*>(state.attention_value.data()),
                 static_cast<std::streamsize>(state.attention_value.size() *
                                              sizeof(float)));
    output.write(reinterpret_cast<const char*>(logits.data()),
                 static_cast<std::streamsize>(logits.size() * sizeof(float)));
    output.write(reinterpret_cast<const char*>(&sampler),
                 static_cast<std::streamsize>(sizeof(sampler)));
    if (!output) {
      return {StatusCode::kIoError, "failed while writing host checkpoint"};
    }
  }
  std::filesystem::rename(staging, destination, error);
  if (error) {
    std::filesystem::remove(staging, error);
    return {StatusCode::kIoError, "cannot publish host checkpoint atomically"};
  }
  return Status::ok();
}

Status restore_host_checkpoint(const std::string& path,
                               const ModelGeometry& geometry,
                               ScalarSessionState* state,
                               std::vector<Token>* tokens,
                               std::vector<float>* logits,
                               HostSamplerState* sampler) noexcept {
  if (path.empty() || state == nullptr || tokens == nullptr ||
      logits == nullptr || sampler == nullptr || !geometry_is_valid(geometry)) {
    return {StatusCode::kInvalidArgument,
            "host checkpoint restore arguments are invalid"};
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {StatusCode::kIoError, "cannot open host checkpoint"};
  }
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  std::uint32_t version = 0;
  std::uint32_t identity = 0;
  std::uint64_t layers = 0;
  std::uint64_t capacity = 0;
  std::uint64_t frontier = 0;
  std::uint64_t token_count = 0;
  if (!input || std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0 ||
      !read_u32(&input, &version) || version != kVersion ||
      !read_u32(&input, &identity) || identity != geometry.identity ||
      !read_u64(&input, &layers) || layers != geometry.layer_count ||
      !read_u64(&input, &capacity) || capacity != state->capacity ||
      !read_u64(&input, &frontier) || frontier > state->capacity ||
      !read_u64(&input, &token_count) || token_count != frontier) {
    return {StatusCode::kIncompatibleArtifact,
            "host checkpoint header does not match the open model"};
  }
  std::vector<Token> restored_tokens(static_cast<std::size_t>(token_count));
  if (token_count != 0) {
    input.read(reinterpret_cast<char*>(restored_tokens.data()),
               static_cast<std::streamsize>(token_count * sizeof(Token)));
  }
  auto read_floats = [&input](std::vector<float>* destination,
                              std::size_t expected) -> bool {
    std::uint64_t count = 0;
    if (!read_u64(&input, &count) || count != expected) return false;
    input.read(reinterpret_cast<char*>(destination->data()),
               static_cast<std::streamsize>(expected * sizeof(float)));
    return static_cast<bool>(input);
  };
  if (!read_floats(&state->gdn_convolution, state->gdn_convolution.size()) ||
      !read_floats(&state->gdn_recurrent, state->gdn_recurrent.size()) ||
      !read_floats(&state->attention_key, state->attention_key.size()) ||
      !read_floats(&state->attention_value, state->attention_value.size())) {
    return {StatusCode::kIncompatibleArtifact,
            "host checkpoint state payload is truncated or mismatched"};
  }
  std::vector<float> restored_logits(geometry.vocabulary);
  input.read(reinterpret_cast<char*>(restored_logits.data()),
             static_cast<std::streamsize>(restored_logits.size() * sizeof(float)));
  HostSamplerState restored_sampler{};
  input.read(reinterpret_cast<char*>(&restored_sampler),
             static_cast<std::streamsize>(sizeof(restored_sampler)));
  if (!input) {
    return {StatusCode::kIoError, "failed while reading host checkpoint"};
  }
  state->frontier = static_cast<std::size_t>(frontier);
  *tokens = std::move(restored_tokens);
  *logits = std::move(restored_logits);
  *sampler = restored_sampler;
  return Status::ok();
}

}  // namespace qw38::internal
