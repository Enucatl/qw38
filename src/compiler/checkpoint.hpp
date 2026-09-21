#pragma once

#include "compiler/error.hpp"
#include "compiler/identity.hpp"
#include "format/schema.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw38::compiler {

struct ArchitectureConfig {
  std::string architecture;
  std::string model_type;
  bool tie_word_embeddings{true};
  bool mtp_use_dedicated_embeddings{true};
  std::uint64_t hidden_size{};
  std::uint64_t intermediate_size{};
  std::uint64_t vocab_size{};
  std::uint32_t num_hidden_layers{};
  std::uint32_t num_attention_heads{};
  std::uint32_t num_key_value_heads{};
  std::uint64_t head_dim{};
  std::uint32_t full_attention_interval{};
  std::uint64_t linear_conv_kernel_dim{};
  std::uint64_t linear_key_head_dim{};
  std::uint64_t linear_value_head_dim{};
  std::uint64_t linear_num_key_heads{};
  std::uint64_t linear_num_value_heads{};
  double rope_theta{};
  double partial_rotary_factor{};
  std::vector<std::uint32_t> mrope_section;
  bool mrope_interleaved{};
  std::string dtype;
  std::string mamba_ssm_dtype;
  std::uint32_t mtp_num_hidden_layers{};
  std::vector<std::string> layer_types;
};

struct ShardTensor {
  SourceTensor source;
  std::uint64_t header_bytes{};  // JSON header length
};

struct OpenShard {
  std::filesystem::path path;
  std::uint64_t header_bytes{};
  std::uint64_t payload_base{};  // 8 + header_bytes
  std::uint64_t file_size{};
};

[[nodiscard]] std::expected<ArchitectureConfig, CompilerError> parse_text_config(
    std::string_view json_text);

[[nodiscard]] std::expected<void, CompilerError> validate_architecture(
    ArchitectureConfig const& cfg);

[[nodiscard]] std::expected<std::string, CompilerError> read_text_file(
    std::filesystem::path const& path, std::string_view field);

[[nodiscard]] std::expected<qw38::format::Hash256, CompilerError> hash_file(
    std::filesystem::path const& path, std::string_view field);

class MappedShard {
 public:
  MappedShard() = default;
  MappedShard(MappedShard&&) noexcept;
  MappedShard& operator=(MappedShard&&) noexcept;
  ~MappedShard();

  MappedShard(MappedShard const&) = delete;
  MappedShard& operator=(MappedShard const&) = delete;

  [[nodiscard]] static std::expected<MappedShard, CompilerError> open(
      std::filesystem::path const& path);

  [[nodiscard]] std::span<std::byte const> bytes() const noexcept;
  [[nodiscard]] std::expected<std::span<std::byte const>, CompilerError>
  tensor_bytes(SourceTensor const& tensor) const;

 private:
  int fd_{-1};
  void* addr_{nullptr};
  std::size_t size_{0};
};

struct Checkpoint {
  std::filesystem::path root;
  ArchitectureConfig config{};
  qw38::format::Hash256 source_hash{};
  qw38::format::Hash256 config_hash{};
  qw38::format::Hash256 tokenizer_hash{};
  ClassifiedCheckpoint classified{};
  std::unordered_map<std::string, OpenShard> shards;
};

[[nodiscard]] std::expected<Checkpoint, CompilerError> open_checkpoint(
    std::filesystem::path const& root);

}  // namespace qw38::compiler
