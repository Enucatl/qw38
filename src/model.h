#ifndef QW38_MODEL_H_
#define QW38_MODEL_H_

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "qw38/status.h"

namespace qw38::internal {

struct TensorInfo final {
  std::string name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type = 0;
  std::uint64_t offset = 0;
  std::uint64_t storage_bytes = 0;
  std::uint64_t padded_span_bytes = 0;
  std::string semantic_role;
};

class MappedFile final {
 public:
  MappedFile() noexcept;
  ~MappedFile();
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  Status open(const std::string& path) noexcept;
  const unsigned char* data() const noexcept;
  std::size_t size() const noexcept;

 private:
  void close() noexcept;
  int descriptor_ = -1;
  unsigned char* data_ = nullptr;
  std::size_t size_ = 0;
};

struct ModelInfo final {
  std::uint32_t gguf_version = 0;
  std::uint64_t metadata_count = 0;
  std::string architecture;
  std::string name;
  std::uint64_t alignment = 32;
  std::uint64_t block_count = 0;
  std::uint64_t context_length = 0;
  std::uint64_t embedding_length = 0;
  std::uint64_t query_heads = 0;
  std::uint64_t kv_heads = 0;
  std::uint64_t rope_dimensions = 0;
  std::uint64_t data_offset = 0;
  std::vector<TensorInfo> tensors;
  std::string tokenizer_model;
  std::vector<std::string> tokenizer_tokens;
  std::vector<std::uint32_t> tokenizer_token_types;
  std::vector<std::string> tokenizer_merges;
};

Status inspect_gguf(const std::string& path, ModelInfo* info) noexcept;
Status validate_qwen38_contract(ModelInfo* info) noexcept;
const char* ggml_type_name(std::uint32_t type) noexcept;

}  // namespace qw38::internal

#endif  // QW38_MODEL_H_
