#ifndef QW38_MODEL_H_
#define QW38_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "qw38/status.h"

namespace qw38::internal {

struct TensorInfo final {
  std::string name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
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
};

Status inspect_gguf(const std::string& path, ModelInfo* info) noexcept;
Status validate_qwen38_contract(const ModelInfo& info) noexcept;

}  // namespace qw38::internal

#endif  // QW38_MODEL_H_
