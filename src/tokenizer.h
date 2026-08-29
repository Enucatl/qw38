#ifndef QW38_TOKENIZER_H_
#define QW38_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "qw38/status.h"

namespace qw38::internal {

class Tokenizer final {
 public:
  Status build(const ModelInfo& model) noexcept;
  Status encode(const std::string& utf8, std::vector<std::uint32_t>* ids) const noexcept;

 private:
  Status encode_ordinary(const std::string& utf8,
                         std::vector<std::uint32_t>* ids) const noexcept;
  Status encode_piece(const std::string& bytes,
                      std::vector<std::uint32_t>* ids) const noexcept;

  std::unordered_map<std::string, std::uint32_t> vocabulary_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  std::vector<std::pair<std::string, std::uint32_t>> special_tokens_;
  std::vector<std::string> byte_tokens_;
};

}  // namespace qw38::internal

#endif  // QW38_TOKENIZER_H_
