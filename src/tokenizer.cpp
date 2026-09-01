#include "tokenizer.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <utility>

#include "utf8proc.h"

namespace qw38::internal {
namespace {

struct Codepoint final {
  utf8proc_int32_t value;
  std::size_t begin;
  std::size_t end;
};

bool is_letter(utf8proc_int32_t value) noexcept {
  const utf8proc_category_t category = utf8proc_category(value);
  return category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO;
}

bool is_mark(utf8proc_int32_t value) noexcept {
  const utf8proc_category_t category = utf8proc_category(value);
  return category >= UTF8PROC_CATEGORY_MN && category <= UTF8PROC_CATEGORY_ME;
}

bool is_number(utf8proc_int32_t value) noexcept {
  const utf8proc_category_t category = utf8proc_category(value);
  return category >= UTF8PROC_CATEGORY_ND && category <= UTF8PROC_CATEGORY_NO;
}

bool is_space(utf8proc_int32_t value) noexcept {
  const utf8proc_category_t category = utf8proc_category(value);
  return (category >= UTF8PROC_CATEGORY_ZS && category <= UTF8PROC_CATEGORY_ZP) ||
         (value >= 0x09 && value <= 0x0d) || value == 0x85;
}

bool is_line(utf8proc_int32_t value) noexcept {
  return value == '\r' || value == '\n';
}

bool contraction_at(const std::vector<Codepoint>& points, std::size_t index,
                    std::size_t* length) noexcept {
  if (points[index].value != '\'' || index + 1 >= points.size()) return false;
  static constexpr std::array<const char*, 7> suffixes = {"s", "t", "re", "ve",
                                                          "m", "ll", "d"};
  for (const char* suffix : suffixes) {
    std::size_t cursor = index + 1;
    bool matches = true;
    for (const char* character = suffix; *character != '\0'; ++character, ++cursor) {
      if (cursor >= points.size() ||
          utf8proc_tolower(points[cursor].value) != *character) {
        matches = false;
        break;
      }
    }
    if (matches) {
      *length = cursor - index;
      return true;
    }
  }
  return false;
}

Status normalize(const std::string& input, std::string* output) noexcept {
  utf8proc_uint8_t* normalized = nullptr;
  const utf8proc_ssize_t length = utf8proc_map(
      reinterpret_cast<const utf8proc_uint8_t*>(input.data()),
      static_cast<utf8proc_ssize_t>(input.size()), &normalized,
      static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
  if (length < 0) {
    return {StatusCode::kInvalidArgument,
            std::string("invalid UTF-8: ") + utf8proc_errmsg(length)};
  }
  output->assign(reinterpret_cast<const char*>(normalized),
                 static_cast<std::size_t>(length));
  std::free(normalized);
  return Status::ok();
}

Status codepoints(const std::string& input,
                  std::vector<Codepoint>* output) noexcept {
  std::size_t offset = 0;
  while (offset < input.size()) {
    utf8proc_int32_t value = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(input.data() + offset),
        static_cast<utf8proc_ssize_t>(input.size() - offset), &value);
    if (length <= 0) {
      return {StatusCode::kInvalidArgument, "invalid UTF-8 after normalization"};
    }
    output->push_back(
        {value, offset, offset + static_cast<std::size_t>(length)});
    offset += static_cast<std::size_t>(length);
  }
  return Status::ok();
}

std::string pair_key(const std::string& left, const std::string& right) {
  std::string key = left;
  key.push_back('\0');
  key += right;
  return key;
}

}  // namespace

Status Tokenizer::build(const ModelInfo& model) noexcept {
  if (model.tokenizer_tokens.empty() ||
      model.tokenizer_token_types.size() != model.tokenizer_tokens.size()) {
    return {StatusCode::kIncompatibleArtifact, "tokenizer metadata is incomplete"};
  }
  vocabulary_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  byte_tokens_.assign(256, {});
  byte_values_.clear();
  id_tokens_ = model.tokenizer_tokens;
  special_ids_.assign(model.tokenizer_tokens.size(), false);
  for (std::size_t index = 0; index < model.tokenizer_tokens.size(); ++index) {
    if (!vocabulary_.emplace(model.tokenizer_tokens[index], index).second) {
      return {StatusCode::kIncompatibleArtifact, "duplicate tokenizer token"};
    }
    const std::uint32_t type = model.tokenizer_token_types[index];
    if (type == 3 || type == 4) {
      special_tokens_.emplace_back(model.tokenizer_tokens[index], index);
      special_ids_[index] = true;
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& left, const auto& right) {
              return left.first.size() > right.first.size();
            });
  for (std::size_t rank = 0; rank < model.tokenizer_merges.size(); ++rank) {
    const std::string& merge = model.tokenizer_merges[rank];
    const std::size_t separator = merge.find(' ');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= merge.size()) {
      return {StatusCode::kIncompatibleArtifact, "invalid tokenizer merge"};
    }
    merge_ranks_.emplace(
        pair_key(merge.substr(0, separator), merge.substr(separator + 1)), rank);
  }

  std::array<bool, 256> direct{};
  for (std::uint32_t value = 33; value <= 126; ++value) direct[value] = true;
  for (std::uint32_t value = 161; value <= 172; ++value) direct[value] = true;
  for (std::uint32_t value = 174; value <= 255; ++value) direct[value] = true;
  std::uint32_t extra = 0;
  for (std::uint32_t value = 0; value < 256; ++value) {
    const utf8proc_int32_t mapped =
        direct[value] ? static_cast<utf8proc_int32_t>(value)
                      : static_cast<utf8proc_int32_t>(256 + extra++);
    std::array<utf8proc_uint8_t, 4> encoded{};
    const utf8proc_ssize_t length = utf8proc_encode_char(mapped, encoded.data());
    byte_tokens_[value].assign(reinterpret_cast<const char*>(encoded.data()),
                               static_cast<std::size_t>(length));
    byte_values_.emplace(byte_tokens_[value], static_cast<std::uint8_t>(value));
  }
  return Status::ok();
}

Status Tokenizer::encode(const std::string& utf8,
                         std::vector<std::uint32_t>* ids) const noexcept {
  if (ids == nullptr) {
    return {StatusCode::kInvalidArgument, "token output is required"};
  }
  ids->clear();
  std::size_t offset = 0;
  while (offset < utf8.size()) {
    const std::pair<std::string, std::uint32_t>* matched = nullptr;
    std::size_t next_special = utf8.size();
    for (const auto& special : special_tokens_) {
      const std::size_t found = utf8.find(special.first, offset);
      if (found < next_special) {
        next_special = found;
        matched = &special;
      } else if (found == next_special && matched != nullptr &&
                 special.first.size() > matched->first.size()) {
        matched = &special;
      }
    }
    if (next_special > offset) {
      Status status = encode_ordinary(utf8.substr(offset, next_special - offset), ids);
      if (!status.is_ok()) return status;
    }
    if (matched == nullptr) break;
    ids->push_back(matched->second);
    offset = next_special + matched->first.size();
  }
  return Status::ok();
}

Status Tokenizer::encode_ordinary(const std::string& utf8,
                                  std::vector<std::uint32_t>* ids) const noexcept {
  std::string normalized;
  Status status = normalize(utf8, &normalized);
  if (!status.is_ok()) return status;
  std::vector<Codepoint> points;
  status = codepoints(normalized, &points);
  if (!status.is_ok()) return status;
  std::size_t index = 0;
  while (index < points.size()) {
    const std::size_t begin = index;
    std::size_t contraction_length = 0;
    if (contraction_at(points, index, &contraction_length)) {
      index += contraction_length;
    } else {
      std::size_t body = index;
      if (!is_line(points[body].value) && !is_letter(points[body].value) &&
          !is_number(points[body].value) && body + 1 < points.size() &&
          (is_letter(points[body + 1].value) || is_mark(points[body + 1].value))) {
        ++body;
      }
      if (is_letter(points[body].value) || is_mark(points[body].value)) {
        index = body + 1;
        while (index < points.size() &&
               (is_letter(points[index].value) || is_mark(points[index].value))) {
          ++index;
        }
      } else if (is_number(points[index].value)) {
        ++index;
      } else {
        std::size_t symbol = index;
        if (points[symbol].value == ' ' && symbol + 1 < points.size() &&
            !is_space(points[symbol + 1].value) &&
            !is_letter(points[symbol + 1].value) &&
            !is_mark(points[symbol + 1].value) &&
            !is_number(points[symbol + 1].value)) {
          ++symbol;
        }
        if (!is_space(points[symbol].value) && !is_letter(points[symbol].value) &&
            !is_mark(points[symbol].value) && !is_number(points[symbol].value)) {
          index = symbol + 1;
          while (index < points.size() && !is_space(points[index].value) &&
                 !is_letter(points[index].value) && !is_mark(points[index].value) &&
                 !is_number(points[index].value)) {
            ++index;
          }
          while (index < points.size() && is_line(points[index].value)) ++index;
        } else if (is_space(points[index].value)) {
          std::size_t end = index;
          std::size_t last_line = index;
          bool has_line = false;
          while (end < points.size() && is_space(points[end].value)) {
            if (is_line(points[end].value)) {
              has_line = true;
              last_line = end;
            }
            ++end;
          }
          if (has_line) {
            index = last_line + 1;
            while (index < end && is_line(points[index].value)) ++index;
          } else if (end < points.size() && end - index > 1) {
            // `\s+(?!\S)` backtracks one character before non-whitespace. The
            // remaining ASCII space is then consumed by the optional prefix of
            // the following word/symbol alternative.
            index = end - 1;
          } else {
            index = end;
          }
        } else {
          ++index;
        }
      }
    }
    const std::size_t byte_begin = points[begin].begin;
    const std::size_t byte_end = points[index - 1].end;
    status = encode_piece(normalized.substr(byte_begin, byte_end - byte_begin), ids);
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

Status Tokenizer::encode_piece(const std::string& bytes,
                               std::vector<std::uint32_t>* ids) const noexcept {
  std::vector<std::string> symbols;
  symbols.reserve(bytes.size());
  for (unsigned char byte : bytes) symbols.push_back(byte_tokens_[byte]);
  while (symbols.size() > 1) {
    std::size_t best = symbols.size();
    std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
      const auto rank = merge_ranks_.find(pair_key(symbols[index], symbols[index + 1]));
      if (rank != merge_ranks_.end() && rank->second < best_rank) {
        best = index;
        best_rank = rank->second;
      }
    }
    if (best == symbols.size()) break;
    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (std::size_t index = 0; index < symbols.size();) {
      if (index + 1 < symbols.size() && symbols[index] == left &&
          symbols[index + 1] == right) {
        merged.push_back(left + right);
        index += 2;
      } else {
        merged.push_back(std::move(symbols[index++]));
      }
    }
    symbols = std::move(merged);
  }
  for (const std::string& symbol : symbols) {
    const auto token = vocabulary_.find(symbol);
    if (token == vocabulary_.end()) {
      return {StatusCode::kIncompatibleArtifact,
              "BPE produced a token absent from the vocabulary"};
    }
    ids->push_back(token->second);
  }
  return Status::ok();
}

Status Tokenizer::decode(const std::vector<std::uint32_t>& ids,
                         bool skip_special_tokens,
                         std::string* utf8) const noexcept {
  if (utf8 == nullptr) {
    return {StatusCode::kInvalidArgument, "decoded text output is required"};
  }
  utf8->clear();
  for (std::uint32_t id : ids) {
    if (id >= id_tokens_.size()) {
      return {StatusCode::kInvalidArgument, "token ID is outside the vocabulary"};
    }
    if (special_ids_[id]) {
      if (!skip_special_tokens) *utf8 += id_tokens_[id];
      continue;
    }
    std::vector<Codepoint> points;
    Status status = codepoints(id_tokens_[id], &points);
    if (!status.is_ok()) return status;
    for (const Codepoint& point : points) {
      const std::string symbol =
          id_tokens_[id].substr(point.begin, point.end - point.begin);
      const auto byte = byte_values_.find(symbol);
      if (byte == byte_values_.end()) {
        return {StatusCode::kIncompatibleArtifact,
                "token contains a symbol outside the GPT-2 byte map"};
      }
      utf8->push_back(static_cast<char>(byte->second));
    }
  }
  return Status::ok();
}

}  // namespace qw38::internal
