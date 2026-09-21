#include "compiler/json.hpp"

#include <cctype>
#include <charconv>
#include <system_error>

namespace qw38::compiler {
namespace {

class Parser {
 public:
  Parser(std::string_view text, std::string_view field)
      : text_(text), field_(field) {}

  std::expected<Json, CompilerError> parse() {
    skip();
    auto value = parse_value();
    if (!value) {
      return std::unexpected(value.error());
    }
    skip();
    if (pos_ != text_.size()) {
      return std::unexpected(err("trailing bytes after JSON value"));
    }
    return *value;
  }

 private:
  std::string_view text_;
  std::string_view field_;
  std::size_t pos_{0};

  CompilerError err(std::string_view detail) const {
    return make_error(CompilerErrorCode::InvalidJson, field_, detail, pos_);
  }

  void skip() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  std::expected<char, CompilerError> peek() {
    skip();
    if (pos_ >= text_.size()) {
      return std::unexpected(err("unexpected end of JSON"));
    }
    return text_[pos_];
  }

  std::expected<char, CompilerError> take() {
    auto c = peek();
    if (!c) {
      return std::unexpected(c.error());
    }
    ++pos_;
    return *c;
  }

  std::expected<void, CompilerError> expect_char(char wanted) {
    auto c = take();
    if (!c) {
      return std::unexpected(c.error());
    }
    if (*c != wanted) {
      return std::unexpected(err("unexpected character"));
    }
    return {};
  }

  std::expected<Json, CompilerError> parse_value() {
    auto c = peek();
    if (!c) {
      return std::unexpected(c.error());
    }
    switch (*c) {
      case '{':
        return parse_object();
      case '[':
        return parse_array();
      case '"':
        return parse_string();
      case 't':
      case 'f':
        return parse_bool();
      case 'n':
        return parse_null();
      default:
        if (*c == '-' || (*c >= '0' && *c <= '9')) {
          return parse_number();
        }
        return std::unexpected(err("invalid JSON value"));
    }
  }

  std::expected<Json, CompilerError> parse_null() {
    if (text_.substr(pos_, 4) != "null") {
      return std::unexpected(err("expected null"));
    }
    pos_ += 4;
    return Json{nullptr};
  }

  std::expected<Json, CompilerError> parse_bool() {
    if (text_.substr(pos_, 4) == "true") {
      pos_ += 4;
      return Json{true};
    }
    if (text_.substr(pos_, 5) == "false") {
      pos_ += 5;
      return Json{false};
    }
    return std::unexpected(err("expected boolean"));
  }

  std::expected<Json, CompilerError> parse_number() {
    auto const begin = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') {
      ++pos_;
    }
    if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      return std::unexpected(err("invalid number"));
    }
    if (text_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      if (pos_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        return std::unexpected(err("invalid fraction"));
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= text_.size() ||
          !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        return std::unexpected(err("invalid exponent"));
      }
      while (pos_ < text_.size() &&
             std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    auto const token = text_.substr(begin, pos_ - begin);
    double value = 0;
    auto const conv =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (conv.ec != std::errc{} || conv.ptr != token.data() + token.size()) {
      return std::unexpected(err("number is not a finite IEEE value"));
    }
    return Json{value};
  }

  std::expected<Json, CompilerError> parse_string() {
    if (auto st = expect_char('"'); !st) {
      return std::unexpected(st.error());
    }
    std::string out;
    while (pos_ < text_.size()) {
      char const c = text_[pos_++];
      if (c == '"') {
        return Json{std::move(out)};
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          return std::unexpected(err("unterminated escape"));
        }
        char const e = text_[pos_++];
        switch (e) {
          case '"':
          case '\\':
          case '/':
            out.push_back(e);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'u':
            if (pos_ + 4 > text_.size()) {
              return std::unexpected(err("truncated unicode escape"));
            }
            {
              unsigned code = 0;
              for (int i = 0; i < 4; ++i) {
                char const h = text_[pos_++];
                code <<= 4;
                if (h >= '0' && h <= '9') {
                  code += static_cast<unsigned>(h - '0');
                } else if (h >= 'a' && h <= 'f') {
                  code += static_cast<unsigned>(h - 'a' + 10);
                } else if (h >= 'A' && h <= 'F') {
                  code += static_cast<unsigned>(h - 'A' + 10);
                } else {
                  return std::unexpected(err("invalid unicode escape"));
                }
              }
              if (code <= 0x7F) {
                out.push_back(static_cast<char>(code));
              } else if (code <= 0x7FF) {
                out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
              } else {
                out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
              }
            }
            break;
          default:
            return std::unexpected(err("unknown escape"));
        }
        continue;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        return std::unexpected(err("unescaped control character"));
      }
      out.push_back(c);
    }
    return std::unexpected(err("unterminated string"));
  }

  std::expected<Json, CompilerError> parse_array() {
    if (auto st = expect_char('['); !st) {
      return std::unexpected(st.error());
    }
    Json::Array arr;
    skip();
    auto next = peek();
    if (!next) {
      return std::unexpected(next.error());
    }
    if (*next == ']') {
      ++pos_;
      return Json{std::move(arr)};
    }
    for (;;) {
      auto value = parse_value();
      if (!value) {
        return std::unexpected(value.error());
      }
      arr.push_back(std::move(*value));
      skip();
      auto sep = take();
      if (!sep) {
        return std::unexpected(sep.error());
      }
      if (*sep == ']') {
        return Json{std::move(arr)};
      }
      if (*sep != ',') {
        return std::unexpected(err("expected comma in array"));
      }
    }
  }

  std::expected<Json, CompilerError> parse_object() {
    if (auto st = expect_char('{'); !st) {
      return std::unexpected(st.error());
    }
    Json::Object obj;
    skip();
    auto next = peek();
    if (!next) {
      return std::unexpected(next.error());
    }
    if (*next == '}') {
      ++pos_;
      return Json{std::move(obj)};
    }
    for (;;) {
      auto key = parse_string();
      if (!key) {
        return std::unexpected(key.error());
      }
      skip();
      if (auto st = expect_char(':'); !st) {
        return std::unexpected(st.error());
      }
      auto value = parse_value();
      if (!value) {
        return std::unexpected(value.error());
      }
      auto const inserted =
          obj.emplace(key->as_string(), std::move(*value)).second;
      if (!inserted) {
        return std::unexpected(err("duplicate object key"));
      }
      skip();
      auto sep = take();
      if (!sep) {
        return std::unexpected(sep.error());
      }
      if (*sep == '}') {
        return Json{std::move(obj)};
      }
      if (*sep != ',') {
        return std::unexpected(err("expected comma in object"));
      }
      skip();
    }
  }
};

}  // namespace

std::expected<Json, CompilerError> parse_json(std::string_view text,
                                              std::string_view field) {
  Parser p{text, field};
  return p.parse();
}

}  // namespace qw38::compiler
