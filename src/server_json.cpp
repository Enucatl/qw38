#include "server_json.h"

#include <cctype>
#include <cstdint>
#include <utility>

#include "utf8proc.h"

namespace qw38::server {
namespace {

void append_utf8(std::uint32_t value, std::string* output) {
  if (value <= 0x7fU) {
    output->push_back(static_cast<char>(value));
  } else if (value <= 0x7ffU) {
    output->push_back(static_cast<char>(0xc0U | (value >> 6U)));
    output->push_back(static_cast<char>(0x80U | (value & 0x3fU)));
  } else if (value <= 0xffffU) {
    output->push_back(static_cast<char>(0xe0U | (value >> 12U)));
    output->push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
    output->push_back(static_cast<char>(0x80U | (value & 0x3fU)));
  } else {
    output->push_back(static_cast<char>(0xf0U | (value >> 18U)));
    output->push_back(static_cast<char>(0x80U | ((value >> 12U) & 0x3fU)));
    output->push_back(static_cast<char>(0x80U | ((value >> 6U) & 0x3fU)));
    output->push_back(static_cast<char>(0x80U | (value & 0x3fU)));
  }
}

int hex_value(char value) noexcept {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

bool valid_utf8(const std::string& value) noexcept {
  std::size_t offset = 0;
  while (offset < value.size()) {
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t count = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(value.data() + offset),
        static_cast<utf8proc_ssize_t>(value.size() - offset), &codepoint);
    if (count <= 0 || (codepoint >= 0xd800 && codepoint <= 0xdfff)) return false;
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

class Parser final {
 public:
  Parser(const std::string& input, std::size_t maximum_depth) noexcept
      : input_(input), maximum_depth_(maximum_depth) {}

  Status parse(Json* output) noexcept {
    if (output == nullptr || maximum_depth_ == 0) {
      return {StatusCode::kInvalidArgument, "JSON output and depth are required"};
    }
    skip_space();
    Status status = parse_value(0, output);
    skip_space();
    if (status.is_ok() && offset_ != input_.size()) {
      return error("unexpected bytes after JSON value");
    }
    return status;
  }

 private:
  Status error(const char* message) const noexcept {
    return {StatusCode::kInvalidArgument,
            std::string("invalid JSON at byte ") + std::to_string(offset_) +
                ": " + message};
  }

  void skip_space() noexcept {
    while (offset_ < input_.size() &&
           (input_[offset_] == ' ' || input_[offset_] == '\t' ||
            input_[offset_] == '\r' || input_[offset_] == '\n')) {
      ++offset_;
    }
  }

  bool consume(char expected) noexcept {
    if (offset_ >= input_.size() || input_[offset_] != expected) return false;
    ++offset_;
    return true;
  }

  Status parse_value(std::size_t depth, Json* output) noexcept {
    if (depth > maximum_depth_) return error("nesting limit exceeded");
    if (offset_ >= input_.size()) return error("value is missing");
    const char leading = input_[offset_];
    if (leading == 'n') return parse_literal("null", Json::null(), output);
    if (leading == 't') {
      return parse_literal("true", Json::boolean_value(true), output);
    }
    if (leading == 'f') {
      return parse_literal("false", Json::boolean_value(false), output);
    }
    if (leading == '"') {
      std::string value;
      Status status = parse_string(&value);
      if (status.is_ok()) *output = Json::string(std::move(value));
      return status;
    }
    if (leading == '[') return parse_array(depth, output);
    if (leading == '{') return parse_object(depth, output);
    if (leading == '-' || std::isdigit(static_cast<unsigned char>(leading))) {
      return parse_number(output);
    }
    return error("unexpected value prefix");
  }

  Status parse_literal(const char* literal, Json value, Json* output) noexcept {
    const std::string expected(literal);
    if (input_.compare(offset_, expected.size(), expected) != 0) {
      return error("invalid literal");
    }
    offset_ += expected.size();
    *output = std::move(value);
    return Status::ok();
  }

  Status parse_hex_quad(std::uint32_t* output) noexcept {
    if (offset_ + 4 > input_.size()) return error("short Unicode escape");
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      const int digit = hex_value(input_[offset_++]);
      if (digit < 0) return error("invalid Unicode escape");
      value = (value << 4U) | static_cast<std::uint32_t>(digit);
    }
    *output = value;
    return Status::ok();
  }

  Status parse_string(std::string* output) noexcept {
    if (!consume('"')) return error("string must begin with a quote");
    output->clear();
    while (offset_ < input_.size()) {
      const unsigned char value = static_cast<unsigned char>(input_[offset_++]);
      if (value == '"') {
        return valid_utf8(*output) ? Status::ok()
                                   : error("string is not valid UTF-8");
      }
      if (value < 0x20U) return error("unescaped control byte in string");
      if (value != '\\') {
        output->push_back(static_cast<char>(value));
        continue;
      }
      if (offset_ >= input_.size()) return error("short string escape");
      const char escaped = input_[offset_++];
      if (escaped == '"' || escaped == '\\' || escaped == '/') {
        output->push_back(escaped);
      } else if (escaped == 'b') {
        output->push_back('\b');
      } else if (escaped == 'f') {
        output->push_back('\f');
      } else if (escaped == 'n') {
        output->push_back('\n');
      } else if (escaped == 'r') {
        output->push_back('\r');
      } else if (escaped == 't') {
        output->push_back('\t');
      } else if (escaped == 'u') {
        std::uint32_t codepoint = 0;
        Status status = parse_hex_quad(&codepoint);
        if (!status.is_ok()) return status;
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
          if (offset_ + 2 > input_.size() || input_[offset_] != '\\' ||
              input_[offset_ + 1] != 'u') {
            return error("high surrogate has no low surrogate");
          }
          offset_ += 2;
          std::uint32_t low = 0;
          status = parse_hex_quad(&low);
          if (!status.is_ok()) return status;
          if (low < 0xdc00U || low > 0xdfffU) {
            return error("invalid low surrogate");
          }
          codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                      (low - 0xdc00U);
        } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
          return error("unexpected low surrogate");
        }
        append_utf8(codepoint, output);
      } else {
        return error("unknown string escape");
      }
    }
    return error("unterminated string");
  }

  Status parse_number(Json* output) noexcept {
    const std::size_t begin = offset_;
    consume('-');
    if (consume('0')) {
      if (offset_ < input_.size() &&
          std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
        return error("leading zero in number");
      }
    } else {
      if (offset_ >= input_.size() || input_[offset_] < '1' ||
          input_[offset_] > '9') {
        return error("invalid integer part");
      }
      while (offset_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
        ++offset_;
      }
    }
    if (consume('.')) {
      if (offset_ >= input_.size() ||
          !std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
        return error("fraction has no digits");
      }
      while (offset_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
        ++offset_;
      }
    }
    if (offset_ < input_.size() &&
        (input_[offset_] == 'e' || input_[offset_] == 'E')) {
      ++offset_;
      if (offset_ < input_.size() &&
          (input_[offset_] == '+' || input_[offset_] == '-')) {
        ++offset_;
      }
      if (offset_ >= input_.size() ||
          !std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
        return error("exponent has no digits");
      }
      while (offset_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[offset_]))) {
        ++offset_;
      }
    }
    *output = Json::number(input_.substr(begin, offset_ - begin));
    return Status::ok();
  }

  Status parse_array(std::size_t depth, Json* output) noexcept {
    consume('[');
    skip_space();
    std::vector<Json> values;
    if (consume(']')) {
      *output = Json::array_value(std::move(values));
      return Status::ok();
    }
    for (;;) {
      Json value;
      Status status = parse_value(depth + 1, &value);
      if (!status.is_ok()) return status;
      values.push_back(std::move(value));
      skip_space();
      if (consume(']')) break;
      if (!consume(',')) return error("array requires comma or closing bracket");
      skip_space();
    }
    *output = Json::array_value(std::move(values));
    return Status::ok();
  }

  Status parse_object(std::size_t depth, Json* output) noexcept {
    consume('{');
    skip_space();
    std::map<std::string, Json> values;
    if (consume('}')) {
      *output = Json::object_value(std::move(values));
      return Status::ok();
    }
    for (;;) {
      if (offset_ >= input_.size() || input_[offset_] != '"') {
        return error("object key must be a string");
      }
      std::string key;
      Status status = parse_string(&key);
      if (!status.is_ok()) return status;
      skip_space();
      if (!consume(':')) return error("object key requires colon");
      skip_space();
      Json value;
      status = parse_value(depth + 1, &value);
      if (!status.is_ok()) return status;
      if (!values.emplace(std::move(key), std::move(value)).second) {
        return error("duplicate object key");
      }
      skip_space();
      if (consume('}')) break;
      if (!consume(',')) return error("object requires comma or closing brace");
      skip_space();
    }
    *output = Json::object_value(std::move(values));
    return Status::ok();
  }

  const std::string& input_;
  std::size_t maximum_depth_ = 0;
  std::size_t offset_ = 0;
};

void dump_value(const Json& value, bool spaced, std::string* output) {
  if (value.kind == JsonKind::kNull) {
    *output += "null";
  } else if (value.kind == JsonKind::kBoolean) {
    *output += value.boolean ? "true" : "false";
  } else if (value.kind == JsonKind::kNumber) {
    *output += value.text;
  } else if (value.kind == JsonKind::kString) {
    *output += quote_json(value.text);
  } else if (value.kind == JsonKind::kArray) {
    output->push_back('[');
    for (std::size_t index = 0; index < value.array.size(); ++index) {
      if (index != 0) *output += spaced ? ", " : ",";
      dump_value(value.array[index], spaced, output);
    }
    output->push_back(']');
  } else {
    output->push_back('{');
    std::size_t index = 0;
    for (const auto& entry : value.object) {
      if (index++ != 0) *output += spaced ? ", " : ",";
      *output += quote_json(entry.first);
      *output += spaced ? ": " : ":";
      dump_value(entry.second, spaced, output);
    }
    output->push_back('}');
  }
}

}  // namespace

Json Json::null() { return {}; }

Json Json::boolean_value(bool value) {
  Json result;
  result.kind = JsonKind::kBoolean;
  result.boolean = value;
  return result;
}

Json Json::number(std::string value) {
  Json result;
  result.kind = JsonKind::kNumber;
  result.text = std::move(value);
  return result;
}

Json Json::string(std::string value) {
  Json result;
  result.kind = JsonKind::kString;
  result.text = std::move(value);
  return result;
}

Json Json::array_value(std::vector<Json> value) {
  Json result;
  result.kind = JsonKind::kArray;
  result.array = std::move(value);
  return result;
}

Json Json::object_value(std::map<std::string, Json> value) {
  Json result;
  result.kind = JsonKind::kObject;
  result.object = std::move(value);
  return result;
}

const Json* Json::find(const std::string& name) const noexcept {
  if (kind != JsonKind::kObject) return nullptr;
  const auto found = object.find(name);
  return found == object.end() ? nullptr : &found->second;
}

Status parse_json(const std::string& input, std::size_t maximum_depth,
                  Json* output) noexcept {
  return Parser(input, maximum_depth).parse(output);
}

std::string quote_json(const std::string& value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output = "\"";
  for (unsigned char byte : value) {
    if (byte == '"') output += "\\\"";
    else if (byte == '\\') output += "\\\\";
    else if (byte == '\b') output += "\\b";
    else if (byte == '\f') output += "\\f";
    else if (byte == '\n') output += "\\n";
    else if (byte == '\r') output += "\\r";
    else if (byte == '\t') output += "\\t";
    else if (byte < 0x20U) {
      output += "\\u00";
      output.push_back(kHex[byte >> 4U]);
      output.push_back(kHex[byte & 0x0fU]);
    } else {
      output.push_back(static_cast<char>(byte));
    }
  }
  output.push_back('"');
  return output;
}

std::string dump_json(const Json& value, bool spaced) {
  std::string output;
  dump_value(value, spaced, &output);
  return output;
}

}  // namespace qw38::server
