#pragma once

#include "compiler/error.hpp"

#include <cstdint>
#include <expected>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace qw38::compiler {

class Json {
 public:
  using Object = std::map<std::string, Json>;
  using Array = std::vector<Json>;

  Json() = default;
  Json(std::nullptr_t) : value_(nullptr) {}
  Json(bool v) : value_(v) {}
  Json(double v) : value_(v) {}
  Json(std::string v) : value_(std::move(v)) {}
  Json(Array v) : value_(std::move(v)) {}
  Json(Object v) : value_(std::move(v)) {}

  [[nodiscard]] bool is_null() const noexcept {
    return std::holds_alternative<std::nullptr_t>(value_);
  }
  [[nodiscard]] bool is_bool() const noexcept {
    return std::holds_alternative<bool>(value_);
  }
  [[nodiscard]] bool is_number() const noexcept {
    return std::holds_alternative<double>(value_);
  }
  [[nodiscard]] bool is_string() const noexcept {
    return std::holds_alternative<std::string>(value_);
  }
  [[nodiscard]] bool is_array() const noexcept {
    return std::holds_alternative<Array>(value_);
  }
  [[nodiscard]] bool is_object() const noexcept {
    return std::holds_alternative<Object>(value_);
  }

  [[nodiscard]] bool as_bool() const { return std::get<bool>(value_); }
  [[nodiscard]] double as_number() const { return std::get<double>(value_); }
  [[nodiscard]] std::string const& as_string() const {
    return std::get<std::string>(value_);
  }
  [[nodiscard]] Array const& as_array() const { return std::get<Array>(value_); }
  [[nodiscard]] Object const& as_object() const {
    return std::get<Object>(value_);
  }

  [[nodiscard]] Json const* find(std::string_view key) const {
    if (!is_object()) {
      return nullptr;
    }
    auto const& obj = as_object();
    auto it = obj.find(std::string(key));
    if (it == obj.end()) {
      return nullptr;
    }
    return &it->second;
  }

 private:
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value_{
      nullptr};
};

[[nodiscard]] std::expected<Json, CompilerError> parse_json(
    std::string_view text, std::string_view field = "json");

}  // namespace qw38::compiler
