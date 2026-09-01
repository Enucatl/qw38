#ifndef QW38_SERVER_JSON_H_
#define QW38_SERVER_JSON_H_

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "qw38/status.h"

namespace qw38::server {

enum class JsonKind {
  kNull,
  kBoolean,
  kNumber,
  kString,
  kArray,
  kObject,
};

struct Json final {
  JsonKind kind = JsonKind::kNull;
  bool boolean = false;
  std::string text;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  static Json null();
  static Json boolean_value(bool value);
  static Json number(std::string value);
  static Json string(std::string value);
  static Json array_value(std::vector<Json> value);
  static Json object_value(std::map<std::string, Json> value);
  const Json* find(const std::string& name) const noexcept;
};

Status parse_json(const std::string& input, std::size_t maximum_depth,
                  Json* output) noexcept;
std::string dump_json(const Json& value, bool spaced = false);
std::string quote_json(const std::string& value);

}  // namespace qw38::server

#endif  // QW38_SERVER_JSON_H_
