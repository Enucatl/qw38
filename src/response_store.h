#ifndef QW38_RESPONSE_STORE_H_
#define QW38_RESPONSE_STORE_H_

#include <string>
#include <vector>

#include "qw38/engine.h"
#include "server_json.h"

namespace qw38::server {

struct ResponseRecord final {
  std::vector<Token> tokens;
  std::vector<Json> tool_schemas;
};

class ResponseStore final {
 public:
  explicit ResponseStore(std::string directory) noexcept;
  Status open() noexcept;
  Status save(const std::string& id, const ResponseRecord& record) noexcept;
  Status load(const std::string& id, ResponseRecord* record) const noexcept;
  const std::string& directory() const noexcept;

 private:
  std::string directory_;
};

}  // namespace qw38::server

#endif  // QW38_RESPONSE_STORE_H_
