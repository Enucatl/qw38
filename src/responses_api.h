#ifndef QW38_RESPONSES_API_H_
#define QW38_RESPONSES_API_H_

#include <string>

#include "server_api.h"

namespace qw38::server {

struct ResponsesRequest final {
  ChatRequest generation;
  std::string previous_response_id;
  bool store = true;
};

Status parse_responses_request(const Json& root,
                               ResponsesRequest* request) noexcept;

}  // namespace qw38::server

#endif  // QW38_RESPONSES_API_H_
