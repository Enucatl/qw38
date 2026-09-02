#ifndef QW38_SERVER_API_H_
#define QW38_SERVER_API_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qw38/engine.h"
#include "qw38/status.h"
#include "server_json.h"

namespace qw38::server {

enum class ToolChoice {
  kAuto,
  kNone,
  kRequired,
  kNamed,
};

struct ChatRequest final {
  std::vector<ChatMessage> messages;
  std::vector<Token> prepared_prompt;
  ChatOptions chat;
  SamplerConfig sampler;
  std::vector<std::string> stops;
  std::vector<Json> tool_schemas;
  std::size_t maximum_tokens = 256;
  bool stream = false;
  bool include_usage = false;
  ToolChoice tool_choice = ToolChoice::kAuto;
  std::string named_tool;
};

struct OutputToolCall final {
  std::string name;
  std::string arguments_json;
};

struct AssistantOutput final {
  std::string reasoning;
  std::string content;
  std::vector<OutputToolCall> tool_calls;
};

Status parse_chat_request(const Json& root, ChatRequest* request) noexcept;
AssistantOutput parse_assistant_output(const std::string& decoded,
                                       bool thinking,
                                       const std::vector<Json>* tool_schemas);
std::size_t stopped_visible_bytes(const std::string& decoded,
                                  const std::vector<std::string>& stops,
                                  bool final);
std::size_t complete_utf8_bytes(const std::string& value,
                                std::size_t limit) noexcept;

}  // namespace qw38::server

#endif  // QW38_SERVER_API_H_
