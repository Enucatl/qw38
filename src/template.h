#ifndef QW38_TEMPLATE_H_
#define QW38_TEMPLATE_H_

#include <string>
#include <utility>
#include <vector>

#include "qw38/status.h"

namespace qw38::internal {

enum class MessageRole { kSystem, kDeveloper, kUser, kAssistant, kTool };

struct ToolCall final {
  std::string name;
  std::vector<std::pair<std::string, std::string>> rendered_arguments;
};

struct Message final {
  Message() = default;
  Message(MessageRole message_role, std::string message_content)
      : role(message_role), content(std::move(message_content)) {}

  MessageRole role = MessageRole::kUser;
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  bool has_unsupported_content = false;
};

struct TemplateOptions final {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
  std::string reasoning_effort = "xhigh";
  bool preserve_thinking = true;
};

struct TemplateInput final {
  std::vector<Message> messages;
  std::vector<std::string> canonical_tool_json;
  TemplateOptions options;
};

Status render_chat(const TemplateInput& input, std::string* rendered) noexcept;
Status render_user_turn(const std::string& content, bool enable_thinking,
                        std::string* rendered) noexcept;

}  // namespace qw38::internal

#endif  // QW38_TEMPLATE_H_
