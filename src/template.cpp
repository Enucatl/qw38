#include "template.h"

#include <cstddef>

#include "utf8proc.h"

namespace qw38::internal {
namespace {

constexpr const char* kXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";
constexpr const char* kLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration.";
constexpr const char* kToolInstructions = R"(# Tools

You have access to the following functions:

<tools>)";
constexpr const char* kToolSuffix = R"(
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>)";

bool unicode_space(utf8proc_int32_t value) noexcept {
  const utf8proc_category_t category = utf8proc_category(value);
  return (category >= UTF8PROC_CATEGORY_ZS && category <= UTF8PROC_CATEGORY_ZP) ||
         (value >= 0x09 && value <= 0x0d) || value == 0x85;
}

std::string trim(const std::string& input) {
  std::size_t offset = 0;
  std::size_t first = input.size();
  std::size_t last = 0;
  while (offset < input.size()) {
    utf8proc_int32_t value = 0;
    const utf8proc_ssize_t length = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(input.data() + offset),
        static_cast<utf8proc_ssize_t>(input.size() - offset), &value);
    if (length <= 0) return input;
    const std::size_t end = offset + static_cast<std::size_t>(length);
    if (!unicode_space(value)) {
      if (first == input.size()) first = offset;
      last = end;
    }
    offset = end;
  }
  return first == input.size() ? std::string() : input.substr(first, last - first);
}

Status reasoning_instruction(const TemplateOptions& options,
                             std::string* instruction) noexcept {
  instruction->clear();
  if (!options.enable_thinking) return Status::ok();
  if (options.reasoning_effort == "xhigh") *instruction = kXhigh;
  else if (options.reasoning_effort == "low") *instruction = kLow;
  else if (options.reasoning_effort != "medium") {
    return {StatusCode::kInvalidArgument,
            "Unexpected reasoning effort " + options.reasoning_effort +
                ". Supported types are xhigh (default), medium, and low."};
  }
  return Status::ok();
}

void append_system(const std::string& instruction, const std::string& content,
                   std::string* output) {
  if (instruction.empty() && content.empty()) return;
  *output += "<|im_start|>system\n";
  if (!instruction.empty()) *output += instruction;
  if (!instruction.empty() && !content.empty()) *output += "\n\n";
  *output += content;
  *output += "<|im_end|>\n";
}

}  // namespace

Status render_chat(const TemplateInput& input, std::string* rendered) noexcept {
  if (rendered == nullptr) {
    return {StatusCode::kInvalidArgument, "rendered template output is required"};
  }
  rendered->clear();
  if (input.messages.empty()) {
    return {StatusCode::kInvalidArgument, "No messages provided."};
  }
  for (const Message& message : input.messages) {
    if (message.has_unsupported_content) {
      return {StatusCode::kUnimplemented, "vision content is unsupported in v1"};
    }
  }
  std::string instruction;
  Status status = reasoning_instruction(input.options, &instruction);
  if (!status.is_ok()) return status;

  std::size_t first_non_system = 0;
  std::string system_content;
  while (first_non_system < input.messages.size() &&
         (input.messages[first_non_system].role == MessageRole::kSystem ||
          input.messages[first_non_system].role == MessageRole::kDeveloper)) {
    if (input.messages[first_non_system].role == MessageRole::kSystem &&
        first_non_system != 0) {
      return {StatusCode::kInvalidArgument,
              "System message must be at the beginning."};
    }
    const std::string content = trim(input.messages[first_non_system].content);
    if (!content.empty()) {
      if (!system_content.empty()) system_content += "\n\n";
      system_content += content;
    }
    ++first_non_system;
  }
  for (std::size_t index = first_non_system; index < input.messages.size(); ++index) {
    if (input.messages[index].role == MessageRole::kSystem) {
      return {StatusCode::kInvalidArgument,
              "System message must be at the beginning."};
    }
    if (input.messages[index].role == MessageRole::kDeveloper) {
      return {StatusCode::kInvalidArgument,
              "Developer message must be at the beginning."};
    }
  }

  if (!input.canonical_tool_json.empty()) {
    *rendered += "<|im_start|>system\n";
    if (!instruction.empty()) *rendered += instruction + "\n\n";
    *rendered += kToolInstructions;
    for (const std::string& tool : input.canonical_tool_json) *rendered += "\n" + tool;
    *rendered += kToolSuffix;
    if (!system_content.empty()) *rendered += "\n\n" + system_content;
    *rendered += "<|im_end|>\n";
  } else {
    append_system(instruction, system_content, rendered);
  }

  std::size_t last_query = input.messages.size();
  for (std::size_t index = input.messages.size(); index-- > first_non_system;) {
    if (input.messages[index].role == MessageRole::kUser) {
      last_query = index;
      break;
    }
  }
  if (last_query == input.messages.size()) {
    return {StatusCode::kInvalidArgument, "No user query found in messages."};
  }

  bool tool_group_open = false;
  for (std::size_t index = first_non_system; index < input.messages.size(); ++index) {
    const Message& message = input.messages[index];
    const std::string content = trim(message.content);
    if (message.role == MessageRole::kUser) {
      if (tool_group_open) {
        *rendered += "<|im_end|>\n";
        tool_group_open = false;
      }
      *rendered += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (message.role == MessageRole::kAssistant) {
      if (tool_group_open) {
        *rendered += "<|im_end|>\n";
        tool_group_open = false;
      }
      *rendered += "<|im_start|>assistant\n";
      if (input.options.preserve_thinking || index > last_query) {
        *rendered += "<think>\n" + trim(message.reasoning_content) +
                     "\n</think>\n\n";
      }
      *rendered += content;
      for (std::size_t call_index = 0; call_index < message.tool_calls.size();
           ++call_index) {
        const ToolCall& call = message.tool_calls[call_index];
        if (call_index == 0 && !content.empty()) *rendered += "\n\n";
        if (call_index != 0) *rendered += "\n";
        *rendered += "<tool_call>\n<function=" + call.name + ">\n";
        for (const auto& argument : call.rendered_arguments) {
          *rendered += "<parameter=" + argument.first + ">\n" + argument.second +
                       "\n</parameter>\n";
        }
        *rendered += "</function>\n</tool_call>";
      }
      *rendered += "<|im_end|>\n";
    } else if (message.role == MessageRole::kTool) {
      if (!tool_group_open) {
        *rendered += "<|im_start|>user";
        tool_group_open = true;
      }
      *rendered += "\n<tool_response>\n" + content + "\n</tool_response>";
      const bool next_is_tool = index + 1 < input.messages.size() &&
                                input.messages[index + 1].role == MessageRole::kTool;
      if (!next_is_tool) {
        *rendered += "<|im_end|>\n";
        tool_group_open = false;
      }
    }
  }
  if (tool_group_open) *rendered += "<|im_end|>\n";
  if (input.options.add_generation_prompt) {
    *rendered += "<|im_start|>assistant\n";
    *rendered += input.options.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  return Status::ok();
}

Status render_user_turn(const std::string& content, bool enable_thinking,
                        std::string* rendered) noexcept {
  if (rendered == nullptr) {
    return {StatusCode::kInvalidArgument, "rendered turn output is required"};
  }
  const std::string query = trim(content);
  if (query.empty()) {
    return {StatusCode::kInvalidArgument, "user turn cannot be empty"};
  }
  *rendered = "<|im_start|>user\n" + query +
              "<|im_end|>\n<|im_start|>assistant\n";
  *rendered += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return Status::ok();
}

}  // namespace qw38::internal
