#include "server_api.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

#include "utf8proc.h"

namespace qw38::server {
namespace {

constexpr const char* kModelId = "qwen3.8-27b-q4_k_m";
constexpr std::size_t kMaximumMessages = 4096;
constexpr std::size_t kMaximumTools = 128;
constexpr std::size_t kMaximumStops = 4;
constexpr std::size_t kMaximumGeneratedTokens = 8192;

Status invalid(const std::string& message) noexcept {
  return {StatusCode::kInvalidArgument, message};
}

Status unsupported(const std::string& message) noexcept {
  return {StatusCode::kUnimplemented, message};
}

bool exact_keys(const Json& value, const std::set<std::string>& allowed,
                std::string* unknown) {
  if (value.kind != JsonKind::kObject) return false;
  for (const auto& entry : value.object) {
    if (allowed.find(entry.first) == allowed.end()) {
      *unknown = entry.first;
      return false;
    }
  }
  return true;
}

bool as_boolean(const Json* value, bool* output) noexcept {
  if (value == nullptr) return false;
  if (value->kind != JsonKind::kBoolean) return false;
  *output = value->boolean;
  return true;
}

bool as_double(const Json* value, double* output) noexcept {
  if (value == nullptr || value->kind != JsonKind::kNumber) return false;
  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(value->text.c_str(), &end);
  if (errno != 0 || end == nullptr || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  *output = parsed;
  return true;
}

bool as_size(const Json* value, std::size_t* output) noexcept {
  if (value == nullptr || value->kind != JsonKind::kNumber ||
      value->text.empty() || value->text.front() == '-') {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed =
      std::strtoull(value->text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *output = static_cast<std::size_t>(parsed);
  return static_cast<unsigned long long>(*output) == parsed;
}

Status parse_content(const Json* value, bool allow_null,
                     std::string* content) noexcept {
  content->clear();
  if (value == nullptr || value->kind == JsonKind::kNull) {
    return allow_null ? Status::ok() : invalid("message content is required");
  }
  if (value->kind == JsonKind::kString) {
    *content = value->text;
    return Status::ok();
  }
  if (value->kind != JsonKind::kArray) {
    return invalid("message content must be text or an array of text parts");
  }
  for (const Json& part : value->array) {
    if (part.kind != JsonKind::kObject) {
      return invalid("message content part must be an object");
    }
    const Json* type = part.find("type");
    if (type == nullptr || type->kind != JsonKind::kString) {
      return invalid("message content part requires a string type");
    }
    if (type->text != "text" && type->text != "input_text") {
      if (type->text.find("image") != std::string::npos) {
        return unsupported("image content is unsupported in text-only v1");
      }
      if (type->text.find("audio") != std::string::npos) {
        return unsupported("audio content is unsupported in text-only v1");
      }
      if (type->text.find("file") != std::string::npos) {
        return unsupported("file content is unsupported in text-only v1");
      }
      return unsupported("unsupported message content part type: " + type->text);
    }
    const Json* text = part.find("text");
    if (text == nullptr || text->kind != JsonKind::kString) {
      return invalid("text content part requires string text");
    }
    *content += text->text;
  }
  return Status::ok();
}

Status parse_arguments(const Json& arguments,
                       std::vector<std::pair<std::string, std::string>>* output) {
  Json parsed;
  const Json* object = &arguments;
  if (arguments.kind == JsonKind::kString) {
    Status status = parse_json(arguments.text, 32, &parsed);
    if (!status.is_ok()) return invalid("tool call arguments are not valid JSON");
    object = &parsed;
  }
  if (object->kind != JsonKind::kObject) {
    return invalid("tool call arguments must encode a JSON object");
  }
  output->clear();
  for (const auto& argument : object->object) {
    output->push_back(
        {argument.first, argument.second.kind == JsonKind::kString
                             ? argument.second.text
                             : dump_json(argument.second)});
  }
  return Status::ok();
}

Status parse_history_tool_calls(const Json* value,
                                ChatMessage* message) noexcept {
  if (value == nullptr) return Status::ok();
  if (value->kind != JsonKind::kArray) {
    return invalid("assistant tool_calls must be an array");
  }
  for (const Json& call : value->array) {
    if (call.kind != JsonKind::kObject) {
      return invalid("assistant tool call must be an object");
    }
    const Json* type = call.find("type");
    const Json* function = call.find("function");
    if (type == nullptr || type->kind != JsonKind::kString ||
        type->text != "function" || function == nullptr ||
        function->kind != JsonKind::kObject) {
      return unsupported("only function tool calls are supported");
    }
    const Json* name = function->find("name");
    const Json* arguments = function->find("arguments");
    if (name == nullptr || name->kind != JsonKind::kString || name->text.empty() ||
        arguments == nullptr) {
      return invalid("function tool call requires name and arguments");
    }
    ChatMessage::ToolCall converted;
    converted.name = name->text;
    Status status = parse_arguments(*arguments, &converted.rendered_arguments);
    if (!status.is_ok()) return status;
    message->tool_calls.push_back(std::move(converted));
  }
  return Status::ok();
}

Status parse_messages(const Json* value,
                      std::vector<ChatMessage>* messages) noexcept {
  if (value == nullptr || value->kind != JsonKind::kArray ||
      value->array.empty() || value->array.size() > kMaximumMessages) {
    return invalid("messages must be a non-empty bounded array");
  }
  messages->clear();
  for (const Json& item : value->array) {
    if (item.kind != JsonKind::kObject) return invalid("message must be an object");
    const Json* role = item.find("role");
    if (role == nullptr || role->kind != JsonKind::kString) {
      return invalid("message role must be a string");
    }
    ChatMessage message;
    bool allow_null = false;
    if (role->text == "system") message.role = ChatRole::kSystem;
    else if (role->text == "developer") message.role = ChatRole::kDeveloper;
    else if (role->text == "user") message.role = ChatRole::kUser;
    else if (role->text == "assistant") {
      message.role = ChatRole::kAssistant;
      allow_null = true;
    } else if (role->text == "tool") {
      message.role = ChatRole::kTool;
    } else {
      return invalid("unsupported message role: " + role->text);
    }
    std::set<std::string> allowed = {"role", "content"};
    if (message.role == ChatRole::kAssistant) {
      allowed.insert("reasoning_content");
      allowed.insert("tool_calls");
    }
    if (message.role == ChatRole::kTool) allowed.insert("tool_call_id");
    std::string unknown;
    if (!exact_keys(item, allowed, &unknown)) {
      return unsupported("unsupported message field: " + unknown);
    }
    Status status = parse_content(item.find("content"), allow_null,
                                  &message.content);
    if (!status.is_ok()) return status;
    const Json* reasoning = item.find("reasoning_content");
    if (reasoning != nullptr) {
      if (reasoning->kind != JsonKind::kString) {
        return invalid("reasoning_content must be a string");
      }
      message.reasoning_content = reasoning->text;
    }
    if (message.role == ChatRole::kAssistant) {
      status = parse_history_tool_calls(item.find("tool_calls"), &message);
      if (!status.is_ok()) return status;
    }
    if (message.role == ChatRole::kTool) {
      const Json* call_id = item.find("tool_call_id");
      if (call_id == nullptr || call_id->kind != JsonKind::kString ||
          call_id->text.empty()) {
        return invalid("tool message requires tool_call_id");
      }
    }
    messages->push_back(std::move(message));
  }
  bool saw_user = false;
  bool saw_non_instruction = false;
  for (const ChatMessage& message : *messages) {
    if (message.role == ChatRole::kUser) saw_user = true;
    if (message.role == ChatRole::kSystem ||
        message.role == ChatRole::kDeveloper) {
      if (saw_non_instruction) {
        return invalid("system and developer messages must be at the beginning");
      }
    } else {
      saw_non_instruction = true;
    }
  }
  if (!saw_user) return invalid("messages require at least one user query");
  return Status::ok();
}

Status parse_tools(const Json* value, ChatOptions* options,
                   std::set<std::string>* names,
                   std::vector<Json>* schemas) noexcept {
  if (value == nullptr) return Status::ok();
  if (value->kind != JsonKind::kArray || value->array.size() > kMaximumTools) {
    return invalid("tools must be a bounded array");
  }
  for (const Json& tool : value->array) {
    if (tool.kind != JsonKind::kObject) return invalid("tool must be an object");
    std::string unknown;
    if (!exact_keys(tool, {"type", "function"}, &unknown)) {
      return unsupported("unsupported tool field: " + unknown);
    }
    const Json* type = tool.find("type");
    const Json* function = tool.find("function");
    if (type == nullptr || type->kind != JsonKind::kString ||
        type->text != "function" || function == nullptr ||
        function->kind != JsonKind::kObject) {
      return unsupported("only function tools are supported");
    }
    if (!exact_keys(*function, {"name", "description", "parameters"},
                    &unknown)) {
      return unsupported("unsupported function definition field: " + unknown);
    }
    const Json* name = function->find("name");
    const Json* parameters = function->find("parameters");
    if (name == nullptr || name->kind != JsonKind::kString || name->text.empty() ||
        (parameters != nullptr && parameters->kind != JsonKind::kObject)) {
      return invalid("function tool requires a name and object parameters");
    }
    if (!names->insert(name->text).second) {
      return invalid("function tool names must be unique");
    }
    options->canonical_tools.push_back(dump_json(tool, true));
    schemas->push_back(tool);
  }
  return Status::ok();
}

Status parse_tool_choice(const Json* value, const std::set<std::string>& names,
                         ChatRequest* request) noexcept {
  if (value == nullptr) return Status::ok();
  if (value->kind == JsonKind::kString) {
    if (value->text == "auto") request->tool_choice = ToolChoice::kAuto;
    else if (value->text == "none") request->tool_choice = ToolChoice::kNone;
    else if (value->text == "required") {
      request->tool_choice = ToolChoice::kRequired;
    } else {
      return invalid("tool_choice must be auto, none, required, or a function");
    }
  } else if (value->kind == JsonKind::kObject) {
    const Json* type = value->find("type");
    const Json* function = value->find("function");
    const Json* name = function == nullptr ? nullptr : function->find("name");
    if (type == nullptr || type->kind != JsonKind::kString ||
        type->text != "function" || function == nullptr ||
        function->kind != JsonKind::kObject || name == nullptr ||
        name->kind != JsonKind::kString || names.find(name->text) == names.end()) {
      return invalid("named tool_choice must select a declared function");
    }
    request->tool_choice = ToolChoice::kNamed;
    request->named_tool = name->text;
  } else {
    return invalid("tool_choice has an invalid type");
  }
  if ((request->tool_choice == ToolChoice::kRequired ||
       request->tool_choice == ToolChoice::kNamed) &&
      names.empty()) {
    return invalid("tool_choice requires at least one declared tool");
  }
  if (request->tool_choice == ToolChoice::kNone) {
    request->chat.canonical_tools.clear();
  }
  return Status::ok();
}

void insert_tool_instruction(ChatRequest* request) {
  if (request->tool_choice != ToolChoice::kRequired &&
      request->tool_choice != ToolChoice::kNamed) {
    return;
  }
  const std::string content =
      request->tool_choice == ToolChoice::kNamed
          ? "For this request, call the function named " + request->named_tool +
                ". Do not answer normally."
          : "For this request, call one available function. Do not answer normally.";
  std::size_t position = 0;
  while (position < request->messages.size() &&
         (request->messages[position].role == ChatRole::kSystem ||
          request->messages[position].role == ChatRole::kDeveloper)) {
    ++position;
  }
  request->messages.insert(request->messages.begin() + position,
                           {ChatRole::kDeveloper, content, std::string(), {}});
}

std::string trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

const Json* parameter_schema(const std::vector<Json>& schemas,
                             const std::string& function_name,
                             const std::string& parameter_name) {
  for (const Json& tool : schemas) {
    const Json* function = tool.find("function");
    const Json* name = function == nullptr ? nullptr : function->find("name");
    if (name == nullptr || name->kind != JsonKind::kString ||
        name->text != function_name) continue;
    const Json* parameters = function->find("parameters");
    const Json* properties =
        parameters == nullptr ? nullptr : parameters->find("properties");
    return properties == nullptr ? nullptr : properties->find(parameter_name);
  }
  return nullptr;
}

bool declared_function(const std::vector<Json>& schemas,
                       const std::string& function_name) {
  for (const Json& tool : schemas) {
    const Json* function = tool.find("function");
    const Json* name = function == nullptr ? nullptr : function->find("name");
    if (name != nullptr && name->kind == JsonKind::kString &&
        name->text == function_name) {
      return true;
    }
  }
  return false;
}

bool typed_argument(const std::string& content, const Json* schema,
                    Json* value) {
  const Json* type = schema == nullptr ? nullptr : schema->find("type");
  if (type == nullptr || type->kind != JsonKind::kString ||
      type->text == "string") {
    *value = Json::string(content);
    return true;
  }
  Status status = parse_json(content, 32, value);
  if (!status.is_ok()) return false;
  if (type->text == "integer") {
    return value->kind == JsonKind::kNumber &&
           value->text.find_first_of(".eE") == std::string::npos;
  }
  if (type->text == "number") return value->kind == JsonKind::kNumber;
  if (type->text == "boolean") return value->kind == JsonKind::kBoolean;
  if (type->text == "object") return value->kind == JsonKind::kObject;
  if (type->text == "array") return value->kind == JsonKind::kArray;
  if (type->text == "null") return value->kind == JsonKind::kNull;
  return false;
}

bool parse_tool_xml(const std::string& value,
                    const std::vector<Json>& schemas,
                    std::vector<OutputToolCall>* calls) {
  std::size_t offset = 0;
  while (offset < value.size()) {
    const std::size_t call_begin = value.find("<tool_call>", offset);
    if (call_begin == std::string::npos) break;
    const std::size_t call_end = value.find("</tool_call>", call_begin);
    const std::size_t function_begin = value.find("<function=", call_begin);
    if (call_end == std::string::npos || function_begin == std::string::npos ||
        function_begin > call_end) return false;
    const std::size_t name_end = value.find('>', function_begin + 10);
    if (name_end == std::string::npos || name_end > call_end) return false;
    OutputToolCall call;
    call.name = value.substr(function_begin + 10,
                             name_end - (function_begin + 10));
    if (call.name.empty() || !declared_function(schemas, call.name)) return false;
    std::map<std::string, Json> arguments;
    std::size_t parameter = name_end + 1;
    while ((parameter = value.find("<parameter=", parameter)) !=
               std::string::npos &&
           parameter < call_end) {
      const std::size_t parameter_name_end = value.find('>', parameter + 11);
      if (parameter_name_end == std::string::npos ||
          parameter_name_end > call_end) return false;
      const std::string name = value.substr(
          parameter + 11, parameter_name_end - (parameter + 11));
      const std::size_t parameter_end =
          value.find("</parameter>", parameter_name_end + 1);
      if (name.empty() || parameter_end == std::string::npos ||
          parameter_end > call_end) return false;
      const std::string content = trim(value.substr(
          parameter_name_end + 1, parameter_end - parameter_name_end - 1));
      Json argument;
      if (!typed_argument(content,
                          parameter_schema(schemas, call.name, name),
                          &argument) ||
          !arguments.emplace(name, std::move(argument)).second) {
        return false;
      }
      parameter = parameter_end + 12;
    }
    call.arguments_json = dump_json(Json::object_value(std::move(arguments)));
    calls->push_back(std::move(call));
    offset = call_end + 12;
  }
  return !calls->empty();
}

}  // namespace

Status parse_chat_request(const Json& root, ChatRequest* request) noexcept {
  return parse_chat_request(root, kModelId, request);
}

Status parse_chat_request(const Json& root, const char* model_id,
                          ChatRequest* request) noexcept {
  if (request == nullptr || root.kind != JsonKind::kObject) {
    return invalid("Chat Completions request must be a JSON object");
  }
  static const std::set<std::string> kAllowed = {
      "model", "messages", "stream", "stream_options", "max_tokens",
      "max_completion_tokens", "temperature", "top_p", "top_k", "seed",
      "stop", "tools", "tool_choice", "parallel_tool_calls",
      "reasoning_effort", "chat_template_kwargs", "n", "user", "logprobs",
      "top_logprobs", "presence_penalty", "frequency_penalty", "response_format",
      "modalities", "audio", "functions", "function_call", "store", "metadata"};
  std::string unknown;
  if (!exact_keys(root, kAllowed, &unknown)) {
    return unsupported("unsupported Chat Completions field: " + unknown);
  }
  *request = {};
  request->sampler.temperature = 1.0F;
  const Json* model = root.find("model");
  const char* expected = model_id == nullptr ? kModelId : model_id;
  if (model == nullptr || model->kind != JsonKind::kString ||
      model->text != expected) {
    return invalid(std::string("model must be ") + expected);
  }
  Status status = parse_messages(root.find("messages"), &request->messages);
  if (!status.is_ok()) return status;
  std::set<std::string> tool_names;
  status = parse_tools(root.find("tools"), &request->chat, &tool_names,
                       &request->tool_schemas);
  if (!status.is_ok()) return status;
  status = parse_tool_choice(root.find("tool_choice"), tool_names, request);
  if (!status.is_ok()) return status;
  if (request->tool_choice == ToolChoice::kNone) request->tool_schemas.clear();

  const Json* maximum = root.find("max_completion_tokens");
  if (maximum != nullptr && root.find("max_tokens") != nullptr) {
    return invalid("use only one of max_tokens and max_completion_tokens");
  }
  if (maximum == nullptr) maximum = root.find("max_tokens");
  if (maximum != nullptr &&
      (!as_size(maximum, &request->maximum_tokens) ||
       request->maximum_tokens == 0 ||
       request->maximum_tokens > kMaximumGeneratedTokens)) {
    return invalid("completion token limit must be between 1 and 8192");
  }
  double number = 0.0;
  if (root.find("temperature") != nullptr) {
    if (!as_double(root.find("temperature"), &number) || number < 0.0 ||
        number > 2.0) return invalid("temperature must be between 0 and 2");
    request->sampler.temperature = static_cast<float>(number);
  }
  if (root.find("top_p") != nullptr) {
    if (!as_double(root.find("top_p"), &number) || number <= 0.0 || number > 1.0) {
      return invalid("top_p must be greater than 0 and at most 1");
    }
    request->sampler.top_p = static_cast<float>(number);
  }
  std::size_t integer = 0;
  if (root.find("top_k") != nullptr) {
    if (!as_size(root.find("top_k"), &integer) || integer > 248320) {
      return invalid("top_k must be between 0 and the vocabulary size");
    }
    request->sampler.top_k = static_cast<std::uint32_t>(integer);
  }
  if (root.find("seed") != nullptr) {
    if (!as_size(root.find("seed"), &integer)) return invalid("seed must be unsigned");
    request->sampler.seed = static_cast<std::uint64_t>(integer);
  }
  if (root.find("stream") != nullptr &&
      !as_boolean(root.find("stream"), &request->stream)) {
    return invalid("stream must be boolean");
  }
  const Json* stream_options = root.find("stream_options");
  if (stream_options != nullptr) {
    if (!request->stream || stream_options->kind != JsonKind::kObject ||
        !as_boolean(stream_options->find("include_usage"),
                    &request->include_usage)) {
      return invalid("stream_options requires stream=true and include_usage boolean");
    }
    std::string unknown_stream_option;
    if (!exact_keys(*stream_options, {"include_usage"},
                    &unknown_stream_option)) {
      return unsupported("unsupported stream_options field: " +
                         unknown_stream_option);
    }
  }
  const Json* stop = root.find("stop");
  if (stop != nullptr) {
    if (stop->kind == JsonKind::kString) request->stops.push_back(stop->text);
    else if (stop->kind == JsonKind::kArray &&
             stop->array.size() <= kMaximumStops) {
      for (const Json& item : stop->array) {
        if (item.kind != JsonKind::kString) return invalid("stop array must contain strings");
        request->stops.push_back(item.text);
      }
    } else return invalid("stop must be a string or up to four strings");
    for (const std::string& item : request->stops) {
      if (item.empty()) return invalid("stop strings cannot be empty");
    }
  }
  const Json* effort = root.find("reasoning_effort");
  if (effort != nullptr) {
    if (effort->kind != JsonKind::kString) return invalid("reasoning_effort must be a string");
    if (effort->text == "none") request->chat.enable_thinking = false;
    else if (effort->text == "low" || effort->text == "medium") {
      request->chat.reasoning_effort = effort->text;
    } else if (effort->text == "high" || effort->text == "xhigh") {
      request->chat.reasoning_effort = "xhigh";
    } else return invalid("reasoning_effort must be none, low, medium, high, or xhigh");
  }
  const Json* template_options = root.find("chat_template_kwargs");
  if (template_options != nullptr) {
    if (template_options->kind != JsonKind::kObject ||
        !as_boolean(template_options->find("enable_thinking"),
                    &request->chat.enable_thinking)) {
      return invalid("chat_template_kwargs supports only enable_thinking boolean");
    }
    for (const auto& entry : template_options->object) {
      if (entry.first != "enable_thinking") {
        return unsupported("unsupported chat_template_kwargs field: " + entry.first);
      }
    }
  }
  if (root.find("n") != nullptr &&
      (!as_size(root.find("n"), &integer) || integer != 1)) {
    return unsupported("only n=1 is supported");
  }
  const Json* parallel = root.find("parallel_tool_calls");
  bool parallel_value = false;
  if (parallel != nullptr &&
      (!as_boolean(parallel, &parallel_value) || parallel_value)) {
    return unsupported("parallel_tool_calls=true is unsupported");
  }
  for (const char* field : {"response_format", "modalities", "audio", "functions",
                            "function_call", "metadata"}) {
    if (root.find(field) != nullptr) return unsupported(std::string(field) + " is unsupported");
  }
  for (const char* field : {"presence_penalty", "frequency_penalty"}) {
    if (root.find(field) != nullptr &&
        (!as_double(root.find(field), &number) || number != 0.0)) {
      return unsupported(std::string(field) + " is unsupported unless zero");
    }
  }
  bool flag = false;
  if (root.find("logprobs") != nullptr &&
      (!as_boolean(root.find("logprobs"), &flag) || flag)) {
    return unsupported("logprobs are unsupported");
  }
  if (root.find("top_logprobs") != nullptr) return unsupported("top_logprobs are unsupported");
  if (root.find("store") != nullptr &&
      (!as_boolean(root.find("store"), &flag) || flag)) {
    return unsupported("store=true is unsupported");
  }
  insert_tool_instruction(request);
  return Status::ok();
}

AssistantOutput parse_assistant_output(const std::string& decoded,
                                       bool thinking,
                                       const std::vector<Json>* tool_schemas) {
  AssistantOutput output;
  std::string visible = decoded;
  if (thinking) {
    const std::size_t marker = visible.find("</think>");
    if (marker == std::string::npos) {
      output.reasoning = trim(visible);
      return output;
    }
    output.reasoning = trim(visible.substr(0, marker));
    visible = visible.substr(marker + 8);
  }
  visible = trim(visible);
  if (tool_schemas != nullptr && !tool_schemas->empty() &&
      parse_tool_xml(visible, *tool_schemas, &output.tool_calls)) {
    return output;
  }
  output.content = visible;
  return output;
}

std::size_t stopped_visible_bytes(const std::string& decoded,
                                  const std::vector<std::string>& stops,
                                  bool final) {
  std::size_t visible = decoded.size();
  for (const std::string& stop : stops) {
    if (stop.empty()) continue;
    const std::size_t found = decoded.find(stop);
    if (found != std::string::npos) visible = std::min(visible, found);
  }
  if (visible != decoded.size() || final) return visible;
  std::size_t withheld = 0;
  for (const std::string& stop : stops) {
    if (stop.empty()) continue;
    const std::size_t maximum = std::min(stop.size() - 1, decoded.size());
    for (std::size_t count = maximum; count > withheld; --count) {
      if (decoded.compare(decoded.size() - count, count, stop, 0, count) == 0) {
        withheld = count;
        break;
      }
    }
  }
  return visible - withheld;
}

std::size_t complete_utf8_bytes(const std::string& value,
                                std::size_t limit) noexcept {
  std::size_t offset = 0;
  const std::size_t bounded = std::min(limit, value.size());
  while (offset < bounded) {
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t count = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(value.data() + offset),
        static_cast<utf8proc_ssize_t>(bounded - offset), &codepoint);
    if (count <= 0 || offset + static_cast<std::size_t>(count) > bounded) break;
    offset += static_cast<std::size_t>(count);
  }
  return offset;
}

}  // namespace qw38::server
