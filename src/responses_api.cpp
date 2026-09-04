#include "responses_api.h"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace qw38::server {
namespace {

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

Status text_content(const Json* value, std::string* output) noexcept {
  output->clear();
  if (value == nullptr) return invalid("message content is required");
  if (value->kind == JsonKind::kString) {
    *output = value->text;
    return Status::ok();
  }
  if (value->kind != JsonKind::kArray) {
    return invalid("message content must be text or text parts");
  }
  for (const Json& part : value->array) {
    if (part.kind != JsonKind::kObject) {
      return invalid("input content part must be an object");
    }
    const Json* type = part.find("type");
    if (type == nullptr || type->kind != JsonKind::kString) {
      return invalid("input content part requires a string type");
    }
    if (type->text != "input_text" && type->text != "output_text" &&
        type->text != "text") {
      if (type->text.find("image") != std::string::npos) {
        return unsupported("image content is unsupported in text-only v1");
      }
      if (type->text.find("audio") != std::string::npos) {
        return unsupported("audio content is unsupported in text-only v1");
      }
      if (type->text.find("file") != std::string::npos) {
        return unsupported("file content is unsupported in text-only v1");
      }
      return unsupported("unsupported Responses content type: " + type->text);
    }
    const Json* text = part.find("text");
    if (text == nullptr || text->kind != JsonKind::kString) {
      return invalid("text content part requires string text");
    }
    *output += text->text;
  }
  return Status::ok();
}

Json chat_message(const std::string& role, const std::string& content) {
  return Json::object_value({{"content", Json::string(content)},
                             {"role", Json::string(role)}});
}

Status append_input_item(const Json& item, std::vector<Json>* messages) noexcept {
  if (item.kind != JsonKind::kObject) {
    return invalid("Responses input item must be an object");
  }
  const Json* type = item.find("type");
  if (type == nullptr || type->kind != JsonKind::kString) {
    return invalid("Responses input item requires a string type");
  }
  if (type->text == "message") {
    std::string unknown;
    if (!exact_keys(item, {"type", "role", "content"}, &unknown)) {
      return unsupported("unsupported message item field: " + unknown);
    }
    const Json* role = item.find("role");
    if (role == nullptr || role->kind != JsonKind::kString ||
        (role->text != "system" && role->text != "developer" &&
         role->text != "user" && role->text != "assistant")) {
      return invalid("message item role must be system, developer, user, or assistant");
    }
    std::string content;
    Status status = text_content(item.find("content"), &content);
    if (!status.is_ok()) return status;
    messages->push_back(chat_message(role->text, content));
    return Status::ok();
  }
  if (type->text == "function_call") {
    std::string unknown;
    if (!exact_keys(item, {"type", "id", "call_id", "name", "arguments",
                           "status"}, &unknown)) {
      return unsupported("unsupported function_call field: " + unknown);
    }
    const Json* name = item.find("name");
    const Json* arguments = item.find("arguments");
    if (name == nullptr || name->kind != JsonKind::kString ||
        name->text.empty() || arguments == nullptr ||
        arguments->kind != JsonKind::kString) {
      return invalid("function_call requires string name and arguments");
    }
    Json parsed_arguments;
    Status status = parse_json(arguments->text, 32, &parsed_arguments);
    if (!status.is_ok() || parsed_arguments.kind != JsonKind::kObject) {
      return invalid("function_call arguments must encode a JSON object");
    }
    Json function = Json::object_value(
        {{"arguments", Json::string(arguments->text)},
         {"name", Json::string(name->text)}});
    Json call = Json::object_value(
        {{"function", std::move(function)}, {"type", Json::string("function")}});
    messages->push_back(Json::object_value(
        {{"content", Json::null()},
         {"role", Json::string("assistant")},
         {"tool_calls", Json::array_value({std::move(call)})}}));
    return Status::ok();
  }
  if (type->text == "function_call_output") {
    std::string unknown;
    if (!exact_keys(item, {"type", "call_id", "output"}, &unknown)) {
      return unsupported("unsupported function_call_output field: " + unknown);
    }
    const Json* call_id = item.find("call_id");
    const Json* output = item.find("output");
    if (call_id == nullptr || call_id->kind != JsonKind::kString ||
        call_id->text.empty() || output == nullptr ||
        output->kind != JsonKind::kString) {
      return invalid("function_call_output requires string call_id and output");
    }
    messages->push_back(Json::object_value(
        {{"content", Json::string(output->text)},
         {"role", Json::string("tool")},
         {"tool_call_id", Json::string(call_id->text)}}));
    return Status::ok();
  }
  if (type->text.find("image") != std::string::npos ||
      type->text.find("audio") != std::string::npos ||
      type->text.find("file") != std::string::npos) {
    return unsupported(type->text + " is unsupported in text-only v1");
  }
  return unsupported("unsupported Responses input item type: " + type->text);
}

Status convert_tools(const Json* tools, Json* output) noexcept {
  if (tools == nullptr) return Status::ok();
  if (tools->kind != JsonKind::kArray) return invalid("tools must be an array");
  std::vector<Json> converted;
  for (const Json& tool : tools->array) {
    if (tool.kind != JsonKind::kObject) return invalid("tool must be an object");
    std::string unknown;
    if (!exact_keys(tool, {"type", "name", "description", "parameters",
                           "strict"}, &unknown)) {
      return unsupported("unsupported tool field: " + unknown);
    }
    const Json* type = tool.find("type");
    const Json* name = tool.find("name");
    const Json* parameters = tool.find("parameters");
    const Json* strict = tool.find("strict");
    if (type == nullptr || type->kind != JsonKind::kString ||
        type->text != "function") {
      return unsupported("only function tools are supported");
    }
    if (strict != nullptr &&
        (strict->kind != JsonKind::kBoolean || strict->boolean)) {
      return unsupported("strict function output is unsupported");
    }
    if (name == nullptr || name->kind != JsonKind::kString ||
        name->text.empty() ||
        (parameters != nullptr && parameters->kind != JsonKind::kObject)) {
      return invalid("function tool requires a name and object parameters");
    }
    std::map<std::string, Json> function;
    function.emplace("name", *name);
    if (const Json* description = tool.find("description")) {
      if (description->kind != JsonKind::kString) {
        return invalid("tool description must be a string");
      }
      function.emplace("description", *description);
    }
    if (parameters != nullptr) function.emplace("parameters", *parameters);
    converted.push_back(Json::object_value(
        {{"function", Json::object_value(std::move(function))},
         {"type", Json::string("function")}}));
  }
  *output = Json::array_value(std::move(converted));
  return Status::ok();
}

Status convert_tool_choice(const Json* choice, Json* output) noexcept {
  if (choice == nullptr) return Status::ok();
  if (choice->kind == JsonKind::kString) {
    *output = *choice;
    return Status::ok();
  }
  if (choice->kind != JsonKind::kObject) {
    return invalid("tool_choice has an invalid type");
  }
  std::string unknown;
  if (!exact_keys(*choice, {"type", "name"}, &unknown)) {
    return unsupported("unsupported named tool_choice field: " + unknown);
  }
  const Json* type = choice->find("type");
  const Json* name = choice->find("name");
  if (type == nullptr || type->kind != JsonKind::kString ||
      type->text != "function" || name == nullptr ||
      name->kind != JsonKind::kString) {
    return invalid("named tool_choice requires type=function and name");
  }
  *output = Json::object_value(
      {{"function", Json::object_value({{"name", *name}})},
       {"type", Json::string("function")}});
  return Status::ok();
}

}  // namespace

Status parse_responses_request(const Json& root,
                               ResponsesRequest* request) noexcept {
  if (request == nullptr || root.kind != JsonKind::kObject) {
    return invalid("Responses request must be a JSON object");
  }
  static const std::set<std::string> kAllowed = {
      "model", "input", "instructions", "previous_response_id", "store",
      "stream", "max_output_tokens", "temperature", "top_p", "top_k",
      "seed", "stop", "reasoning", "tools", "tool_choice",
      "parallel_tool_calls", "text", "include", "truncation", "metadata",
      "user", "service_tier", "background"};
  std::string unknown;
  if (!exact_keys(root, kAllowed, &unknown)) {
    return unsupported("unsupported Responses field: " + unknown);
  }
  *request = {};
  std::map<std::string, Json> chat;
  const Json* model = root.find("model");
  chat.emplace("model", model == nullptr ? Json::null() : *model);
  std::vector<Json> messages;
  const Json* previous = root.find("previous_response_id");
  if (previous != nullptr) {
    if (previous->kind != JsonKind::kString || previous->text.empty()) {
      return invalid("previous_response_id must be a non-empty string");
    }
    request->previous_response_id = previous->text;
    if (root.find("instructions") != nullptr) {
      return unsupported("instructions cannot change during exact continuation");
    }
  }
  if (const Json* instructions = root.find("instructions")) {
    if (instructions->kind != JsonKind::kString) {
      return invalid("instructions must be a string");
    }
    messages.push_back(chat_message("developer", instructions->text));
  }
  const Json* input = root.find("input");
  if (input == nullptr) return invalid("input is required");
  if (input->kind == JsonKind::kString) {
    messages.push_back(chat_message("user", input->text));
  } else if (input->kind == JsonKind::kArray && !input->array.empty()) {
    for (const Json& item : input->array) {
      Status status = append_input_item(item, &messages);
      if (!status.is_ok()) return status;
    }
  } else {
    return invalid("input must be text or a non-empty item array");
  }
  bool has_user = false;
  for (const Json& message : messages) {
    const Json* role = message.find("role");
    if (role != nullptr && role->kind == JsonKind::kString &&
        role->text == "user") {
      has_user = true;
    }
  }
  const bool inserted_continuation_user = previous != nullptr && !has_user;
  if (inserted_continuation_user) {
    messages.insert(messages.begin(), chat_message("user", "continuation"));
  }
  chat.emplace("messages", Json::array_value(std::move(messages)));
  for (const auto& mapping :
       std::vector<std::pair<const char*, const char*>>{
           {"stream", "stream"}, {"max_output_tokens", "max_completion_tokens"},
           {"temperature", "temperature"}, {"top_p", "top_p"},
           {"top_k", "top_k"}, {"seed", "seed"}, {"stop", "stop"}}) {
    if (const Json* value = root.find(mapping.first)) {
      chat.emplace(mapping.second, *value);
    }
  }
  Json tools;
  Status status = convert_tools(root.find("tools"), &tools);
  if (!status.is_ok()) return status;
  if (root.find("tools") != nullptr) chat.emplace("tools", std::move(tools));
  Json choice;
  status = convert_tool_choice(root.find("tool_choice"), &choice);
  if (!status.is_ok()) return status;
  if (root.find("tool_choice") != nullptr) {
    chat.emplace("tool_choice", std::move(choice));
  }
  if (const Json* reasoning = root.find("reasoning")) {
    if (reasoning->kind != JsonKind::kObject) {
      return invalid("reasoning must be an object");
    }
    if (!exact_keys(*reasoning, {"effort", "summary"}, &unknown)) {
      return unsupported("unsupported reasoning field: " + unknown);
    }
    const Json* effort = reasoning->find("effort");
    if (effort != nullptr) chat.emplace("reasoning_effort", *effort);
    if (reasoning->find("summary") != nullptr) {
      return unsupported("reasoning summaries are unsupported in v1");
    }
  }
  if (const Json* store = root.find("store")) {
    if (store->kind != JsonKind::kBoolean) return invalid("store must be boolean");
    request->store = store->boolean;
  }
  const Json* parallel = root.find("parallel_tool_calls");
  if (parallel != nullptr &&
      (parallel->kind != JsonKind::kBoolean || parallel->boolean)) {
    return unsupported("parallel_tool_calls=true is unsupported");
  }
  const Json* text = root.find("text");
  if (text != nullptr) {
    if (text->kind != JsonKind::kObject ||
        !exact_keys(*text, {"format"}, &unknown)) {
      return unsupported("text supports only the format field");
    }
    const Json* format = text->kind == JsonKind::kObject ? text->find("format") : nullptr;
    if (format == nullptr || format->kind != JsonKind::kObject ||
        !exact_keys(*format, {"type"}, &unknown)) {
      return unsupported("text.format supports only type=text");
    }
    const Json* type = format != nullptr && format->kind == JsonKind::kObject
                           ? format->find("type")
                           : nullptr;
    if (type == nullptr || type->kind != JsonKind::kString ||
        type->text != "text") {
      return unsupported("structured text output is unsupported; use type=text");
    }
  }
  const Json* include = root.find("include");
  if (include != nullptr &&
      (include->kind != JsonKind::kArray || !include->array.empty())) {
    return unsupported("Responses include expansions are unsupported");
  }
  const Json* truncation = root.find("truncation");
  if (truncation != nullptr &&
      (truncation->kind != JsonKind::kString || truncation->text != "disabled")) {
    return unsupported("only truncation=disabled is supported");
  }
  for (const char* field : {"metadata", "user", "service_tier", "background"}) {
    if (root.find(field) != nullptr) {
      return unsupported(std::string(field) + " is unsupported");
    }
  }
  status = parse_chat_request(Json::object_value(std::move(chat)),
                              &request->generation);
  if (!status.is_ok()) return status;
  if (inserted_continuation_user) {
    request->generation.messages.erase(request->generation.messages.begin());
  }
  return Status::ok();
}

}  // namespace qw38::server
