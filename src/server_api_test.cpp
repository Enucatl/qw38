#include <iostream>
#include <string>

#include "server_api.h"
#include "server_json.h"

namespace {

bool parse_request(const std::string& text, qw38::server::ChatRequest* request,
                   qw38::Status* status) {
  qw38::server::Json root;
  *status = qw38::server::parse_json(text, 64, &root);
  if (status->is_ok()) {
    *status = qw38::server::parse_chat_request(root, request);
  }
  return status->is_ok();
}

int fail(const char* name) {
  std::cerr << "failed_case=" << name << '\n';
  return 1;
}

}  // namespace

int main() {
  using qw38::server::ChatRequest;
  using qw38::server::Json;
  using qw38::server::JsonKind;

  Json unicode;
  const std::string emoji =
      std::string("\xF0") + "\x9F" + "\x98" + "\x82";
  qw38::Status status = qw38::server::parse_json(
      R"({"z":"\uD83D\uDE02","a":[true,null,-1.25e2]})", 8, &unicode);
  if (!status.is_ok() || unicode.kind != JsonKind::kObject ||
      unicode.find("z") == nullptr || unicode.find("z")->text != emoji ||
      qw38::server::dump_json(unicode, true) !=
          std::string(R"({"a": [true, null, -1.25e2], "z": ")") + emoji +
              R"("})") {
    return fail("unicode_canonical");
  }
  Json malformed;
  status = qw38::server::parse_json(R"({"x":1,"x":2})", 8, &malformed);
  if (status.code() != qw38::StatusCode::kInvalidArgument) return fail("duplicate");
  status = qw38::server::parse_json("[[[0]]]", 2, &malformed);
  if (status.code() != qw38::StatusCode::kInvalidArgument) return fail("depth");

  ChatRequest request;
  const std::string valid = R"({
    "model":"qwen3.8-27b-q4_k_m",
    "messages":[
      {"role":"developer","content":"Be exact."},
      {"role":"user","content":[{"type":"text","text":"Weather?"}]},
      {"role":"assistant","content":null,"tool_calls":[{"id":"call_old","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Bern\"}"}}]},
      {"role":"tool","tool_call_id":"call_old","content":"18 C"},
      {"role":"user","content":"Again"}
    ],
    "tools":[{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
    "tool_choice":{"type":"function","function":{"name":"weather"}},
    "reasoning_effort":"high","temperature":0,"top_p":0.9,"top_k":40,
    "seed":7,"max_completion_tokens":32,"stop":["END","STOP"],
    "stream":true,"stream_options":{"include_usage":true}
  })";
  if (!parse_request(valid, &request, &status) || request.messages.size() != 6 ||
      request.chat.canonical_tools.size() != 1 ||
      request.chat.reasoning_effort != "xhigh" ||
      request.sampler.temperature != 0.0F || request.sampler.top_k != 40 ||
      request.maximum_tokens != 32 || request.stops.size() != 2 ||
      !request.stream || !request.include_usage ||
      request.named_tool != "weather") {
    std::cerr << status.message() << '\n';
    return fail("valid_request");
  }
  if (request.chat.canonical_tools[0] !=
      R"({"function": {"description": "Get weather", "name": "weather", "parameters": {"properties": {"city": {"type": "string"}}, "type": "object"}}, "type": "function"})") {
    return fail("canonical_tool");
  }

  ChatRequest rejected;
  if (parse_request(
          R"({"model":"qwen3.8-27b-q4_k_m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})",
          &rejected, &status) || status.code() != qw38::StatusCode::kUnimplemented) {
    return fail("image_rejection");
  }
  if (parse_request(
          R"({"model":"qwen3.8-27b-q4_k_m","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}})",
          &rejected, &status) || status.code() != qw38::StatusCode::kUnimplemented) {
    return fail("structured_rejection");
  }

  const qw38::server::AssistantOutput output =
      qw38::server::parse_assistant_output(
          "check\n</think>\n\n<tool_call>\n<function=weather>\n"
          "<parameter=city>\nBern\n</parameter>\n</function>\n</tool_call>",
          true, &request.tool_schemas);
  if (output.reasoning != "check" || !output.content.empty() ||
      output.tool_calls.size() != 1 || output.tool_calls[0].name != "weather" ||
      output.tool_calls[0].arguments_json != R"({"city":"Bern"})") {
    return fail("tool_output");
  }
  Json numeric_tool;
  status = qw38::server::parse_json(
      R"({"type":"function","function":{"name":"measure","parameters":{"type":"object","properties":{"count":{"type":"integer"},"exact":{"type":"boolean"}}}}})",
      16, &numeric_tool);
  const std::vector<Json> numeric_schemas{numeric_tool};
  const qw38::server::AssistantOutput typed_output =
      qw38::server::parse_assistant_output(
          "<tool_call><function=measure><parameter=count>18</parameter>"
          "<parameter=exact>true</parameter></function></tool_call>",
          false, &numeric_schemas);
  if (!status.is_ok() || typed_output.tool_calls.size() != 1 ||
      typed_output.tool_calls[0].arguments_json !=
          R"({"count":18,"exact":true})") {
    return fail("typed_tool_output");
  }
  const std::string incomplete_utf8 = std::string("a") + "\xE2" + "\x82";
  if (qw38::server::stopped_visible_bytes("answer<ST", {"<STOP>"}, false) != 6 ||
      qw38::server::stopped_visible_bytes("answer<STOP>tail", {"<STOP>"}, true) != 6 ||
      qw38::server::complete_utf8_bytes(incomplete_utf8, 3) != 1) {
    return fail("stop_utf8");
  }

  std::cout << "json_case=unicode_canonical_duplicate_depth passed=true\n"
            << "request_case=roles_tools_sampling_stream passed=true\n"
            << "rejection_case=image_structured_output passed=true\n"
            << "output_case=reasoning_tool_stop_utf8 passed=true\n"
            << "status=passed\n";
  return 0;
}
