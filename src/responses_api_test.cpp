#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "response_store.h"
#include "responses_api.h"
#include "server_json.h"

namespace {

int fail(const char* name, const qw38::Status& status = qw38::Status::ok()) {
  std::cerr << "failed_case=" << name;
  if (!status.is_ok()) std::cerr << " message=" << status.message();
  std::cerr << '\n';
  return 1;
}

qw38::Status parse(const std::string& text,
                   qw38::server::ResponsesRequest* request) {
  qw38::server::Json root;
  qw38::Status status = qw38::server::parse_json(text, 64, &root);
  if (status.is_ok()) {
    status = qw38::server::parse_responses_request(root, request);
  }
  return status;
}

}  // namespace

int main() {
  qw38::server::ResponsesRequest request;
  qw38::Status status = parse(
      R"({"model":"qwen3.8-27b-q4_k_m","instructions":"Be exact.","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Weather?"}]}],"tools":[{"type":"function","name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}},"strict":false}],"tool_choice":{"type":"function","name":"weather"},"reasoning":{"effort":"high"},"temperature":0,"max_output_tokens":32,"stream":true})",
      &request);
  if (!status.is_ok() || request.generation.messages.size() != 3 ||
      request.generation.messages.front().role != qw38::ChatRole::kDeveloper ||
      request.generation.tool_schemas.size() != 1 ||
      request.generation.named_tool != "weather" ||
      request.generation.chat.reasoning_effort != "xhigh" ||
      request.generation.maximum_tokens != 32 ||
      !request.generation.stream || !request.store) {
    return fail("request_mapping", status);
  }

  qw38::server::ResponsesRequest continuation;
  status = parse(
      R"({"model":"qwen3.8-27b-q4_k_m","previous_response_id":"resp_qw38_123_1","input":[{"type":"function_call_output","call_id":"call_1","output":"18 C"}],"store":false})",
      &continuation);
  if (!status.is_ok() || continuation.previous_response_id != "resp_qw38_123_1" ||
      continuation.store || continuation.generation.messages.size() != 1 ||
      continuation.generation.messages.front().role != qw38::ChatRole::kTool ||
      continuation.generation.messages.front().content != "18 C") {
    return fail("continuation_mapping", status);
  }

  qw38::server::ResponsesRequest rejected;
  status = parse(
      R"({"model":"qwen3.8-27b-q4_k_m","input":[{"type":"input_image","image_url":"x"}]})",
      &rejected);
  if (status.code() != qw38::StatusCode::kUnimplemented) {
    return fail("image_rejection", status);
  }
  status = parse(
      R"({"model":"qwen3.8-27b-q4_k_m","input":"x","text":{"format":{"type":"json_schema"}}})",
      &rejected);
  if (status.code() != qw38::StatusCode::kUnimplemented) {
    return fail("structured_rejection", status);
  }
  status = parse(
      R"({"model":"qwen3.8-27b-q4_k_m","previous_response_id":"resp_qw38_123_1","instructions":"replace","input":"x"})",
      &rejected);
  if (status.code() != qw38::StatusCode::kUnimplemented) {
    return fail("instruction_replacement", status);
  }

  char directory_template[] = "/tmp/qw38-responses-XXXXXX";
  char* directory = mkdtemp(directory_template);
  if (directory == nullptr) return fail("temporary_directory");
  qw38::server::ResponseStore store(directory);
  status = store.open();
  qw38::server::ResponseRecord original;
  original.tokens = {17, 18, 248319};
  original.tool_schemas = request.generation.tool_schemas;
  if (status.is_ok()) status = store.save("resp_qw38_123_1", original);
  qw38::server::ResponseRecord restored;
  if (status.is_ok()) status = store.load("resp_qw38_123_1", &restored);
  if (!status.is_ok() || restored.tokens != original.tokens ||
      restored.tool_schemas.size() != 1) {
    return fail("atomic_roundtrip", status);
  }
  status = store.load("resp_qw38_missing", &restored);
  if (status.code() != qw38::StatusCode::kInvalidArgument) {
    return fail("missing_record", status);
  }
  const std::string record_path =
      std::string(directory) + "/resp_qw38_123_1.json";
  const int descriptor = open(record_path.c_str(), O_WRONLY | O_TRUNC);
  if (descriptor < 0 || write(descriptor, "{}", 2) != 2 || close(descriptor) != 0) {
    return fail("corrupt_record_setup");
  }
  status = store.load("resp_qw38_123_1", &restored);
  if (status.code() != qw38::StatusCode::kIncompatibleArtifact) {
    return fail("corrupt_record", status);
  }
  unlink(record_path.c_str());
  rmdir(directory);

  std::cout << "request_case=text_tools_reasoning_stream passed=true\n"
            << "continuation_case=function_output_store_false passed=true\n"
            << "rejection_case=media_structured_instruction passed=true\n"
            << "store_case=atomic_missing_corrupt passed=true\n"
            << "status=passed\n";
  return 0;
}
