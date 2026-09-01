#include "server_generation.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace qw38::server {
namespace {

constexpr std::size_t kContextCapacity = 131072;
constexpr const char* kThinkingEnd = "</think>";

std::size_t first_stop(const std::string& value,
                       const std::vector<std::string>& stops) {
  std::size_t found = std::string::npos;
  for (const std::string& stop : stops) {
    const std::size_t candidate = value.find(stop);
    if (candidate != std::string::npos) found = std::min(found, candidate);
  }
  return found;
}

std::size_t safe_marker_bytes(const std::string& value) {
  const std::string marker(kThinkingEnd);
  const std::size_t maximum = std::min(marker.size() - 1, value.size());
  for (std::size_t count = maximum; count > 0; --count) {
    if (value.compare(value.size() - count, count, marker, 0, count) == 0) {
      return value.size() - count;
    }
  }
  return value.size();
}

std::string content_after_marker(const std::string& value,
                                 std::size_t marker) {
  std::size_t begin = marker + 8;
  while (begin < value.size() &&
         (value[begin] == '\r' || value[begin] == '\n')) {
    ++begin;
  }
  return value.substr(begin);
}

bool emit_suffix(DeltaSink sink, void* context, DeltaKind kind,
                 const std::string& complete, std::string* emitted) noexcept {
  if (complete.size() <= emitted->size()) return true;
  if (complete.compare(0, emitted->size(), *emitted) != 0) return false;
  const std::string suffix = complete.substr(emitted->size());
  if (sink != nullptr && !suffix.empty() && !sink(context, kind, suffix)) {
    return false;
  }
  *emitted = complete;
  return true;
}

bool emit_visible(const std::string& decoded, std::size_t visible_bytes,
                  const ChatRequest& request, bool final, DeltaSink sink,
                  void* context, std::string* emitted_reasoning,
                  std::string* emitted_content) noexcept {
  std::size_t safe = complete_utf8_bytes(decoded, visible_bytes);
  std::string visible = decoded.substr(0, safe);
  if (!request.chat.enable_thinking) {
    if (!request.chat.canonical_tools.empty() && !final) return true;
    const std::string content =
        request.tool_schemas.empty()
            ? visible
            : parse_assistant_output(visible, false, &request.tool_schemas).content;
    return emit_suffix(sink, context, DeltaKind::kContent, content,
                       emitted_content);
  }
  const std::size_t marker = visible.find(kThinkingEnd);
  if (marker == std::string::npos) {
    if (final) {
      return emit_suffix(sink, context, DeltaKind::kReasoning, visible,
                         emitted_reasoning);
    }
    visible.resize(safe_marker_bytes(visible));
    return emit_suffix(sink, context, DeltaKind::kReasoning, visible,
                       emitted_reasoning);
  }
  if (!emit_suffix(sink, context, DeltaKind::kReasoning,
                   visible.substr(0, marker), emitted_reasoning)) {
    return false;
  }
  if (!request.chat.canonical_tools.empty() && !final) return true;
  const std::string content = content_after_marker(visible, marker);
  const std::string rendered =
      request.tool_schemas.empty()
          ? content
          : parse_assistant_output(content, false, &request.tool_schemas).content;
  return emit_suffix(sink, context, DeltaKind::kContent, rendered,
                     emitted_content);
}

}  // namespace

Status generate_chat(const Engine& engine, Session* session,
                     const ChatRequest& request, Token end_token,
                     std::atomic<bool>* cancelled, DeltaSink sink,
                     void* sink_context, GenerationResult* result) noexcept {
  if (session == nullptr || cancelled == nullptr || result == nullptr) {
    return {StatusCode::kInvalidArgument,
            "generation session, cancellation, and output are required"};
  }
  *result = {};
  std::vector<Token> prompt;
  Status status = engine.render_chat(request.messages, request.chat, &prompt);
  if (!status.is_ok()) return status;
  result->prompt_tokens = prompt.size();
  if (prompt.size() + request.maximum_tokens + 1 > kContextCapacity) {
    return {StatusCode::kResourceExhausted,
            "prompt plus requested completion exceeds 131072 tokens"};
  }
  status = session->sync(prompt, cancelled);
  if (!status.is_ok()) return status;

  std::vector<Token> generated;
  std::string decoded;
  std::string emitted_reasoning;
  std::string emitted_content;
  bool closed = false;
  bool stopped = false;
  for (std::size_t index = 0;
       status.is_ok() && index < request.maximum_tokens; ++index) {
    if (cancelled->load(std::memory_order_relaxed)) {
      status = {StatusCode::kCancelled, "request was cancelled"};
      break;
    }
    Token token = 0;
    status = session->sample(request.sampler, &token);
    if (!status.is_ok()) break;
    status = session->eval(token, cancelled);
    if (!status.is_ok()) break;
    ++result->completion_tokens;
    if (token == end_token) {
      closed = true;
      break;
    }
    generated.push_back(token);
    status = engine.decode(generated, true, &decoded);
    if (!status.is_ok()) break;
    const std::size_t stop = first_stop(decoded, request.stops);
    stopped = stop != std::string::npos;
    const std::size_t visible = stopped
                                    ? stop
                                    : stopped_visible_bytes(
                                          decoded, request.stops, false);
    if (!emit_visible(decoded, visible, request, stopped, sink, sink_context,
                      &emitted_reasoning, &emitted_content)) {
      cancelled->store(true, std::memory_order_relaxed);
      status = {StatusCode::kCancelled, "client disconnected during stream"};
      break;
    }
    if (stopped) break;
  }
  if (!status.is_ok()) return status;
  if (!closed) {
    status = session->eval(end_token, cancelled);
    if (!status.is_ok()) return status;
  }
  const std::size_t visible = stopped_visible_bytes(decoded, request.stops, true);
  decoded.resize(complete_utf8_bytes(decoded, visible));
  result->output = parse_assistant_output(
      decoded, request.chat.enable_thinking,
      &request.tool_schemas);
  if (request.tool_choice == ToolChoice::kRequired &&
      result->output.tool_calls.empty()) {
    return {StatusCode::kInternal,
            "model did not produce the required function call"};
  }
  if (request.tool_choice == ToolChoice::kNamed) {
    if (result->output.tool_calls.empty() ||
        result->output.tool_calls.front().name != request.named_tool) {
      return {StatusCode::kInternal,
              "model did not produce the selected function call"};
    }
  }
  if (!emit_visible(decoded, decoded.size(), request, true, sink, sink_context,
                    &emitted_reasoning, &emitted_content)) {
    cancelled->store(true, std::memory_order_relaxed);
    return {StatusCode::kCancelled, "client disconnected during stream"};
  }
  if (!result->output.tool_calls.empty()) result->finish_reason = "tool_calls";
  else if (!closed && !stopped &&
           result->completion_tokens == request.maximum_tokens) {
    result->finish_reason = "length";
  }
  return Status::ok();
}

}  // namespace qw38::server
