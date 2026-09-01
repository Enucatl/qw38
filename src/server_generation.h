#ifndef QW38_SERVER_GENERATION_H_
#define QW38_SERVER_GENERATION_H_

#include <atomic>
#include <cstddef>
#include <string>

#include "qw38/engine.h"
#include "qw38/status.h"
#include "server_api.h"

namespace qw38::server {

enum class DeltaKind {
  kReasoning,
  kContent,
};

using DeltaSink = bool (*)(void* context, DeltaKind kind,
                           const std::string& bytes) noexcept;

struct GenerationResult final {
  AssistantOutput output;
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  std::string finish_reason = "stop";
};

Status generate_chat(const Engine& engine, Session* session,
                     const ChatRequest& request, Token end_token,
                     std::atomic<bool>* cancelled, DeltaSink sink,
                     void* sink_context, GenerationResult* result) noexcept;

}  // namespace qw38::server

#endif  // QW38_SERVER_GENERATION_H_
