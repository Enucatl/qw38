#ifndef QW38_ENGINE_H_
#define QW38_ENGINE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qw38/status.h"

namespace qw38 {

using Token = std::uint32_t;

struct SamplerConfig final {
  float temperature = 1.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0;
};

enum class ChatRole : std::uint8_t {
  kSystem,
  kDeveloper,
  kUser,
  kAssistant,
  kTool,
};

struct ChatMessage final {
  ChatRole role = ChatRole::kUser;
  std::string content;
  std::string reasoning_content;
  struct ToolCall final {
    std::string name;
    std::vector<std::pair<std::string, std::string>> rendered_arguments;
  };
  std::vector<ToolCall> tool_calls;
};

struct ChatOptions final {
  bool enable_thinking = true;
  std::string reasoning_effort = "xhigh";
  bool preserve_thinking = true;
  std::vector<std::string> canonical_tools;
};

class Session final {
 public:
  Session() noexcept;
  ~Session();
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  Status sync(const std::vector<Token>& tokens) noexcept;
  Status sync(const std::vector<Token>& tokens,
              const std::atomic<bool>* cancelled) noexcept;
  Status logits(std::vector<float>* output) const noexcept;
  Status sample(const SamplerConfig& config, Token* token) const noexcept;
  Status eval(Token token) noexcept;
  Status eval(Token token, const std::atomic<bool>* cancelled) noexcept;
  Status save(const std::string& path) const noexcept;
  Status restore(const std::string& path) noexcept;
  Status tokens(std::vector<Token>* output) const noexcept;

 private:
  struct Impl;
  explicit Session(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
  friend class Engine;
};

class Engine final {
 public:
  Engine() noexcept;
  ~Engine();
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  static Status open(const std::string& model_path, Engine* engine) noexcept;
  static Status open(const std::string& model_path, std::size_t context,
                     Engine* engine) noexcept;
  Status create_session(std::unique_ptr<Session>* session) const noexcept;
  Status admitted_model(std::string* identity) const noexcept;
  Status context_capacity(std::size_t* capacity) const noexcept;
  Status encode(const std::string& utf8,
                std::vector<Token>* tokens) const noexcept;
  Status decode(const std::vector<Token>& tokens, bool skip_special_tokens,
                std::string* utf8) const noexcept;
  Status render_chat(const std::vector<ChatMessage>& messages,
                     const ChatOptions& options,
                     std::vector<Token>* tokens) const noexcept;
  Status render_user_turn(const std::string& content, bool enable_thinking,
                          std::vector<Token>* tokens) const noexcept;

 private:
  struct Impl;
  explicit Engine(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qw38

#endif  // QW38_ENGINE_H_
