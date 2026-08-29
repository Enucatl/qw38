#ifndef QW38_ENGINE_H_
#define QW38_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
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

class Session final {
 public:
  Session() noexcept;
  ~Session();
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  Status sync(const std::vector<Token>& tokens) noexcept;
  Status logits(std::vector<float>* output) const noexcept;
  Status sample(const SamplerConfig& config, Token* token) const noexcept;
  Status eval(Token token) noexcept;
  Status save(const std::string& path) const noexcept;
  Status restore(const std::string& path) noexcept;

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
  Status create_session(std::unique_ptr<Session>* session) const noexcept;

 private:
  struct Impl;
  explicit Engine(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qw38

#endif  // QW38_ENGINE_H_
