#include "qw38/engine.h"

#include <filesystem>
#include <memory>
#include <utility>

#include "model.h"
#include "sha256.h"

namespace qw38 {
namespace {

constexpr std::uintmax_t kPinnedModelBytes = 18973870432ULL;
constexpr const char* kPinnedModelSha256 =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";

Status unavailable(const char* operation) noexcept {
  return {StatusCode::kUnimplemented,
          std::string(operation) + " has not passed its delivery gate"};
}

}  // namespace

struct Engine::Impl final {
  std::string model_path;
  internal::ModelInfo model;
  internal::MappedFile mapping;
};

struct Session::Impl final {};

Engine::Engine() noexcept = default;
Engine::~Engine() = default;
Engine::Engine(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Status Engine::open(const std::string& model_path, Engine* engine) noexcept {
  if (engine == nullptr || model_path.empty()) {
    return {StatusCode::kInvalidArgument, "model path and engine are required"};
  }
  std::error_code error;
  const std::uintmax_t bytes = std::filesystem::file_size(model_path, error);
  if (error) {
    return {StatusCode::kIoError, "cannot stat model: " + error.message()};
  }
  if (bytes != kPinnedModelBytes) {
    return {StatusCode::kIncompatibleArtifact,
            "model byte size does not match the pinned artifact"};
  }
  internal::ModelInfo model;
  Status status = internal::inspect_gguf(model_path, &model);
  if (!status.is_ok()) {
    return status;
  }
  status = internal::validate_qwen38_contract(&model);
  if (!status.is_ok()) {
    return status;
  }
  std::string digest;
  status = internal::sha256_file(model_path, &digest);
  if (!status.is_ok()) {
    return status;
  }
  if (digest != kPinnedModelSha256) {
    return {StatusCode::kIncompatibleArtifact,
            "model SHA-256 does not match the pinned artifact"};
  }
  auto impl = std::make_unique<Impl>();
  status = impl->mapping.open(model_path);
  if (!status.is_ok()) {
    return status;
  }
  impl->model_path = model_path;
  impl->model = std::move(model);
  *engine = Engine(std::move(impl));
  return Status::ok();
}

Status Engine::create_session(std::unique_ptr<Session>* session) const noexcept {
  if (session == nullptr) {
    return {StatusCode::kInvalidArgument, "session output is required"};
  }
  if (!impl_) {
    return {StatusCode::kInvalidArgument, "engine is not open"};
  }
  return unavailable("session creation");
}

Session::Session() noexcept = default;
Session::~Session() = default;
Session::Session(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Status Session::sync(const std::vector<Token>&) noexcept {
  return unavailable("prefix synchronization");
}
Status Session::logits(std::vector<float>* output) const noexcept {
  if (output == nullptr) {
    return {StatusCode::kInvalidArgument, "logits output is required"};
  }
  return unavailable("logits");
}
Status Session::sample(const SamplerConfig&, Token* token) const noexcept {
  if (token == nullptr) {
    return {StatusCode::kInvalidArgument, "sample output is required"};
  }
  return unavailable("sampling");
}
Status Session::eval(Token) noexcept { return unavailable("evaluation"); }
Status Session::save(const std::string&) const noexcept {
  return unavailable("checkpoint save");
}
Status Session::restore(const std::string&) noexcept {
  return unavailable("checkpoint restore");
}

}  // namespace qw38
