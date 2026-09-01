#include "qw38/engine.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <numeric>
#include <utility>

#include "model.h"
#include "sha256.h"
#include "template.h"
#include "tokenizer.h"
#include "weights.h"

#ifdef QW38_CUDA_RUNTIME
#include "full_scheduler.h"
#endif

namespace qw38 {
namespace {

constexpr std::uintmax_t kPinnedModelBytes = 18973870432ULL;
constexpr const char* kPinnedModelSha256 =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr std::size_t kSessionCapacity = 131072;
constexpr std::size_t kVocabularySize = 248320;
constexpr std::size_t kResidualWidth = 5120;
constexpr std::uint64_t kDefaultSeed = 0x5157385357495353ULL;

#ifndef QW38_CUDA_RUNTIME
Status unavailable(const char* operation) noexcept {
  return {StatusCode::kUnimplemented,
          std::string(operation) + " has not passed its delivery gate"};
}
#endif

}  // namespace

struct Engine::Impl final {
  std::string model_path;
  internal::ModelInfo model;
  internal::MappedFile mapping;
  internal::Tokenizer tokenizer;
#ifdef QW38_CUDA_RUNTIME
  internal::ModelWeights weights;
  std::shared_ptr<cuda::ResidentModel> resident;
#endif
};

struct Session::Impl final {
#ifdef QW38_CUDA_RUNTIME
  std::shared_ptr<const cuda::ResidentModel> model;
  cuda::SchedulerSession session;
  cuda::SchedulerWorkspace workspace;
  cuda::SchedulerGraphs graphs;
  std::vector<float> logits;
  std::vector<float> hidden;
  mutable bool has_pending_sampler = false;
  mutable Token pending_token = 0;
  mutable cuda::SamplerState pending_sampler;
#endif
};

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
  status = impl->tokenizer.build(impl->model);
  if (!status.is_ok()) {
    return status;
  }
#ifdef QW38_CUDA_RUNTIME
  status = internal::bind_model_weights(impl->model, impl->mapping,
                                        &impl->weights);
  if (!status.is_ok()) return status;
  impl->resident = std::make_shared<cuda::ResidentModel>();
  status = impl->resident->upload(impl->weights, impl->mapping.data(),
                                  impl->mapping.size());
  if (!status.is_ok()) return status;
#endif
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
#ifdef QW38_CUDA_RUNTIME
  auto session_impl = std::make_unique<Session::Impl>();
  session_impl->model = impl_->resident;
  session_impl->logits.resize(kVocabularySize);
  session_impl->hidden.resize(kResidualWidth);
  Status status = session_impl->session.create(kSessionCapacity);
  if (status.is_ok()) status = session_impl->workspace.create(kSessionCapacity);
  if (status.is_ok()) {
    status = session_impl->graphs.create(*impl_->resident,
                                         &session_impl->workspace);
  }
  if (!status.is_ok()) return status;
  *session = std::unique_ptr<Session>(new Session(std::move(session_impl)));
  return Status::ok();
#else
  return unavailable("session creation");
#endif
}

Status Engine::encode(const std::string& utf8,
                      std::vector<Token>* tokens) const noexcept {
  if (!impl_) return {StatusCode::kInvalidArgument, "engine is not open"};
  return impl_->tokenizer.encode(utf8, tokens);
}

Status Engine::decode(const std::vector<Token>& tokens,
                      bool skip_special_tokens,
                      std::string* utf8) const noexcept {
  if (!impl_) return {StatusCode::kInvalidArgument, "engine is not open"};
  return impl_->tokenizer.decode(tokens, skip_special_tokens, utf8);
}

Status Engine::render_chat(const std::vector<ChatMessage>& messages,
                           const ChatOptions& options,
                           std::vector<Token>* tokens) const noexcept {
  if (!impl_ || tokens == nullptr) {
    return {StatusCode::kInvalidArgument,
            "open engine and rendered token output are required"};
  }
  internal::TemplateInput input;
  input.options.enable_thinking = options.enable_thinking;
  input.options.reasoning_effort = options.reasoning_effort;
  input.options.preserve_thinking = options.preserve_thinking;
  input.messages.reserve(messages.size());
  for (const ChatMessage& message : messages) {
    internal::Message converted;
    if (message.role == ChatRole::kSystem) {
      converted.role = internal::MessageRole::kSystem;
    } else if (message.role == ChatRole::kDeveloper) {
      converted.role = internal::MessageRole::kDeveloper;
    } else if (message.role == ChatRole::kUser) {
      converted.role = internal::MessageRole::kUser;
    } else if (message.role == ChatRole::kAssistant) {
      converted.role = internal::MessageRole::kAssistant;
    } else {
      converted.role = internal::MessageRole::kTool;
    }
    converted.content = message.content;
    converted.reasoning_content = message.reasoning_content;
    input.messages.push_back(std::move(converted));
  }
  std::string rendered;
  Status status = internal::render_chat(input, &rendered);
  if (!status.is_ok()) return status;
  return impl_->tokenizer.encode(rendered, tokens);
}

Status Engine::render_user_turn(const std::string& content,
                                bool enable_thinking,
                                std::vector<Token>* tokens) const noexcept {
  if (!impl_ || tokens == nullptr) {
    return {StatusCode::kInvalidArgument,
            "open engine and rendered token output are required"};
  }
  std::string rendered;
  Status status =
      internal::render_user_turn(content, enable_thinking, &rendered);
  if (!status.is_ok()) return status;
  return impl_->tokenizer.encode(rendered, tokens);
}

Session::Session() noexcept = default;
Session::~Session() = default;
Session::Session(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Status Session::sync(const std::vector<Token>& tokens) noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  impl_->has_pending_sampler = false;
  std::vector<std::size_t> widened(tokens.begin(), tokens.end());
  cuda::SyncResult result;
  if (widened.empty()) {
    return cuda::sync_tokens(*impl_->model, nullptr, 0, &impl_->session,
                             &impl_->workspace, nullptr, 0, nullptr, 0,
                             &result);
  }
  return cuda::sync_tokens(
      *impl_->model, widened.data(), widened.size(), &impl_->session,
      &impl_->workspace, impl_->logits.data(), impl_->logits.size(),
      impl_->hidden.data(), impl_->hidden.size(), &result);
#else
  (void)tokens;
  return unavailable("prefix synchronization");
#endif
}
Status Session::logits(std::vector<float>* output) const noexcept {
  if (output == nullptr) {
    return {StatusCode::kInvalidArgument, "logits output is required"};
  }
#ifdef QW38_CUDA_RUNTIME
  if (!impl_ || impl_->session.frontier() == 0) {
    return {StatusCode::kInvalidArgument, "session has no committed logits"};
  }
  *output = impl_->logits;
  return Status::ok();
#else
  return unavailable("logits");
#endif
}
Status Session::sample(const SamplerConfig& config, Token* token) const noexcept {
  if (token == nullptr) {
    return {StatusCode::kInvalidArgument, "sample output is required"};
  }
#ifdef QW38_CUDA_RUNTIME
  if (!impl_ || impl_->session.frontier() == 0 ||
      !std::isfinite(config.temperature) || config.temperature < 0.0F ||
      !std::isfinite(config.top_p) || config.top_p <= 0.0F ||
      config.top_p > 1.0F || config.top_k > kVocabularySize) {
    return {StatusCode::kInvalidArgument, "sampler configuration is invalid"};
  }
  if (config.temperature == 0.0F) {
    std::size_t selected = 0;
    Status status = cuda::greedy_sample(impl_->session, &selected);
    if (status.is_ok()) {
      impl_->has_pending_sampler = false;
      *token = static_cast<Token>(selected);
    }
    return status;
  }
  cuda::SamplerState state = impl_->session.sampler_state();
  if (state.temperature != config.temperature || state.top_p != config.top_p ||
      state.top_k != config.top_k || state.seed != config.seed ||
      state.rng_state == 0) {
    state = {config.temperature, config.top_p, config.top_k, config.seed,
             config.seed == 0 ? kDefaultSeed : config.seed};
  }
  std::vector<std::size_t> candidates(kVocabularySize);
  std::iota(candidates.begin(), candidates.end(), 0);
  const bool ordered = config.top_k > 0 || config.top_p < 1.0F;
  if (ordered) {
    const std::size_t retained =
        config.top_k == 0 ? candidates.size()
                          : std::min<std::size_t>(config.top_k,
                                                  candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + retained, candidates.end(),
        [this](std::size_t left, std::size_t right) {
          return impl_->logits[left] > impl_->logits[right];
        });
    candidates.resize(retained);
  }
  float maximum = -INFINITY;
  for (std::size_t index : candidates) {
    maximum = std::max(maximum, impl_->logits[index]);
  }
  std::vector<double> weights(candidates.size());
  double total = 0.0;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    weights[index] = std::exp(static_cast<double>(
        (impl_->logits[candidates[index]] - maximum) / config.temperature));
    total += weights[index];
  }
  if (ordered && config.top_p < 1.0F) {
    const double threshold = total * config.top_p;
    double cumulative = 0.0;
    std::size_t retained = 0;
    do {
      cumulative += weights[retained++];
    } while (retained < weights.size() && cumulative < threshold);
    candidates.resize(retained);
    weights.resize(retained);
    total = cumulative;
  }
  state.rng_state ^= state.rng_state >> 12U;
  state.rng_state ^= state.rng_state << 25U;
  state.rng_state ^= state.rng_state >> 27U;
  const std::uint64_t random =
      state.rng_state * 0x2545F4914F6CDD1DULL;
  const double unit = static_cast<double>(random >> 11U) /
                      static_cast<double>(1ULL << 53U);
  const double target = unit * total;
  double cumulative = 0.0;
  std::size_t selected = candidates.size() - 1;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    cumulative += weights[index];
    if (target < cumulative) {
      selected = index;
      break;
    }
  }
  impl_->pending_token = static_cast<Token>(candidates[selected]);
  impl_->pending_sampler = state;
  impl_->has_pending_sampler = true;
  *token = impl_->pending_token;
  return Status::ok();
#else
  (void)config;
  return unavailable("sampling");
#endif
}
Status Session::eval(Token token) noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  float elapsed = 0.0F;
  Status status = cuda::execute_token(
      *impl_->model, token, &impl_->session, &impl_->workspace,
      impl_->logits.data(), impl_->logits.size(), impl_->hidden.data(),
      impl_->hidden.size(), &elapsed, nullptr, nullptr,
      cuda::PointwisePath::kFused, &impl_->graphs);
  if (status.is_ok() && impl_->has_pending_sampler &&
      impl_->pending_token == token) {
    status = impl_->session.set_sampler_state(impl_->pending_sampler);
  }
  impl_->has_pending_sampler = false;
  return status;
#else
  (void)token;
  return unavailable("evaluation");
#endif
}
Status Session::save(const std::string& path) const noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  return impl_->session.save_checkpoint(path);
#else
  (void)path;
  return unavailable("checkpoint save");
#endif
}
Status Session::restore(const std::string& path) noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  impl_->has_pending_sampler = false;
  Status status = impl_->session.restore_checkpoint(path, &impl_->workspace);
  if (status.is_ok() && impl_->session.frontier() > 0) {
    status = impl_->session.copy_last_outputs(
        impl_->logits.data(), impl_->logits.size(), impl_->hidden.data(),
        impl_->hidden.size());
  }
  return status;
#else
  (void)path;
  return unavailable("checkpoint restore");
#endif
}

Status Session::tokens(std::vector<Token>* output) const noexcept {
  if (output == nullptr) {
    return {StatusCode::kInvalidArgument, "token history output is required"};
  }
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  std::vector<std::size_t> widened(impl_->session.token_count());
  Status status = impl_->session.copy_tokens(widened.data(), widened.size());
  if (!status.is_ok()) return status;
  output->assign(widened.begin(), widened.end());
  return Status::ok();
#else
  return unavailable("token history");
#endif
}

}  // namespace qw38
