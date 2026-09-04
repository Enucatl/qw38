#include "qw38/engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <utility>

#include "model.h"
#include "scalar_runtime.h"
#ifndef QW38_CUDA_RUNTIME
#include "host_checkpoint.h"
#endif
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
#ifdef QW38_CUDA_RUNTIME
constexpr std::size_t kVocabularySize = 248320;
constexpr std::size_t kResidualWidth = 5120;
#endif
constexpr std::uint64_t kDefaultSeed = 0x5157385357495353ULL;
constexpr const char* kCudaModelId = "qwen3.8-27b-q4_k_m";
constexpr const char* kHostModelId = "qwen3.5-2b-q4_k_m";
constexpr std::size_t kHostRssBudgetBytes = 10ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uintmax_t kPinnedCpuModelBytes = 1280835840ULL;
constexpr const char* kPinnedCpuModelSha256 =
    "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223";

#ifndef QW38_CUDA_RUNTIME
Status unavailable(const char* operation) noexcept {
  return {StatusCode::kUnimplemented,
          std::string(operation) + " has not passed its delivery gate"};
}
#endif

}  // namespace

struct Engine::Impl final {
  std::string model_path;
  std::string model_id;
  std::size_t context = 0;
  internal::ModelInfo model;
  internal::MappedFile mapping;
  internal::Tokenizer tokenizer;
#ifdef QW38_CUDA_RUNTIME
  internal::ModelWeights weights;
  std::shared_ptr<cuda::ResidentModel> resident;
#else
  internal::ModelWeights weights;
  internal::ScalarModelParameters parameters;
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
#else
  const internal::ModelWeights* weights = nullptr;
  const internal::ScalarModelParameters* parameters = nullptr;
  std::size_t context = 0;
  internal::ScalarSessionState state;
  internal::ScalarWorkspace workspace;
  std::vector<float> logits;
  std::vector<Token> history;
  mutable bool has_pending_sampler = false;
  mutable Token pending_token = 0;
  mutable internal::HostSamplerState pending_sampler;
#endif
};

Engine::Engine() noexcept = default;
Engine::~Engine() = default;
Engine::Engine(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

Status Engine::open(const std::string& model_path, Engine* engine) noexcept {
  return open(model_path, 0, engine);
}

Status Engine::open(const std::string& model_path, std::size_t context,
                    Engine* engine) noexcept {
  if (engine == nullptr || model_path.empty()) {
    return {StatusCode::kInvalidArgument, "model path and engine are required"};
  }
  std::error_code error;
  const std::uintmax_t bytes = std::filesystem::file_size(model_path, error);
  if (error) {
    return {StatusCode::kIoError, "cannot stat model: " + error.message()};
  }
  internal::ModelInfo model;
  Status status = internal::inspect_gguf(model_path, &model);
  if (!status.is_ok()) {
    return status;
  }
  internal::ModelGeometry geometry;
  status = internal::admit_pinned_geometry(&model, &geometry);
  if (!status.is_ok()) {
    return status;
  }
  if (geometry.identity == internal::kGeometryQwen38_27B) {
    if (bytes != kPinnedModelBytes) {
      return {StatusCode::kIncompatibleArtifact,
              "model byte size does not match the pinned artifact"};
    }
  }
  std::string digest;
  status = internal::sha256_file(model_path, &digest);
  if (!status.is_ok()) {
    return status;
  }
  if (geometry.identity == internal::kGeometryQwen38_27B &&
      digest != kPinnedModelSha256) {
    return {StatusCode::kIncompatibleArtifact,
            "model SHA-256 does not match the pinned artifact"};
  }
  if (geometry.identity == internal::kGeometryQwen35_2B) {
    if (kPinnedCpuModelBytes != 0 && bytes != kPinnedCpuModelBytes) {
      return {StatusCode::kIncompatibleArtifact,
              "2B model byte size does not match the pinned laptop artifact"};
    }
    if (kPinnedCpuModelSha256[0] != '\0' && digest != kPinnedCpuModelSha256) {
      return {StatusCode::kIncompatibleArtifact,
              "2B model SHA-256 does not match the pinned laptop artifact"};
    }
  }
#ifndef QW38_CUDA_RUNTIME
  if (geometry.identity == internal::kGeometryQwen38_27B) {
    return {StatusCode::kUnimplemented,
            "Qwen3.8-27B requires the CUDA production build"};
  }
#endif
  auto impl = std::make_unique<Impl>();
  status = impl->mapping.open(model_path);
  if (!status.is_ok()) {
    return status;
  }
  impl->model_path = model_path;
  impl->model = std::move(model);
  impl->model_id = geometry.identity == internal::kGeometryQwen35_2B
                       ? kHostModelId
                       : kCudaModelId;
  if (geometry.identity == internal::kGeometryQwen35_2B) {
    impl->context = context == 0 ? internal::kHostDefaultContext : context;
    if (impl->context == 0 || impl->context > internal::kHostMaximumContext) {
      return {StatusCode::kInvalidArgument,
              "host context must be between 1 and 8192 tokens"};
    }
  } else {
    impl->context = kSessionCapacity;
  }
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
#else
  status = internal::bind_model_weights(impl->model, impl->mapping,
                                        &impl->weights);
  if (!status.is_ok()) return status;
  status = internal::prepare_scalar_model_parameters(impl->weights,
                                                     &impl->parameters);
  if (!status.is_ok()) return status;
  const std::size_t estimated =
      impl->mapping.size() +
      geometry.gdn_layer_count * geometry.gdn_recurrent_values() * sizeof(float) +
      geometry.attention_layer_count * impl->context *
          geometry.attention_kv_width() * 2 * sizeof(float) +
      geometry.vocabulary * sizeof(float) * 2;
  if (estimated > kHostRssBudgetBytes) {
    return {StatusCode::kResourceExhausted,
            "host session would exceed the 10 GiB laptop RSS budget"};
  }
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
  if (impl_->weights.geometry.identity != internal::kGeometryQwen35_2B) {
    return unavailable("session creation");
  }
  auto session_impl = std::make_unique<Session::Impl>();
  session_impl->weights = &impl_->weights;
  session_impl->parameters = &impl_->parameters;
  session_impl->context = impl_->context;
  session_impl->logits.assign(impl_->weights.geometry.vocabulary, 0.0F);
  Status status = internal::create_scalar_session_state(
      impl_->weights.geometry, impl_->context, &session_impl->state);
  if (status.is_ok()) {
    status = internal::create_scalar_workspace(
        impl_->weights.geometry, impl_->context, &session_impl->workspace);
  }
  if (!status.is_ok()) return status;
  *session = std::unique_ptr<Session>(new Session(std::move(session_impl)));
  return Status::ok();
#endif
}

Status Engine::admitted_model(std::string* identity) const noexcept {
  if (identity == nullptr) {
    return {StatusCode::kInvalidArgument, "model identity output is required"};
  }
  if (!impl_) return {StatusCode::kInvalidArgument, "engine is not open"};
  *identity = impl_->model_id;
  return Status::ok();
}

Status Engine::context_capacity(std::size_t* capacity) const noexcept {
  if (capacity == nullptr) {
    return {StatusCode::kInvalidArgument, "context capacity output is required"};
  }
  if (!impl_) return {StatusCode::kInvalidArgument, "engine is not open"};
  *capacity = impl_->context;
  return Status::ok();
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
  input.canonical_tool_json = options.canonical_tools;
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
    converted.tool_calls.reserve(message.tool_calls.size());
    for (const ChatMessage::ToolCall& call : message.tool_calls) {
      converted.tool_calls.push_back({call.name, call.rendered_arguments});
    }
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
  return sync(tokens, nullptr);
}

namespace {
#ifdef QW38_CUDA_RUNTIME
Status poll_cancelled(void* context) noexcept {
  const auto* cancelled = static_cast<const std::atomic<bool>*>(context);
  return cancelled != nullptr && cancelled->load(std::memory_order_relaxed)
             ? Status(StatusCode::kCancelled, "request was cancelled")
             : Status::ok();
}
#else
Status host_cancelled(const std::atomic<bool>* cancelled) noexcept {
  return cancelled != nullptr && cancelled->load(std::memory_order_relaxed)
             ? Status(StatusCode::kCancelled, "request was cancelled")
             : Status::ok();
}
#endif
}  // namespace

Status Session::sync(const std::vector<Token>& tokens,
                     const std::atomic<bool>* cancelled) noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  impl_->has_pending_sampler = false;
  std::vector<std::size_t> widened(tokens.begin(), tokens.end());
  cuda::SyncResult result;
  const cuda::EvalControl control{poll_cancelled,
                                  const_cast<std::atomic<bool>*>(cancelled)};
  const cuda::EvalControl* control_pointer =
      cancelled == nullptr ? nullptr : &control;
  if (widened.empty()) {
    return cuda::sync_tokens(*impl_->model, nullptr, 0, &impl_->session,
                             &impl_->workspace, nullptr, 0, nullptr, 0,
                             &result, control_pointer);
  }
  return cuda::sync_tokens(
      *impl_->model, widened.data(), widened.size(), &impl_->session,
      &impl_->workspace, impl_->logits.data(), impl_->logits.size(),
      impl_->hidden.data(), impl_->hidden.size(), &result, control_pointer);
#else
  if (!impl_ || impl_->weights == nullptr) {
    return {StatusCode::kInvalidArgument, "session is not initialized"};
  }
  impl_->has_pending_sampler = false;
  Status status = host_cancelled(cancelled);
  if (!status.is_ok()) return status;
  std::size_t common = 0;
  const std::size_t limit =
      std::min(impl_->history.size(), tokens.size());
  while (common < limit && impl_->history[common] == tokens[common]) {
    ++common;
  }
  if (tokens.empty() || common < impl_->history.size()) {
    status = internal::create_scalar_session_state(
        impl_->weights->geometry, impl_->context, &impl_->state);
    if (status.is_ok()) {
      status = internal::create_scalar_workspace(
          impl_->weights->geometry, impl_->context,
          &impl_->workspace);
    }
    if (!status.is_ok()) return status;
    impl_->history.clear();
    impl_->logits.assign(impl_->weights->geometry.vocabulary, 0.0F);
    common = 0;
  }
  if (tokens.size() > impl_->context) {
    return {StatusCode::kInvalidArgument,
            "prompt exceeds the host context capacity"};
  }
  if (tokens.size() == common) {
    return Status::ok();
  }
  for (std::size_t index = common; index < tokens.size(); ++index) {
    status = host_cancelled(cancelled);
    if (!status.is_ok()) return status;
    status = internal::execute_scalar_token(
        *impl_->weights, *impl_->parameters, tokens[index],
        &impl_->state, &impl_->workspace, impl_->logits.data(),
        impl_->logits.size());
    if (!status.is_ok()) return status;
  }
  impl_->history.assign(tokens.begin(), tokens.end());
  return host_cancelled(cancelled);
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
  if (!impl_ || impl_->history.empty()) {
    return {StatusCode::kInvalidArgument, "session has no committed logits"};
  }
  *output = impl_->logits;
  return Status::ok();
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
  if (!impl_ || impl_->history.empty() ||
      !std::isfinite(config.temperature) || config.temperature < 0.0F ||
      !std::isfinite(config.top_p) || config.top_p <= 0.0F ||
      config.top_p > 1.0F ||
      config.top_k > impl_->weights->geometry.vocabulary) {
    return {StatusCode::kInvalidArgument, "sampler configuration is invalid"};
  }
  const std::size_t vocabulary = impl_->logits.size();
  if (config.temperature == 0.0F) {
    std::size_t selected = 0;
    float best = impl_->logits[0];
    for (std::size_t index = 1; index < vocabulary; ++index) {
      if (impl_->logits[index] > best) {
        best = impl_->logits[index];
        selected = index;
      }
    }
    impl_->has_pending_sampler = false;
    *token = static_cast<Token>(selected);
    return Status::ok();
  }
  internal::HostSamplerState state = impl_->pending_sampler;
  if (state.temperature != config.temperature || state.top_p != config.top_p ||
      state.top_k != config.top_k || state.seed != config.seed ||
      state.rng_state == 0) {
    state = {config.temperature, config.top_p, config.top_k, config.seed,
             config.seed == 0 ? kDefaultSeed : config.seed};
  }
  std::vector<std::size_t> candidates(vocabulary);
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
  const std::uint64_t random = state.rng_state * 0x2545F4914F6CDD1DULL;
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
#endif
}
Status Session::eval(Token token) noexcept {
  return eval(token, nullptr);
}

Status Session::eval(Token token,
                     const std::atomic<bool>* cancelled) noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  float elapsed = 0.0F;
  const cuda::EvalControl control{poll_cancelled,
                                  const_cast<std::atomic<bool>*>(cancelled)};
  Status status = cuda::execute_token(
      *impl_->model, token, &impl_->session, &impl_->workspace,
      impl_->logits.data(), impl_->logits.size(), impl_->hidden.data(),
      impl_->hidden.size(), &elapsed,
      cancelled == nullptr ? nullptr : &control, nullptr,
      cuda::PointwisePath::kFused, &impl_->graphs);
  if (status.is_ok() && impl_->has_pending_sampler &&
      impl_->pending_token == token) {
    status = impl_->session.set_sampler_state(impl_->pending_sampler);
  }
  impl_->has_pending_sampler = false;
  return status;
#else
  if (!impl_ || impl_->weights == nullptr) {
    return {StatusCode::kInvalidArgument, "session is not initialized"};
  }
  Status status = host_cancelled(cancelled);
  if (!status.is_ok()) return status;
  status = internal::execute_scalar_token(
      *impl_->weights, *impl_->parameters, token, &impl_->state,
      &impl_->workspace, impl_->logits.data(), impl_->logits.size());
  if (!status.is_ok()) return status;
  impl_->history.push_back(token);
  impl_->has_pending_sampler = false;
  return host_cancelled(cancelled);
#endif
}
Status Session::save(const std::string& path) const noexcept {
#ifdef QW38_CUDA_RUNTIME
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  return impl_->session.save_checkpoint(path);
#else
  if (!impl_ || impl_->weights == nullptr) {
    return {StatusCode::kInvalidArgument, "session is not initialized"};
  }
  return internal::save_host_checkpoint(
      path, impl_->weights->geometry, impl_->state, impl_->history,
      impl_->logits, impl_->pending_sampler);
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
  if (!impl_ || impl_->weights == nullptr) {
    return {StatusCode::kInvalidArgument, "session is not initialized"};
  }
  impl_->has_pending_sampler = false;
  return internal::restore_host_checkpoint(
      path, impl_->weights->geometry, &impl_->state, &impl_->history,
      &impl_->logits, &impl_->pending_sampler);
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
  if (!impl_) return {StatusCode::kInvalidArgument, "session is not initialized"};
  *output = impl_->history;
  return Status::ok();
#endif
}

}  // namespace qw38
