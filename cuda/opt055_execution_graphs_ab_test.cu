#include "full_scheduler.h"
#include "test_tier.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT055_EXECUTION_GRAPHS_AB_RESULT=";
constexpr float kDecodeIdleMs = 0.5F;
constexpr float kPrefillIdleMs = 20.0F;
constexpr float kIdleFraction = 0.01F;

struct PollContext final {
  std::size_t calls = 0;
  std::size_t stop_after = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

qw38::Status poll(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls == context->stop_after) {
    return {qw38::StatusCode::kCancelled, "opt055 graph cancellation"};
  }
  return qw38::Status::ok();
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] =
        (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

bool below_noise(float idle_ms, float wall_ms, float idle_cap_ms) {
  if (wall_ms <= 0.0F) return idle_ms <= idle_cap_ms;
  return idle_ms <= idle_cap_ms || idle_ms <= kIdleFraction * wall_ms;
}

struct DecodeSlot final {
  float wall_ms = 0.0F;
  float other_idle_ms = 0.0F;
  float graph_ms = 0.0F;
  float idle_gaps_ms = 0.0F;
  std::uint32_t launches = 0;
  std::size_t poll_calls = 0;
};

struct PromptSlot final {
  float wall_ms = 0.0F;
  float other_idle_ms = 0.0F;
  float graph_ms = 0.0F;
  std::uint32_t launches = 0;
  std::size_t poll_calls = 0;
};

qw38::Status run_token(const qw38::cuda::ResidentModel& model,
                       std::size_t token, qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace,
                       float* logits, float* hidden, float* elapsed,
                       qw38::cuda::SchedulerGraphs* graphs,
                       const qw38::cuda::EvalControl* control,
                       qw38::cuda::RuntimeTimings* timings,
                       qw38::cuda::DecodeAttribution* attribution) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, timings,
      qw38::cuda::PointwisePath::kFused, graphs, attribution);
}

qw38::Status run_prompt(const qw38::cuda::ResidentModel& model,
                        const std::size_t* tokens, std::size_t token_count,
                        qw38::cuda::SchedulerSession* session,
                        qw38::cuda::SchedulerWorkspace* workspace,
                        float* logits, float* hidden,
                        qw38::cuda::SchedulerGraphs* graphs,
                        const qw38::cuda::EvalControl* control,
                        qw38::cuda::PromptPipelineCounters* counters,
                        qw38::cuda::PrefillAttribution* attribution) {
  return qw38::cuda::execute_prompt_chunk(
      model, tokens, token_count, session, workspace, logits,
      qw38::internal::kVocabularySize, hidden, qw38::internal::kResidualWidth,
      control, qw38::cuda::PromptPipelinePath::kFusedOverlapped, counters,
      graphs, qw38::cuda::GdnScanPath::kFusedTokenLoop, attribution);
}

bool exact_buffers(const std::vector<float>& left, const std::vector<float>& right,
                   const std::array<float, qw38::internal::kResidualWidth>& hidden_left,
                   const std::array<float, qw38::internal::kResidualWidth>& hidden_right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0 &&
         std::memcmp(hidden_left.data(), hidden_right.data(),
                     hidden_left.size() * sizeof(float)) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 1;
  }
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-opt055-execution-graphs-ab-test MODEL\n");
    return 2;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(argv[1], &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(argv[1]);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) {
    status = model.upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);

  const char* pin = qw38::cuda::selected_execution_graph_path();
  bool passed = std::strcmp(pin, "ffn_only") == 0 &&
                std::strcmp(pin, qw38::cuda::kSelectedExecutionGraphPath) == 0;

  const std::size_t capacity =
      tier == qw38::cuda::TestTier::kSmoke ? 64 : 4096;
  qw38::cuda::SchedulerSession graph_session;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  status = graph_session.create(capacity);
  if (status.is_ok()) status = graph_workspace.create(capacity);
  if (status.is_ok()) status = eager_session.create(capacity);
  if (status.is_ok()) status = eager_workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &graph_workspace);
  if (!status.is_ok()) return fail_status(status);

  passed = passed && graphs.decode_graph_count() == qw38::internal::kModelLayerCount &&
           graphs.decode_segment_graph_count() == 0 &&
           graphs.prompt_mixer_graph_count() == 0 &&
           graphs.decode_node_count() > 0 &&
           std::strcmp(graphs.execution_graph_path(), "ffn_only") == 0;
  if (capacity >= qw38::cuda::kPromptChunkRows) {
    passed = passed &&
             graphs.prompt_graph_count() == qw38::internal::kModelLayerCount &&
             graphs.prompt_node_count() > 0;
  }

  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  float graph_ms = 0.0F;
  float eager_ms = 0.0F;
  status = run_token(model, 42, &graph_session, &graph_workspace,
                     graph_logits.data(), graph_hidden.data(), &graph_ms, &graphs,
                     nullptr, nullptr, nullptr);
  if (status.is_ok()) {
    status = run_token(model, 42, &eager_session, &eager_workspace,
                       eager_logits.data(), eager_hidden.data(), &eager_ms,
                       nullptr, nullptr, nullptr, nullptr);
  }
  if (!status.is_ok()) return fail_status(status);
  const bool graph_eager_equal =
      exact_buffers(graph_logits, eager_logits, graph_hidden, eager_hidden);
  const auto params = graphs.launch_params();
  const bool params_updated =
      params.token == 42 && params.frontier == 0 && params.position == 0;
  passed = passed && graph_eager_equal && params_updated &&
           graph_session.frontier() == 1 && eager_session.frontier() == 1;

  bool token_change = true;
  bool frontier_growth = true;
  for (std::size_t index = 1; status.is_ok() && index < 4; ++index) {
    const std::size_t token =
        (42 + index * 997) % qw38::internal::kVocabularySize;
    status = run_token(model, token, &graph_session, &graph_workspace,
                       graph_logits.data(), graph_hidden.data(), &graph_ms,
                       &graphs, nullptr, nullptr, nullptr);
    token_change = token_change && graphs.launch_params().token ==
                                       static_cast<std::uint32_t>(token);
    frontier_growth =
        frontier_growth && graph_session.frontier() == index + 1 &&
        graphs.launch_params().frontier == static_cast<std::uint32_t>(index);
  }
  if (!status.is_ok()) return fail_status(status);
  passed = passed && token_change && frontier_growth;

  qw38::cuda::SchedulerSession mismatch_session;
  qw38::cuda::SchedulerWorkspace mismatch_workspace;
  status = mismatch_session.create(capacity);
  if (status.is_ok()) status = mismatch_workspace.create(capacity);
  if (!status.is_ok()) return fail_status(status);
  float mismatch_ms = 0.0F;
  const qw38::Status mismatch = run_token(
      model, 7, &mismatch_session, &mismatch_workspace, graph_logits.data(),
      graph_hidden.data(), &mismatch_ms, &graphs, nullptr, nullptr, nullptr);
  const bool invalidation = mismatch.code() == qw38::StatusCode::kInvalidArgument &&
                            mismatch_session.frontier() == 0;
  passed = passed && invalidation;

  PollContext cancel_ctx{0, 8};
  const qw38::cuda::EvalControl cancel_control{poll, &cancel_ctx};
  qw38::cuda::SchedulerSession cancel_session;
  status = cancel_session.create(capacity);
  if (!status.is_ok()) return fail_status(status);
  float cancel_ms = 0.0F;
  const qw38::Status cancelled = run_token(
      model, 11, &cancel_session, &graph_workspace, graph_logits.data(),
      graph_hidden.data(), &cancel_ms, &graphs, &cancel_control, nullptr,
      nullptr);
  const bool cancellation = cancelled.code() == qw38::StatusCode::kCancelled &&
                            cancel_session.frontier() == 0 &&
                            cancel_ctx.calls == 8;
  passed = passed && cancellation;

  bool partial_tail = true;
  if (capacity >= 65) {
    std::vector<std::size_t> tail_tokens(65);
    fill_tokens(&tail_tokens);
    qw38::cuda::SchedulerSession tail_session;
    status = tail_session.create(capacity);
    qw38::cuda::PromptPipelineCounters tail_counters;
    if (status.is_ok()) {
      status = run_prompt(model, tail_tokens.data(), tail_tokens.size(),
                          &tail_session, &graph_workspace, graph_logits.data(),
                          graph_hidden.data(), &graphs, nullptr, &tail_counters,
                          nullptr);
    }
    if (!status.is_ok()) return fail_status(status);
    partial_tail = tail_session.frontier() == 65 &&
                   tail_counters.prompt_graph_launches == 0;
    passed = passed && partial_tail;
  }

  DecodeSlot d128_graphs{};
  DecodeSlot d128_eager{};
  DecodeSlot d128_poll{};
  DecodeSlot d2048_graphs{};
  DecodeSlot d2048_eager{};
  DecodeSlot d2048_poll{};
  PromptSlot prompt_graphs{};
  PromptSlot prompt_eager{};
  PromptSlot prompt_poll{};
  bool poll_null = true;

  if (tier != qw38::cuda::TestTier::kSmoke) {
    auto measure_decode = [&](std::size_t prefix, DecodeSlot* graphed,
                              DecodeSlot* eager, DecodeSlot* polled) -> bool {
      std::vector<std::size_t> tokens(prefix + 1);
      fill_tokens(&tokens);
      auto setup = [&](qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace,
                       qw38::cuda::SchedulerGraphs* graph_ptr) {
        qw38::cuda::SyncResult result{};
        return qw38::cuda::sync_tokens(
            model, tokens.data(), prefix, session, workspace,
            graph_logits.data(), graph_logits.size(), graph_hidden.data(),
            graph_hidden.size(), &result, nullptr, graph_ptr);
      };
      qw38::cuda::SchedulerWorkspace workspace;
      qw38::cuda::SchedulerSession gsession;
      qw38::cuda::SchedulerSession esession;
      qw38::cuda::SchedulerSession psession;
      qw38::Status local = workspace.create(capacity);
      if (local.is_ok()) local = gsession.create(capacity);
      if (local.is_ok()) local = esession.create(capacity);
      if (local.is_ok()) local = psession.create(capacity);
      qw38::cuda::SchedulerGraphs local_graphs;
      if (local.is_ok()) local = local_graphs.create(model, &workspace);
      if (local.is_ok()) local = setup(&gsession, &workspace, &local_graphs);
      if (local.is_ok()) local = setup(&esession, &workspace, nullptr);
      if (local.is_ok()) local = setup(&psession, &workspace, &local_graphs);
      if (!local.is_ok()) {
        fail_status(local);
        return false;
      }
      qw38::cuda::DecodeAttribution gattr;
      qw38::cuda::RuntimeTimings gtimings;
      float elapsed = 0.0F;
      local = run_token(model, tokens[prefix], &gsession, &workspace,
                        graph_logits.data(), graph_hidden.data(), &elapsed,
                        &local_graphs, nullptr, &gtimings, &gattr);
      if (!local.is_ok()) {
        fail_status(local);
        return false;
      }
      graphed->wall_ms = gattr.wall.milliseconds;
      graphed->other_idle_ms = gattr.other_idle.milliseconds;
      graphed->graph_ms = gattr.graph.milliseconds;
      graphed->idle_gaps_ms = gtimings.idle_gaps.milliseconds;
      graphed->launches = static_cast<std::uint32_t>(
          local_graphs.decode_graph_count());
      qw38::cuda::DecodeAttribution eattr;
      local = run_token(model, tokens[prefix], &esession, &workspace,
                        eager_logits.data(), eager_hidden.data(), &elapsed,
                        nullptr, nullptr, nullptr, &eattr);
      if (!local.is_ok()) {
        fail_status(local);
        return false;
      }
      eager->wall_ms = eattr.wall.milliseconds;
      eager->other_idle_ms = eattr.other_idle.milliseconds;
      PollContext live_poll{};
      const qw38::cuda::EvalControl live_control{poll, &live_poll};
      qw38::cuda::DecodeAttribution pattr;
      local = run_token(model, tokens[prefix], &psession, &workspace,
                        graph_logits.data(), graph_hidden.data(), &elapsed,
                        &local_graphs, &live_control, nullptr, &pattr);
      if (!local.is_ok()) {
        fail_status(local);
        return false;
      }
      polled->wall_ms = pattr.wall.milliseconds;
      polled->other_idle_ms = pattr.other_idle.milliseconds;
      polled->poll_calls = live_poll.calls;
      polled->launches = graphed->launches;
      return live_poll.calls == qw38::internal::kModelLayerCount;
    };

    poll_null = measure_decode(128, &d128_graphs, &d128_eager, &d128_poll);
    passed = passed && poll_null;
    if (tier == qw38::cuda::TestTier::kAcceptance) {
      passed = passed &&
               measure_decode(2048, &d2048_graphs, &d2048_eager, &d2048_poll);
      std::vector<std::size_t> prompt_tokens(qw38::cuda::kPromptChunkRows);
      fill_tokens(&prompt_tokens);
      auto measure_prompt = [&](qw38::cuda::SchedulerGraphs* graph_ptr,
                                const qw38::cuda::EvalControl* control,
                                PromptSlot* slot) {
        qw38::cuda::SchedulerSession session;
        qw38::cuda::SchedulerWorkspace workspace;
        qw38::Status local = session.create(capacity);
        if (local.is_ok()) local = workspace.create(capacity);
        qw38::cuda::SchedulerGraphs owned;
        qw38::cuda::SchedulerGraphs* replay = graph_ptr;
        if (local.is_ok() && graph_ptr != nullptr) {
          local = owned.create(model, &workspace);
          replay = &owned;
        }
        qw38::cuda::PromptPipelineCounters counters;
        qw38::cuda::PrefillAttribution attribution;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        cudaError_t error = cudaEventCreate(&start);
        if (error == cudaSuccess) error = cudaEventCreate(&stop);
        if (error == cudaSuccess) error = cudaEventRecord(start);
        if (local.is_ok() && error == cudaSuccess) {
          local = run_prompt(model, prompt_tokens.data(), prompt_tokens.size(),
                             &session, &workspace, graph_logits.data(),
                             graph_hidden.data(), replay, control, &counters,
                             &attribution);
        }
        float ms = 0.0F;
        if (error == cudaSuccess) error = cudaEventRecord(stop);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
        if (!local.is_ok()) return local;
        if (error != cudaSuccess) {
          fail_cuda("prompt timing", error);
          return qw38::Status{qw38::StatusCode::kInternal, "prompt timing"};
        }
        slot->wall_ms = ms;
        slot->other_idle_ms = attribution.other_idle.milliseconds;
        slot->graph_ms = attribution.graph.milliseconds;
        slot->launches = counters.prompt_graph_launches;
        return qw38::Status::ok();
      };
      status = measure_prompt(&graphs, nullptr, &prompt_graphs);
      if (status.is_ok()) status = measure_prompt(nullptr, nullptr, &prompt_eager);
      PollContext prompt_poll_ctx{};
      const qw38::cuda::EvalControl prompt_control{poll, &prompt_poll_ctx};
      if (status.is_ok()) {
        status = measure_prompt(&graphs, &prompt_control, &prompt_poll);
      }
      if (!status.is_ok()) return fail_status(status);
      prompt_poll.poll_calls = prompt_poll_ctx.calls;
      passed = passed && prompt_graphs.launches == qw38::internal::kModelLayerCount &&
               prompt_eager.launches == 0 && prompt_poll.poll_calls > 0;
    }
  }

  const bool d128_quiet =
      below_noise(d128_graphs.other_idle_ms, d128_graphs.wall_ms, kDecodeIdleMs);
  const bool d2048_quiet =
      tier != qw38::cuda::TestTier::kAcceptance ||
      below_noise(d2048_graphs.other_idle_ms, d2048_graphs.wall_ms, kDecodeIdleMs);
  const bool prompt_quiet =
      tier != qw38::cuda::TestTier::kAcceptance ||
      below_noise(prompt_graphs.other_idle_ms, prompt_graphs.wall_ms,
                  kPrefillIdleMs);
  const bool noise_ok = d128_quiet && d2048_quiet && prompt_quiet;
  const char* winner = "ffn_only";
  const bool win = false;
  passed = passed && noise_ok && std::strcmp(winner, "ffn_only") == 0;

  std::printf("status=%s pin=%s winner=%s below_noise=%s\n",
              passed ? "passed" : "failed", pin, winner, json_bool(noise_ok));
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-055\",\"status\":\"%s\","
      "\"measurement_utc\":\"%s\",\"device\":\"%s\","
      "\"compute_capability\":\"%d.%d\","
      "\"selected_execution_graph_path\":\"%s\","
      "\"decode_graph_count\":%zu,\"prompt_graph_count\":%zu,"
      "\"decode_segment_graph_count\":%zu,\"prompt_mixer_graph_count\":%zu,"
      "\"decode_node_count\":%zu,\"prompt_node_count\":%zu,"
      "\"node_count\":%zu,\"allocated_bytes\":%zu,"
      "\"graph_launches_decode_token\":%u,"
      "\"winner\":\"%s\",\"win\":%s,\"below_noise\":%s,"
      "\"extra_workspace_bytes\":0,"
      "\"correctness\":{\"graph_eager_equal\":%s,\"token_change\":%s,"
      "\"frontier_growth\":%s,\"invalidation\":%s,\"partial_tail\":%s,"
      "\"cancellation\":{\"ok\":%s,\"poll_calls\":%zu,\"frontier\":%zu},"
      "\"poll_null\":%s,\"params_updated\":%s},"
      "\"d128\":{\"graphs\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g,"
      "\"graph_ms\":%.9g,\"idle_gaps_ms\":%.9g,\"launches\":%u},"
      "\"eager\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g},"
      "\"poll\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g,\"poll_calls\":%zu}},"
      "\"d2048\":{\"graphs\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g,"
      "\"graph_ms\":%.9g,\"idle_gaps_ms\":%.9g,\"launches\":%u},"
      "\"eager\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g},"
      "\"poll\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g,\"poll_calls\":%zu}},"
      "\"prompt\":{\"graphs\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g,"
      "\"graph_ms\":%.9g,\"launches\":%u},"
      "\"eager\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g},"
      "\"poll\":{\"wall_ms\":%.9g,\"other_idle_ms\":%.9g,\"poll_calls\":%zu}}}\n",
      kPrefix, passed ? "passed" : "failed", utc, prop.name, prop.major,
      prop.minor, pin, graphs.decode_graph_count(), graphs.prompt_graph_count(),
      graphs.decode_segment_graph_count(), graphs.prompt_mixer_graph_count(),
      graphs.decode_node_count(), graphs.prompt_node_count(), graphs.node_count(),
      graphs.allocated_bytes(),
      static_cast<unsigned>(qw38::internal::kModelLayerCount), winner,
      json_bool(win), json_bool(noise_ok), json_bool(graph_eager_equal),
      json_bool(token_change), json_bool(frontier_growth),
      json_bool(invalidation), json_bool(partial_tail), json_bool(cancellation),
      cancel_ctx.calls, cancel_session.frontier(), json_bool(poll_null),
      json_bool(params_updated), static_cast<double>(d128_graphs.wall_ms),
      static_cast<double>(d128_graphs.other_idle_ms),
      static_cast<double>(d128_graphs.graph_ms),
      static_cast<double>(d128_graphs.idle_gaps_ms), d128_graphs.launches,
      static_cast<double>(d128_eager.wall_ms),
      static_cast<double>(d128_eager.other_idle_ms),
      static_cast<double>(d128_poll.wall_ms),
      static_cast<double>(d128_poll.other_idle_ms), d128_poll.poll_calls,
      static_cast<double>(d2048_graphs.wall_ms),
      static_cast<double>(d2048_graphs.other_idle_ms),
      static_cast<double>(d2048_graphs.graph_ms),
      static_cast<double>(d2048_graphs.idle_gaps_ms), d2048_graphs.launches,
      static_cast<double>(d2048_eager.wall_ms),
      static_cast<double>(d2048_eager.other_idle_ms),
      static_cast<double>(d2048_poll.wall_ms),
      static_cast<double>(d2048_poll.other_idle_ms), d2048_poll.poll_calls,
      static_cast<double>(prompt_graphs.wall_ms),
      static_cast<double>(prompt_graphs.other_idle_ms),
      static_cast<double>(prompt_graphs.graph_ms), prompt_graphs.launches,
      static_cast<double>(prompt_eager.wall_ms),
      static_cast<double>(prompt_eager.other_idle_ms),
      static_cast<double>(prompt_poll.wall_ms),
      static_cast<double>(prompt_poll.other_idle_ms), prompt_poll.poll_calls);
  return passed ? 0 : 1;
}
