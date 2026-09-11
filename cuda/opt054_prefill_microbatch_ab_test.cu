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

constexpr char kPrefix[] = "QW38_OPT054_PREFILL_MICROBATCH_AB_RESULT=";
constexpr std::size_t kLegal[] = {512, 1024, 2048, 4096};
constexpr int kLegalCount = 4;
constexpr std::size_t kPrompt = 4096;

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
    return {qw38::StatusCode::kCancelled, "opt054 microbatch cancellation"};
  }
  return qw38::Status::ok();
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] =
        (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

qw38::Status run_chunk(const qw38::cuda::ResidentModel& model,
                       const std::size_t* tokens, std::size_t token_count,
                       std::size_t capacity, std::size_t microbatch,
                       bool use_graphs, const qw38::cuda::EvalControl* control,
                       std::size_t* frontier) {
  qw38::cuda::PromptMicrobatchRowsScope scope(microbatch);
  qw38::cuda::SchedulerSession local;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = local.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  qw38::cuda::SchedulerGraphs* graph_ptr = nullptr;
  if (status.is_ok() && use_graphs) {
    status = graphs.create(model, &workspace);
    if (status.is_ok()) graph_ptr = &graphs;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok()) {
    status = qw38::cuda::execute_prompt_chunk(
        model, tokens, token_count, &local, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), control,
        qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, graph_ptr);
  }
  if (frontier != nullptr) *frontier = local.frontier();
  return status;
}

float time_prefixed_chunk(const qw38::cuda::ResidentModel& model,
                          const std::size_t* prefix_tokens, std::size_t prefix,
                          const std::size_t* chunk_tokens,
                          std::size_t chunk_count, std::size_t capacity,
                          std::size_t microbatch, bool use_graphs,
                          cudaError_t* error) {
  qw38::cuda::PromptMicrobatchRowsScope scope(microbatch);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  qw38::cuda::SchedulerGraphs* graph_ptr = nullptr;
  if (status.is_ok() && use_graphs) {
    status = graphs.create(model, &workspace);
    if (status.is_ok()) graph_ptr = &graphs;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok() && prefix > 0) {
    status = qw38::cuda::execute_prompt_chunk(
        model, prefix_tokens, prefix, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), nullptr,
        qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, graph_ptr);
  }
  if (!status.is_ok()) {
    *error = cudaErrorUnknown;
    std::fprintf(stderr, "%s\n", status.message().c_str());
    return 0.0F;
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  *error = cudaEventCreate(&start);
  if (*error == cudaSuccess) *error = cudaEventCreate(&stop);
  if (*error == cudaSuccess) *error = cudaEventRecord(start);
  if (*error == cudaSuccess) {
    status = qw38::cuda::execute_prompt_chunk(
        model, chunk_tokens, chunk_count, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), nullptr,
        qw38::cuda::PromptPipelinePath::kFusedOverlapped, nullptr, graph_ptr);
    if (!status.is_ok()) {
      std::fprintf(stderr, "%s\n", status.message().c_str());
      *error = cudaErrorUnknown;
    }
  }
  float ms = 0.0F;
  if (*error == cudaSuccess) *error = cudaEventRecord(stop);
  if (*error == cudaSuccess) *error = cudaEventSynchronize(stop);
  if (*error == cudaSuccess) *error = cudaEventElapsedTime(&ms, start, stop);
  if (start != nullptr) cudaEventDestroy(start);
  if (stop != nullptr) cudaEventDestroy(stop);
  if (*error == cudaSuccess && session.frontier() != prefix + chunk_count) {
    *error = cudaErrorUnknown;
  }
  return ms;
}

float time_chunk(const qw38::cuda::ResidentModel& model,
                 const std::size_t* tokens, std::size_t token_count,
                 std::size_t capacity, std::size_t microbatch, bool use_graphs,
                 cudaError_t* error) {
  return time_prefixed_chunk(model, nullptr, 0, tokens, token_count, capacity,
                             microbatch, use_graphs, error);
}

bool cancel_ok(const qw38::cuda::ResidentModel& model,
               const std::size_t* tokens, std::size_t token_count,
               std::size_t capacity, std::size_t microbatch,
               std::size_t stop_after, std::size_t* poll_calls,
               std::size_t* frontier) {
  PollContext context{0, stop_after};
  const qw38::cuda::EvalControl control{poll, &context};
  const qw38::Status status =
      run_chunk(model, tokens, token_count, capacity, microbatch, false,
                &control, frontier);
  *poll_calls = context.calls;
  return status.code() == qw38::StatusCode::kCancelled && *frontier == 0 &&
         context.calls == stop_after;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 1;
  }
  if (argc != 2) {
    std::fprintf(stderr, "usage: qw38-cuda-opt054-prefill-microbatch-ab-test MODEL\n");
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

  const std::size_t pin = qw38::cuda::selected_prompt_microbatch_rows();
  bool passed = pin == qw38::cuda::kSelectedPromptMicrobatchRows && pin == 4096;

  const std::size_t smoke_rows = 16;
  std::vector<std::size_t> smoke_tokens(smoke_rows);
  fill_tokens(&smoke_tokens);
  std::size_t smoke_frontier = 1;
  status = run_chunk(model, smoke_tokens.data(), smoke_rows, 64, 4096, false,
                     nullptr, &smoke_frontier);
  passed = passed && status.is_ok() && smoke_frontier == smoke_rows;
  std::size_t smoke_poll = 0;
  std::size_t smoke_cancel_frontier = 1;
  passed = passed && cancel_ok(model, smoke_tokens.data(), smoke_rows, 64, 4096,
                               8, &smoke_poll, &smoke_cancel_frontier);

  bool batch1_ok = true;
  bool batch2_ok = true;
  bool final_ok = true;
  std::size_t batch1_poll = 0;
  std::size_t batch2_poll = 0;
  std::size_t final_poll = 0;
  std::size_t batch1_frontier = 1;
  std::size_t batch2_frontier = 1;
  std::size_t final_frontier = 1;
  if (tier != qw38::cuda::TestTier::kSmoke) {
    const std::size_t iso_rows = 2048;
    std::vector<std::size_t> iso_tokens(iso_rows);
    fill_tokens(&iso_tokens);
    std::size_t one_frontier = 0;
    std::size_t split_frontier = 0;
    status = run_chunk(model, iso_tokens.data(), iso_rows, iso_rows, 2048, false,
                       nullptr, &one_frontier);
    const qw38::Status split_status =
        run_chunk(model, iso_tokens.data(), iso_rows, iso_rows, 512, false,
                  nullptr, &split_frontier);
    passed = passed && status.is_ok() && split_status.is_ok() &&
             one_frontier == iso_rows && split_frontier == iso_rows;
    batch1_ok = cancel_ok(model, iso_tokens.data(), iso_rows, iso_rows, 512, 64,
                          &batch1_poll, &batch1_frontier);
    batch2_ok = cancel_ok(model, iso_tokens.data(), iso_rows, iso_rows, 512, 128,
                          &batch2_poll, &batch2_frontier);
    final_ok = cancel_ok(model, iso_tokens.data(), iso_rows, iso_rows, 512, 256,
                         &final_poll, &final_frontier);
    passed = passed && batch1_ok && batch2_ok && final_ok;
  }

  struct Candidate {
    std::size_t rows = 0;
    std::vector<float> samples;
    float mean_ms = 0.0F;
    float tok_s = 0.0F;
  };
  Candidate off[kLegalCount]{};
  Candidate graphs4096{};
  graphs4096.rows = 4096;
  std::size_t graphs_off_winner = 4096;
  bool win = false;
  float p2k_ms = 0.0F;
  float p2k_tok_s = 0.0F;
  float append_2048_ms = 0.0F;
  float append_2048_tok_s = 0.0F;
  float append_4096_ms = 0.0F;
  float append_4096_tok_s = 0.0F;
  if (tier == qw38::cuda::TestTier::kAcceptance) {
    std::vector<std::size_t> tokens(kPrompt);
    fill_tokens(&tokens);
    const int replicates = 3;
    cudaError_t error = cudaSuccess;
    for (int index = 0; index < kLegalCount; ++index) {
      off[index].rows = kLegal[index];
      for (int replicate = 0; replicate < replicates; ++replicate) {
        const float ms =
            time_chunk(model, tokens.data(), kPrompt, kPrompt, kLegal[index],
                       false, &error);
        if (error != cudaSuccess) return fail_cuda("graphs-off A/B", error);
        off[index].samples.push_back(ms);
        off[index].mean_ms += ms;
      }
      off[index].mean_ms /= static_cast<float>(replicates);
      off[index].tok_s = off[index].mean_ms > 0.0F
                             ? static_cast<float>(kPrompt) * 1000.0F /
                                   off[index].mean_ms
                             : 0.0F;
    }
    for (int replicate = 0; replicate < replicates; ++replicate) {
      const float ms =
          time_chunk(model, tokens.data(), kPrompt, kPrompt, 4096, true, &error);
      if (error != cudaSuccess) return fail_cuda("graphs-on 4096", error);
      graphs4096.samples.push_back(ms);
      graphs4096.mean_ms += ms;
    }
    graphs4096.mean_ms /= static_cast<float>(replicates);
    graphs4096.tok_s =
        graphs4096.mean_ms > 0.0F
            ? static_cast<float>(kPrompt) * 1000.0F / graphs4096.mean_ms
            : 0.0F;
    const float eager4096 = off[kLegalCount - 1].mean_ms;
    graphs_off_winner = 4096;
    for (int index = kLegalCount - 2; index >= 0; --index) {
      if (off[index].mean_ms > 0.0F && off[index].mean_ms < eager4096 &&
          off[index].mean_ms < graphs4096.mean_ms) {
        graphs_off_winner = kLegal[index];
        win = true;
        break;
      }
    }
    std::vector<std::size_t> diag_tokens(8192);
    fill_tokens(&diag_tokens);
    p2k_ms = time_chunk(model, diag_tokens.data(), 2048, 2048, 4096, true,
                        &error);
    if (error != cudaSuccess) return fail_cuda("diagnostic 2K empty", error);
    p2k_tok_s = p2k_ms > 0.0F ? 2048.0F * 1000.0F / p2k_ms : 0.0F;
    append_2048_ms = time_prefixed_chunk(
        model, diag_tokens.data(), 2048, diag_tokens.data() + 2048, kPrompt,
        8192, 4096, true, &error);
    if (error != cudaSuccess) return fail_cuda("diagnostic 4K@2048", error);
    append_2048_tok_s =
        append_2048_ms > 0.0F ? static_cast<float>(kPrompt) * 1000.0F /
                                    append_2048_ms
                              : 0.0F;
    append_4096_ms = time_prefixed_chunk(
        model, diag_tokens.data(), kPrompt, diag_tokens.data() + kPrompt,
        kPrompt, 8192, 4096, true, &error);
    if (error != cudaSuccess) return fail_cuda("diagnostic 4K@4096", error);
    append_4096_tok_s =
        append_4096_ms > 0.0F ? static_cast<float>(kPrompt) * 1000.0F /
                                    append_4096_ms
                              : 0.0F;
  } else {
    graphs_off_winner = 4096;
  }

  const std::size_t selected =
      (tier == qw38::cuda::TestTier::kAcceptance && win) ? graphs_off_winner
                                                         : 4096;
  passed = passed && (selected == 4096 || win);

  std::printf("status=%s pin=%zu graphs_off_winner=%zu selected=%zu\n",
              passed ? "passed" : "failed", pin, graphs_off_winner, selected);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-054\",\"status\":\"%s\","
      "\"measurement_utc\":\"%s\",\"device\":\"%s\","
      "\"compute_capability\":\"%d.%d\","
      "\"selected_prompt_microbatch_rows\":%zu,"
      "\"graphs_off_winner\":%zu,\"win\":%s,\"pin\":%zu,"
      "\"smoke\":{\"frontier\":%zu,\"cancel_frontier\":%zu,\"poll_calls\":%zu},"
      "\"cancellation\":{\"after_batch_1\":{\"ok\":%s,\"poll_calls\":%zu,"
      "\"frontier\":%zu},\"after_batch_2\":{\"ok\":%s,\"poll_calls\":%zu,"
      "\"frontier\":%zu},\"after_final_batch\":{\"ok\":%s,\"poll_calls\":%zu,"
      "\"frontier\":%zu}},\"graphs_off\":{",
      kPrefix, passed ? "passed" : "failed", utc, prop.name, prop.major,
      prop.minor, selected, graphs_off_winner, json_bool(win), pin,
      smoke_frontier, smoke_cancel_frontier, smoke_poll,
      json_bool(batch1_ok), batch1_poll, batch1_frontier, json_bool(batch2_ok),
      batch2_poll, batch2_frontier, json_bool(final_ok), final_poll,
      final_frontier);
  for (int index = 0; index < kLegalCount; ++index) {
    if (index != 0) std::printf(",");
    std::printf("\"%zu\":{\"mean_ms\":%.9g,\"tok_s\":%.9g,\"samples\":[",
                kLegal[index], static_cast<double>(off[index].mean_ms),
                static_cast<double>(off[index].tok_s));
    for (std::size_t sample = 0; sample < off[index].samples.size(); ++sample) {
      if (sample != 0) std::printf(",");
      std::printf("%.9g", static_cast<double>(off[index].samples[sample]));
    }
    std::printf("]}");
  }
  std::printf("},\"graphs_on\":{\"4096\":{\"mean_ms\":%.9g,\"tok_s\":%.9g,"
              "\"samples\":[",
              static_cast<double>(graphs4096.mean_ms),
              static_cast<double>(graphs4096.tok_s));
  for (std::size_t sample = 0; sample < graphs4096.samples.size(); ++sample) {
    if (sample != 0) std::printf(",");
    std::printf("%.9g", static_cast<double>(graphs4096.samples[sample]));
  }
  std::printf("]}},\"diagnostic\":{\"p2k_empty\":{\"mean_ms\":%.9g,\"tok_s\":%.9g},"
              "\"append_4k_at_2048\":{\"mean_ms\":%.9g,\"tok_s\":%.9g},"
              "\"append_4k_at_4096\":{\"mean_ms\":%.9g,\"tok_s\":%.9g},"
              "\"label\":\"diagnostic_not_historical_p\"},"
              "\"extra_workspace_bytes\":0}\n",
              static_cast<double>(p2k_ms), static_cast<double>(p2k_tok_s),
              static_cast<double>(append_2048_ms),
              static_cast<double>(append_2048_tok_s),
              static_cast<double>(append_4096_ms),
              static_cast<double>(append_4096_tok_s));
  return passed ? 0 : 1;
}
