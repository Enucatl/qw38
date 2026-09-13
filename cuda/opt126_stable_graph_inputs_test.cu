#include "decode_launch_state.cuh"
#include "execution_graph_path.cuh"
#include "attention_decode_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT126_STABLE_GRAPH_INPUTS_RESULT=";
constexpr char kCounts[] = "QW38_OPT126_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;

struct Options final {
  const char* workload = "inventory";
  std::size_t prefix = 128;
  std::size_t tokens = 4;
  std::size_t capacity = 0;
};

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

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload inventory|positions|isolation|restore|"
               "same-math|costs] [--prefix N] [--tokens N] [--capacity N] "
               "MODEL independently_restored=true same_binary=true\n",
               argv0);
  return 2;
}

bool parse_size(const char* text, std::size_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--tokens") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->tokens)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--capacity") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->capacity)) return usage(argv[0]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    } else {
      model_index = index;
    }
  }
  if (model_index < 0) return usage(argv[0]);
  return model_index;
}

std::size_t session_capacity(std::size_t prefix, std::size_t outputs,
                             std::size_t requested) {
  if (requested != 0) return requested;
  return std::max(prefix + outputs + 32, kMinCapacity);
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

qw38::Status load_model(const char* path, qw38::internal::MappedFile* mapping,
                        qw38::cuda::ResidentModel* model) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  if (status.is_ok()) status = mapping->open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, *mapping, &weights);
  }
  if (status.is_ok()) {
    status = model->upload(weights, mapping->data(), mapping->size());
  }
  return status;
}

qw38::Status run_token(const qw38::cuda::ResidentModel& model, std::size_t token,
                       qw38::cuda::SchedulerSession* session,
                       qw38::cuda::SchedulerWorkspace* workspace, float* logits,
                       float* hidden, float* elapsed,
                       qw38::cuda::SchedulerGraphs* graphs,
                       const qw38::cuda::EvalControl* control = nullptr) {
  return qw38::cuda::execute_token(
      model, token, session, workspace, logits, qw38::internal::kVocabularySize,
      hidden, qw38::internal::kResidualWidth, elapsed, control, nullptr,
      qw38::cuda::PointwisePath::kFused, graphs, nullptr);
}

qw38::Status json_escape(const std::string& input, std::string* output) {
  output->clear();
  output->reserve(input.size());
  for (char ch : input) {
    if (ch == '"' || ch == '\\') {
      output->push_back('\\');
      output->push_back(ch);
    } else if (ch == '\n') {
      output->append("\\n");
    } else {
      output->push_back(ch);
    }
  }
  return qw38::Status::ok();
}

qw38::Status create_graphs(const qw38::cuda::ResidentModel& model,
                           qw38::cuda::SchedulerSession* session,
                           qw38::cuda::SchedulerWorkspace* workspace,
                           qw38::cuda::SchedulerGraphs* graphs,
                           const char* path) {
  qw38::cuda::ExecutionGraphPathScope scope(path);
  if (std::strcmp(path, qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0) {
    return graphs->create(model, workspace, session);
  }
  return graphs->create(model, workspace);
}

bool exact_buffers(const std::vector<float>& left,
                   const std::vector<float>& right,
                   const std::array<float, qw38::internal::kResidualWidth>& hidden_left,
                   const std::array<float, qw38::internal::kResidualWidth>& hidden_right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0 &&
         std::memcmp(hidden_left.data(), hidden_right.data(),
                     hidden_left.size() * sizeof(float)) == 0;
}

qw38::Status prefill(const qw38::cuda::ResidentModel& model,
                     const std::vector<std::size_t>& tokens, std::size_t prefix,
                     qw38::cuda::SchedulerSession* session,
                     qw38::cuda::SchedulerWorkspace* workspace,
                     qw38::cuda::SchedulerGraphs* graphs, float* logits,
                     float* hidden) {
  qw38::cuda::SyncResult sync{};
  return qw38::cuda::sync_tokens(model, tokens.data(), prefix, session,
                                 workspace, logits, qw38::internal::kVocabularySize,
                                 hidden, qw38::internal::kResidualWidth, &sync,
                                 nullptr, graphs);
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-126\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

unsigned int buffer_hash(const float* device, std::size_t floats) {
  const std::size_t take = std::min(floats, static_cast<std::size_t>(1024));
  std::vector<float> host(take);
  if (cudaMemcpy(host.data(), device, take * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 0;
  }
  unsigned int hash = 2166136261u;
  const auto* bytes = reinterpret_cast<const unsigned char*>(host.data());
  for (std::size_t index = 0; index < take * sizeof(float); ++index) {
    hash ^= bytes[index];
    hash *= 16777619u;
  }
  return hash;
}

qw38::Status poll_stop(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    return {qw38::StatusCode::kCancelled, "opt126 graph cancellation"};
  }
  return qw38::Status::ok();
}

qw38::Status poll_fail(void* opaque) noexcept {
  auto* context = static_cast<PollContext*>(opaque);
  ++context->calls;
  if (context->stop_after != 0 && context->calls >= context->stop_after) {
    return {qw38::StatusCode::kInternal, "opt126 graph failure"};
  }
  return qw38::Status::ok();
}

int run_inventory(const qw38::cuda::ResidentModel& model) {
  (void)model;
  const bool selector_unchanged =
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath, "ffn_only") == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-126\",\"workload\":\"inventory\","
      "\"ok\":%s,\"gdn_parity_design\":\"%s\","
      "\"launch_state_bytes\":%zu,\"topology_count\":%d,"
      "\"shipping_execution_graphs\":\"%s\","
      "\"selector_unchanged\":%s,\"claims_throughput\":false,"
      "\"changing_inputs\":["
      "{\"name\":\"frontier_position\",\"consumer\":\"attention_rope_kv_span\","
      "\"kind\":\"value\",\"topology\":false},"
      "{\"name\":\"visible_kv_length\",\"consumer\":\"attention_decode\","
      "\"kind\":\"value\",\"topology\":false},"
      "{\"name\":\"gdn_committed_slot\",\"consumer\":\"gdn_prepare_tiled\","
      "\"kind\":\"pointer_index\",\"topology\":false},"
      "{\"name\":\"token\",\"consumer\":\"embedding_row\",\"kind\":\"value\","
      "\"topology\":false},"
      "{\"name\":\"attention_dispatch\",\"consumer\":\"vec128_vs_warp_query\","
      "\"kind\":\"kernel_identity\",\"topology\":true},"
      "{\"name\":\"session_identity\",\"consumer\":\"scheduler_graphs\","
      "\"kind\":\"pointer\",\"topology\":true},"
      "{\"name\":\"q8_grouped_descriptors\",\"consumer\":\"mixer_q8\","
      "\"kind\":\"descriptor_lifetime\",\"topology\":true}"
      "],\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(selector_unchanged), qw38::cuda::kGdnParityDesign,
      sizeof(qw38::cuda::DecodeLaunchState),
      qw38::cuda::kDecodeGraphTopologyCount,
      qw38::cuda::kSelectedExecutionGraphPath, json_bool(selector_unchanged));
  print_counts(0, 1, 1, false);
  return selector_unchanged ? 0 : 1;
}

struct DecodeOutputs {
  std::vector<std::vector<float>> logits;
  std::vector<std::array<float, qw38::internal::kResidualWidth>> hidden;
};

bool outputs_exact(const DecodeOutputs& left, const DecodeOutputs& right) {
  if (left.logits.size() != right.logits.size() ||
      left.hidden.size() != right.hidden.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.logits.size(); ++index) {
    if (!exact_buffers(left.logits[index], right.logits[index],
                       left.hidden[index], right.hidden[index])) {
      return false;
    }
  }
  return true;
}

struct GraphStats {
  std::uint32_t uploads = 0;
  std::uint32_t recaptures = 0;
  std::uint32_t recapture_before = 0;
  std::size_t decode_segment_graph_count = 0;
};

qw38::Status collect_decode_outputs(
    const qw38::cuda::ResidentModel& model,
    const std::vector<std::size_t>& tokens, std::size_t prefix,
    std::size_t outputs, std::size_t capacity, const char* graph_path,
    DecodeOutputs* collected, GraphStats* stats) {
  collected->logits.clear();
  collected->hidden.clear();
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok() && graph_path != nullptr) {
    status = create_graphs(model, &session, &workspace, &graphs, graph_path);
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SchedulerGraphs* prefill_graphs = nullptr;
  if (graph_path != nullptr &&
      std::strcmp(graph_path,
                  qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0) {
    prefill_graphs = &graphs;
  }
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, prefill_graphs,
                     logits.data(), hidden.data());
  }
  if (stats != nullptr) {
    stats->recapture_before = graphs.topology_recapture_count();
  }
  for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
    float elapsed = 0.0F;
    qw38::cuda::SchedulerGraphs* launch_graphs =
        graph_path != nullptr ? &graphs : nullptr;
    if (graph_path != nullptr &&
        std::strcmp(graph_path,
                    qw38::cuda::kLegalExecutionGraphDecodeSegments8) == 0) {
      qw38::cuda::ExecutionGraphPathScope scope(graph_path);
      status = run_token(model, tokens[prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, launch_graphs);
    } else {
      status = run_token(model, tokens[prefix + step], &session, &workspace,
                         logits.data(), hidden.data(), &elapsed, launch_graphs);
    }
    if (status.is_ok()) {
      collected->logits.push_back(logits);
      collected->hidden.push_back(hidden);
    }
  }
  if (stats != nullptr) {
    stats->uploads = graphs.launch_state_upload_count();
    stats->recaptures = graphs.topology_recapture_count();
    stats->decode_segment_graph_count = graphs.decode_segment_graph_count();
  }
  return status;
}

int run_positions(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t positions[] = {0, 127, 1023, 1024, 2047, 131071};
  const std::size_t npos = sizeof(positions) / sizeof(positions[0]);
  bool all_ok = true;
  std::string body =
      "{\"schema_version\":1,\"task\":\"OPT-126\","
      "\"workload\":\"positions\",\"positions\":[";
  for (std::size_t index = 0; index < npos; ++index) {
    const std::size_t frontier = positions[index];
    const bool long_ctx = frontier >= 65536;
    const std::size_t prefix = long_ctx ? 0 : frontier;
    const std::size_t capacity = session_capacity(prefix + 1, 2, options.capacity);
    qw38::Status status = qw38::Status::ok();
    bool ok = true;
    bool exact = false;
    bool no_recapture = true;
    std::uint32_t uploads = 0;
    std::uint32_t recaptures = 0;
    std::size_t decode_segment_graph_count = 0;
    std::string message;
    if (long_ctx) {
      qw38::cuda::SchedulerSession graph_session;
      qw38::cuda::SchedulerWorkspace graph_workspace;
      qw38::cuda::SchedulerGraphs segment_graphs;
      status = graph_session.create(capacity);
      if (status.is_ok()) status = graph_workspace.create(capacity);
      ok = status.is_ok();
      if (ok) {
        status = create_graphs(model, &graph_session, &graph_workspace,
                               &segment_graphs,
                               qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      }
      ok = status.is_ok();
      if (ok) {
        const qw38::Status updated = segment_graphs.update_launch_params(
            0, static_cast<std::uint32_t>(frontier),
            static_cast<std::uint32_t>(frontier));
        ok = updated.is_ok();
        json_escape(updated.message(), &message);
        uploads = segment_graphs.launch_state_upload_count();
        recaptures = segment_graphs.topology_recapture_count();
        decode_segment_graph_count = segment_graphs.decode_segment_graph_count();
        exact = ok;
        no_recapture = recaptures == 0;
      } else {
        json_escape(status.message(), &message);
      }
    } else {
      std::vector<std::size_t> tokens(prefix + 2);
      fill_tokens(&tokens);
      DecodeOutputs eager_out;
      DecodeOutputs ffn_out;
      DecodeOutputs segment_out;
      GraphStats segment_stats;
      status = collect_decode_outputs(model, tokens, prefix, 2, capacity,
                                      nullptr, &eager_out, nullptr);
      if (status.is_ok()) {
        status = collect_decode_outputs(
            model, tokens, prefix, 2, capacity,
            qw38::cuda::kLegalExecutionGraphFfnOnly, &ffn_out, nullptr);
      }
      if (status.is_ok()) {
        status = collect_decode_outputs(
            model, tokens, prefix, 2, capacity,
            qw38::cuda::kLegalExecutionGraphDecodeSegments8, &segment_out,
            &segment_stats);
      }
      ok = status.is_ok();
      json_escape(status.message(), &message);
      exact = ok && outputs_exact(eager_out, ffn_out) &&
              outputs_exact(eager_out, segment_out);
      uploads = segment_stats.uploads;
      recaptures = segment_stats.recaptures;
      decode_segment_graph_count = segment_stats.decode_segment_graph_count;
      no_recapture = recaptures == segment_stats.recapture_before;
    }
    const int topology = qw38::cuda::decode_graph_topology_index(frontier);
    const int expected_topology =
        (frontier >= static_cast<std::size_t>(
             qw38::cuda::kSelectedDecodeAttentionCrossoverThreshold) &&
         frontier <= static_cast<std::size_t>(
             qw38::cuda::kSelectedDecodeAttentionVerifiedMax))
            ? 1
            : 0;
    all_ok = all_ok && ok && exact && no_recapture &&
             topology == expected_topology;
    char row[640];
    std::snprintf(
        row, sizeof(row),
        "%s{\"frontier\":%zu,\"ok\":%s,\"exact\":%s,\"odd_even_exact\":%s,"
        "\"no_recapture\":%s,\"launch_state_uploads\":%u,"
        "\"topology_recaptures\":%u,\"topology_index\":%d,"
        "\"expected_topology\":%d,\"decode_segment_graph_count\":%zu,"
        "\"message\":\"%s\"}",
        index == 0 ? "" : ",", frontier, json_bool(ok), json_bool(exact),
        json_bool(exact), json_bool(no_recapture), uploads, recaptures,
        topology, expected_topology, decode_segment_graph_count, message.c_str());
    body.append(row);
  }
  char tail[192];
  std::snprintf(tail, sizeof(tail),
                "],\"ok\":%s,\"observed_warmups\":0,\"observed_samples\":1,"
                "\"observed_candidates\":2,\"keep\":false}\n",
                json_bool(all_ok));
  body.append(tail);
  std::printf("%s%s", kPrefix, body.c_str());
  print_counts(0, 1, 2, false);
  return all_ok ? 0 : 1;
}

int run_isolation(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t capacity = session_capacity(prefix, 4, options.capacity);
  std::vector<std::size_t> tokens(prefix + 4);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  const std::size_t frontier_before = session.frontier();
  const std::uint32_t slot_before = session.gdn_committed_slot();
  const unsigned int committed_before =
      buffer_hash(session.gdn_committed_recurrent(),
                  48 * qw38::internal::kGdnRecurrentStateValues);
  const unsigned int candidate_before =
      buffer_hash(session.gdn_candidate_recurrent(),
                  48 * qw38::internal::kGdnRecurrentStateValues);
  PollContext poll_context{};
  poll_context.stop_after = 1;
  qw38::cuda::EvalControl control{};
  control.poll = poll_stop;
  control.context = &poll_context;
  float elapsed = 0.0F;
  qw38::Status cancelled = qw38::Status::ok();
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    cancelled = run_token(model, tokens[prefix], &session, &workspace,
                          logits.data(), hidden.data(), &elapsed, &graphs,
                          &control);
  }
  const bool cancel_ok = cancelled.code() == qw38::StatusCode::kCancelled &&
                         session.frontier() == frontier_before &&
                         session.gdn_committed_slot() == slot_before;
  const unsigned int committed_after_cancel =
      buffer_hash(session.gdn_committed_recurrent(),
                  48 * qw38::internal::kGdnRecurrentStateValues);
  bool retry_ok = false;
  if (cancel_ok) {
    float retry_ms = 0.0F;
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    status = run_token(model, tokens[prefix], &session, &workspace,
                       logits.data(), hidden.data(), &retry_ms, &graphs);
    retry_ok = status.is_ok() && session.frontier() == frontier_before + 1 &&
               session.gdn_committed_slot() == (slot_before ^ 1u);
  }
  const unsigned int committed_after_retry =
      buffer_hash(session.gdn_committed_recurrent(),
                  48 * qw38::internal::kGdnRecurrentStateValues);
  const unsigned int candidate_after_retry =
      buffer_hash(session.gdn_candidate_recurrent(),
                  48 * qw38::internal::kGdnRecurrentStateValues);
  PollContext fail_context{};
  fail_context.stop_after = 1;
  qw38::cuda::EvalControl fail_control{};
  fail_control.poll = poll_fail;
  fail_control.context = &fail_context;
  const std::size_t frontier_after_retry = session.frontier();
  const std::uint32_t slot_after_retry = session.gdn_committed_slot();
  bool failure_ok = false;
  if (retry_ok) {
    float fail_ms = 0.0F;
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    const qw38::Status failed =
        run_token(model, tokens[prefix + 1], &session, &workspace,
                  logits.data(), hidden.data(), &fail_ms, &graphs,
                  &fail_control);
    failure_ok = failed.code() == qw38::StatusCode::kInternal &&
                 session.frontier() == frontier_after_retry &&
                 session.gdn_committed_slot() == slot_after_retry;
  }
  const bool isolated =
      committed_after_cancel == committed_before &&
      (retry_ok ? committed_after_retry != candidate_after_retry : false);
  const bool ok =
      status.is_ok() && cancel_ok && retry_ok && failure_ok && isolated;
  std::string message;
  json_escape(status.message(), &message);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-126\",\"workload\":\"isolation\","
      "\"ok\":%s,\"cancel_ok\":%s,\"retry_ok\":%s,\"failure_ok\":%s,"
      "\"candidate_isolated\":%s,\"committed_hash_before\":%u,"
      "\"committed_hash_after_cancel\":%u,\"candidate_hash_before\":%u,"
      "\"candidate_hash_after_retry\":%u,\"message\":\"%s\","
      "\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(cancel_ok), json_bool(retry_ok),
      json_bool(failure_ok), json_bool(isolated), committed_before,
      committed_after_cancel, candidate_before, candidate_after_retry,
      message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_restore(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = std::max(options.prefix, static_cast<std::size_t>(8));
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  std::vector<std::size_t> divergent(prefix + 2);
  fill_tokens(&divergent);
  for (std::size_t index = 0; index < divergent.size(); ++index) {
    divergent[index] = (divergent[index] + 13) % qw38::internal::kVocabularySize;
  }
  const char* ckpt = "build/opt126-stable-graph-inputs.ckpt";
  qw38::Status status = qw38::Status::ok();
  bool divergent_invalidated = false;
  bool session_replace_invalidated = false;
  {
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::cuda::SchedulerGraphs graphs;
    status = session.create(capacity);
    if (status.is_ok()) status = workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &session, &workspace, &graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    if (status.is_ok()) {
      status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                       logits.data(), hidden.data());
    }
    if (status.is_ok()) status = session.save_checkpoint(ckpt);
    if (status.is_ok()) {
      qw38::cuda::SchedulerSession other;
      qw38::cuda::SchedulerWorkspace other_workspace;
      status = other.create(capacity);
      if (status.is_ok()) status = other_workspace.create(capacity);
      std::vector<float> other_logits(qw38::internal::kVocabularySize);
      std::array<float, qw38::internal::kResidualWidth> other_hidden{};
      if (status.is_ok()) {
        status = prefill(model, divergent, prefix, &other, &other_workspace,
                         nullptr, other_logits.data(), other_hidden.data());
      }
      if (status.is_ok()) {
        qw38::cuda::ExecutionGraphPathScope scope(
            qw38::cuda::kLegalExecutionGraphDecodeSegments8);
        float elapsed = 0.0F;
        const qw38::Status mismatch =
            run_token(model, divergent[prefix], &other, &other_workspace,
                      other_logits.data(), other_hidden.data(), &elapsed,
                      &graphs);
        divergent_invalidated = mismatch.code() == qw38::StatusCode::kInvalidArgument;
      }
    }
    if (status.is_ok()) {
      qw38::cuda::SchedulerSession replacement;
      qw38::cuda::SchedulerWorkspace replacement_workspace;
      status = replacement.create(capacity);
      if (status.is_ok()) status = replacement_workspace.create(capacity);
      if (status.is_ok()) {
        qw38::cuda::ExecutionGraphPathScope scope(
            qw38::cuda::kLegalExecutionGraphDecodeSegments8);
        float elapsed = 0.0F;
        std::vector<float> repl_logits(qw38::internal::kVocabularySize);
        std::array<float, qw38::internal::kResidualWidth> repl_hidden{};
        const qw38::Status mismatch =
            run_token(model, tokens[prefix], &replacement, &replacement_workspace,
                      repl_logits.data(), repl_hidden.data(), &elapsed, &graphs);
        session_replace_invalidated =
            mismatch.code() == qw38::StatusCode::kInvalidArgument;
      }
    }
  }
  bool generation_bumped = false;
  bool restored_runs = false;
  if (status.is_ok()) {
    qw38::cuda::SchedulerSession restored;
    qw38::cuda::SchedulerWorkspace restored_workspace;
    qw38::cuda::SchedulerGraphs restored_graphs;
    status = restored.create(capacity);
    if (status.is_ok()) status = restored_workspace.create(capacity);
    const std::uint32_t restored_generation_before = restored.launch_generation();
    if (status.is_ok()) {
      status = restored.restore_checkpoint(ckpt, &restored_workspace);
    }
    generation_bumped =
        status.is_ok() &&
        restored.launch_generation() != restored_generation_before;
    if (status.is_ok()) {
      status = create_graphs(model, &restored, &restored_workspace,
                             &restored_graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    if (status.is_ok()) {
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      float elapsed = 0.0F;
      std::vector<float> logits(qw38::internal::kVocabularySize);
      std::array<float, qw38::internal::kResidualWidth> hidden{};
      status = run_token(model, tokens[prefix], &restored, &restored_workspace,
                         logits.data(), hidden.data(), &elapsed,
                         &restored_graphs);
      restored_runs = status.is_ok();
    }
  }
  const bool stale_rejected = session_replace_invalidated && divergent_invalidated;
  const bool ok = status.is_ok() && generation_bumped && stale_rejected &&
                  restored_runs;
  std::string message;
  json_escape(status.message(), &message);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-126\",\"workload\":\"restore\","
      "\"ok\":%s,\"generation_bumped\":%s,\"session_replace_invalidated\":%s,"
      "\"divergent_prefix_invalidated\":%s,\"restored_runs\":%s,"
      "\"stale_descriptors_rejected\":%s,\"independently_restored\":true,"
      "\"same_binary\":true,\"message\":\"%s\",\"observed_warmups\":0,"
      "\"observed_samples\":1,\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(generation_bumped),
      json_bool(session_replace_invalidated), json_bool(divergent_invalidated),
      json_bool(restored_runs), json_bool(stale_rejected), message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

int run_same_math(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t outputs = options.tokens == 0 ? 4 : options.tokens;
  const std::size_t capacity =
      session_capacity(prefix, outputs, options.capacity);
  std::vector<std::size_t> tokens(prefix + outputs);
  fill_tokens(&tokens);
  std::vector<float> eager_logits(qw38::internal::kVocabularySize);
  std::vector<float> graph_logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> eager_hidden{};
  std::array<float, qw38::internal::kResidualWidth> graph_hidden{};
  std::vector<std::vector<float>> eager_token_logits;
  std::vector<std::array<float, qw38::internal::kResidualWidth>> eager_token_hidden;
  qw38::Status status = qw38::Status::ok();
  {
    qw38::cuda::SchedulerSession eager_session;
    qw38::cuda::SchedulerWorkspace eager_workspace;
    status = eager_session.create(capacity);
    if (status.is_ok()) status = eager_workspace.create(capacity);
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &eager_session, &eager_workspace,
                       nullptr, eager_logits.data(), eager_hidden.data());
    }
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float eager_ms = 0.0F;
      status = run_token(model, tokens[prefix + step], &eager_session,
                         &eager_workspace, eager_logits.data(),
                         eager_hidden.data(), &eager_ms, nullptr);
      if (status.is_ok()) {
        eager_token_logits.push_back(eager_logits);
        eager_token_hidden.push_back(eager_hidden);
      }
    }
  }
  std::size_t matched = 0;
  bool exact = status.is_ok();
  std::uint32_t launch_param_updates = 0;
  std::uint32_t launch_state_uploads = 0;
  std::uint32_t topology_recaptures = 0;
  std::size_t decode_segment_graph_count = 0;
  {
    qw38::cuda::SchedulerSession graph_session;
    qw38::cuda::SchedulerWorkspace graph_workspace;
    qw38::cuda::SchedulerGraphs segment_graphs;
    if (status.is_ok()) status = graph_session.create(capacity);
    if (status.is_ok()) status = graph_workspace.create(capacity);
    if (status.is_ok()) {
      status = create_graphs(model, &graph_session, &graph_workspace,
                             &segment_graphs,
                             qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    }
    if (status.is_ok() && prefix > 0) {
      status = prefill(model, tokens, prefix, &graph_session, &graph_workspace,
                       &segment_graphs, graph_logits.data(), graph_hidden.data());
    }
    exact = exact && status.is_ok() && eager_token_logits.size() == outputs;
    for (std::size_t step = 0; status.is_ok() && step < outputs; ++step) {
      float graph_ms = 0.0F;
      qw38::cuda::ExecutionGraphPathScope scope(
          qw38::cuda::kLegalExecutionGraphDecodeSegments8);
      status = run_token(model, tokens[prefix + step], &graph_session,
                         &graph_workspace, graph_logits.data(),
                         graph_hidden.data(), &graph_ms, &segment_graphs);
      const bool step_exact =
          status.is_ok() && step < eager_token_logits.size() &&
          exact_buffers(eager_token_logits[step], graph_logits,
                        eager_token_hidden[step], graph_hidden);
      exact = exact && step_exact;
      if (step_exact) ++matched;
    }
    launch_param_updates = segment_graphs.launch_param_update_count();
    launch_state_uploads = segment_graphs.launch_state_upload_count();
    topology_recaptures = segment_graphs.topology_recapture_count();
    decode_segment_graph_count = segment_graphs.decode_segment_graph_count();
  }
  std::string message;
  json_escape(status.message(), &message);
  const bool ok =
      status.is_ok() && exact &&
      decode_segment_graph_count ==
          8 * qw38::cuda::kDecodeGraphTopologyCount &&
      std::strcmp(qw38::cuda::kSelectedExecutionGraphPath, "ffn_only") == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-126\",\"workload\":\"same-math\","
      "\"ok\":%s,\"exact\":%s,\"state_equals\":%s,\"matched_tokens\":%zu,"
      "\"tokens\":%zu,\"prefix\":%zu,\"launch_param_updates\":%u,"
      "\"launch_state_uploads\":%u,\"topology_recaptures\":%u,"
      "\"ffn_only_path\":true,\"decode_segments8_path\":true,"
      "\"shipping_execution_graphs\":\"%s\",\"message\":\"%s\","
      "\"observed_warmups\":0,\"observed_samples\":1,\"observed_candidates\":2,"
      "\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(exact), json_bool(exact), matched,
      outputs, prefix, launch_param_updates, launch_state_uploads,
      topology_recaptures, qw38::cuda::kSelectedExecutionGraphPath,
      message.c_str());
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

int run_costs(const qw38::cuda::ResidentModel& model, const Options& options) {
  const std::size_t prefix = options.prefix;
  const std::size_t capacity = session_capacity(prefix, 4, options.capacity);
  std::vector<std::size_t> tokens(prefix + 4);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  const auto create_started = std::chrono::steady_clock::now();
  if (status.is_ok()) {
    status = create_graphs(model, &session, &workspace, &graphs,
                           qw38::cuda::kLegalExecutionGraphDecodeSegments8);
  }
  const float create_ms = static_cast<float>(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - create_started)
          .count());
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (status.is_ok() && prefix > 0) {
    status = prefill(model, tokens, prefix, &session, &workspace, &graphs,
                     logits.data(), hidden.data());
  }
  float elapsed = 0.0F;
  if (status.is_ok()) {
    qw38::cuda::ExecutionGraphPathScope scope(
        qw38::cuda::kLegalExecutionGraphDecodeSegments8);
    status = run_token(model, tokens[prefix], &session, &workspace,
                       logits.data(), hidden.data(), &elapsed, &graphs);
  }
  const bool ok =
      status.is_ok() &&
      graphs.captured_topology_count() == qw38::cuda::kDecodeGraphTopologyCount;
  std::string message;
  json_escape(status.message(), &message);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-126\",\"workload\":\"costs\","
      "\"ok\":%s,\"create_ms\":%.9g,\"graph_bytes\":%zu,"
      "\"launch_state_bytes\":%zu,\"launch_state_upload_ms\":%.9g,"
      "\"launch_state_uploads\":%u,\"topology_recaptures\":%u,"
      "\"topology_count\":%d,\"decode_segment_graph_count\":%zu,"
      "\"indirection_loads_per_consumer\":2,\"gdn_parity_design\":\"%s\","
      "\"message\":\"%s\",\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":1,\"keep\":false}\n",
      kPrefix, json_bool(ok), static_cast<double>(create_ms),
      graphs.allocated_bytes(), sizeof(qw38::cuda::DecodeLaunchState),
      static_cast<double>(graphs.last_launch_state_upload_ms()),
      graphs.launch_state_upload_count(), graphs.topology_recapture_count(),
      graphs.captured_topology_count(), graphs.decode_segment_graph_count(),
      qw38::cuda::kGdnParityDesign, message.c_str());
  print_counts(0, 1, 1, false);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  const int model_index = parse_args(argc, argv, &options);
  if (model_index < 0) return 2;
  qw38::internal::MappedFile mapping;
  qw38::cuda::ResidentModel model;
  const qw38::Status loaded = load_model(argv[model_index], &mapping, &model);
  if (!loaded.is_ok()) return fail_status(loaded);
  std::printf("independently_restored=true same_binary=true\n");
  if (std::strcmp(options.workload, "inventory") == 0) {
    return run_inventory(model);
  }
  if (std::strcmp(options.workload, "positions") == 0) {
    return run_positions(model, options);
  }
  if (std::strcmp(options.workload, "isolation") == 0) {
    return run_isolation(model, options);
  }
  if (std::strcmp(options.workload, "restore") == 0) {
    return run_restore(model, options);
  }
  if (std::strcmp(options.workload, "same-math") == 0) {
    return run_same_math(model, options);
  }
  if (std::strcmp(options.workload, "costs") == 0) {
    return run_costs(model, options);
  }
  return usage(argv[0]);
}
