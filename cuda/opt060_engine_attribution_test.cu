#include "engine_attribution.h"
#include "full_scheduler.h"
#include "model.h"
#include "scheduler.h"
#include "test_tier.h"
#include "weights.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT060_ENGINE_ATTRIBUTION_RESULT=";
constexpr std::size_t kCapacity = 8192;
constexpr std::size_t kRecordCap = 24576;
constexpr char kLlamaBin[] =
    ".cache/authorities/llama-build-opt060/bin/qw38-llama-engine-attribution";
constexpr char kEvidenceDir[] =
    "evidence/optimization/opt060-engine-attribution";

struct Options final {
  const char* model = nullptr;
  const char* workload = nullptr;
  const char* llama_bin = kLlamaBin;
  int repetitions = 1;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|decode|prefill|"
               "acceptance] [MODEL] [--llama-bin PATH] [--repetitions N]\n",
               argv0);
  return 2;
}

void release_gpu();

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--llama-bin") == 0 && index + 1 < argc) {
      options->llama_bin = argv[++index];
    } else if (std::strcmp(arg, "--repetitions") == 0 && index + 1 < argc) {
      options->repetitions = std::atoi(argv[++index]);
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* default_workload(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kCorrectness) return "correctness";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "acceptance";
  return "decode";
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

__global__ void tiny_add(const float* a, const float* b, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int run_smoke() {
  qw38::cuda::EngineEventPool<4> pool;
  qw38::cuda::EngineOpRecord storage[8]{};
  qw38::cuda::EngineAttribution dest{};
  dest.records = storage;
  dest.capacity = 8;
  dest.record = true;
  qw38::cuda::copy_cstr(dest.engine, sizeof(dest.engine), "quartz");
  qw38::cuda::copy_cstr(dest.phase, sizeof(dest.phase), "smoke");
  qw38::cuda::copy_cstr(dest.graph_mode, sizeof(dest.graph_mode),
                       "eager_diagnostic");

  constexpr int n = 256;
  float *a = nullptr, *b = nullptr, *c = nullptr;
  cudaError_t error = cudaMalloc(&a, n * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&b, n * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&c, n * sizeof(float));
  if (error != cudaSuccess) return 1;
  cudaMemset(a, 0, n * sizeof(float));
  cudaMemset(b, 0, n * sizeof(float));
  if (pool.record_epoch(nullptr) != cudaSuccess) return 1;

  qw38::cuda::EngineOpRecord spec{};
  qw38::cuda::copy_cstr(spec.engine, sizeof(spec.engine), "quartz");
  qw38::cuda::copy_cstr(spec.role, sizeof(spec.role), "tiny_a");
  qw38::cuda::copy_cstr(spec.attribution_role, sizeof(spec.attribution_role),
                       "member");
  qw38::cuda::copy_cstr(spec.launch_family, sizeof(spec.launch_family),
                       "tiny_kernel");
  if (pool.begin(spec, nullptr) != cudaSuccess) return 1;
  tiny_add<<<1, 256>>>(a, b, c, n);
  if (pool.end() != cudaSuccess) return 1;

  qw38::cuda::copy_cstr(spec.role, sizeof(spec.role), "tiny_fused");
  qw38::cuda::copy_cstr(spec.fused_member_ids, sizeof(spec.fused_member_ids),
                       "tiny_b,tiny_c");
  spec.fused_member_count = 2;
  qw38::cuda::copy_cstr(spec.attribution_role, sizeof(spec.attribution_role),
                       "enclosing");
  if (pool.begin(spec, nullptr) != cudaSuccess) return 1;
  tiny_add<<<1, 256>>>(c, b, c, n);
  tiny_add<<<1, 256>>>(c, a, c, n);
  if (pool.end() != cudaSuccess) return 1;

  qw38::cuda::copy_cstr(spec.role, sizeof(spec.role), "tiny_d");
  spec.fused_member_count = 1;
  spec.fused_member_ids[0] = '\0';
  qw38::cuda::copy_cstr(spec.attribution_role, sizeof(spec.attribution_role),
                       "member");
  if (pool.begin(spec, nullptr) != cudaSuccess) return 1;
  tiny_add<<<1, 256>>>(c, b, c, n);
  if (pool.end() != cudaSuccess) return 1;
  if (pool.collect(&dest) != cudaSuccess) return 1;
  if (dest.count != 3) {
    std::fprintf(stderr, "smoke expected 3 records, got %zu\n", dest.count);
    return 1;
  }

  qw38::cuda::EngineEventPool<1> tiny;
  tiny.record_epoch(nullptr);
  qw38::cuda::copy_cstr(spec.role, sizeof(spec.role), "overflow");
  if (tiny.begin(spec, nullptr) != cudaSuccess) return 1;
  tiny_add<<<1, 256>>>(a, b, c, n);
  if (tiny.end() != cudaSuccess) return 1;
  const bool overflowed = tiny.begin(spec, nullptr) != cudaSuccess && tiny.overflow();
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  if (!overflowed) {
    std::fprintf(stderr, "event pool overflow was not detected\n");
    return 1;
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-060\",\"workload\":\"smoke\","
      "\"engine\":\"quartz\",\"operations\":3,\"fused_pair\":true,"
      "\"pool_overflow\":true,\"model\":false}\n",
      kPrefix);
  return 0;
}

int load_model(const char* path, qw38::cuda::ResidentModel* model) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (status.is_ok()) {
    status = model->upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

int discover_nsight() {
  const int nsys = std::system("command -v nsys >/dev/null 2>&1");
  const int ncu = std::system("command -v ncu >/dev/null 2>&1");
  std::printf("nsys_available=%s ncu_available=%s full_ncu_sweep=false "
              "owner=OPT-061\n",
              nsys == 0 ? "true" : "false", ncu == 0 ? "true" : "false");
  if (nsys != 0) std::printf("nsys_error=nsys not found\n");
  if (ncu != 0) std::printf("ncu_error=ncu not found\n");
  return 0;
}

void release_gpu() {
  cudaDeviceSynchronize();
  cudaDeviceReset();
}

void dump_families(const qw38::cuda::EngineAttribution& families,
                   const char* phase, int rep) {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/quartz-%s-rep%d-records.json",
                kEvidenceDir, phase, rep);
  FILE* out = std::fopen(path, "w");
  if (out == nullptr) {
    std::printf("quartz_records_path=unavailable count=%zu overflow=%s\n",
                families.count, families.pool_overflow ? "true" : "false");
    return;
  }
  std::fprintf(
      out,
      "{\"schema_version\":1,\"task\":\"OPT-060\",\"engine\":\"quartz\","
      "\"phase\":\"%s\",\"rep\":%d,\"graph_mode\":\"eager_diagnostic\","
      "\"pool_overflow\":%s,\"count\":%zu,\"records\":[",
      phase, rep, families.pool_overflow ? "true" : "false", families.count);
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(
        out, families.records[index], index + 1 == families.count);
  }
  std::fprintf(out, "]}\n");
  std::fclose(out);
  std::printf("quartz_records_path=%s count=%zu overflow=%s\n", path,
              families.count, families.pool_overflow ? "true" : "false");
}

int run_llama(const Options& options, const char* workload, const char* mode,
              int prompt, int prefix, int output_tokens, int rep) {
  if (options.llama_bin == nullptr || options.llama_bin[0] == '\0') {
    std::printf("llama_status=unavailable reason=missing_binary\n");
    return 0;
  }
  if (access(options.llama_bin, X_OK) != 0) {
    std::printf("llama_status=unavailable reason=not_executable path=%s "
                "build=tools/llama_authority/build_opt060_instrumented.sh\n",
                options.llama_bin);
    return 0;
  }
  char log_path[512];
  std::snprintf(log_path, sizeof(log_path),
                "%s/llama-%s-%s-rep%d.log", kEvidenceDir, workload, mode, rep);
  char cmd[2048];
  std::snprintf(cmd, sizeof(cmd),
                "QW38_OPT060_ATTRIBUTION=%s %s %s --workload %s --mode %s "
                "--batch-policy matched --prompt %d --prefix %d "
                "--output-tokens %d --warmups 1 --samples 1 "
                "> %s 2>&1",
                std::strcmp(mode, "eager_diagnostic") == 0 ? "1" : "0",
                options.llama_bin, options.model != nullptr ? options.model : "",
                workload, mode, prompt, prefix, output_tokens, log_path);
  const int rc = std::system(cmd);
  std::printf("llama_status=%s rc=%d mode=%s log=%s\n",
              rc == 0 ? "ok" : "failed", rc, mode, log_path);
  return rc == 0 ? 0 : 1;
}

int run_correctness(const Options& options) {
  if (options.model == nullptr) {
    std::fprintf(stderr, "model required for correctness\n");
    return 2;
  }
  qw38::cuda::ResidentModel model;
  const int loaded = load_model(options.model, &model);
  if (loaded != 0) return loaded;
  std::vector<std::size_t> tokens(3);
  fill_tokens(&tokens);
  qw38::cuda::SchedulerSession session_graph;
  qw38::cuda::SchedulerSession session_eager;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(64);
  if (status.is_ok()) status = session_graph.create(64);
  if (status.is_ok()) status = session_eager.create(64);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);
  std::vector<float> logits_graph(qw38::internal::kVocabularySize);
  std::vector<float> logits_eager(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  float elapsed = 0.0F;
  qw38::cuda::SyncResult result{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), 2, &session_graph, &workspace, logits_graph.data(),
      logits_graph.size(), hidden.data(), hidden.size(), &result, nullptr,
      &graphs);
  if (status.is_ok()) {
    status = qw38::cuda::execute_token(
        model, tokens[2], &session_graph, &workspace, logits_graph.data(),
        logits_graph.size(), hidden.data(), hidden.size(), &elapsed, nullptr,
        nullptr, qw38::cuda::PointwisePath::kFused, &graphs);
  }
  if (!status.is_ok()) return fail_status(status);
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), 2, &session_eager, &workspace, logits_eager.data(),
      logits_eager.size(), hidden.data(), hidden.size(), &result, nullptr,
      nullptr);
  if (status.is_ok()) {
    status = qw38::cuda::execute_token(
        model, tokens[2], &session_eager, &workspace, logits_eager.data(),
        logits_eager.size(), hidden.data(), hidden.size(), &elapsed, nullptr,
        nullptr, qw38::cuda::PointwisePath::kFused, nullptr);
  }
  if (!status.is_ok()) return fail_status(status);
  const bool equal = std::memcmp(logits_graph.data(), logits_eager.data(),
                                 logits_graph.size() * sizeof(float)) == 0;
  if (!equal) {
    std::fprintf(stderr, "graph/eager logits differ\n");
    return 1;
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-060\",\"workload\":\"correctness\","
      "\"graph_modes\":[\"cuda_graph\",\"eager_diagnostic\"],\"warmup\":1,"
      "\"sample\":1,\"exact_output\":true}\n",
      kPrefix);
  return 0;
}

int run_quartz_decode(const qw38::cuda::ResidentModel& model,
                      const std::vector<std::size_t>& tokens, std::size_t prefix,
                      std::size_t outputs, bool diagnostic, bool use_graph,
                      qw38::cuda::EngineAttribution* families, float* wall_ms,
                      std::vector<float>* logits) {
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(kCapacity);
  if (status.is_ok()) status = session.create(kCapacity);
  qw38::cuda::SchedulerGraphs graphs;
  qw38::cuda::SchedulerGraphs* graph_ptr = nullptr;
  if (status.is_ok() && use_graph) {
    status = graphs.create(model, &workspace);
    graph_ptr = &graphs;
  }
  if (!status.is_ok()) return fail_status(status);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult result{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prefix, &session, &workspace, logits->data(),
      logits->size(), hidden.data(), hidden.size(), &result, nullptr,
      graph_ptr);
  if (!status.is_ok()) return fail_status(status);
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t step = 0; step < outputs; ++step) {
    qw38::cuda::DecodeAttribution attribution;
    attribution.record_leaves = diagnostic;
    attribution.families = diagnostic ? families : nullptr;
    if (families != nullptr) {
      families->record = diagnostic;
      families->token_position = static_cast<int>(prefix + step);
      qw38::cuda::copy_cstr(families->phase, sizeof(families->phase), "decode");
    }
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, tokens[prefix + step], &session, &workspace, logits->data(),
        logits->size(), hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, graph_ptr,
        diagnostic ? &attribution : nullptr);
    if (!status.is_ok()) return fail_status(status);
  }
  *wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count());
  return 0;
}

int run_quartz_prefill(const qw38::cuda::ResidentModel& model,
                       const std::vector<std::size_t>& tokens, bool diagnostic,
                       bool use_graph, qw38::cuda::EngineAttribution* families,
                       float* wall_ms, std::vector<float>* logits) {
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(kCapacity);
  if (status.is_ok()) status = session.create(kCapacity);
  qw38::cuda::SchedulerGraphs graphs;
  qw38::cuda::SchedulerGraphs* graph_ptr = nullptr;
  if (status.is_ok() && use_graph) {
    status = graphs.create(model, &workspace);
    graph_ptr = &graphs;
  }
  if (!status.is_ok()) return fail_status(status);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::PrefillAttribution attribution;
  attribution.record_leaves = diagnostic;
  attribution.families = diagnostic ? families : nullptr;
  if (families != nullptr) {
    families->record = diagnostic;
    qw38::cuda::copy_cstr(families->phase, sizeof(families->phase), "prefill");
  }
  qw38::cuda::SyncResult result{};
  const auto started = std::chrono::steady_clock::now();
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), tokens.size(), &session, &workspace, logits->data(),
      logits->size(), hidden.data(), hidden.size(), &result, nullptr, graph_ptr,
      qw38::cuda::GdnScanPath::kFusedTokenLoop,
      diagnostic ? &attribution : nullptr);
  *wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count());
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

int run_screen(const Options& options, const char* phase, int repetitions) {
  if (options.model == nullptr) {
    std::fprintf(stderr, "model required for %s\n", phase);
    return 2;
  }
  discover_nsight();
  const bool prefill = std::strcmp(phase, "prefill") == 0;
  const std::size_t prompt = prefill ? 4096 : 2048;
  const std::size_t outputs = prefill ? 0 : 16;
  std::vector<std::size_t> tokens(prompt + outputs);
  fill_tokens(&tokens);
  std::vector<qw38::cuda::EngineOpRecord> storage(kRecordCap);
  qw38::cuda::EngineAttribution families{};
  families.records = storage.data();
  families.capacity = storage.size();
  std::vector<float> logits(qw38::internal::kVocabularySize);
  int rc = 0;
  for (int rep = 0; rep < repetitions; ++rep) {
    const bool quartz_first = (rep % 2) == 0;
    float unperturbed = 0.0F;
    float diagnostic = 0.0F;
    auto quartz = [&]() {
      families.count = 0;
      families.pool_overflow = false;
      qw38::cuda::ResidentModel model;
      const int loaded = load_model(options.model, &model);
      if (loaded != 0) return loaded;
      int local = 0;
      if (prefill) {
        local |= run_quartz_prefill(model, tokens, false, true, nullptr,
                                    &unperturbed, &logits);
        local |= run_quartz_prefill(model, tokens, true, false, &families,
                                    &diagnostic, &logits);
      } else {
        local |= run_quartz_decode(model, tokens, prompt, outputs, false, true,
                                   nullptr, &unperturbed, &logits);
        local |= run_quartz_decode(model, tokens, prompt, outputs, true, false,
                                   &families, &diagnostic, &logits);
      }
      dump_families(families, phase, rep);
      std::printf(
          "quartz_rep=%d phase=%s unperturbed_ms=%.9g diagnostic_ms=%.9g "
          "overhead_ms=%.9g records=%zu graph_mode_diagnostic=eager_diagnostic "
          "unperturbed_graph_mode=cuda_graph exclusive_gpu=true\n",
          rep, phase, unperturbed, diagnostic, diagnostic - unperturbed,
          families.count);
      return local;
    };
    auto llama = [&]() {
      int local = 0;
      local |= run_llama(options, phase, "unperturbed",
                         prefill ? 4096 : 0, prefill ? 0 : 2048,
                         static_cast<int>(outputs), rep);
      local |= run_llama(options, phase, "eager_diagnostic",
                         prefill ? 4096 : 0, prefill ? 0 : 2048,
                         static_cast<int>(outputs), rep);
      return local;
    };
    if (quartz_first) {
      rc |= quartz();
      release_gpu();
      rc |= llama();
    } else {
      rc |= llama();
      rc |= quartz();
      release_gpu();
    }
  }
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-060\",\"workload\":\"%s\","
      "\"engines\":2,\"modes\":[\"unperturbed\",\"eager_diagnostic\"],"
      "\"repetitions\":%d,\"records\":%zu,\"claims_throughput\":false}\n",
      kPrefix, phase, repetitions, families.count);
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr,
                 "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
                 "screen, or acceptance\n");
    return 2;
  }
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload;
  if (workload == nullptr) workload = default_workload(qw38::cuda::test_tier());
  if (std::strcmp(workload, "smoke") == 0) return run_smoke();
  if (std::strcmp(workload, "correctness") == 0) {
    const int rc = run_correctness(options);
    release_gpu();
    return rc;
  }
  if (std::strcmp(workload, "decode") == 0) {
    return run_screen(options, "decode", options.repetitions);
  }
  if (std::strcmp(workload, "prefill") == 0) {
    return run_screen(options, "prefill", options.repetitions);
  }
  if (std::strcmp(workload, "acceptance") == 0) {
    int rc = run_smoke();
    if (rc == 0) rc = run_correctness(options);
    release_gpu();
    if (rc == 0) rc = run_screen(options, "decode", 3);
    if (rc == 0) rc = run_screen(options, "prefill", 3);
    return rc;
  }
  return usage(argv[0]);
}
