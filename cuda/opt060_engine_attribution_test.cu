#include "engine_attribution.h"
#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "model.h"
#include "q4k_decode_path.cuh"
#include "scheduler.h"
#include "test_tier.h"
#include "weights.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT060_ENGINE_ATTRIBUTION_RESULT=";
constexpr char kOpt090Prefix[] = "QW38_OPT090_ATTRIBUTION_RESULT=";
constexpr std::size_t kCapacity = 8192;
constexpr std::size_t kRecordCap = 49152;
constexpr char kLlamaBin[] =
    ".cache/authorities/llama-build-opt060/bin/qw38-llama-engine-attribution";
constexpr char kEvidenceDir[] =
    "evidence/optimization/opt060-engine-attribution";

struct Options final {
  const char* model = nullptr;
  const char* workload = nullptr;
  const char* llama_bin = kLlamaBin;
  const char* ident = "shipping";
  const char* idents = nullptr;
  const char* evidence_dir = kEvidenceDir;
  const char* q4_decode = nullptr;
  const char* ffn_decode = nullptr;
  const char* selected_q4 = "integer_q8_late";
  const char* selected_ffn = "paired_integer";
  int repetitions = 1;
  int prefix = 0;
  int prompt = 0;
  int output_tokens = 0;
  int warmups = 0;
  int samples = 0;
  bool skip_llama = false;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|decode|prefill|"
               "acceptance] [MODEL] [--llama-bin PATH] [--repetitions N] "
               "[--prefix N] [--prompt N] [--output-tokens N] [--warmups N] "
               "[--samples N] [--q4-decode packed|integer_q8|integer_q8_late] "
               "[--ffn-decode paired_staged|paired_integer] [--ident NAME] "
               "[--idents A,B] [--selected-q4 PATH] [--selected-ffn PATH] "
               "[--evidence-dir DIR] [--skip-llama]\n",
               argv0);
  return 2;
}

void release_gpu();

bool parse_int(const char* text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--llama-bin") == 0 && index + 1 < argc) {
      options->llama_bin = argv[++index];
    } else if (std::strcmp(arg, "--repetitions") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->repetitions)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--prompt") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->prompt)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--output-tokens") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->output_tokens)) {
        return usage(argv[0]);
      }
    } else if (std::strcmp(arg, "--warmups") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->warmups)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--samples") == 0 && index + 1 < argc) {
      if (!parse_int(argv[++index], &options->samples)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--q4-decode") == 0 && index + 1 < argc) {
      options->q4_decode = argv[++index];
    } else if (std::strcmp(arg, "--ffn-decode") == 0 && index + 1 < argc) {
      options->ffn_decode = argv[++index];
    } else if (std::strcmp(arg, "--ident") == 0 && index + 1 < argc) {
      options->ident = argv[++index];
    } else if (std::strcmp(arg, "--idents") == 0 && index + 1 < argc) {
      options->idents = argv[++index];
    } else if (std::strcmp(arg, "--selected-q4") == 0 && index + 1 < argc) {
      options->selected_q4 = argv[++index];
    } else if (std::strcmp(arg, "--selected-ffn") == 0 && index + 1 < argc) {
      options->selected_ffn = argv[++index];
    } else if (std::strcmp(arg, "--evidence-dir") == 0 && index + 1 < argc) {
      options->evidence_dir = argv[++index];
    } else if (std::strcmp(arg, "--skip-llama") == 0) {
      options->skip_llama = true;
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
}

int apply_selector_overrides(const char* q4_decode, const char* ffn_decode) {
  if (q4_decode != nullptr && std::strcmp(q4_decode, "selected") != 0 &&
      !qw38::cuda::apply_q4_decode_ident(q4_decode, 4U)) {
    std::fprintf(stderr, "invalid --q4-decode %s\n", q4_decode);
    return 2;
  }
  if (ffn_decode != nullptr && std::strcmp(ffn_decode, "selected") != 0 &&
      !qw38::cuda::apply_ffn_decode_ident(ffn_decode)) {
    std::fprintf(stderr, "invalid --ffn-decode %s\n", ffn_decode);
    return 2;
  }
  return 0;
}

struct IdentConfig final {
  char id[64]{};
  char q4[32]{};
  char ffn[32]{};
};

void fill_ident(IdentConfig* config, const char* id, const char* q4,
                const char* ffn) {
  std::snprintf(config->id, sizeof(config->id), "%s", id);
  std::snprintf(config->q4, sizeof(config->q4), "%s", q4);
  std::snprintf(config->ffn, sizeof(config->ffn), "%s", ffn);
}

int load_ident_configs(const Options& options, IdentConfig* configs,
                       int* count) {
  if (options.idents == nullptr || options.idents[0] == '\0') {
    const char* q4 =
        options.q4_decode != nullptr ? options.q4_decode : "selected";
    const char* ffn =
        options.ffn_decode != nullptr ? options.ffn_decode : "selected";
    fill_ident(&configs[0], options.ident, q4, ffn);
    *count = 1;
    return 0;
  }
  std::string raw(options.idents);
  int n = 0;
  std::size_t start = 0;
  while (start <= raw.size() && n < 2) {
    const std::size_t comma = raw.find(',', start);
    const std::string token =
        raw.substr(start, comma == std::string::npos ? std::string::npos
                                                     : comma - start);
    if (!token.empty()) {
      if (token == "opt088_control") {
        fill_ident(&configs[n], token.c_str(), "packed", "paired_staged");
      } else if (token == "opt089_selected") {
        fill_ident(&configs[n], token.c_str(), options.selected_q4,
                   options.selected_ffn);
      } else {
        fill_ident(&configs[n], token.c_str(),
                   options.q4_decode != nullptr ? options.q4_decode : "selected",
                   options.ffn_decode != nullptr ? options.ffn_decode
                                                 : "selected");
      }
      n += 1;
    }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (n == 0) return 2;
  *count = n;
  return 0;
}

float percentile_ms(std::vector<float>* values, double fraction) {
  if (values == nullptr || values->empty()) return 0.0F;
  std::sort(values->begin(), values->end());
  const double index = fraction * static_cast<double>(values->size() - 1);
  const std::size_t lo = static_cast<std::size_t>(index);
  const std::size_t hi = std::min(lo + 1, values->size() - 1);
  const double mix = index - static_cast<double>(lo);
  return static_cast<float>((1.0 - mix) * (*values)[lo] + mix * (*values)[hi]);
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

void dump_families(const Options& options, const char* ident,
                   const qw38::cuda::EngineAttribution& families,
                   const char* phase, int rep) {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/quartz-%s-%s-rep%d-records.json",
                options.evidence_dir, ident, phase, rep);
  FILE* out = std::fopen(path, "w");
  if (out == nullptr) {
    std::printf("quartz_records_path=unavailable count=%zu overflow=%s\n",
                families.count, families.pool_overflow ? "true" : "false");
    return;
  }
  std::fprintf(
      out,
      "{\"schema_version\":1,\"task\":\"OPT-090\",\"engine\":\"quartz\","
      "\"ident\":\"%s\",\"phase\":\"%s\",\"rep\":%d,"
      "\"graph_mode\":\"eager_diagnostic\",\"pool_overflow\":%s,"
      "\"count\":%zu,\"records\":[",
      ident, phase, rep, families.pool_overflow ? "true" : "false",
      families.count);
  for (std::size_t index = 0; index < families.count; ++index) {
    qw38::cuda::write_engine_record_json(
        out, families.records[index], index + 1 == families.count);
  }
  std::fprintf(out, "]}\n");
  std::fclose(out);
  std::printf("quartz_records_path=%s count=%zu overflow=%s ident=%s\n", path,
              families.count, families.pool_overflow ? "true" : "false",
              ident);
}

int run_llama(const Options& options, const char* workload, const char* mode,
              int prompt, int prefix, int output_tokens, int warmups,
              int samples) {
  if (options.skip_llama) {
    std::printf("llama_status=skipped ident=%s\n", options.ident);
    return 0;
  }
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
                "%s/llama-%s-%s-%s-samples%d.log", options.evidence_dir,
                options.ident, workload, mode, samples);
  char cmd[2048];
  std::snprintf(cmd, sizeof(cmd),
                "QW38_OPT060_ATTRIBUTION=%s %s %s --workload %s --mode %s "
                "--batch-policy matched --prompt %d --prefix %d "
                "--output-tokens %d --warmups %d --samples %d "
                "> %s 2>&1",
                std::strcmp(mode, "eager_diagnostic") == 0 ? "1" : "0",
                options.llama_bin, options.model != nullptr ? options.model : "",
                workload, mode, prompt, prefix, output_tokens, warmups, samples,
                log_path);
  const int rc = std::system(cmd);
  std::printf("llama_status=%s rc=%d mode=%s samples=%d log=%s ident=%s\n",
              rc == 0 ? "ok" : "failed", rc, mode, samples, log_path,
              options.ident);
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
                      float* itl_p50_ms, float* itl_p95_ms,
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
  if (families != nullptr && diagnostic) {
    families->record = true;
    qw38::cuda::copy_cstr(families->engine, sizeof(families->engine), "quartz");
    qw38::cuda::copy_cstr(families->phase, sizeof(families->phase), "decode");
    qw38::cuda::copy_cstr(families->graph_mode, sizeof(families->graph_mode),
                          use_graph ? "cuda_graph" : "eager_diagnostic");
    cudaDeviceSynchronize();
    if (qw38::cuda::record_sequence_epoch(families, nullptr) != cudaSuccess) {
      return 1;
    }
  }
  const auto started = std::chrono::steady_clock::now();
  std::vector<float> itl;
  itl.reserve(outputs);
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
    itl.push_back(elapsed);
  }
  *wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count());
  if (itl_p50_ms != nullptr) {
    std::vector<float> copy = itl;
    *itl_p50_ms = percentile_ms(&copy, 0.50);
  }
  if (itl_p95_ms != nullptr) {
    std::vector<float> copy = itl;
    *itl_p95_ms = percentile_ms(&copy, 0.95);
  }
  if (families != nullptr) {
    qw38::cuda::destroy_sequence_epoch(families);
    if (families->pool_overflow) {
      std::fprintf(stderr, "quartz attribution dropped records\n");
      return 1;
    }
  }
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
    qw38::cuda::copy_cstr(families->engine, sizeof(families->engine), "quartz");
    qw38::cuda::copy_cstr(families->phase, sizeof(families->phase), "prefill");
    qw38::cuda::copy_cstr(families->graph_mode, sizeof(families->graph_mode),
                          use_graph ? "cuda_graph" : "eager_diagnostic");
  }
  qw38::cuda::SyncResult result{};
  if (diagnostic && families != nullptr) {
    cudaDeviceSynchronize();
    if (qw38::cuda::record_sequence_epoch(families, nullptr) != cudaSuccess) {
      return 1;
    }
  }
  const auto started = std::chrono::steady_clock::now();
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), tokens.size(), &session, &workspace, logits->data(),
      logits->size(), hidden.data(), hidden.size(), &result, nullptr, graph_ptr,
      qw38::cuda::GdnScanPath::kFusedTokenLoop,
      diagnostic ? &attribution : nullptr);
  *wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count());
  if (families != nullptr) {
    qw38::cuda::destroy_sequence_epoch(families);
    if (families->pool_overflow) {
      std::fprintf(stderr, "quartz attribution dropped records\n");
      return 1;
    }
  }
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

int run_screen(const Options& options, const char* phase, int repetitions) {
  if (options.model == nullptr) {
    std::fprintf(stderr, "model required for %s\n", phase);
    return 2;
  }
  IdentConfig configs[2]{};
  int ident_count = 0;
  if (load_ident_configs(options, configs, &ident_count) != 0) {
    std::fprintf(stderr, "invalid --idents\n");
    return 2;
  }
  const int applied =
      apply_selector_overrides(configs[0].q4, configs[0].ffn);
  if (applied != 0) return applied;
  char mkdir_cmd[640];
  std::snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s",
                options.evidence_dir);
  const int made = std::system(mkdir_cmd);
  (void)made;
  discover_nsight();
  const bool prefill = std::strcmp(phase, "prefill") == 0;
  const std::size_t prompt =
      prefill ? (options.prompt > 0 ? static_cast<std::size_t>(options.prompt)
                                    : 4096)
              : 0;
  const std::size_t prefix =
      prefill ? 0
              : (options.prefix > 0 ? static_cast<std::size_t>(options.prefix)
                                    : 2048);
  const std::size_t outputs =
      prefill ? 0
              : (options.output_tokens > 0
                     ? static_cast<std::size_t>(options.output_tokens)
                     : 16);
  const int warmups = options.warmups > 0 ? options.warmups : 0;
  const int samples = options.samples > 0 ? options.samples : repetitions;
  std::vector<std::size_t> tokens((prefill ? prompt : prefix) + outputs);
  fill_tokens(&tokens);
  std::vector<qw38::cuda::EngineOpRecord> storage(kRecordCap);
  qw38::cuda::EngineAttribution families{};
  families.records = storage.data();
  families.capacity = storage.size();
  std::vector<float> logits(qw38::internal::kVocabularySize);
  int rc = 0;
  int observed_warmups = 0;
  int observed_samples = 0;
  {
    qw38::cuda::ResidentModel model;
    const int loaded = load_model(options.model, &model);
    if (loaded != 0) return loaded;
    for (int warmup = 0; warmup < warmups; ++warmup) {
      for (int ident_index = 0; ident_index < ident_count; ++ident_index) {
        rc |= apply_selector_overrides(configs[ident_index].q4,
                                       configs[ident_index].ffn);
        float discarded = 0.0F;
        float p50 = 0.0F;
        float p95 = 0.0F;
        if (prefill) {
          rc |= run_quartz_prefill(model, tokens, false, true, nullptr,
                                   &discarded, &logits);
        } else {
          rc |= run_quartz_decode(model, tokens, prefix, outputs, false, true,
                                  nullptr, &discarded, &p50, &p95, &logits);
        }
        std::printf(
            "quartz_warmup=%d phase=%s ident=%s wall_ms=%.9g "
            "window=warmup prefix_excluded=true graph_create_excluded=true "
            "order=AB\n",
            warmup, phase, configs[ident_index].id, discarded);
        if (rc != 0) break;
      }
      observed_warmups += 1;
      if (rc != 0) break;
    }
    for (int rep = 0; rep < samples; ++rep) {
      const bool ba = ident_count > 1 && (rep % 2) == 1;
      std::printf("quartz_pair_order=%s sample_index=%d ident_count=%d\n",
                  ba ? "BA" : "AB", rep, ident_count);
      for (int step = 0; step < ident_count; ++step) {
        const int ident_index = ba ? (ident_count - 1 - step) : step;
        rc |= apply_selector_overrides(configs[ident_index].q4,
                                       configs[ident_index].ffn);
        families.count = 0;
        families.pool_overflow = false;
        float unperturbed = 0.0F;
        float diagnostic = 0.0F;
        float graph_p50 = 0.0F;
        float graph_p95 = 0.0F;
        float eager_p50 = 0.0F;
        float eager_p95 = 0.0F;
        if (prefill) {
          rc |= run_quartz_prefill(model, tokens, false, true, nullptr,
                                   &unperturbed, &logits);
          rc |= run_quartz_prefill(model, tokens, true, false, &families,
                                   &diagnostic, &logits);
        } else {
          rc |= run_quartz_decode(model, tokens, prefix, outputs, false, true,
                                  nullptr, &unperturbed, &graph_p50, &graph_p95,
                                  &logits);
          rc |= run_quartz_decode(model, tokens, prefix, outputs, true, false,
                                  &families, &diagnostic, &eager_p50, &eager_p95,
                                  &logits);
        }
        dump_families(options, configs[ident_index].id, families, phase, rep);
        std::printf(
            "quartz_rep=%d phase=%s ident=%s shipping_graph_wall_ms=%.9g "
            "eager_diagnostic_wall_ms=%.9g instrumentation_overhead_ms=%.9g "
            "graph_itl_p50_ms=%.9g graph_itl_p95_ms=%.9g "
            "eager_itl_p50_ms=%.9g eager_itl_p95_ms=%.9g "
            "records=%zu graph_mode_diagnostic=eager_diagnostic "
            "unperturbed_graph_mode=cuda_graph exclusive_gpu=true "
            "prefix_excluded=true graph_create_excluded=true "
            "window=measured independently_restored=true order=%s "
            "q4_decode=%s ffn_decode=%s\n",
            rep, phase, configs[ident_index].id, unperturbed, diagnostic,
            diagnostic - unperturbed, graph_p50, graph_p95, eager_p50,
            eager_p95, families.count, ba ? "BA" : "AB",
            configs[ident_index].q4, configs[ident_index].ffn);
        if (rc != 0) break;
      }
      observed_samples += 1;
      if (rc != 0) break;
    }
  }
  release_gpu();
  rc |= run_llama(options, phase, "unperturbed",
                  prefill ? static_cast<int>(prompt) : 0,
                  prefill ? 0 : static_cast<int>(prefix),
                  static_cast<int>(outputs), warmups > 0 ? warmups : 1,
                  samples);
  rc |= run_llama(options, phase, "eager_diagnostic",
                  prefill ? static_cast<int>(prompt) : 0,
                  prefill ? 0 : static_cast<int>(prefix),
                  static_cast<int>(outputs), warmups > 0 ? warmups : 1,
                  samples);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-071\",\"workload\":\"%s\","
      "\"engines\":2,\"modes\":[\"unperturbed\",\"eager_diagnostic\"],"
      "\"repetitions\":%d,\"records\":%zu,\"claims_throughput\":false,"
      "\"shipping_graph_wall\":true}\n",
      kPrefix, phase, samples, families.count);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-090\",\"workload\":\"%s\","
      "\"ident\":\"%s\",\"engines\":2,"
      "\"modes\":[\"unperturbed\",\"eager_diagnostic\"],"
      "\"warmups\":%d,\"samples\":%d,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":%d,\"observed_shapes\":1,"
      "\"observed_tier\":\"screen\",\"prefix\":%zu,\"prompt\":%zu,"
      "\"output_tokens\":%zu,\"records\":%zu,\"claims_throughput\":false,"
      "\"keep\":false,\"ident_count\":%d}\n",
      kOpt090Prefix, phase, configs[0].id, warmups, samples, observed_warmups,
      observed_samples, ident_count, prefix, prompt, outputs, families.count,
      ident_count);
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
