#include "full_scheduler.h"

#include "ffn_decode_path.cuh"
#include "q4k_decode_path.cuh"
#include "q8_decode_path.cuh"
#include "rms_norm.cuh"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "diagnostic_trace.h"
#include "model.h"
#include "scheduler.h"
#include "sha256.h"
#include "tokenizer.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT058_RESULT=";
constexpr std::array<std::size_t, 2> kTokens{42, 3649};
constexpr std::size_t kSavedTaps = 5;
constexpr std::size_t kMaxGenerate = 16;

struct Options final {
  const char* model = nullptr;
  const char* workload = nullptr;
  const char* bundle = nullptr;
  const char* llama_oracle = nullptr;
  const char* prompt_set = "v2";
};

struct BundleCase final {
  std::string name;
  std::vector<std::uint32_t> context;
  std::vector<std::uint32_t> target;
};

struct TapCapture final {
  float* residuals;
  float* logits;
  std::size_t token_row;
  int received[kSavedTaps]{};
};

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|scheduler|functional|"
               "quality-baseline] [MODEL] [--bundle PATH] "
               "[--llama-oracle PATH] [--prompt-set v2|original]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--bundle") == 0 && index + 1 < argc) {
      options->bundle = argv[++index];
    } else if (std::strcmp(arg, "--llama-oracle") == 0 && index + 1 < argc) {
      options->llama_oracle = argv[++index];
    } else if (std::strcmp(arg, "--prompt-set") == 0 && index + 1 < argc) {
      options->prompt_set = argv[++index];
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
  if (tier == qw38::cuda::TestTier::kAcceptance) return "quality-baseline";
  if (tier == qw38::cuda::TestTier::kCorrectness) return "functional";
  return "scheduler";
}

std::string json_escape(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (unsigned char ch : input) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (ch < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
          out += buf;
        } else {
          out += static_cast<char>(ch);
        }
        break;
    }
  }
  return out;
}

const char* expected_choice(const std::string& name) {
  if (name == "task_arithmetic" || name == "task_sort") return "B";
  if (name == "task_python_len" || name == "task_json") return "A";
  if (name == "task_inference" || name == "task_reading") return "C";
  if (name == "task_minutes" || name == "task_sequence") return "D";
  return "A";
}

std::string trim_ascii(const std::string& input) {
  std::size_t begin = 0;
  std::size_t end = input.size();
  while (begin < end && (input[begin] == ' ' || input[begin] == '\n' ||
                         input[begin] == '\r' || input[begin] == '\t')) {
    ++begin;
  }
  while (end > begin && (input[end - 1] == ' ' || input[end - 1] == '\n' ||
                         input[end - 1] == '\r' || input[end - 1] == '\t')) {
    --end;
  }
  return input.substr(begin, end - begin);
}

const char* parse_answer(const std::string& text, const char* expected) {
  const std::string stripped = trim_ascii(text);
  if (stripped.empty()) return "empty";
  if (stripped == "A" || stripped == "B" || stripped == "C" || stripped == "D") {
    return stripped == expected ? "exact_choice" : "wrong_choice";
  }
  bool seen = false;
  for (char ch : stripped) {
    if (ch == 'A' || ch == 'B' || ch == 'C' || ch == 'D') {
      if (seen) return "second_answer";
      seen = true;
    }
  }
  return "extra_text";
}

struct FiniteSummary final {
  std::size_t count = 0;
  std::size_t finite = 0;
  std::size_t first_nonfinite = std::numeric_limits<std::size_t>::max();
  bool ok = true;
};

FiniteSummary inspect_finite(const float* values, std::size_t count) {
  FiniteSummary summary;
  summary.count = count;
  for (std::size_t index = 0; index < count; ++index) {
    if (std::isfinite(values[index])) {
      ++summary.finite;
    } else if (summary.first_nonfinite == std::numeric_limits<std::size_t>::max()) {
      summary.first_nonfinite = index;
    }
  }
  summary.ok = summary.finite == count;
  return summary;
}

std::string hex_sha(const unsigned char* data, std::size_t size) {
  std::string digest;
  const qw38::Status status = qw38::internal::sha256_bytes(data, size, &digest);
  return status.is_ok() ? digest : std::string("unavailable");
}

void print_finite(const char* name, const FiniteSummary& summary) {
  std::printf(
      "\"%s\":{\"count\":%zu,\"finite_count\":%zu,\"nonfinite_count\":%zu,"
      "\"first_nonfinite\":",
      name, summary.count, summary.finite, summary.count - summary.finite);
  if (summary.first_nonfinite == std::numeric_limits<std::size_t>::max()) {
    std::printf("null,\"finite\":%s}", summary.ok ? "true" : "false");
  } else {
    std::printf("%zu,\"finite\":%s}", summary.first_nonfinite,
                summary.ok ? "true" : "false");
  }
}

bool get_u32(std::ifstream* file, std::uint32_t* value) {
  std::uint8_t bytes[4];
  if (!file->read(reinterpret_cast<char*>(bytes), 4)) return false;
  *value = static_cast<std::uint32_t>(bytes[0] | (bytes[1] << 8) |
                                      (bytes[2] << 16) | (bytes[3] << 24));
  return true;
}

bool get_u16(std::ifstream* file, std::uint16_t* value) {
  std::uint8_t bytes[2];
  if (!file->read(reinterpret_cast<char*>(bytes), 2)) return false;
  *value = static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
  return true;
}

bool read_bundle(const char* path, std::vector<BundleCase>* cases) {
  std::ifstream file(path, std::ios::binary);
  char magic[8]{};
  const char expected[8] = {'Q', 'W', '3', '8', 'Q', 1, 0, 0};
  if (!file.read(magic, 8) || std::memcmp(magic, expected, 8) != 0) return false;
  std::uint32_t count = 0;
  if (!get_u32(&file, &count) || count == 0 || count > 64) return false;
  for (std::uint32_t index = 0; index < count; ++index) {
    std::uint16_t name_len = 0;
    std::uint32_t n_context = 0;
    std::uint32_t n_target = 0;
    if (!get_u16(&file, &name_len) || name_len == 0 || name_len > 128) {
      return false;
    }
    BundleCase item;
    item.name.resize(name_len);
    if (!file.read(item.name.data(), name_len) || !get_u32(&file, &n_context) ||
        !get_u32(&file, &n_target) || n_context == 0 || n_target == 0) {
      return false;
    }
    item.context.resize(n_context);
    item.target.resize(n_target);
    for (auto* values : {&item.context, &item.target}) {
      for (auto& token : *values) {
        std::uint32_t packed = 0;
        if (!get_u32(&file, &packed)) return false;
        token = packed;
      }
    }
    cases->push_back(std::move(item));
  }
  return file.peek() == std::ifstream::traits_type::eof();
}

qw38::Status capture_bundle_tap(const qw38::internal::TraceTensorView& tensor,
                                void* context) noexcept {
  auto* capture = static_cast<TapCapture*>(context);
  int slot = -1;
  if (std::strcmp(tensor.name, "layer_residual") == 0) {
    if (tensor.layer == 0) slot = 0;
    if (tensor.layer == 3) slot = 1;
    if (tensor.layer == 63) slot = 2;
  } else if (std::strcmp(tensor.name, "final_norm") == 0) {
    slot = 3;
  } else if (std::strcmp(tensor.name, "logits") == 0) {
    std::copy(tensor.values, tensor.values + tensor.value_count,
              capture->logits + capture->token_row *
                                    qw38::internal::kVocabularySize);
    capture->received[4] = 1;
    return qw38::Status::ok();
  }
  if (slot >= 0) {
    std::copy(tensor.values, tensor.values + tensor.value_count,
              capture->residuals +
                  (capture->token_row * 4 + static_cast<std::size_t>(slot)) *
                      qw38::internal::kResidualWidth);
    capture->received[static_cast<std::size_t>(slot)] = 1;
  }
  return qw38::Status::ok();
}

int load_model(const char* path, qw38::cuda::ResidentModel* model,
               qw38::internal::Tokenizer* tokenizer) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (status.is_ok() && tokenizer != nullptr) status = tokenizer->build(info);
  if (status.is_ok()) {
    status = model->upload(weights, mapping.data(), mapping.size());
  }
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

void print_selectors() {
  std::printf(
      "\"selectors\":{\"q4_decode\":\"%s\",\"q8_decode\":\"%s\","
      "\"ffn_decode\":\"%s\",\"execution_graph\":\"%s\",\"rms_norm\":\"%s\"}",
      qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::selected_q8_decode_path(),
      qw38::cuda::selected_ffn_decode_path(),
      qw38::cuda::selected_execution_graph_path(),
      qw38::cuda::selected_rms_norm_path());
}

int run_smoke() {
  const char* reasons[4] = {
      parse_answer("  B\n", "B"),
      parse_answer("", "A"),
      parse_answer("B please", "B"),
      parse_answer("B\nC", "B"),
  };
  const bool parser_ok = std::strcmp(reasons[0], "exact_choice") == 0 &&
                         std::strcmp(reasons[1], "empty") == 0 &&
                         std::strcmp(reasons[2], "extra_text") == 0 &&
                         std::strcmp(reasons[3], "second_answer") == 0;
  std::array<float, 4> finite_tap{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> nonfinite_tap{1.0F, NAN, 3.0F, 4.0F};
  const FiniteSummary finite = inspect_finite(finite_tap.data(), finite_tap.size());
  const FiniteSummary nonfinite =
      inspect_finite(nonfinite_tap.data(), nonfinite_tap.size());
  const bool tap_ok = finite.ok && !nonfinite.ok && nonfinite.first_nonfinite == 1;
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-058\","
              "\"workload\":\"smoke\",\"model_loaded\":false,"
              "\"parser_cases\":4,\"parser_ok\":%s,",
              kPrefix, parser_ok ? "true" : "false");
  print_finite("finite_tap", finite);
  std::printf(",");
  print_finite("nonfinite_tap", nonfinite);
  std::printf(",\"status\":\"%s\"}\n", parser_ok && tap_ok ? "passed" : "failed");
  std::printf("status=%s\n", parser_ok && tap_ok ? "passed" : "failed");
  return parser_ok && tap_ok ? 0 : 1;
}

int replay_prefix(const qw38::cuda::ResidentModel& model,
                  qw38::cuda::SchedulerSession* session,
                  qw38::cuda::SchedulerWorkspace* workspace,
                  qw38::cuda::SchedulerGraphs* graphs, std::size_t count,
                  float* logits, float* hidden) {
  qw38::Status status = session->reset();
  if (!status.is_ok()) return fail_status(status);
  for (std::size_t index = 0; index < count; ++index) {
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, kTokens[index], session, workspace, logits,
        qw38::internal::kVocabularySize, hidden, qw38::internal::kResidualWidth,
        &elapsed, nullptr, nullptr, qw38::cuda::PointwisePath::kFused, graphs);
    if (!status.is_ok()) return fail_status(status);
  }
  return 0;
}

int run_scheduler(const Options& options) {
  qw38::cuda::ResidentModel model;
  const int loaded = load_model(options.model, &model, nullptr);
  if (loaded != 0) return loaded;
  constexpr std::size_t kCapacity = 64;
  qw38::cuda::SchedulerWorkspace eager_workspace;
  qw38::cuda::SchedulerWorkspace graph_workspace;
  qw38::Status status = eager_workspace.create(kCapacity);
  if (status.is_ok()) status = graph_workspace.create(kCapacity);
  qw38::cuda::SchedulerSession eager_session;
  qw38::cuda::SchedulerSession graph_session;
  if (status.is_ok()) status = eager_session.create(kCapacity);
  if (status.is_ok()) status = graph_session.create(kCapacity);
  if (!status.is_ok()) return fail_status(status);

  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  std::vector<float> traced_logits(kTokens.size() *
                                  qw38::internal::kVocabularySize);
  std::vector<float> taps(kTokens.size() * 4 * qw38::internal::kResidualWidth);
  const char* modes[] = {"eager", "traced", "graph"};
  bool all_finite = true;
  qw38::cuda::SchedulerGraphs graphs;
  bool graphs_ready = false;
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-058\","
              "\"workload\":\"scheduler\",\"tokens\":[42,3649],"
              "\"modes\":[\"eager\",\"traced\",\"graph\"],"
              "\"model_loaded_once\":true,\"single_traversal_taps\":true,",
              kPrefix);
  print_selectors();
  std::printf(",\"runs\":[");
  bool first_run = true;
  for (std::size_t row = 0; row < kTokens.size(); ++row) {
    for (const char* mode : modes) {
      const bool use_graph = std::strcmp(mode, "graph") == 0;
      if (use_graph && !graphs_ready) {
        status = graphs.create(model, &graph_workspace);
        if (!status.is_ok()) return fail_status(status);
        graphs_ready = true;
        status = graph_session.reset();
        if (!status.is_ok()) return fail_status(status);
        float capture_ms = 0.0F;
        status = qw38::cuda::execute_token(
            model, kTokens[0], &graph_session, &graph_workspace, logits.data(),
            qw38::internal::kVocabularySize, hidden.data(), hidden.size(),
            &capture_ms, nullptr, nullptr, qw38::cuda::PointwisePath::kFused,
            nullptr);
        if (!status.is_ok()) return fail_status(status);
        const FiniteSummary after_capture =
            inspect_finite(logits.data(), logits.size());
        if (!after_capture.ok) {
          std::fprintf(stderr,
                       "eager after graph capture is nonfinite first=%zu\n",
                       after_capture.first_nonfinite);
          return 1;
        }
        status = graph_session.reset();
        if (!status.is_ok()) return fail_status(status);
      }
      qw38::cuda::SchedulerSession* session =
          use_graph ? &graph_session : &eager_session;
      qw38::cuda::SchedulerWorkspace* workspace =
          use_graph ? &graph_workspace : &eager_workspace;
      qw38::cuda::SchedulerGraphs* graph_ptr = use_graph ? &graphs : nullptr;
      const int prefix =
          replay_prefix(model, session, workspace, nullptr, row, logits.data(),
                        hidden.data());
      if (prefix != 0) return prefix;
      float elapsed = 0.0F;
      TapCapture capture{taps.data(), traced_logits.data(), row};
      if (std::strcmp(mode, "traced") == 0) {
        status = qw38::cuda::execute_token_traced_bundle(
            model, kTokens[row], session, workspace, logits.data(),
            qw38::internal::kVocabularySize, hidden.data(), hidden.size(),
            &elapsed, capture_bundle_tap, &capture, nullptr,
            qw38::cuda::PointwisePath::kUnfused);
      } else {
        status = qw38::cuda::execute_token(
            model, kTokens[row], session, workspace, logits.data(),
            qw38::internal::kVocabularySize, hidden.data(), hidden.size(),
            &elapsed, nullptr, nullptr, qw38::cuda::PointwisePath::kFused,
            graph_ptr);
      }
      if (!status.is_ok()) return fail_status(status);
      const FiniteSummary summary =
          inspect_finite(logits.data(), logits.size());
      if (!summary.ok) all_finite = false;
      if (!first_run) std::printf(",");
      first_run = false;
      std::printf("{\"token\":%zu,\"mode\":\"%s\",\"elapsed_ms\":%.9g,"
                  "\"frontier\":%zu,\"taps_received\":%d,",
                  kTokens[row], mode, static_cast<double>(elapsed),
                  session->frontier(),
                  capture.received[0] + capture.received[1] +
                      capture.received[2] + capture.received[3] +
                      capture.received[4]);
      print_finite("logits", summary);
      std::printf(",\"logits_sha256\":\"%s\"}",
                  hex_sha(reinterpret_cast<const unsigned char*>(logits.data()),
                          logits.size() * sizeof(float))
                      .c_str());
      if (!summary.ok) {
        std::printf("],\"status\":\"nonfinite\",\"first_nonfinite_mode\":\"%s\","
                    "\"first_nonfinite_token\":%zu}\n",
                    mode, kTokens[row]);
        std::printf("status=failed\n");
        return 1;
      }
      if (std::strcmp(mode, "traced") == 0) {
        int received = 0;
        for (int flag : capture.received) received += flag;
        if (received != static_cast<int>(kSavedTaps)) {
          std::fprintf(stderr, "traced bundle missed taps token=%zu got=%d\n",
                       kTokens[row], received);
          return 1;
        }
      }
    }
  }
  std::printf("],\"saved_taps\":5,\"all_finite\":%s,\"historical_opt046_nan\":"
              "\"unreproduced_on_current_packed_production\",\"status\":\"%s\"}\n",
              all_finite ? "true" : "false", all_finite ? "passed" : "failed");
  std::printf("status=%s\n", all_finite ? "passed" : "failed");
  return all_finite ? 0 : 1;
}

double target_nll(const std::vector<float>& logits, std::uint32_t target) {
  double maximum = static_cast<double>(
      *std::max_element(logits.begin(), logits.end()));
  double sum = 0.0;
  for (float value : logits) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
    sum += std::exp(static_cast<double>(value) - maximum);
  }
  return -(static_cast<double>(logits[target]) - maximum - std::log(sum));
}

void top_two(const std::vector<float>& logits, std::size_t* first,
             std::size_t* second) {
  *first = 0;
  *second = 1;
  for (std::size_t index = 1; index < logits.size(); ++index) {
    if (logits[index] > logits[*first] ||
        (logits[index] == logits[*first] && index < *first)) {
      *second = *first;
      *first = index;
    } else if (index != *first &&
               (logits[index] > logits[*second] ||
                (logits[index] == logits[*second] && index < *second))) {
      *second = index;
    }
  }
}

int generate_case(const qw38::cuda::ResidentModel& model,
                  const BundleCase& item, qw38::cuda::SchedulerSession* session,
                  qw38::cuda::SchedulerWorkspace* workspace,
                  qw38::cuda::SchedulerGraphs* graphs,
                  qw38::internal::Tokenizer* tokenizer, std::size_t max_new,
                  std::vector<std::uint32_t>* tokens, std::string* text,
                  std::size_t* greedy0, std::size_t* runner0, float* logit0,
                  float* runner_logit0) {
  qw38::Status status = session->reset();
  if (!status.is_ok()) return fail_status(status);
  std::vector<std::size_t> context(item.context.begin(), item.context.end());
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (!context.empty()) {
    qw38::cuda::SyncResult sync{};
    status = qw38::cuda::sync_tokens(
        model, context.data(), context.size(), session, workspace,
        logits.data(), logits.size(), hidden.data(), hidden.size(), &sync,
        nullptr, graphs);
    if (!status.is_ok()) return fail_status(status);
  }
  tokens->clear();
  for (std::size_t step = 0; step < max_new; ++step) {
    std::size_t first = 0;
    std::size_t second = 1;
    top_two(logits, &first, &second);
    if (step == 0) {
      *greedy0 = first;
      *runner0 = second;
      *logit0 = logits[first];
      *runner_logit0 = logits[second];
    }
    const FiniteSummary summary = inspect_finite(logits.data(), logits.size());
    if (!summary.ok) {
      std::fprintf(stderr, "nonfinite logits during generate %s first=%zu\n",
                   item.name.c_str(), summary.first_nonfinite);
      return 1;
    }
    tokens->push_back(static_cast<std::uint32_t>(first));
    if (first == 248046 || first == 248044) {
      break;
    }
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, first, session, workspace, logits.data(), logits.size(),
        hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, graphs);
    if (!status.is_ok()) return fail_status(status);
  }
  if (tokenizer != nullptr) {
    status = tokenizer->decode(*tokens, true, text);
    if (!status.is_ok()) return fail_status(status);
  }
  return 0;
}

int run_llama_generate(const Options& options, const char* bundle_path) {
  if (options.llama_oracle == nullptr) return 0;
  std::string directory(options.llama_oracle);
  const std::size_t slash = directory.find_last_of('/');
  if (slash != std::string::npos) directory.resize(slash);
  const char* out_path = "build/opt058-llama-generate.tsv";
  const char* err_path = "build/opt058-llama-generate.err";
  std::string command = "LD_LIBRARY_PATH=" + directory + ":${LD_LIBRARY_PATH} ";
  command += std::string(options.llama_oracle) + " " + options.model + " " +
             bundle_path + " --generate 16 >" + out_path + " 2>" + err_path;
  const int code = std::system(command.c_str());
  std::ifstream in(out_path);
  if (!in) {
    std::fprintf(stderr, "cannot read llama generate output\n");
    return 1;
  }
  std::printf(",\"llama_generate\":[");
  bool first = true;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("gen\t", 0) != 0) continue;
    if (!first) std::printf(",");
    first = false;
    char name[128];
    unsigned long pos = 0;
    unsigned long token = 0;
    unsigned long runner = 0;
    double logit = 0.0;
    double runner_logit = 0.0;
    if (std::sscanf(line.c_str(), "gen\t%127s\t%lu\t%lu\t%lf\t%lu\t%lf", name,
                    &pos, &token, &logit, &runner, &runner_logit) != 6) {
      continue;
    }
    std::printf("{\"case\":\"%s\",\"position\":%lu,\"token\":%lu,\"logit\":%.9g,"
                "\"runner_up_token\":%lu,\"runner_up_logit\":%.9g}",
                name, pos, token, logit, runner, runner_logit);
  }
  std::printf("]");
  return code == 0 ? 0 : 1;
}

int run_functional(const Options& options) {
  if (options.bundle == nullptr) {
    std::fprintf(stderr, "functional workload requires --bundle\n");
    return 2;
  }
  std::vector<BundleCase> cases;
  if (!read_bundle(options.bundle, &cases) || cases.size() != 8) {
    std::fprintf(stderr, "functional bundle must contain eight cases\n");
    return 2;
  }
  std::string token_271_hex;
  std::string quartz_json = "[";
  std::size_t output_tokens = 0;
  {
    qw38::cuda::ResidentModel model;
    qw38::internal::Tokenizer tokenizer;
    const int loaded = load_model(options.model, &model, &tokenizer);
    if (loaded != 0) return loaded;
    std::size_t capacity = 64;
    for (const auto& item : cases) {
      capacity = std::max(capacity, item.context.size() + kMaxGenerate);
    }
    qw38::cuda::SchedulerWorkspace workspace;
    qw38::Status status = workspace.create(capacity);
    qw38::cuda::SchedulerSession session;
    if (status.is_ok()) status = session.create(capacity);
    qw38::cuda::SchedulerGraphs graphs;
    if (status.is_ok()) status = graphs.create(model, &workspace);
    if (!status.is_ok()) return fail_status(status);
    std::string token_271;
    status = tokenizer.decode({271}, false, &token_271);
    if (!status.is_ok()) return fail_status(status);
    for (unsigned char ch : token_271) {
      char buf[3];
      std::snprintf(buf, sizeof(buf), "%02x", ch);
      token_271_hex += buf;
    }
    bool first = true;
    for (const auto& item : cases) {
      std::vector<std::uint32_t> tokens;
      std::string text;
      std::size_t greedy = 0;
      std::size_t runner = 0;
      float logit = 0.0F;
      float runner_logit = 0.0F;
      const int generated = generate_case(
          model, item, &session, &workspace, &graphs, &tokenizer, kMaxGenerate,
          &tokens, &text, &greedy, &runner, &logit, &runner_logit);
      if (generated != 0) return generated;
      output_tokens += tokens.size();
      if (!first) quartz_json += ",";
      first = false;
      quartz_json += "{\"case\":\"" + json_escape(item.name) +
                     "\",\"next_token\":" + std::to_string(greedy) +
                     ",\"top_two\":[" + std::to_string(greedy) + "," +
                     std::to_string(runner) + "],\"text\":\"" + json_escape(text) +
                     "\",\"parser\":\"" +
                     parse_answer(text, expected_choice(item.name)) +
                     "\",\"tokens\":[";
      for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index) quartz_json += ",";
        quartz_json += std::to_string(tokens[index]);
      }
      quartz_json += "]}";
    }
    quartz_json += "]";
  }
  cudaDeviceSynchronize();
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-058\","
              "\"workload\":\"functional\",\"prompt_set\":\"%s\","
              "\"engines\":[\"quartz\",\"llama\"],\"max_new_tokens\":16,"
              "\"token_271\":271,\"token_271_hex\":\"%s\",",
              kPrefix, options.prompt_set, token_271_hex.c_str());
  print_selectors();
  std::printf(",\"quartz\":%s,\"quartz_output_tokens\":%zu", quartz_json.c_str(),
              output_tokens);
  int llama = 0;
  if (options.llama_oracle != nullptr) {
    llama = run_llama_generate(options, options.bundle);
  } else {
    std::printf(",\"llama_generate\":null,\"llama_oracle\":\"missing\"");
  }
  const bool ok = llama == 0 && output_tokens <= 256;
  std::printf(",\"status\":\"%s\"}\n", ok ? "passed" : "failed");
  std::printf("status=%s\n", ok ? "passed" : "failed");
  return ok ? 0 : 1;
}

int score_case(const qw38::cuda::ResidentModel& model, const BundleCase& item,
               qw38::cuda::SchedulerSession* session,
               qw38::cuda::SchedulerWorkspace* workspace,
               qw38::cuda::SchedulerGraphs* graphs, double* mean_nll,
               std::size_t* scored) {
  qw38::Status status = session->reset();
  if (!status.is_ok()) return fail_status(status);
  std::vector<std::size_t> context(item.context.begin(), item.context.end());
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  if (!context.empty()) {
    qw38::cuda::SyncResult sync{};
    status = qw38::cuda::sync_tokens(
        model, context.data(), context.size(), session, workspace,
        logits.data(), logits.size(), hidden.data(), hidden.size(), &sync,
        nullptr, graphs);
    if (!status.is_ok()) return fail_status(status);
  }
  double total = 0.0;
  *scored = 0;
  for (std::uint32_t target : item.target) {
    const FiniteSummary summary = inspect_finite(logits.data(), logits.size());
    if (!summary.ok) {
      std::fprintf(stderr, "nonfinite logits scoring %s first=%zu\n",
                   item.name.c_str(), summary.first_nonfinite);
      return 1;
    }
    total += target_nll(logits, target);
    ++*scored;
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, target, session, workspace, logits.data(), logits.size(),
        hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, graphs);
    if (!status.is_ok()) return fail_status(status);
  }
  *mean_nll = total / static_cast<double>(*scored);
  return 0;
}

int run_quality_baseline(const Options& options) {
  if (options.bundle == nullptr) {
    std::fprintf(stderr, "quality-baseline requires --bundle\n");
    return 2;
  }
  std::vector<BundleCase> cases;
  if (!read_bundle(options.bundle, &cases)) {
    std::fprintf(stderr, "cannot read NLL bundle\n");
    return 2;
  }
  qw38::cuda::ResidentModel model;
  const int loaded = load_model(options.model, &model, nullptr);
  if (loaded != 0) return loaded;
  std::size_t capacity = 64;
  for (const auto& item : cases) {
    capacity = std::max(capacity, item.context.size() + item.target.size());
  }
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::Status status = workspace.create(capacity);
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(capacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);
  std::printf("%s{\"schema_version\":1,\"task\":\"OPT-058\","
              "\"workload\":\"quality-baseline\",\"engine\":\"quartz\","
              "\"repetitions\":1,",
              kPrefix);
  print_selectors();
  std::printf(",\"cases\":[");
  bool first = true;
  for (const auto& item : cases) {
    double mean = 0.0;
    std::size_t scored = 0;
    const int rc =
        score_case(model, item, &session, &workspace, &graphs, &mean, &scored);
    if (rc != 0) return rc;
    if (!first) std::printf(",");
    first = false;
    std::printf("{\"name\":\"%s\",\"scored\":%zu,\"mean_nll\":%.17g}",
                item.name.c_str(), scored, mean);
  }
  std::printf("],\"status\":\"passed\"}\n");
  std::printf("status=passed\n");
  return 0;
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
  const int parsed = parse_args(argc, argv, &options);
  if (parsed != 0) return parsed;
  if (options.workload == nullptr) {
    options.workload = default_workload(qw38::cuda::test_tier());
  }
  if (std::strcmp(options.workload, "smoke") == 0) return run_smoke();
  if (options.model == nullptr) {
    std::fprintf(stderr, "model path is required for %s\n", options.workload);
    return 2;
  }
  if (std::strcmp(options.workload, "scheduler") == 0) {
    return run_scheduler(options);
  }
  if (std::strcmp(options.workload, "functional") == 0) {
    return run_functional(options);
  }
  if (std::strcmp(options.workload, "quality-baseline") == 0) {
    return run_quality_baseline(options);
  }
  return usage(argv[0]);
}
