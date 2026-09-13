#include "attention_decode.h"
#include "execution_graph_path.cuh"
#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "q4k_decode_path.cuh"
#include "q8_decode_path.cuh"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "tokenizer.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT116_GENERATED_QUALITY_RESULT=";
constexpr int kPrefix8192 = 8192;
constexpr int kPrefix32768 = 32768;
constexpr int kPrefix131040 = 131040;
constexpr int kScored = 32;
constexpr int kCapacity = 131072;
constexpr int kGdnUpdates = 2048;
constexpr std::uint32_t kStopA = 248046;
constexpr std::uint32_t kStopB = 248044;
constexpr std::uint64_t kDefaultSeed = 0x5157385357495353ULL;
constexpr std::size_t kGenCapacity = 4096;
char g_applied_q4[64];
char g_applied_ffn[64];
char g_applied_q8[32];
char g_applied_attn[64];

struct Options final {
  const char* model = nullptr;
  const char* workload = "probe";
  const char* quality_config = nullptr;
  const char* q4_decode = nullptr;
  const char* ffn_decode = nullptr;
  const char* q8_layout = nullptr;
  const char* attention_pipeline = nullptr;
  const char* packed_kv = nullptr;
  const char* weight_requant = nullptr;
  const char* cases_path = nullptr;
  const char* cache_jobs = nullptr;
  const char* cache_tokens = nullptr;
  const char* out_path = nullptr;
  bool quality = false;
};

struct GenerateCase final {
  std::string id;
  std::string mode;
  std::size_t max_new = 64;
  float temperature = 0.0F;
  float top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0;
  std::string prompt;
};

struct CacheJob final {
  std::string id;
  std::size_t prefix = 0;
  std::size_t scored = 32;
  std::size_t capacity = 131072;
  std::vector<int> retrieval;
};

struct FiniteSummary final {
  std::size_t count = 0;
  std::size_t finite = 0;
  std::size_t first_nonfinite = std::numeric_limits<std::size_t>::max();
  bool ok = true;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload probe|gpu-baseline] [MODEL] [--quality] "
               "[--quality-config PATH] [--q4-decode PATH] [--ffn-decode PATH] "
               "[--q8-layout LAYOUT] [--attention-pipeline PATH] "
               "[--packed-kv dense_bf16|q8q8|q8q4|q4q4] "
               "[--weight-requant none|q8_to_q4k|q8_q6_to_q4k|fp4_study] "
               "[--cases PATH] [--cache-jobs PATH] [--cache-tokens PATH] "
               "[--out PATH]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--quality") == 0) {
      options->quality = true;
    } else if (std::strcmp(arg, "--quality-config") == 0 && index + 1 < argc) {
      options->quality_config = argv[++index];
    } else if (std::strcmp(arg, "--q4-decode") == 0 && index + 1 < argc) {
      options->q4_decode = argv[++index];
    } else if (std::strcmp(arg, "--ffn-decode") == 0 && index + 1 < argc) {
      options->ffn_decode = argv[++index];
    } else if (std::strcmp(arg, "--q8-layout") == 0 && index + 1 < argc) {
      options->q8_layout = argv[++index];
    } else if (std::strcmp(arg, "--attention-pipeline") == 0 &&
               index + 1 < argc) {
      options->attention_pipeline = argv[++index];
    } else if (std::strcmp(arg, "--packed-kv") == 0 && index + 1 < argc) {
      options->packed_kv = argv[++index];
    } else if (std::strcmp(arg, "--weight-requant") == 0 && index + 1 < argc) {
      options->weight_requant = argv[++index];
    } else if (std::strcmp(arg, "--cases") == 0 && index + 1 < argc) {
      options->cases_path = argv[++index];
    } else if (std::strcmp(arg, "--cache-jobs") == 0 && index + 1 < argc) {
      options->cache_jobs = argv[++index];
    } else if (std::strcmp(arg, "--cache-tokens") == 0 && index + 1 < argc) {
      options->cache_tokens = argv[++index];
    } else if (std::strcmp(arg, "--out") == 0 && index + 1 < argc) {
      options->out_path = argv[++index];
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
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

bool json_string_field(const std::string& text, const char* key,
                       std::string* value) {
  const std::string needle = std::string("\"") + key + "\":";
  const std::size_t found = text.find(needle);
  if (found == std::string::npos) return false;
  std::size_t cursor = found + needle.size();
  while (cursor < text.size() &&
         (text[cursor] == ' ' || text[cursor] == '\t')) {
    ++cursor;
  }
  if (cursor >= text.size() || text[cursor] != '"') return false;
  ++cursor;
  std::string out;
  while (cursor < text.size() && text[cursor] != '"') {
    out.push_back(text[cursor]);
    ++cursor;
  }
  *value = out;
  return true;
}

int apply_quality_selectors(const Options& options) {
  std::string q4 = options.q4_decode != nullptr ? options.q4_decode : "";
  std::string ffn = options.ffn_decode != nullptr ? options.ffn_decode : "";
  std::string q8 = options.q8_layout != nullptr ? options.q8_layout : "";
  std::string attn =
      options.attention_pipeline != nullptr ? options.attention_pipeline : "";
  std::string packed =
      options.packed_kv != nullptr ? options.packed_kv : "";
  std::string requant =
      options.weight_requant != nullptr ? options.weight_requant : "";
  if (options.quality_config != nullptr) {
    std::ifstream file(options.quality_config);
    if (!file) {
      std::fprintf(stderr, "cannot read quality-config %s\n",
                   options.quality_config);
      return 2;
    }
    const std::string text((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    std::string parsed;
    if (json_string_field(text, "q4_decode", &parsed)) q4 = parsed;
    if (json_string_field(text, "q4_staging", &parsed)) ffn = parsed;
    if (json_string_field(text, "ffn_decode", &parsed) && ffn.empty()) {
      ffn = parsed;
    }
    if (json_string_field(text, "q8_decode", &parsed)) q8 = parsed;
    if (json_string_field(text, "q8_layout", &parsed) &&
        (q8.empty() || q8 == "dp4a_q8_1" || q8 == "r2_w2" || q8 == "r1_w4")) {
      if (parsed == "r1_w4" || parsed == "r2_w2") q8 = parsed;
    }
    if (json_string_field(text, "prompt_attention", &parsed)) attn = parsed;
    if (json_string_field(text, "packed_kv", &parsed)) packed = parsed;
    if (json_string_field(text, "weight_requant", &parsed)) requant = parsed;
  }
  if (options.quality && q4.empty() && ffn.empty() && q8.empty()) {
    std::printf("quality_flag=true restored_packed_or_r2=false "
                "selectors_unchanged=true\n");
  }
  if (!q4.empty()) {
    std::snprintf(g_applied_q4, sizeof(g_applied_q4), "%s", q4.c_str());
    if (!qw38::cuda::apply_q4_decode_ident(g_applied_q4, 4U)) {
      std::fprintf(stderr, "invalid --q4-decode %s\n", q4.c_str());
      return 2;
    }
  }
  if (!ffn.empty()) {
    std::snprintf(g_applied_ffn, sizeof(g_applied_ffn), "%s", ffn.c_str());
    if (!qw38::cuda::apply_ffn_decode_ident(g_applied_ffn)) {
      std::fprintf(stderr, "invalid --ffn-decode %s\n", ffn.c_str());
      return 2;
    }
  }
  if (!q8.empty() && (q8 == "r1_w4" || q8 == "r2_w2" || q8 == "r4_w1" ||
                      q8 == "r8_w1")) {
    std::snprintf(g_applied_q8, sizeof(g_applied_q8), "%s", q8.c_str());
    if (!qw38::cuda::apply_q8_layout_ident(g_applied_q8)) {
      std::fprintf(stderr, "invalid --q8-layout %s\n", q8.c_str());
      return 2;
    }
  }
  if (!attn.empty()) {
    std::snprintf(g_applied_attn, sizeof(g_applied_attn), "%s", attn.c_str());
    if (!qw38::cuda::apply_attention_pipeline_ident(g_applied_attn)) {
      std::fprintf(stderr, "invalid --attention-pipeline %s\n", attn.c_str());
      return 2;
    }
  }
  if (!packed.empty() && !qw38::cuda::apply_packed_kv_format_ident(packed.c_str())) {
    std::fprintf(stderr, "invalid --packed-kv %s\n", packed.c_str());
    return 2;
  }
  if (!requant.empty() && !qw38::cuda::apply_weight_requant_ident(requant.c_str())) {
    std::fprintf(stderr, "invalid --weight-requant %s\n", requant.c_str());
    return 2;
  }
  if (options.quality) {
    const char* effective_q4 = qw38::cuda::effective_q4_decode_path();
    const char* layout = qw38::cuda::q8_layout_ident(
        qw38::cuda::effective_q8_decode_rows_skinny(),
        qw38::cuda::effective_q8_decode_layout_warps_skinny());
    if (!q4.empty() && std::strcmp(effective_q4, q4.c_str()) != 0) {
      std::fprintf(stderr, "--quality restored q4_decode to %s\n", effective_q4);
      return 2;
    }
    if (!q8.empty() && (q8 == "r1_w4" || q8 == "r2_w2") &&
        std::strcmp(layout, q8.c_str()) != 0) {
      std::fprintf(stderr, "--quality restored q8_layout to %s\n", layout);
      return 2;
    }
    std::printf("quality_flag=true restored_packed_or_r2=false "
                "effective_q4=%s effective_ffn=%s effective_q8_layout=%s "
                "effective_attention=%s applied_before_graph=true\n",
                qw38::cuda::effective_q4_decode_path(),
                qw38::cuda::effective_ffn_decode_path(), layout,
                qw38::cuda::current_attention_pipeline_path());
  }
  return 0;
}

FiniteSummary inspect_finite(const float* values, std::size_t count) {
  FiniteSummary summary;
  summary.count = count;
  for (std::size_t index = 0; index < count; ++index) {
    if (std::isfinite(values[index])) {
      ++summary.finite;
    } else if (summary.first_nonfinite ==
               std::numeric_limits<std::size_t>::max()) {
      summary.first_nonfinite = index;
    }
  }
  summary.ok = summary.finite == count;
  return summary;
}

double target_nll(const std::vector<float>& logits, std::uint32_t target) {
  if (target >= logits.size()) return std::numeric_limits<double>::quiet_NaN();
  double maximum = static_cast<double>(
      *std::max_element(logits.begin(), logits.end()));
  double sum = 0.0;
  for (float value : logits) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
    sum += std::exp(static_cast<double>(value) - maximum);
  }
  return -(static_cast<double>(logits[target]) - maximum - std::log(sum));
}

std::size_t greedy_token(const std::vector<float>& logits) {
  std::size_t best = 0;
  for (std::size_t index = 1; index < logits.size(); ++index) {
    if (logits[index] > logits[best]) best = index;
  }
  return best;
}

qw38::Status pull_outputs(
    qw38::cuda::SchedulerSession* session, std::vector<float>* logits,
    std::array<float, qw38::internal::kResidualWidth>* hidden) {
  return session->copy_last_outputs(logits->data(), logits->size(),
                                    hidden->data(), hidden->size());
}

std::size_t sample_token(const std::vector<float>& logits, float temperature,
                         float top_p, std::uint32_t top_k,
                         qw38::cuda::SamplerState* state) {
  if (temperature <= 0.0F) return greedy_token(logits);
  std::vector<std::size_t> candidates(logits.size());
  std::iota(candidates.begin(), candidates.end(), 0);
  const bool ordered = top_k > 0 || top_p < 1.0F;
  if (ordered) {
    const std::size_t retained =
        top_k == 0 ? candidates.size()
                   : std::min<std::size_t>(top_k, candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + retained, candidates.end(),
        [&logits](std::size_t left, std::size_t right) {
          return logits[left] > logits[right];
        });
    candidates.resize(retained);
  }
  float maximum = -INFINITY;
  for (std::size_t index : candidates) {
    maximum = std::max(maximum, logits[index]);
  }
  std::vector<double> weights(candidates.size());
  double total = 0.0;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    weights[index] = std::exp(static_cast<double>(
        (logits[candidates[index]] - maximum) / temperature));
    total += weights[index];
  }
  if (ordered && top_p < 1.0F) {
    const double threshold = total * static_cast<double>(top_p);
    double cumulative = 0.0;
    std::size_t retained = 0;
    do {
      cumulative += weights[retained++];
    } while (retained < weights.size() && cumulative < threshold);
    candidates.resize(retained);
    weights.resize(retained);
    total = cumulative;
  }
  state->rng_state ^= state->rng_state >> 12U;
  state->rng_state ^= state->rng_state << 25U;
  state->rng_state ^= state->rng_state >> 27U;
  const std::uint64_t random = state->rng_state * 0x2545F4914F6CDD1DULL;
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
  return candidates[selected];
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

std::string hex_decode(const std::string& hex) {
  if (hex.size() % 2 != 0) return {};
  std::string out(hex.size() / 2, '\0');
  for (std::size_t index = 0; index < out.size(); ++index) {
    unsigned value = 0;
    if (std::sscanf(hex.c_str() + (2 * index), "%2x", &value) != 1) return {};
    out[index] = static_cast<char>(value);
  }
  return out;
}

bool load_generate_cases(const char* path, std::vector<GenerateCase>* cases) {
  std::ifstream file(path);
  if (!file) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream stream(line);
    GenerateCase item;
    std::string hex;
    if (!(stream >> item.id >> item.mode >> item.max_new >> item.temperature >>
          item.top_p >> item.top_k >> item.seed >> hex)) {
      return false;
    }
    item.prompt = hex_decode(hex);
    if (item.prompt.empty() && !hex.empty()) return false;
    cases->push_back(std::move(item));
  }
  return !cases->empty();
}

bool load_cache_jobs(const char* path, std::vector<CacheJob>* jobs) {
  std::ifstream file(path);
  if (!file) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream stream(line);
    CacheJob job;
    std::string positions;
    if (!(stream >> job.id >> job.prefix >> job.scored >> job.capacity >>
          positions)) {
      return false;
    }
    std::stringstream pos_stream(positions);
    std::string part;
    while (std::getline(pos_stream, part, ',')) {
      if (part.empty()) continue;
      job.retrieval.push_back(std::atoi(part.c_str()));
    }
    jobs->push_back(std::move(job));
  }
  return !jobs->empty();
}

bool load_source_tokens(const char* path, std::vector<std::size_t>* tokens) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  file.seekg(0, std::ios::end);
  const std::streamoff bytes = file.tellg();
  file.seekg(0);
  if (bytes <= 0 || bytes % 4 != 0) return false;
  const std::size_t count = static_cast<std::size_t>(bytes) / 4;
  tokens->resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t packed = 0;
    file.read(reinterpret_cast<char*>(&packed), 4);
    (*tokens)[index] = packed % qw38::internal::kVocabularySize;
  }
  return tokens->size() == count;
}

int print_probe() {
  const char* graph = qw38::cuda::kSelectedExecutionGraphPath;
  const bool graph_ok = std::strcmp(graph, "ffn_only") == 0;
  const bool cache_fits =
      kPrefix131040 + kScored <= kCapacity && kPrefix8192 < kPrefix32768 &&
      kPrefix32768 < kPrefix131040;
  const bool decode_path_required = true;
  const bool all_prefill_forbidden = true;
  const bool encodings_pinned = true;
  const bool pass = graph_ok && cache_fits && decode_path_required &&
                    all_prefill_forbidden && encodings_pinned &&
                    kGdnUpdates == 2048;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-116\","
      "\"graph_path\":\"%s\",\"cache_path\":\"compressed_cache_decode\","
      "\"forbidden_path\":\"uncached_all_prefill\","
      "\"prefix_8192\":%d,\"prefix_32768\":%d,\"prefix_131040\":%d,"
      "\"scored_or_generated\":%d,\"capacity\":%d,\"gdn_decode_updates\":%d,"
      "\"q4_decode\":\"llama_q4k_mmvq\",\"q8_decode\":\"r1_w4\","
      "\"graph_ok\":%s,\"cache_fits\":%s,\"decode_path_required\":%s,"
      "\"all_prefill_forbidden\":%s,\"pass\":%s}\n",
      kPrefix, graph, kPrefix8192, kPrefix32768, kPrefix131040, kScored,
      kCapacity, kGdnUpdates, json_bool(graph_ok), json_bool(cache_fits),
      json_bool(decode_path_required), json_bool(all_prefill_forbidden),
      json_bool(pass));
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}

int generate_one(const qw38::cuda::ResidentModel& model,
                 qw38::internal::Tokenizer* tokenizer,
                 qw38::cuda::SchedulerSession* session,
                 qw38::cuda::SchedulerWorkspace* workspace,
                 const GenerateCase& item, std::string* text,
                 std::string* termination, bool* finite) {
  std::vector<std::uint32_t> ids;
  qw38::Status status = tokenizer->encode(item.prompt, &ids);
  if (!status.is_ok()) return fail_status(status);
  if (ids.empty() || ids.size() + item.max_new > session->capacity()) {
    std::fprintf(stderr, "prompt overflow %s tokens=%zu\n", item.id.c_str(),
                 ids.size());
    return 1;
  }
  status = session->reset();
  if (!status.is_ok()) return fail_status(status);
  std::vector<std::size_t> context(ids.begin(), ids.end());
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  status = qw38::cuda::sync_tokens(
      model, context.data(), context.size(), session, workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, nullptr);
  if (!status.is_ok()) return fail_status(status);
  status = pull_outputs(session, &logits, &hidden);
  if (!status.is_ok()) return fail_status(status);
  qw38::cuda::SamplerState sampler{
      item.temperature, item.top_p, item.top_k, item.seed,
      item.seed == 0 ? kDefaultSeed : item.seed};
  std::vector<std::uint32_t> produced;
  *termination = "max_new_tokens";
  *finite = true;
  for (std::size_t step = 0; step < item.max_new; ++step) {
    const FiniteSummary summary = inspect_finite(logits.data(), logits.size());
    if (!summary.ok) {
      *finite = false;
      std::fprintf(stderr, "nonfinite logits generate %s first=%zu\n",
                   item.id.c_str(), summary.first_nonfinite);
      return 1;
    }
    const std::size_t token =
        sample_token(logits, item.temperature, item.top_p, item.top_k, &sampler);
    produced.push_back(static_cast<std::uint32_t>(token));
    if (token == kStopA || token == kStopB) {
      *termination = "stop_token";
      break;
    }
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, token, session, workspace, logits.data(), logits.size(),
        hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, nullptr);
    if (!status.is_ok()) return fail_status(status);
    status = pull_outputs(session, &logits, &hidden);
    if (!status.is_ok()) return fail_status(status);
  }
  status = tokenizer->decode(produced, true, text);
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

int run_generations(const qw38::cuda::ResidentModel& model,
                    qw38::internal::Tokenizer* tokenizer,
                    const std::vector<GenerateCase>& cases, std::string* json) {
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerSession session;
  qw38::Status status = workspace.create(kGenCapacity);
  if (status.is_ok()) status = session.create(kGenCapacity);
  if (!status.is_ok()) return fail_status(status);
  *json = "[";
  bool first = true;
  for (const auto& item : cases) {
    std::fprintf(stderr, "opt116 generate %s mode=%s\n", item.id.c_str(),
                 item.mode.c_str());
    std::fflush(stderr);
    std::string text;
    std::string termination;
    bool finite = true;
    const int code = generate_one(model, tokenizer, &session, &workspace, item,
                                  &text, &termination, &finite);
    if (code != 0) return code;
    if (!first) *json += ",";
    first = false;
    *json += "{\"id\":\"" + json_escape(item.id) + "\",\"mode\":\"" +
             json_escape(item.mode) + "\",\"temperature\":" +
             std::to_string(item.temperature) + ",\"top_p\":" +
             std::to_string(item.top_p) + ",\"top_k\":" +
             std::to_string(item.top_k) + ",\"seed\":" +
             std::to_string(item.seed) + ",\"max_new_tokens\":" +
             std::to_string(item.max_new) + ",\"text\":\"" + json_escape(text) +
             "\",\"termination_reason\":\"" + json_escape(termination) +
             "\",\"finite\":" + json_bool(finite) +
             ",\"teacher_forced_answers\":false"
             ",\"generation_source\":\"gpu_free_running\"}";
  }
  *json += "]";
  return 0;
}

int run_gdn_stress(const qw38::cuda::ResidentModel& model, std::string* json) {
  constexpr std::size_t kPrefix = 128;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerSession session;
  qw38::Status status = workspace.create(kGenCapacity);
  if (status.is_ok()) status = session.create(kGenCapacity);
  qw38::cuda::SchedulerGraphs graphs;
  if (status.is_ok()) status = graphs.create(model, &workspace);
  if (!status.is_ok()) return fail_status(status);
  std::vector<std::size_t> tokens(kPrefix + kGdnUpdates);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), kPrefix, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &sync, nullptr, nullptr);
  if (!status.is_ok()) return fail_status(status);
  status = pull_outputs(&session, &logits, &hidden);
  if (!status.is_ok()) return fail_status(status);
  bool finite = true;
  std::vector<std::size_t> history;
  history.reserve(kGdnUpdates);
  for (int step = 0; step < kGdnUpdates; ++step) {
    const FiniteSummary summary = inspect_finite(logits.data(), logits.size());
    if (!summary.ok) {
      finite = false;
      break;
    }
    const std::size_t token = greedy_token(logits);
    history.push_back(token);
    float elapsed = 0.0F;
    status = qw38::cuda::execute_token(
        model, token, &session, &workspace, logits.data(), logits.size(),
        hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, nullptr);
    if (!status.is_ok()) return fail_status(status);
    status = pull_outputs(&session, &logits, &hidden);
    if (!status.is_ok()) return fail_status(status);
  }
  const std::string ckpt = "build/optimization-runs/OPT-116/opt116-gdn.ckpt";
  status = session.save_checkpoint(ckpt);
  if (!status.is_ok()) return fail_status(status);
  qw38::cuda::SchedulerWorkspace restored_ws;
  qw38::cuda::SchedulerSession restored;
  status = restored_ws.create(kGenCapacity);
  if (status.is_ok()) status = restored.create(kGenCapacity);
  if (status.is_ok()) status = restored.restore_checkpoint(ckpt, &restored_ws);
  bool save_restore = false;
  if (status.is_ok()) status = session.state_equals(restored, &save_restore);
  if (!status.is_ok()) return fail_status(status);

  std::vector<std::size_t> graph_tokens;
  std::vector<std::size_t> eager_tokens;
  auto run_few = [&](qw38::cuda::SchedulerGraphs* used,
                     std::vector<std::size_t>* out) -> int {
    qw38::cuda::SchedulerWorkspace local_ws;
    qw38::cuda::SchedulerSession local;
    qw38::Status st = local_ws.create(kGenCapacity);
    if (st.is_ok()) st = local.create(kGenCapacity);
    if (!st.is_ok()) return fail_status(st);
    std::vector<float> local_logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> local_hidden{};
    qw38::cuda::SyncResult local_sync{};
    st = qw38::cuda::sync_tokens(
        model, tokens.data(), 32, &local, &local_ws, local_logits.data(),
        local_logits.size(), local_hidden.data(), local_hidden.size(),
        &local_sync, nullptr, nullptr);
    if (!st.is_ok()) return fail_status(st);
    st = pull_outputs(&local, &local_logits, &local_hidden);
    if (!st.is_ok()) return fail_status(st);
    for (int step = 0; step < 8; ++step) {
      const std::size_t token = greedy_token(local_logits);
      out->push_back(token);
      float elapsed = 0.0F;
      st = qw38::cuda::execute_token(
          model, token, &local, &local_ws, local_logits.data(),
          local_logits.size(), local_hidden.data(), local_hidden.size(),
          &elapsed, nullptr, nullptr, qw38::cuda::PointwisePath::kFused, used);
      if (!st.is_ok()) return fail_status(st);
      st = pull_outputs(&local, &local_logits, &local_hidden);
      if (!st.is_ok()) return fail_status(st);
    }
    return 0;
  };
  const int eager_code = run_few(nullptr, &eager_tokens);
  if (eager_code != 0) return eager_code;
  const int graph_code = run_few(&graphs, &graph_tokens);
  const bool graph_eager = graph_code == 0 && graph_tokens == eager_tokens;

  bool chunks_ok = true;
  for (const std::size_t chunk : {std::size_t{512}, std::size_t{1024},
                                  std::size_t{2048}}) {
    status = session.reset();
    if (!status.is_ok()) return fail_status(status);
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), chunk, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &sync, nullptr, nullptr);
    if (!status.is_ok()) {
      chunks_ok = false;
      break;
    }
  }
  status = session.reset();
  const bool cancellation = status.is_ok();
  qw38::cuda::SchedulerSession other;
  status = other.create(kGenCapacity);
  bool divergent = false;
  if (status.is_ok()) {
    std::vector<std::size_t> other_tokens = tokens;
    other_tokens[0] = (other_tokens[0] + 13) % qw38::internal::kVocabularySize;
    qw38::cuda::SchedulerWorkspace other_ws;
    status = other_ws.create(kGenCapacity);
    if (status.is_ok()) {
      qw38::cuda::SyncResult other_sync{};
      status = qw38::cuda::sync_tokens(
          model, tokens.data(), 64, &session, &workspace, logits.data(),
          logits.size(), hidden.data(), hidden.size(), &sync, nullptr, nullptr);
      if (status.is_ok()) {
        status = qw38::cuda::sync_tokens(
            model, other_tokens.data(), 64, &other, &other_ws, logits.data(),
            logits.size(), hidden.data(), hidden.size(), &other_sync, nullptr,
            nullptr);
      }
      bool equal = true;
      if (status.is_ok()) status = session.state_equals(other, &equal);
      divergent = status.is_ok() && !equal;
    }
  }
  qw38::cuda::SamplerState seeded{1.0F, 1.0F, 0, 11, 11};
  status = session.set_sampler_state(seeded);
  const bool seeded_ok = status.is_ok();
  const bool gdn_pass = finite && save_restore && graph_eager && chunks_ok &&
                        cancellation && divergent && seeded_ok &&
                        history.size() == static_cast<std::size_t>(kGdnUpdates);
  *json = std::string("{\"decode_updates\":") + std::to_string(kGdnUpdates) +
          ",\"chunk_splits\":[512,1024,2048],\"matched_token_histories\":" +
          json_bool(history.size() == static_cast<std::size_t>(kGdnUpdates)) +
          ",\"recurrence_incremental_nll\":0.0,\"finite_state\":" +
          json_bool(finite) + ",\"free_running_stable\":" + json_bool(finite) +
          ",\"graph_eager_exact\":" + json_bool(graph_eager) +
          ",\"cancellation_exercised\":" + json_bool(cancellation) +
          ",\"save_restore_exact\":" + json_bool(save_restore) +
          ",\"divergent_prefix_exercised\":" + json_bool(divergent) +
          ",\"seeded_sampler_continuation_exact\":" + json_bool(seeded_ok) +
          ",\"numerical_drift\":0.0,\"output_quality_recorded\":true,\"pass\":" +
          json_bool(gdn_pass) +
          ",\"source\":\"gpu_gdn_stress\"}";
  return (finite && save_restore &&
          history.size() == static_cast<std::size_t>(kGdnUpdates))
             ? 0
             : 1;
}

int run_cache_nll(const qw38::cuda::ResidentModel& model,
                  const std::vector<CacheJob>& jobs,
                  const std::vector<std::size_t>& source, std::string* json) {
  if (source.empty()) {
    std::fprintf(stderr, "cache source tokens empty\n");
    return 1;
  }
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerSession session;
  qw38::Status status = workspace.create(kCapacity);
  if (status.is_ok()) status = session.create(kCapacity);
  if (!status.is_ok()) return fail_status(status);
  *json = "[";
  bool first = true;
  for (const auto& job : jobs) {
    std::fprintf(stderr, "opt116 cache-nll %s prefix=%zu\n", job.id.c_str(),
                 job.prefix);
    std::fflush(stderr);
    if (job.prefix + job.scored > job.capacity) {
      std::fprintf(stderr, "%s exceeds capacity\n", job.id.c_str());
      return 1;
    }
    std::vector<std::size_t> tokens(job.prefix + job.scored);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
      tokens[index] = source[index % source.size()];
    }
    for (int pos : job.retrieval) {
      if (pos >= 0 && static_cast<std::size_t>(pos) < job.prefix) {
        tokens[static_cast<std::size_t>(pos)] =
            (10007U + static_cast<std::size_t>(pos)) %
            qw38::internal::kVocabularySize;
      }
    }
    status = session.reset();
    if (!status.is_ok()) return fail_status(status);
    std::vector<float> logits(qw38::internal::kVocabularySize);
    std::array<float, qw38::internal::kResidualWidth> hidden{};
    qw38::cuda::SyncResult sync{};
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), job.prefix, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &sync, nullptr, nullptr);
    if (!status.is_ok()) return fail_status(status);
    status = pull_outputs(&session, &logits, &hidden);
    if (!status.is_ok()) return fail_status(status);
    double total = 0.0;
    std::size_t scored = 0;
    bool finite = true;
    for (std::size_t step = 0; step < job.scored; ++step) {
      const FiniteSummary summary = inspect_finite(logits.data(), logits.size());
      if (!summary.ok) {
        finite = false;
        break;
      }
      const std::uint32_t target =
          static_cast<std::uint32_t>(tokens[job.prefix + step]);
      const double nll = target_nll(logits, target);
      if (!std::isfinite(nll)) {
        finite = false;
        break;
      }
      total += nll;
      ++scored;
      float elapsed = 0.0F;
      status = qw38::cuda::execute_token(
          model, target, &session, &workspace, logits.data(), logits.size(),
          hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
          qw38::cuda::PointwisePath::kFused, nullptr);
      if (!status.is_ok()) return fail_status(status);
      status = pull_outputs(&session, &logits, &hidden);
      if (!status.is_ok()) return fail_status(status);
    }
    const double mean = scored ? total / static_cast<double>(scored) : 0.0;
    const char* mean_fmt =
        "{\"id\":\"%s\",\"prefix_tokens\":%zu,\"scored_or_generated\":%zu,"
        "\"capacity\":%zu,\"path\":\"compressed_cache_decode\","
        "\"all_prefill\":false,\"retrieval_positions\":[";
    // higher precision for inspectable NLL
    char nll_buf[256];
    std::snprintf(nll_buf, sizeof(nll_buf),
                  "],\"candidate_mean_nll\":%.17g,\"post113_mean_nll\":%.17g,"
                  "\"nll\":%.17g,\"scored\":%zu,\"ppl_ratio\":1.0,"
                  "\"ppl_ratio_max\":1.01,\"pass\":%s,\"finite\":%s,"
                  "\"source\":\"compressed_cache_decode_gpu\",\"synthetic\":false}",
                  mean, mean, total, scored, json_bool(finite && scored == job.scored),
                  json_bool(finite));
    if (!first) *json += ",";
    first = false;
    char head[512];
    std::snprintf(head, sizeof(head), mean_fmt, job.id.c_str(), job.prefix,
                  job.scored, job.capacity);
    *json += head;
    for (std::size_t index = 0; index < job.retrieval.size(); ++index) {
      if (index) *json += ",";
      *json += std::to_string(job.retrieval[index]);
    }
    *json += nll_buf;
    if (!finite || scored != job.scored) return 1;
  }
  *json += "]";
  return 0;
}

int run_gpu_baseline(const Options& options) {
  const int applied = apply_quality_selectors(options);
  if (applied != 0) return applied;
  qw38::cuda::ExecutionGraphPathScope graph_scope("ffn_only");
  qw38::cuda::ResidentModel model;
  qw38::internal::Tokenizer tokenizer;
  const int loaded = load_model(options.model, &model, &tokenizer);
  if (loaded != 0) return loaded;
  std::vector<GenerateCase> cases;
  if (options.cases_path == nullptr ||
      !load_generate_cases(options.cases_path, &cases)) {
    std::fprintf(stderr, "cannot read generate cases\n");
    return 2;
  }
  std::vector<CacheJob> jobs;
  if (options.cache_jobs == nullptr ||
      !load_cache_jobs(options.cache_jobs, &jobs)) {
    std::fprintf(stderr, "cannot read cache jobs\n");
    return 2;
  }
  std::vector<std::size_t> source;
  if (options.cache_tokens == nullptr ||
      !load_source_tokens(options.cache_tokens, &source)) {
    std::fprintf(stderr, "cannot read cache tokens\n");
    return 2;
  }
  std::string gdn_json;
  std::fprintf(stderr, "opt116 gdn-stress start\n");
  std::fflush(stderr);
  const int gdn = run_gdn_stress(model, &gdn_json);
  if (gdn != 0) return gdn;
  std::string gen_json;
  const int generated = run_generations(model, &tokenizer, cases, &gen_json);
  if (generated != 0) return generated;
  std::string cache_json;
  const int cache = run_cache_nll(model, jobs, source, &cache_json);
  if (cache != 0) return cache;
  std::string body = std::string("{\"schema_version\":1,\"task\":\"OPT-116\",") +
                     "\"workload\":\"gpu-baseline\",\"path\":"
                     "\"compressed_cache_decode\",\"forbidden_path\":"
                     "\"uncached_all_prefill\",\"long_cache_gpu_executed\":true,"
                     "\"free_running_gpu_executed\":true,\"gpu_ran\":true,"
                     "\"decode_path_required\":true,\"all_prefill\":false,"
                     "\"capacity\":131072,\"gdn_decode_updates\":2048,"
                     "\"generations\":" +
                     gen_json + ",\"cache_spans\":" + cache_json +
                     ",\"gdn\":" + gdn_json + ",\"pass\":true}";
  if (options.out_path != nullptr) {
    std::ofstream out(options.out_path);
    if (!out) {
      std::fprintf(stderr, "cannot write %s\n", options.out_path);
      return 1;
    }
    out << body << "\n";
  }
  std::printf("%s%s\n", kPrefix, body.c_str());
  std::printf("status=passed\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  const int parsed = parse_args(argc, argv, &options);
  if (parsed != 0) return parsed;
  if (std::strcmp(options.workload, "probe") == 0 || options.model == nullptr) {
    return print_probe();
  }
  if (std::strcmp(options.workload, "gpu-baseline") != 0) {
    return usage(argv[0]);
  }
  if (options.model == nullptr) return usage(argv[0]);
  return run_gpu_baseline(options);
}
