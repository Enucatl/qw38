#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qw38/engine.h"
#include "server_json.h"
#include "sha256.h"

#ifdef QW38_CUDA_RUNTIME
#include <cuda_runtime_api.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using Json = qw38::server::Json;

constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr const char* kSchema = "qw38.benchmark-result.v1";
constexpr const char* kModelId = "qwen3.8-27b-q4_k_m";
constexpr const char* kModelSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kContainer =
    "qw38-cuda:13.0.2 / nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:"
    "0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c";
constexpr const char* kBuildFlags =
    "-std=c++17 -O2 -arch=sm_120 --fmad=false -fno-exceptions -fno-rtti "
    "-ffp-contract=off";
constexpr std::size_t kCapacity = 131072;

enum class Workload { kPrefill, kDecode };
enum class CachePolicy { kDisabled, kAgentReuse };

struct Options final {
  std::string model;
  std::string prompt;
  std::string prompt_file;
  std::string output;
  std::string context_label;
  std::string source_revision;
  std::string source_state = "unknown";
  Workload workload = Workload::kPrefill;
  CachePolicy cache_policy = CachePolicy::kDisabled;
  std::size_t expected_prompt_tokens = 0;
  std::size_t output_tokens = 256;
  std::size_t warmups = 3;
  std::size_t samples = 30;
  bool workload_set = false;
  bool smoke = false;
};

struct Telemetry final {
  std::string uuid;
  std::string name;
  std::string driver;
  std::string pstate;
  double temperature_c = 0.0;
  double power_w = 0.0;
  double sm_clock_mhz = 0.0;
  double memory_clock_mhz = 0.0;
  double memory_used_mib = 0.0;
  double memory_total_mib = 0.0;
  std::size_t host_rss_kib = 0;
};

qw38::Status invalid(const std::string& message) noexcept {
  return {qw38::StatusCode::kInvalidArgument, message};
}

qw38::Status io_error(const std::string& action) noexcept {
  return {qw38::StatusCode::kIoError, action + ": " + std::strerror(errno)};
}

void usage(std::ostream& output) {
  output
      << "usage: qw38-bench MODEL --workload prefill|decode --output FILE "
         "(--prompt TEXT|--prompt-file FILE) [options]\n"
      << "  --expected-prompt-tokens N  require this rendered token count\n"
      << "  --context-label TEXT        workload label, for example 8K\n"
      << "  --source-revision TEXT      tested git commit (required for release)\n"
      << "  --source-state clean|dirty  tested tree state (clean for release)\n"
      << "  --output-tokens N           decode steps per run (default 256)\n"
      << "  --warmups N                 retained preparation runs (minimum 3)\n"
      << "  --samples N                 measured runs (minimum 30)\n"
      << "  --cache-policy disabled|agent-reuse (default disabled)\n"
      << "  --smoke                     allow smaller non-admission counts\n"
      << "  --self-test                 test statistics without a model\n";
}

bool parse_size(const char* text, std::size_t* value) noexcept {
  if (text == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return static_cast<unsigned long long>(*value) == parsed;
}

qw38::Status parse_options(int argc, char** argv, Options* options) noexcept {
  if (argc < 2 || options == nullptr) return invalid("model and options are required");
  options->model = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--smoke") {
      options->smoke = true;
      continue;
    }
    if (argument == "--help") {
      usage(std::cout);
      std::exit(0);
    }
    if (index + 1 >= argc) return invalid("missing value for " + argument);
    const char* value = argv[++index];
    if (argument == "--workload") {
      options->workload_set = true;
      if (std::string(value) == "prefill") options->workload = Workload::kPrefill;
      else if (std::string(value) == "decode") options->workload = Workload::kDecode;
      else return invalid("workload must be prefill or decode");
    } else if (argument == "--output") options->output = value;
    else if (argument == "--prompt") options->prompt = value;
    else if (argument == "--prompt-file") options->prompt_file = value;
    else if (argument == "--context-label") options->context_label = value;
    else if (argument == "--source-revision") options->source_revision = value;
    else if (argument == "--source-state") {
      options->source_state = value;
      if (options->source_state != "clean" && options->source_state != "dirty") {
        return invalid("source state must be clean or dirty");
      }
    }
    else if (argument == "--cache-policy") {
      if (std::string(value) == "disabled") {
        options->cache_policy = CachePolicy::kDisabled;
      } else if (std::string(value) == "agent-reuse") {
        options->cache_policy = CachePolicy::kAgentReuse;
      } else return invalid("cache policy must be disabled or agent-reuse");
    } else if (argument == "--expected-prompt-tokens") {
      if (!parse_size(value, &options->expected_prompt_tokens)) {
        return invalid("expected prompt token count must be unsigned");
      }
    } else if (argument == "--output-tokens") {
      if (!parse_size(value, &options->output_tokens) ||
          options->output_tokens == 0) {
        return invalid("output token count must be positive");
      }
    } else if (argument == "--warmups") {
      if (!parse_size(value, &options->warmups)) {
        return invalid("warmup count must be unsigned");
      }
    } else if (argument == "--samples") {
      if (!parse_size(value, &options->samples) || options->samples == 0) {
        return invalid("sample count must be positive");
      }
    } else return invalid("unknown benchmark option: " + argument);
  }
  if (!options->workload_set || options->output.empty() ||
      (options->prompt.empty() == options->prompt_file.empty())) {
    return invalid("workload, output, and exactly one prompt source are required");
  }
  if (!options->smoke &&
      (options->warmups < 3 || options->samples < 30 ||
       options->expected_prompt_tokens == 0 || options->source_revision.empty() ||
       options->source_state != "clean" || options->context_label.empty())) {
    return invalid("release runs require >=3 warmups, >=30 samples, an expected prompt token count, a context label, a source revision, and source-state=clean; use --smoke otherwise");
  }
  return qw38::Status::ok();
}

double milliseconds(Clock::time_point begin, Clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

Json number(double value) {
  if (!std::isfinite(value)) return Json::null();
  char text[64];
  std::snprintf(text, sizeof(text), "%.9g", value);
  return Json::number(text);
}

std::string utc_now() {
  const std::time_t now = std::time(nullptr);
  std::tm value{};
  if (gmtime_r(&now, &value) == nullptr) return {};
  char output[32];
  std::strftime(output, sizeof(output), "%Y-%m-%dT%H:%M:%SZ", &value);
  return output;
}

Json size_number(std::size_t value) { return Json::number(std::to_string(value)); }

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return NAN;
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

std::string trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::string command_output(const char* command) {
  FILE* pipe = popen(command, "r");
  if (pipe == nullptr) return {};
  std::string output;
  char buffer[1024];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
  if (pclose(pipe) != 0) return {};
  return trim(output);
}

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> fields;
  std::size_t offset = 0;
  while (offset <= text.size()) {
    const std::size_t comma = text.find(',', offset);
    fields.push_back(trim(text.substr(
        offset, comma == std::string::npos ? std::string::npos : comma - offset)));
    if (comma == std::string::npos) break;
    offset = comma + 1;
  }
  return fields;
}

bool parse_double(const std::string& text, double* value) noexcept {
  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == nullptr || *end != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  *value = parsed;
  return true;
}

std::size_t resident_kib() {
  std::ifstream input("/proc/self/status");
  std::string key;
  while (input >> key) {
    if (key == "VmRSS:") {
      std::size_t value = 0;
      input >> value;
      return value;
    }
    std::string remainder;
    std::getline(input, remainder);
  }
  return 0;
}

qw38::Status read_telemetry(Telemetry* telemetry) noexcept {
  const std::string raw = command_output(
      "nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,"
      "power.draw,clocks.sm,clocks.mem,memory.used,memory.total "
      "--format=csv,noheader,nounits 2>/dev/null");
  const std::vector<std::string> fields = split_csv(raw);
  if (fields.size() != 10) {
    return {qw38::StatusCode::kIoError,
            "nvidia-smi did not return one ten-field GPU row"};
  }
  telemetry->uuid = fields[0];
  telemetry->name = fields[1];
  telemetry->driver = fields[2];
  telemetry->pstate = fields[3];
  if (!parse_double(fields[4], &telemetry->temperature_c) ||
      !parse_double(fields[5], &telemetry->power_w) ||
      !parse_double(fields[6], &telemetry->sm_clock_mhz) ||
      !parse_double(fields[7], &telemetry->memory_clock_mhz) ||
      !parse_double(fields[8], &telemetry->memory_used_mib) ||
      !parse_double(fields[9], &telemetry->memory_total_mib)) {
    return {qw38::StatusCode::kIoError,
            "nvidia-smi returned a non-numeric telemetry field"};
  }
  telemetry->host_rss_kib = resident_kib();
  return qw38::Status::ok();
}

Json telemetry_json(const Telemetry& value) {
  return Json::object_value(
      {{"driver_version", Json::string(value.driver)},
       {"gpu_name", Json::string(value.name)},
       {"gpu_uuid", Json::string(value.uuid)},
       {"host_rss_kib", size_number(value.host_rss_kib)},
       {"memory_clock_mhz", number(value.memory_clock_mhz)},
       {"memory_total_mib", number(value.memory_total_mib)},
       {"memory_used_mib", number(value.memory_used_mib)},
       {"power_w", number(value.power_w)},
       {"pstate", Json::string(value.pstate)},
       {"sm_clock_mhz", number(value.sm_clock_mhz)},
       {"temperature_c", number(value.temperature_c)}});
}

Json timing_value(const qw38::TimingValue& value) {
  return value.measured ? number(value.milliseconds) : Json::null();
}

Json timing_json(const qw38::RuntimeTimings& value) {
  return Json::object_value(
      {{"attention_ms", timing_value(value.attention)},
       {"embedding_ms", timing_value(value.embedding)},
       {"ffn_ms", timing_value(value.ffn)},
       {"gdn_ms", timing_value(value.gdn)},
       {"graph_launch_ms", timing_value(value.graph_launch)},
       {"idle_gaps_ms", timing_value(value.idle_gaps)},
       {"loading_ms", timing_value(value.loading)},
       {"logits_ms", timing_value(value.logits)},
       {"persistence_ms", timing_value(value.persistence)},
       {"queueing_ms", timing_value(value.queueing)},
       {"sampling_ms", timing_value(value.sampling)},
       {"state_commit_ms", timing_value(value.state_commit)},
       {"token_total_ms", timing_value(value.token_total)}});
}

qw38::Status read_prompt(const Options& options, std::string* prompt) noexcept {
  if (!options.prompt_file.empty()) {
    std::ifstream input(options.prompt_file, std::ios::binary);
    if (!input) return io_error("cannot open benchmark prompt");
    prompt->assign(std::istreambuf_iterator<char>(input),
                   std::istreambuf_iterator<char>());
    if (!input.eof()) return io_error("cannot read benchmark prompt");
  } else {
    *prompt = options.prompt;
  }
  return prompt->empty() ? invalid("benchmark prompt cannot be empty")
                         : qw38::Status::ok();
}

qw38::Status write_all(int descriptor, const std::string& bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return io_error("cannot write benchmark result");
    offset += static_cast<std::size_t>(count);
  }
  return qw38::Status::ok();
}

qw38::Status publish(const std::string& path, const Json& root) noexcept {
  const std::string temporary = path + ".tmp-" + std::to_string(getpid());
  const int descriptor =
      open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) return io_error("cannot create benchmark result");
  const std::string bytes = qw38::server::dump_json(root) + "\n";
  qw38::Status status = write_all(descriptor, bytes);
  if (status.is_ok() && fsync(descriptor) != 0) {
    status = io_error("cannot sync benchmark result");
  }
  if (close(descriptor) != 0 && status.is_ok()) {
    status = io_error("cannot close benchmark result");
  }
  if (!status.is_ok()) {
    unlink(temporary.c_str());
    return status;
  }
  if (rename(temporary.c_str(), path.c_str()) != 0) {
    const qw38::Status failure = io_error("cannot publish benchmark result");
    unlink(temporary.c_str());
    return failure;
  }
  const std::size_t slash = path.find_last_of('/');
  const std::string parent = slash == std::string::npos ? "." : path.substr(0, slash);
  const int directory = open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) return io_error("cannot open benchmark result directory");
  if (fsync(directory) != 0) {
    const qw38::Status failure = io_error("cannot sync result directory");
    close(directory);
    return failure;
  }
  if (close(directory) != 0) return io_error("cannot close result directory");
  return qw38::Status::ok();
}

Json status_json(const qw38::Status& status) {
  return Json::object_value(
      {{"code", Json::string(qw38::status_code_name(status.code()))},
       {"message", Json::string(status.message())}});
}

Json base_result(const Options& options) {
  return Json::object_value(
      {{"cache_policy",
        Json::string(options.cache_policy == CachePolicy::kDisabled
                         ? "disabled"
                         : "agent-reuse")},
       {"context_label", Json::string(options.context_label)},
       {"model_id", Json::string(kModelId)},
       {"model_sha256", Json::string(kModelSha)},
       {"schema", Json::string(kSchema)},
       {"started_at_utc", Json::string(utc_now())},
       {"status", Json::string("running")},
       {"workload", Json::string(options.workload == Workload::kPrefill
                                      ? "prefill"
                                      : "decode")}});
}

qw38::Status run_sample(const qw38::Engine& engine, qw38::Session* session,
                        const Options& options,
                        const std::vector<qw38::Token>& initial_prompt,
                        qw38::Token end_token,
                        std::vector<qw38::Token>* reuse_history,
                        Json* sample) noexcept {
  const bool reuse = options.cache_policy == CachePolicy::kAgentReuse &&
                     !reuse_history->empty();
  const std::size_t reused_tokens = reuse ? reuse_history->size() : 0;
  std::vector<qw38::Token> prompt = initial_prompt;
  if (reuse) {
    std::vector<qw38::Token> suffix;
    qw38::Status status =
        engine.render_user_turn("Continue the benchmark.", false, &suffix);
    if (!status.is_ok()) return status;
    prompt = *reuse_history;
    prompt.insert(prompt.end(), suffix.begin(), suffix.end());
  } else {
    const qw38::Status reset = session->sync({});
    if (!reset.is_ok()) return reset;
  }
  const std::size_t required =
      options.workload == Workload::kDecode ? options.output_tokens + 1 : 1;
  if (prompt.size() + required > kCapacity) {
    return {qw38::StatusCode::kResourceExhausted,
            "benchmark prompt plus output exceeds 131072 tokens"};
  }
  Telemetry before;
  qw38::Status status = read_telemetry(&before);
  const auto prefill_start = Clock::now();
  if (status.is_ok()) status = session->sync(prompt);
  const auto prefill_end = Clock::now();
  const double prefill_ms = milliseconds(prefill_start, prefill_end);
  std::vector<Json> token_records;
  std::vector<Json> generated_json;
  std::vector<double> itl;
  double ttft_ms = 0.0;
  double decode_ms = 0.0;
  if (status.is_ok() && options.workload == Workload::kDecode) {
    const auto decode_start = Clock::now();
    Clock::time_point previous_visible;
    for (std::size_t index = 0; status.is_ok() && index < options.output_tokens;
         ++index) {
      const auto token_started = Clock::now();
      qw38::Token token = 0;
      status = session->sample({0.0F, 1.0F, 0, 0}, &token);
      if (status.is_ok()) status = session->eval(token);
      const auto visible = Clock::now();
      if (!status.is_ok()) break;
      generated_json.push_back(size_number(token));
      token_records.push_back(Json::object_value(
          {{"index", size_number(index)},
           {"token", size_number(token)},
           {"wall_ms", number(milliseconds(token_started, visible))}}));
      if (index == 0) {
        ttft_ms = prefill_ms + milliseconds(decode_start, visible);
      } else {
        itl.push_back(milliseconds(previous_visible, visible));
      }
      previous_visible = visible;
    }
    decode_ms = milliseconds(decode_start, Clock::now());
  }
  if (status.is_ok()) status = session->eval(end_token);
  if (status.is_ok() && options.cache_policy == CachePolicy::kAgentReuse) {
    status = session->tokens(reuse_history);
  }
  Telemetry after;
  if (status.is_ok()) status = read_telemetry(&after);
  std::vector<Json> itl_json;
  for (double value : itl) itl_json.push_back(number(value));
  std::map<std::string, Json> fields;
  const std::size_t evaluated_tokens = prompt.size() - reused_tokens;
  fields.emplace("evaluated_prompt_tokens", size_number(evaluated_tokens));
  fields.emplace("input_tokens", size_number(prompt.size()));
  fields.emplace("prefill_ms", number(prefill_ms));
  fields.emplace("prompt_tokens_per_second",
                 number(static_cast<double>(evaluated_tokens) * 1000.0 /
                        prefill_ms));
  fields.emplace("reused_prefix_tokens", size_number(reused_tokens));
  fields.emplace("telemetry_before", telemetry_json(before));
  fields.emplace("telemetry_after", telemetry_json(after));
  fields.emplace("status", Json::string(status.is_ok() ? "success" : "failed"));
  if (!status.is_ok()) fields.emplace("error", status_json(status));
  if (options.workload == Workload::kDecode) {
    fields.emplace("decode_ms", number(decode_ms));
    fields.emplace("generated_tokens", Json::array_value(std::move(generated_json)));
    fields.emplace("itl_ms", Json::array_value(std::move(itl_json)));
    fields.emplace("output_tokens_per_second",
                   number(static_cast<double>(options.output_tokens) * 1000.0 /
                          decode_ms));
    fields.emplace("per_token", Json::array_value(std::move(token_records)));
    fields.emplace("ttft_ms", number(ttft_ms));
  }
  *sample = Json::object_value(std::move(fields));
  return status;
}

qw38::Status component_probe(qw38::Session* session,
                             const std::vector<qw38::Token>& prompt,
                             Json* output) noexcept {
  qw38::Status status = session->sync({});
  if (status.is_ok()) status = session->sync(prompt);
  qw38::Token token = 0;
  qw38::RuntimeTimings sampling;
  qw38::RuntimeTimings evaluation;
  if (status.is_ok()) {
    status = session->sample({0.0F, 1.0F, 0, 0}, &token, &sampling);
  }
  if (status.is_ok()) status = session->eval(token, &evaluation);
  if (!status.is_ok()) return status;
  std::map<std::string, Json> values = timing_json(evaluation).object;
  values["sampling_ms"] = timing_value(sampling.sampling);
  values["token"] = size_number(token);
  values["perturbs_execution"] = Json::boolean_value(true);
  values["used_for_throughput_summary"] = Json::boolean_value(false);
  *output = Json::object_value(std::move(values));
  return qw38::Status::ok();
}

Json summary_json(const std::vector<Json>& samples, Workload workload) {
  std::vector<double> prefills;
  std::vector<double> prompt_rates;
  std::vector<double> ttfts;
  std::vector<double> output_rates;
  std::vector<double> itls;
  double peak_vram = 0.0;
  std::size_t peak_rss = 0;
  for (const Json& sample : samples) {
    const auto numeric = [&sample](const char* name) {
      return std::strtod(sample.find(name)->text.c_str(), nullptr);
    };
    prefills.push_back(numeric("prefill_ms"));
    prompt_rates.push_back(numeric("prompt_tokens_per_second"));
    for (const char* telemetry_name : {"telemetry_before", "telemetry_after"}) {
      const Json* telemetry = sample.find(telemetry_name);
      peak_vram = std::max(
          peak_vram,
          std::strtod(telemetry->find("memory_used_mib")->text.c_str(), nullptr));
      peak_rss = std::max(
          peak_rss, static_cast<std::size_t>(std::strtoull(
                        telemetry->find("host_rss_kib")->text.c_str(), nullptr, 10)));
    }
    if (workload == Workload::kDecode) {
      ttfts.push_back(numeric("ttft_ms"));
      output_rates.push_back(numeric("output_tokens_per_second"));
      for (const Json& value : sample.find("itl_ms")->array) {
        itls.push_back(std::strtod(value.text.c_str(), nullptr));
      }
    }
  }
  std::map<std::string, Json> result;
  result.emplace("peak_device_memory_mib", number(peak_vram));
  result.emplace("peak_host_rss_kib", size_number(peak_rss));
  result.emplace("prefill_ms_p50", number(percentile(prefills, 0.50)));
  result.emplace("prefill_ms_p95", number(percentile(prefills, 0.95)));
  result.emplace("prompt_tokens_per_second_p50",
                 number(percentile(prompt_rates, 0.50)));
  result.emplace("prompt_tokens_per_second_p95",
                 number(percentile(prompt_rates, 0.95)));
  result.emplace("queue_time_ms", Json::null());
  result.emplace("reserved_device_memory_mib", number(peak_vram));
  if (workload == Workload::kDecode) {
    result.emplace("itl_ms_p50", number(percentile(itls, 0.50)));
    result.emplace("itl_ms_p95", number(percentile(itls, 0.95)));
    result.emplace("output_tokens_per_second_p50",
                   number(percentile(output_rates, 0.50)));
    result.emplace("output_tokens_per_second_p95",
                   number(percentile(output_rates, 0.95)));
    result.emplace("ttft_ms_p50", number(percentile(ttfts, 0.50)));
    result.emplace("ttft_ms_p95", number(percentile(ttfts, 0.95)));
  }
  return Json::object_value(std::move(result));
}

int self_test() {
  const std::vector<double> values{5, 1, 4, 2, 3};
  Json result = Json::object_value(
      {{"p50", number(percentile(values, 0.50))},
       {"p95", number(percentile(values, 0.95))},
       {"schema", Json::string(kSchema)},
       {"status", Json::string("passed")}});
  std::cout << qw38::server::dump_json(result) << '\n';
  return percentile(values, 0.50) == 3.0 &&
                 std::fabs(percentile(values, 0.95) - 4.8) < 1.0e-12
             ? 0
             : 1;
}

int report(const qw38::Status& status) {
  std::cerr << qw38::status_code_name(status.code()) << ": "
            << status.message() << '\n';
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << kBrand << '\n';
    usage(std::cerr);
    return 2;
  }
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage(std::cout);
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
  Options options;
  qw38::Status status = parse_options(argc, argv, &options);
  if (!status.is_ok()) {
    usage(std::cerr);
    return report(status);
  }
  std::cout << kBrand << '\n';

  std::string prompt_text;
  status = read_prompt(options, &prompt_text);
  Json result = base_result(options);
  const auto load_started = Clock::now();
  qw38::Engine engine;
  if (status.is_ok()) status = qw38::Engine::open(options.model, &engine);
  const auto engine_loaded = Clock::now();
  std::unique_ptr<qw38::Session> session;
  if (status.is_ok()) status = engine.create_session(&session);
  const auto session_created = Clock::now();
  std::vector<qw38::Token> prompt_tokens;
  if (status.is_ok()) {
    qw38::ChatOptions chat;
    chat.enable_thinking = false;
    status = engine.render_chat(
        {{qw38::ChatRole::kUser, prompt_text, std::string(), {}}}, chat,
        &prompt_tokens);
  }
  if (status.is_ok() && options.expected_prompt_tokens != 0 &&
      prompt_tokens.size() != options.expected_prompt_tokens) {
    status = invalid("rendered prompt token count differs from --expected-prompt-tokens");
  }
  std::vector<qw38::Token> end_tokens;
  if (status.is_ok()) status = engine.encode("<|im_end|>", &end_tokens);
  if (status.is_ok() && end_tokens.size() != 1) {
    status = {qw38::StatusCode::kIncompatibleArtifact,
              "chat terminator is not one token"};
  }
  std::string prompt_sha;
  if (status.is_ok()) {
    status = qw38::internal::sha256_bytes(
        reinterpret_cast<const unsigned char*>(prompt_text.data()),
        prompt_text.size(), &prompt_sha);
  }
  Telemetry environment_telemetry;
  if (status.is_ok()) status = read_telemetry(&environment_telemetry);
  std::vector<Json> token_json;
  for (qw38::Token token : prompt_tokens) token_json.push_back(size_number(token));
  result.object["admission_eligible"] = Json::boolean_value(false);
  result.object["configuration"] = Json::object_value(
      {{"expected_prompt_tokens", size_number(options.expected_prompt_tokens)},
       {"output_tokens", size_number(options.output_tokens)},
       {"samples", size_number(options.samples)},
       {"smoke", Json::boolean_value(options.smoke)},
       {"warmups", size_number(options.warmups)}});
  result.object["environment"] = Json::object_value(
      {{"build_flags", Json::string(kBuildFlags)},
       {"container", Json::string(kContainer)},
       {"cuda_driver_version", Json::null()},
       {"cuda_runtime_version", Json::null()},
       {"source_revision", Json::string(options.source_revision)},
       {"source_state", Json::string(options.source_state)},
       {"host", Json::string(command_output("uname -a"))},
       {"telemetry", telemetry_json(environment_telemetry)}});
#ifdef QW38_CUDA_RUNTIME
  int runtime_version = 0;
  int driver_version = 0;
  if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess &&
      cudaDriverGetVersion(&driver_version) == cudaSuccess) {
    result.object["environment"].object["cuda_runtime_version"] =
        size_number(static_cast<std::size_t>(runtime_version));
    result.object["environment"].object["cuda_driver_version"] =
        size_number(static_cast<std::size_t>(driver_version));
  } else if (status.is_ok()) {
    status = {qw38::StatusCode::kInternal,
              "cannot query CUDA runtime and driver versions"};
  }
#endif
  result.object["load"] = Json::object_value(
      {{"engine_ms", number(milliseconds(load_started, engine_loaded))},
       {"session_ms", number(milliseconds(engine_loaded, session_created))}});
  result.object["prompt"] = Json::object_value(
      {{"bytes", size_number(prompt_text.size())},
       {"sha256", Json::string(prompt_sha)},
       {"token_count", size_number(prompt_tokens.size())},
       {"token_ids", Json::array_value(std::move(token_json))}});

  std::vector<Json> warmups;
  std::vector<Json> samples;
  std::vector<qw38::Token> reuse_history;
  for (std::size_t index = 0; status.is_ok() && index < options.warmups; ++index) {
    Json sample;
    status = run_sample(engine, session.get(), options, prompt_tokens,
                        end_tokens[0], &reuse_history, &sample);
    if (sample.kind != qw38::server::JsonKind::kObject) {
      sample = Json::object_value(
          {{"error", status_json(status)},
           {"status", Json::string("failed")}});
    }
    sample.object["index"] = size_number(index);
    warmups.push_back(std::move(sample));
  }
  reuse_history.clear();
  if (status.is_ok()) status = session->sync({});
  for (std::size_t index = 0; status.is_ok() && index < options.samples; ++index) {
    Json sample;
    status = run_sample(engine, session.get(), options, prompt_tokens,
                        end_tokens[0], &reuse_history, &sample);
    if (sample.kind != qw38::server::JsonKind::kObject) {
      sample = Json::object_value(
          {{"error", status_json(status)},
           {"status", Json::string("failed")}});
    }
    sample.object["index"] = size_number(index);
    samples.push_back(std::move(sample));
  }
  result.object["warmups"] = Json::array_value(std::move(warmups));
  result.object["samples"] = Json::array_value(samples);
  if (status.is_ok()) result.object["summary"] = summary_json(samples, options.workload);
  if (status.is_ok() && options.workload == Workload::kDecode) {
    Json components;
    status = component_probe(session.get(), prompt_tokens, &components);
    if (status.is_ok()) result.object["component_probe"] = std::move(components);
  }
  result.object["status"] = Json::string(status.is_ok() ? "success" : "failed");
  result.object["admission_eligible"] =
      Json::boolean_value(status.is_ok() && !options.smoke);
  if (!status.is_ok()) result.object["error"] = status_json(status);
  const qw38::Status publish_status = publish(options.output, result);
  if (!publish_status.is_ok()) return report(publish_status);
  std::cout << "result=" << options.output << " status="
            << (status.is_ok() ? "success" : "failed") << '\n';
  return status.is_ok() ? 0 : report(status);
}
