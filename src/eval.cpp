#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/resource.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <linux/fs.h>
#endif
#include <unistd.h>

#include "qw38/engine.h"
#include "attention.h"
#include "conversion.h"
#include "gdn.h"
#include "mixer.h"
#include "model.h"
#include "projection.h"
#include "quant.h"
#include "scheduler.h"
#include "scalar_runtime.h"
#include "sha256.h"
#include "template.h"
#include "tensor.h"
#include "tokenizer.h"
#include "weights.h"

#ifdef QW38_DIAGNOSTIC_TRACE
#include "diagnostic_trace.h"
#endif
#ifdef QW38_CUDA_RUNTIME
#include "full_scheduler.h"
#include <cuda_runtime.h>
#endif

namespace {
constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr const char* kModelRevision =
    "0669b98607d47046c7c2b3f801011d54a08cfccf";
constexpr const char* kModelSha256 =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";

int write_inventory(const char* model_path, const char* output_path) {
  std::error_code error;
  if (std::filesystem::exists(output_path, error) || error) {
    std::cerr << "inventory output already exists or cannot be inspected\n";
    return 1;
  }
  qw38::Engine engine;
  qw38::Status status = qw38::Engine::open(model_path, &engine);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  qw38::internal::ModelInfo info;
  status = qw38::internal::inspect_gguf(model_path, &info);
  qw38::internal::ModelGeometry geometry;
  if (status.is_ok()) {
    status = qw38::internal::admit_pinned_geometry(&info, &geometry);
  }
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  const std::string temporary =
      std::string(output_path) + ".tmp." + std::to_string(getpid());
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "cannot create inventory output\n";
    return 1;
  }
  std::string digest;
  status = qw38::internal::sha256_file(model_path, &digest);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  output << "{\n  \"schema_version\": 1,\n";
  output << "  \"model_sha256\": \"" << digest << "\",\n";
  output << "  \"geometry_identity\": " << geometry.identity << ",\n";
  output << "  \"gguf_version\": " << info.gguf_version << ",\n";
  output << "  \"data_offset\": " << info.data_offset << ",\n";
  output << "  \"tensor_count\": " << info.tensors.size() << ",\n";
  output << "  \"tensors\": [\n";
  for (std::size_t index = 0; index < info.tensors.size(); ++index) {
    const qw38::internal::TensorInfo& tensor = info.tensors[index];
    const std::uint64_t absolute = info.data_offset + tensor.offset;
    std::string digest;
    status = qw38::internal::sha256_bytes(mapping.data() + absolute,
                                          tensor.storage_bytes, &digest);
    if (!status.is_ok()) {
      output.close();
      std::remove(temporary.c_str());
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    output << "    {\"name\": \"" << tensor.name << "\", \"role\": \""
           << tensor.semantic_role << "\", \"shape\": [";
    for (std::size_t dimension = 0; dimension < tensor.dimensions.size();
         ++dimension) {
      if (dimension != 0) output << ", ";
      output << tensor.dimensions[dimension];
    }
    output << "], \"dtype\": \""
           << qw38::internal::ggml_type_name(tensor.type)
           << "\", \"offset\": " << tensor.offset
           << ", \"absolute_offset\": " << absolute
           << ", \"storage_bytes\": " << tensor.storage_bytes
           << ", \"padded_span_bytes\": " << tensor.padded_span_bytes
           << ", \"sha256\": \"" << digest << "\"}";
    output << (index + 1 == info.tensors.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  output.close();
  if (!output || std::rename(temporary.c_str(), output_path) != 0) {
    std::remove(temporary.c_str());
    std::cerr << "cannot commit inventory output\n";
    return 1;
  }
  std::cout << "inventory=" << output_path << '\n';
  return 0;
}

int hex_value(char character) {
  if (character >= '0' && character <= '9') return character - '0';
  if (character >= 'a' && character <= 'f') return character - 'a' + 10;
  if (character >= 'A' && character <= 'F') return character - 'A' + 10;
  return -1;
}

bool parse_hex(const std::string& hex, std::vector<std::uint8_t>* bytes) {
  if (hex.size() % 2 != 0) return false;
  bytes->clear();
  bytes->reserve(hex.size() / 2);
  for (std::size_t index = 0; index < hex.size(); index += 2) {
    const int high = hex_value(hex[index]);
    const int low = hex_value(hex[index + 1]);
    if (high < 0 || low < 0) return false;
    bytes->push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return true;
}

bool load_bytes_argument(const std::string& spec, std::vector<std::uint8_t>* bytes) {
  if (!spec.empty() && spec.front() == '@') {
    std::ifstream input(spec.substr(1), std::ios::binary);
    if (!input) return false;
    bytes->assign(std::istreambuf_iterator<char>(input),
                  std::istreambuf_iterator<char>());
    return true;
  }
  return parse_hex(spec, bytes);
}

bool parse_size(const char* text, std::size_t* value) {
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (*end != '\0' || parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  *value = static_cast<std::size_t>(parsed);
  return true;
}

qw38::Status admit_gguf(const char* model_path, qw38::internal::ModelInfo* info) {
  qw38::Status status = qw38::internal::inspect_gguf(model_path, info);
  qw38::internal::ModelGeometry geometry;
  if (status.is_ok()) {
    status = qw38::internal::admit_pinned_geometry(info, &geometry);
  }
  return status;
}

qw38::Status load_admitted_weights(const char* model_path,
                                   qw38::internal::ModelInfo* info,
                                   qw38::internal::MappedFile* mapping,
                                   qw38::internal::ModelWeights* weights) {
  qw38::Status status = admit_gguf(model_path, info);
  if (status.is_ok()) status = mapping->open(model_path);
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(*info, *mapping, weights);
  }
  return status;
}

std::vector<float> tap_values(const std::vector<float>& values,
                              const std::vector<std::size_t>& indices) {
  std::vector<float> selected;
  selected.reserve(indices.size());
  for (std::size_t index : indices) selected.push_back(values[index]);
  return selected;
}

bool is_qwen38_27b(const qw38::internal::ModelGeometry& geometry) {
  return geometry.identity == qw38::internal::kGeometryQwen38_27B;
}

std::vector<std::size_t> residual_taps(
    const qw38::internal::ModelGeometry& geometry) {
  if (is_qwen38_27b(geometry)) return {0, 1, 2559, 5119};
  return {0, 1, geometry.residual_width / 2 - 1, geometry.residual_width - 1};
}

std::vector<std::size_t> residual_end_taps(
    const qw38::internal::ModelGeometry& geometry) {
  if (is_qwen38_27b(geometry)) return {0, 1, 5118, 5119};
  return {0, 1, geometry.residual_width - 2, geometry.residual_width - 1};
}

std::vector<std::size_t> ffn_taps(const qw38::internal::ModelGeometry& geometry) {
  if (is_qwen38_27b(geometry)) return {0, 1, 8703, 8704, 17407};
  return {0, 1, geometry.ffn_width / 2 - 1, geometry.ffn_width / 2,
          geometry.ffn_width - 1};
}

std::vector<std::size_t> gdn_gate_taps(
    const qw38::internal::ModelGeometry& geometry) {
  if (is_qwen38_27b(geometry)) return {0, 1, 3, 47};
  return {0, 1, 3, geometry.gdn_value_heads - 1};
}

std::vector<std::size_t> gdn_convolved_taps(
    const qw38::internal::ModelGeometry& geometry) {
  if (is_qwen38_27b(geometry)) {
    return {0, 127, 2048, 2175, 4096, 4223, 6144, 6271, 4224, 4351};
  }
  return {0, 127, 2048, 2175, 4096, 4223, geometry.gdn_packed_qkv() - 128,
          geometry.gdn_packed_qkv() - 1, 4224, 4351};
}

std::vector<std::size_t> attention_head_taps(
    const qw38::internal::ModelGeometry& geometry) {
  if (is_qwen38_27b(geometry)) {
    return {0, 63, 64, 255, 5 * 256, 6 * 256, 23 * 256 + 255};
  }
  const std::size_t last = geometry.attention_query_width() - 1;
  return {0, 63, 64, 255, 256, 512, last};
}

std::vector<std::size_t> attention_cache_taps(
    const qw38::internal::ModelGeometry& geometry) {
  const std::size_t stride = geometry.attention_kv_width();
  if (is_qwen38_27b(geometry)) {
    return {0, 31, 32, 63, 64, 255, 256, 511, stride, stride + 31, stride + 32,
            stride + 63, stride + 64, stride + 255, stride + 256, stride + 511};
  }
  return {0, 31, 32, 63, 64, 255, 256, 511, stride, stride + 31, stride + 32,
          stride + 63, stride + 64, stride + 255, stride + 256, stride + 511};
}

bool parse_token_csv(const std::string& text, std::vector<qw38::Token>* out) {
  if (text.empty()) return false;
  out->clear();
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const std::size_t end = text.find(',', begin);
    const std::string field = text.substr(begin, end == std::string::npos
                                                   ? std::string::npos
                                                   : end - begin);
    std::size_t value = 0;
    if (field.empty() || field.find_first_not_of("0123456789") !=
                              std::string::npos || !parse_size(field.c_str(), &value) ||
        value >= qw38::internal::kVocabularySize) return false;
    out->push_back(static_cast<qw38::Token>(value));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return !out->empty();
}

int eval_usage(bool diagnostic) {
  std::cerr << (diagnostic ? "usage: qw38-eval-diagnostic" : "usage: qw38-eval")
            << " MODEL --mode logits|checkpoint --tokens CSV --output DIR"
               " --source-revision REVISION --source-state clean|dirty"
               " [--continuation CSV] [--trace-filter LAYER:TAP]\n";
  std::cerr << "qw38-eval: also --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "--verify-model PATH, --check-contract PATH, "
               "--inventory-gguf PATH OUTPUT, --tokenize-hex PATH HEX, "
               "--render-template-case NAME, --check-quant KIND HEX, "
               "--check-gdn COMPONENT CHUNKING, --check-attention LAYER, "
               "--check-conversion COMPONENT, --check-weight-binding MODEL MODE, "
               "--check-projection-layout COMPONENT, "
               "--check-mixer-projections MODEL MODE, "
               "--check-real-gdn-step MODEL MODE, --check-real-ffn-step MODEL MODE, "
               "--check-real-attention-step MODEL MODE, "
               "--check-real-model-boundaries MODEL MODE, "
               "--check-real-scalar-token MODEL MODE, "
               "--check-real-scalar-chunk MODEL MODE, "
               "--dump-real-scalar-logits MODEL OUTPUT, "
               "--measure-host-decode MODEL [PROMPT DECODE], "
               "--check-ffn LAYER, --check-matvec KIND COLUMNS ROWS PAYLOAD "
               "ACTIVATION, or --check-tensor-row MODEL NAME ROW\n";
  return 2;
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    if (character == '\\' || character == '"') escaped += '\\';
    if (character == '\n') escaped += "\\n";
    else if (character == '\r') escaped += "\\r";
    else if (character == '\t') escaped += "\\t";
    else escaped += character;
  }
  return escaped;
}

bool publish_noreplace(const std::string& temporary,
                       const std::string& destination) {
#ifdef __linux__
  const long result = syscall(SYS_renameat2, AT_FDCWD, temporary.c_str(),
                              AT_FDCWD, destination.c_str(), RENAME_NOREPLACE);
  return result == 0;
#else
  std::error_code error;
  if (std::filesystem::exists(destination, error) || error) return false;
  std::filesystem::rename(temporary, destination, error);
  return !error;
#endif
}

std::string tool_identity_json(const std::string& name,
                               const std::string& revision,
                               const std::string& source_state) {
  std::string digest;
  if (!qw38::internal::sha256_file("/proc/self/exe", &digest).is_ok()) return {};
  return "{\"name\":\"" + json_escape(name) + "\",\"revision\":\"" +
         json_escape(revision) + "\",\"source_state\":\"" + source_state +
         "\",\"sha256\":\"" + digest + "\"}";
}

std::string model_identity_json() {
  return std::string("{\"name\":\"qwen3.8-27b-q4_k_m\",\"revision\":\"") +
         kModelRevision + "\",\"sha256\":\"" + kModelSha256 +
         "\",\"byte_count\":18973870432}";
}

std::string runtime_json() {
#ifdef QW38_CUDA_RUNTIME
  int runtime = 0;
  int driver = 0;
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaRuntimeGetVersion(&runtime) != cudaSuccess ||
      cudaDriverGetVersion(&driver) != cudaSuccess ||
      cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) return {};
  return "{\"backend\":\"cuda\",\"cuda_target\":\"sm_120\",\"cuda_runtime_version\":" +
         std::to_string(runtime) + ",\"cuda_driver_version\":" +
         std::to_string(driver) + ",\"device_name\":\"" +
         json_escape(properties.name) + "\",\"compute_capability\":\"" +
         std::to_string(properties.major) + "." + std::to_string(properties.minor) + "\"}";
#else
  return "{\"backend\":\"host\",\"cuda_target\":\"none\",\"cuda_runtime_version\":0,\"cuda_driver_version\":0,\"device_name\":\"host\",\"compute_capability\":\"none\"}";
#endif
}

bool finite_logits(const std::vector<float>& values) {
  return values.size() == qw38::internal::kVocabularySize &&
         std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); });
}

std::size_t greedy_token(const std::vector<float>& values) {
  std::size_t best = 0;
  for (std::size_t index = 1; index < values.size(); ++index)
    if (values[index] > values[best]) best = index;
  return best;
}

std::string summary_json(const std::vector<float>& values) {
  if (values.empty() || !finite_logits(values)) return {};
  double squares = 0.0;
  double mean = 0.0;
  for (float value : values) { mean += value; squares += static_cast<double>(value) * value; }
  mean /= values.size();
  std::ostringstream output;
  output << std::setprecision(17);
  output << "{\"count\":" << values.size() <<
         ",\"finite_count\":" << values.size() <<
         ",\"nan_count\":0,\"positive_infinity_count\":0,\"negative_infinity_count\":0,\"minimum\":"
         << *std::min_element(values.begin(), values.end())
         << ",\"maximum\":" << *std::max_element(values.begin(), values.end())
         << ",\"mean\":" << mean
         << ",\"root_mean_square\":" << std::sqrt(squares / values.size()) << "}";
  return output.str();
}

std::string blob_record_json(const std::string& file, const std::string& digest,
                            const std::vector<float>& values) {
  return "{\"file\":\"" + file + "\",\"dtype\":\"f32-le\",\"shape\":[248320],\"byte_count\":" +
         std::to_string(values.size() * sizeof(float)) + ",\"sha256\":\"" + digest +
         "\",\"summary\":" + summary_json(values) + "}";
}

#if defined(QW38_CUDA_RUNTIME) && defined(QW38_DIAGNOSTIC_TRACE)
bool vector_digest(const std::vector<float>& values, std::string* digest);
struct CudaTraceCapture final {
  std::vector<float> values;
  std::string name;
  std::size_t layer = 0;
  std::array<std::size_t, qw38::internal::kTraceMaximumRank> shape{};
  std::size_t rank = 0;
};
qw38::Status capture_cuda_trace(const qw38::internal::TraceTensorView& view,
                                void* context) noexcept {
  auto* capture = static_cast<CudaTraceCapture*>(context);
  if (capture == nullptr || !capture->values.empty() || view.values == nullptr)
    return {qw38::StatusCode::kInvalidArgument, "trace sink received duplicate tensor"};
  capture->values.assign(view.values, view.values + view.value_count);
  capture->name = view.name;
  capture->layer = view.layer;
  capture->shape = view.shape;
  capture->rank = view.rank;
  return qw38::Status::ok();
}

int run_cuda_trace(const char* model_path, const std::vector<qw38::Token>& tokens,
                   const std::vector<std::string>& filters,
                   const std::string& output_path, const std::string& revision,
                   const std::string& source_state) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) status = qw38::internal::bind_model_weights(info, mapping, &weights);
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) status = model.upload(weights, mapping.data(), mapping.size());
  std::vector<CudaTraceCapture> captures(filters.size());
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::vector<float> reference_logits;
  std::vector<float> hidden(qw38::internal::kResidualWidth);
  float elapsed_milliseconds = 0.0F;
  for (std::size_t index = 0; status.is_ok() && index < filters.size(); ++index) {
    const std::string& text = filters[index];
    const std::size_t colon = text.find(':');
    if (colon == std::string::npos) { status = {qw38::StatusCode::kInvalidArgument, "malformed trace filter"}; break; }
    const std::string layer_text = text.substr(0, colon);
    const std::string tap = text.substr(colon + 1);
    std::size_t layer = qw38::internal::kTraceAllLayers;
    if (layer_text != "global" && !parse_size(layer_text.c_str(), &layer)) { status = {qw38::StatusCode::kInvalidArgument, "malformed trace layer"}; break; }
    qw38::internal::TraceFilter filter{layer, tap.c_str()};
    status = qw38::internal::validate_trace_filter(filter);
    qw38::cuda::SchedulerSession session;
    qw38::cuda::SchedulerWorkspace workspace;
    if (status.is_ok()) status = session.create(tokens.size());
    if (status.is_ok()) status = workspace.create(tokens.size());
    for (std::size_t pos = 0; status.is_ok() && pos + 1 < tokens.size(); ++pos)
      status = qw38::cuda::execute_token(model, tokens[pos], &session, &workspace,
                                         logits.data(), logits.size(), hidden.data(), hidden.size(),
                                         &elapsed_milliseconds,
                                         nullptr, nullptr, qw38::cuda::PointwisePath::kUnfused, nullptr);
    if (status.is_ok()) status = qw38::cuda::execute_token_traced(
        model, tokens.back(), &session, &workspace, logits.data(), logits.size(),
        hidden.data(), hidden.size(), &elapsed_milliseconds, filter,
        capture_cuda_trace, &captures[index]);
    if (status.is_ok() && (session.frontier() != tokens.size() || captures[index].values.empty()))
      status = {qw38::StatusCode::kInternal, "trace frontier or capture mismatch"};
    const std::size_t expected_layer = layer;
    const char* expected_name = tap.c_str();
    const std::size_t expected_values = tap == "logits"
                                             ? qw38::internal::kVocabularySize
                                             : qw38::internal::kResidualWidth;
    if (status.is_ok() &&
        (captures[index].name != expected_name ||
         (expected_layer == qw38::internal::kTraceAllLayers
              ? captures[index].layer != qw38::internal::kTraceAllLayers
              : captures[index].layer != expected_layer) ||
         captures[index].values.size() != expected_values ||
         captures[index].rank != 1 || captures[index].shape[0] != expected_values))
      status = {qw38::StatusCode::kInternal, "trace tensor identity mismatch"};
    if (status.is_ok() && !finite_logits(logits))
      status = {qw38::StatusCode::kInternal, "trace logits are non-finite"};
    if (status.is_ok() && index == 0) reference_logits = logits;
    if (status.is_ok() && index != 0 &&
        std::memcmp(reference_logits.data(), logits.data(),
                    reference_logits.size() * sizeof(float)) != 0)
      status = {qw38::StatusCode::kInternal, "trace logits are not identical"};
  }
  if (!status.is_ok()) { std::cerr << "runtime_failure: " << status.message() << '\n'; return 1; }
  std::error_code error;
  const std::string temporary = output_path + ".tmp." + std::to_string(getpid());
  std::filesystem::create_directories(temporary, error);
  if (error) return 1;
  std::ofstream blob(temporary + "/tensors.f32le.bin", std::ios::binary | std::ios::trunc);
  for (const auto& capture : captures)
    blob.write(reinterpret_cast<const char*>(capture.values.data()), static_cast<std::streamsize>(capture.values.size() * sizeof(float)));
  blob.close();
  std::string digest;
  status = qw38::internal::sha256_file(temporary + "/tensors.f32le.bin", &digest);
  if (!status.is_ok()) { std::filesystem::remove_all(temporary); return 1; }
  std::string executable_digest;
  status = qw38::internal::sha256_file("/proc/self/exe", &executable_digest);
  if (!status.is_ok()) { std::filesystem::remove_all(temporary); return 1; }
  std::ofstream manifest(temporary + "/manifest.json", std::ios::trunc);
  manifest << std::setprecision(17);
  manifest << "{\"schema\":\"qw38.trace\",\"version\":1,\"byte_order\":\"little\",\"model\":{\"name\":\"qwen3.8-27b-q4_k_m\",\"revision\":\"" << kModelRevision << "\",\"sha256\":\"" << kModelSha256 << "\"},\"tool\":{\"name\":\"qw38-eval-diagnostic\",\"revision\":\"" << revision << "\",\"sha256\":\"" << executable_digest << "\"},\"prompt\":{\"encoding\":\"base64\",\"bytes\":\"\",\"byte_count\":0,\"sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\",\"token_ids\":[";
  for (std::size_t i = 0; i < tokens.size(); ++i) manifest << (i ? "," : "") << tokens[i];
  manifest << "],\"positions\":["; for (std::size_t i = 0; i < tokens.size(); ++i) manifest << (i ? "," : "") << i;
  manifest << "]},\"session\":{\"before\":{\"frontier\":0,\"state_sha256\":{}},\"after\":{\"frontier\":" << tokens.size() << ",\"state_sha256\":{}}},\"blob\":{\"file\":\"tensors.f32le.bin\",\"byte_count\":" << std::filesystem::file_size(temporary + "/tensors.f32le.bin") << ",\"sha256\":\"" << digest << "\"},\"tensors\":[";
  std::size_t offset = 0; for (std::size_t i = 0; i < captures.size(); ++i) { const auto& c = captures[i]; const std::string role = filters[i].substr(filters[i].find(':') + 1); const std::string name = filters[i].rfind("global:", 0) == 0 ? filters[i].substr(7) : "layer." + filters[i].substr(0, filters[i].find(':')) + "." + role; std::string tensor_digest; vector_digest(c.values, &tensor_digest); const float minimum = *std::min_element(c.values.begin(), c.values.end()); const float maximum = *std::max_element(c.values.begin(), c.values.end()); const double mean = std::accumulate(c.values.begin(), c.values.end(), 0.0) / c.values.size(); double squares = 0.0; for (float value : c.values) squares += static_cast<double>(value) * value; if (i) manifest << ','; manifest << "{\"name\":\"" << name << "\",\"role\":\"" << role << "\",\"layer\":" << (c.layer == qw38::internal::kTraceAllLayers ? "null" : std::to_string(c.layer)) << ",\"shape\":["; for (std::size_t j=0;j<c.rank;++j) manifest << (j ? "," : "") << c.shape[j]; manifest << "],\"dtype\":\"f32-le\",\"offset_bytes\":" << offset << ",\"length_bytes\":" << c.values.size()*sizeof(float) << ",\"sha256\":\"" << tensor_digest << "\",\"summary\":{\"count\":" << c.values.size() << ",\"finite_count\":" << c.values.size() << ",\"nan_count\":0,\"positive_infinity_count\":0,\"negative_infinity_count\":0,\"minimum\":" << minimum << ",\"maximum\":" << maximum << ",\"mean\":" << mean << ",\"root_mean_square\":" << std::sqrt(squares / c.values.size()) << "}}"; offset += c.values.size()*sizeof(float); }
  manifest << "]}\n"; manifest.close();
  const std::string tool = tool_identity_json("qw38-eval-diagnostic", revision, source_state);
  const std::string runtime = runtime_json();
  if (tool.empty() || runtime.empty()) { std::filesystem::remove_all(temporary); return 1; }
  std::ofstream result(temporary + "/result.json", std::ios::trunc);
  result << "{\"schema\":\"qw38.eval-result\",\"version\":1,\"mode\":\"trace\",\"status\":\"ok\",\"model\":"
         << model_identity_json() << ",\"tool\":" << tool << ",\"runtime\":" << runtime
         << ",\"tokens\":[";
  for (std::size_t i = 0; i < tokens.size(); ++i) result << (i ? "," : "") << tokens[i];
  result << "],\"positions\":[";
  for (std::size_t i = 0; i < tokens.size(); ++i) result << (i ? "," : "") << i;
  result << "],\"frontier\":" << tokens.size()
         << ",\"trace\":{\"manifest\":{\"file\":\"manifest.json\",\"byte_count\":"
         << std::filesystem::file_size(temporary + "/manifest.json") << ",\"sha256\":\"";
  std::string manifest_digest;
  status = qw38::internal::sha256_file(temporary + "/manifest.json", &manifest_digest);
  if (!status.is_ok()) { std::filesystem::remove_all(temporary); return 1; }
  result << manifest_digest << "\"},\"blob\":{\"file\":\"tensors.f32le.bin\",\"byte_count\":"
         << std::filesystem::file_size(temporary + "/tensors.f32le.bin") << ",\"sha256\":\"" << digest
         << "\"},\"filters\":[";
  for (std::size_t i = 0; i < filters.size(); ++i) result << (i ? ",\"" : "\"") << json_escape(filters[i]) << "\"";
  result << "],\"tensor_names\":[";
  for (std::size_t i = 0; i < captures.size(); ++i) {
    const std::string role = filters[i].substr(filters[i].find(':') + 1);
    const std::string name = filters[i].rfind("global:", 0) == 0 ? role : "layer." + filters[i].substr(0, filters[i].find(':')) + "." + role;
    result << (i ? ",\"" : "\"") << json_escape(name) << "\"";
  }
  result << "]}}\n";
  result.close();
  if (!manifest || !result || !publish_noreplace(temporary, output_path)) { std::filesystem::remove_all(temporary); return 1; }
  return 0;
}
#endif

int run_high_level(int argc, char** argv, bool diagnostic) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    eval_usage(diagnostic);
    return 0;
  }
  if (argc < 2) return eval_usage(diagnostic);
  std::string mode;
  std::string token_text;
  std::string continuation_text;
  std::string output_path;
  std::string revision;
  std::string source_state;
  std::vector<std::string> filters;
  std::set<std::string> seen_options;
  for (int i = 2; i < argc; ++i) {
    const std::string option = argv[i];
    if (i + 1 >= argc) return eval_usage(diagnostic);
    if (option != "--trace-filter" && !seen_options.insert(option).second)
      return eval_usage(diagnostic);
    if (option == "--mode") mode = argv[++i];
    else if (option == "--tokens") token_text = argv[++i];
    else if (option == "--continuation") continuation_text = argv[++i];
    else if (option == "--output") output_path = argv[++i];
    else if (option == "--source-revision") revision = argv[++i];
    else if (option == "--source-state") source_state = argv[++i];
    else if (option == "--trace-filter") filters.push_back(argv[++i]);
    else return eval_usage(diagnostic);
  }
  std::vector<qw38::Token> tokens;
  std::vector<qw38::Token> continuation;
  if ((mode != "logits" && mode != "checkpoint" && mode != "trace") ||
      (!diagnostic && mode == "trace") || !parse_token_csv(token_text, &tokens) ||
      (mode == "checkpoint" && !parse_token_csv(continuation_text, &continuation)) ||
      output_path.empty() || revision.empty() ||
      (source_state != "clean" && source_state != "dirty")) {
    std::cerr << "invalid_argument: malformed evaluation request\n";
    return 2;
  }
  std::error_code error;
  if (std::filesystem::exists(output_path, error) || error) {
    std::cerr << "invalid_argument: output already exists\n";
    return 2;
  }
  if (mode == "trace") {
#if defined(QW38_CUDA_RUNTIME) && defined(QW38_DIAGNOSTIC_TRACE)
    if (filters.empty() || filters.size() > 5 ||
        std::adjacent_find(filters.begin(), filters.end()) != filters.end()) {
      std::cerr << "invalid_argument: malformed diagnostic trace filters\n";
      return 2;
    }
    std::set<std::string> unique(filters.begin(), filters.end());
    if (unique.size() != filters.size()) { std::cerr << "invalid_argument: duplicate trace filter\n"; return 2; }
    return run_cuda_trace(argv[1], tokens, filters, output_path, revision, source_state);
#else
    std::cerr << "unsupported_build: diagnostic CUDA trace requires CUDA product\n";
    return 1;
#endif
  }
  qw38::Engine engine;
  qw38::Status status = qw38::Engine::open(argv[1], &engine);
  if (!status.is_ok()) { std::cerr << status.message() << '\n'; return 1; }
  std::unique_ptr<qw38::Session> session;
  status = engine.create_session(&session);
  if (!status.is_ok()) { std::cerr << status.message() << '\n'; return 1; }
  status = session->sync(tokens);
  if (!status.is_ok()) { std::cerr << status.message() << '\n'; return 1; }
  const std::string temporary = output_path + ".tmp." + std::to_string(getpid());
  if (mode == "checkpoint") {
    std::error_code temporary_error;
    std::filesystem::create_directories(temporary, temporary_error);
    if (temporary_error) return 1;
    const std::string checkpoint_path = temporary + "/checkpoint.qw38";
    status = session->save(checkpoint_path);
    std::vector<qw38::Token> prefix_tokens;
    if (status.is_ok()) status = session->tokens(&prefix_tokens);
    const std::size_t prefix_frontier = prefix_tokens.size();
    if (status.is_ok()) status = session->eval(continuation.front());
    for (std::size_t index = 1; status.is_ok() && index < continuation.size(); ++index)
      status = session->eval(continuation[index]);
    std::vector<float> uninterrupted;
    if (status.is_ok()) status = session->logits(&uninterrupted);
    std::vector<qw38::Token> uninterrupted_tokens;
    if (status.is_ok()) status = session->tokens(&uninterrupted_tokens);
    session.reset();
    std::unique_ptr<qw38::Session> restored;
    if (status.is_ok()) status = engine.create_session(&restored);
    if (status.is_ok()) status = restored->restore(checkpoint_path);
    std::vector<qw38::Token> restored_prefix;
    if (status.is_ok()) status = restored->tokens(&restored_prefix);
    const std::size_t restored_prefix_frontier = restored_prefix.size();
    if (status.is_ok() && restored_prefix != prefix_tokens)
      status = {qw38::StatusCode::kInternal, "checkpoint prefix mismatch"};
    if (status.is_ok()) status = restored->eval(continuation.front());
    for (std::size_t index = 1; status.is_ok() && index < continuation.size(); ++index)
      status = restored->eval(continuation[index]);
    std::vector<float> restored_logits;
    std::vector<qw38::Token> restored_tokens;
    if (status.is_ok()) status = restored->logits(&restored_logits);
    if (status.is_ok()) status = restored->tokens(&restored_tokens);
    if (status.is_ok() &&
        (uninterrupted.size() != restored_logits.size() ||
         std::memcmp(uninterrupted.data(), restored_logits.data(),
                     uninterrupted.size() * sizeof(float)) != 0 ||
         uninterrupted_tokens != restored_tokens))
      status = {qw38::StatusCode::kInternal, "checkpoint continuation mismatch"};
    if (!status.is_ok()) {
      std::filesystem::remove_all(temporary);
      std::cerr << "runtime_failure: " << status.message() << '\n';
      return 1;
    }
    std::string digest;
    status = qw38::internal::sha256_file(temporary + "/checkpoint.qw38", &digest);
    if (!status.is_ok()) { std::filesystem::remove_all(temporary); return 1; }
    std::ofstream logits_blob(temporary + "/continuation_logits.f32le.bin", std::ios::binary);
    logits_blob.write(reinterpret_cast<const char*>(uninterrupted.data()),
                      static_cast<std::streamsize>(uninterrupted.size() * sizeof(float)));
    logits_blob.close();
    std::string logits_digest;
    status = qw38::internal::sha256_file(temporary + "/continuation_logits.f32le.bin", &logits_digest);
    if (!status.is_ok()) { std::filesystem::remove_all(temporary); return 1; }
    std::ofstream record(temporary + "/result.json");
    const std::string tool = tool_identity_json("qw38-eval", revision, source_state);
    const std::string runtime = runtime_json();
    if (tool.empty() || runtime.empty() || !finite_logits(uninterrupted)) {
      std::filesystem::remove_all(temporary);
      return 1;
    }
    record << "{\"schema\":\"qw38.eval-result\",\"version\":1,\"mode\":\"checkpoint\",\"status\":\"ok\",\"model\":"
           << model_identity_json() << ",\"tool\":" << tool << ",\"runtime\":" << runtime
           << ",\"prefix_tokens\":[";
    for (std::size_t i = 0; i < tokens.size(); ++i) record << (i ? "," : "") << tokens[i];
    record << "],\"continuation_tokens\":[";
    for (std::size_t i = 0; i < continuation.size(); ++i) record << (i ? "," : "") << continuation[i];
    record << "],\"prefix_positions\":[";
    for (std::size_t i = 0; i < tokens.size(); ++i) record << (i ? "," : "") << i;
    record << "],\"continuation_positions\":[";
    for (std::size_t i = 0; i < continuation.size(); ++i) record << (i ? "," : "") << tokens.size() + i;
    record << "],\"frontiers\":{\"prefix\":" << prefix_frontier
           << ",\"uninterrupted_final\":" << uninterrupted_tokens.size()
           << ",\"restored_prefix\":" << restored_prefix_frontier
           << ",\"restored_final\":" << restored_tokens.size()
           << "},\"checkpoint\":{\"file\":\"checkpoint.qw38\",\"byte_count\":" << std::filesystem::file_size(checkpoint_path)
           << ",\"sha256\":\"" << digest << "\"},\"continuation_logits\":"
           << blob_record_json("continuation_logits.f32le.bin", logits_digest, uninterrupted)
           << ",\"greedy_token\":" << greedy_token(uninterrupted)
           << ",\"equality\":{\"tokens\":true,\"logits\":true}}\n";
    record.close();
    if (!record || !publish_noreplace(temporary, output_path)) {
      std::filesystem::remove_all(temporary); return 1;
    }
    return 0;
  }
  std::vector<float> logits;
  status = session->logits(&logits);
  if (!status.is_ok() || !finite_logits(logits)) {
    std::cerr << "runtime_failure: logits unavailable\n"; return 1;
  }
  std::filesystem::create_directories(temporary, error);
  if (error) return 1;
  std::ofstream blob(temporary + "/logits.f32le.bin", std::ios::binary);
  blob.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
  blob.close();
  if (!blob) { std::filesystem::remove_all(temporary); return 1; }
  std::string digest;
  status = qw38::internal::sha256_file(temporary + "/logits.f32le.bin", &digest);
  if (!status.is_ok()) { std::filesystem::remove_all(temporary); return 1; }
  std::ofstream result(temporary + "/result.json");
  const std::string tool = tool_identity_json("qw38-eval", revision, source_state);
  const std::string runtime = runtime_json();
  if (tool.empty() || runtime.empty()) { std::filesystem::remove_all(temporary); return 1; }
  result << "{\"schema\":\"qw38.eval-result\",\"version\":1,\"mode\":\"logits\",\"status\":\"ok\",\"model\":"
         << model_identity_json() << ",\"tool\":" << tool << ",\"runtime\":" << runtime
         << ",\"tokens\":[";
  for (std::size_t i = 0; i < tokens.size(); ++i) result << (i ? "," : "") << tokens[i];
  result << "],\"positions\":[";
  for (std::size_t i = 0; i < tokens.size(); ++i) result << (i ? "," : "") << i;
  result << "],\"frontier\":" << tokens.size() << ",\"logits\":"
         << blob_record_json("logits.f32le.bin", digest, logits)
         << ",\"greedy_token\":" << greedy_token(logits) << "}\n";
  result.close();
  if (!result || !publish_noreplace(temporary, output_path)) {
    std::filesystem::remove_all(temporary); return 1;
  }
  return 0;
}

#ifdef QW38_DIAGNOSTIC_TRACE
qw38::Status count_trace_tensor(const qw38::internal::TraceTensorView& tensor,
                                void* context) noexcept {
  auto* count = static_cast<std::size_t*>(context);
  ++*count;
  std::cout << "tap=" << tensor.name << ",layer=" << tensor.layer
            << ",count=" << tensor.value_count << '\n';
  return qw38::Status::ok();
}

int check_trace_filter(const char* layer_text, const char* tap) {
  std::size_t layer = qw38::internal::kTraceAllLayers;
  if (std::strcmp(layer_text, "all") != 0 && !parse_size(layer_text, &layer)) {
    std::cerr << "invalid_argument: malformed diagnostic trace layer\n";
    return 1;
  }
  const qw38::internal::TraceFilter filter{layer, tap};
  constexpr std::array<float, 4> kValues{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<qw38::internal::TraceTensorView, 3> tensors{{
      {"embedding", qw38::internal::kTraceAllLayers, kValues.data(), 4,
       {4, 0, 0}, 1},
      {"attention.rope_query", 3, kValues.data(), 4, {2, 2, 0}, 2},
      {"layer_residual", 3, kValues.data(), 4, {4, 0, 0}, 1},
  }};
  std::size_t count = 0;
  qw38::Status status = qw38::internal::validate_trace_filter(filter);
  for (const auto& tensor : tensors) {
    if (status.is_ok()) {
      status = qw38::internal::emit_trace_tensor(filter, count_trace_tensor,
                                                 &count, tensor);
    }
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "matched=" << count << '\n';
  return 0;
}

struct RawTraceCapture final {
  const char* path;
  const char* name = nullptr;
  std::size_t layer = 0;
  std::array<std::size_t, qw38::internal::kTraceMaximumRank> shape{};
  std::size_t rank = 0;
  std::size_t count = 0;
};

struct BundleTraceRecord final {
  const char* name = nullptr;
  std::size_t layer = 0;
  std::size_t position = 0;
  std::array<std::size_t, qw38::internal::kTraceMaximumRank> shape{};
  std::size_t rank = 0;
  std::size_t count = 0;
  std::size_t offset = 0;
};

struct BundleTraceCapture final {
  static constexpr std::size_t kMaximumRecords = 4096;
  std::ofstream* output = nullptr;
  std::array<BundleTraceRecord, kMaximumRecords> records{};
  std::size_t record_count = 0;
  std::size_t position = 0;
  std::size_t bytes = 0;
};

qw38::Status append_raw_trace_tensor(
    const qw38::internal::TraceTensorView& tensor, void* context) noexcept {
  auto* capture = static_cast<BundleTraceCapture*>(context);
  if (capture == nullptr || capture->output == nullptr ||
      capture->record_count == BundleTraceCapture::kMaximumRecords ||
      tensor.value_count > std::numeric_limits<std::size_t>::max() /
                               sizeof(float)) {
    return {qw38::StatusCode::kInvalidArgument,
            "scalar bundle trace capture is invalid or full"};
  }
  const std::size_t tensor_bytes = tensor.value_count * sizeof(float);
  if (capture->bytes >
      std::numeric_limits<std::size_t>::max() - tensor_bytes) {
    return {qw38::StatusCode::kInvalidArgument,
            "scalar bundle trace byte count overflows"};
  }
  capture->output->write(reinterpret_cast<const char*>(tensor.values),
                         static_cast<std::streamsize>(tensor_bytes));
  if (!*capture->output) {
    return {qw38::StatusCode::kInternal,
            "cannot append scalar bundle trace tensor"};
  }
  BundleTraceRecord& record = capture->records[capture->record_count++];
  record = {tensor.name, tensor.layer, capture->position, tensor.shape,
            tensor.rank, tensor.value_count, capture->bytes};
  capture->bytes += tensor_bytes;
  return qw38::Status::ok();
}

qw38::Status write_raw_trace_tensor(
    const qw38::internal::TraceTensorView& tensor, void* context) noexcept {
  auto* capture = static_cast<RawTraceCapture*>(context);
  if (capture->count != 0) {
    return {qw38::StatusCode::kInvalidArgument,
            "exact scalar trace filter matched more than one tensor"};
  }
  std::ofstream output(capture->path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return {qw38::StatusCode::kInternal,
            "cannot create scalar trace tensor output"};
  }
  output.write(reinterpret_cast<const char*>(tensor.values),
               static_cast<std::streamsize>(tensor.value_count * sizeof(float)));
  output.close();
  if (!output) {
    std::remove(capture->path);
    return {qw38::StatusCode::kInternal,
            "cannot write scalar trace tensor output"};
  }
  capture->name = tensor.name;
  capture->layer = tensor.layer;
  capture->shape = tensor.shape;
  capture->rank = tensor.rank;
  capture->count = tensor.value_count;
  return qw38::Status::ok();
}

bool vector_digest(const std::vector<float>& values, std::string* digest) {
  return qw38::internal::sha256_bytes(
             reinterpret_cast<const unsigned char*>(values.data()),
             values.size() * sizeof(float), digest)
      .is_ok();
}

int capture_real_scalar_trace(const char* model_path, const char* token_text,
                              const char* layer_text, const char* tap,
                              const char* output_path) {
  std::size_t token = 0;
  std::size_t layer = qw38::internal::kTraceAllLayers;
  if (!parse_size(token_text, &token) ||
      (std::strcmp(layer_text, "global") != 0 &&
       !parse_size(layer_text, &layer))) {
    std::cerr << "invalid_argument: malformed scalar trace token or layer\n";
    return 1;
  }
  std::error_code error;
  if (std::filesystem::exists(output_path, error) || error) {
    std::cerr << "invalid_argument: scalar trace output already exists\n";
    return 1;
  }
  const qw38::internal::TraceFilter filter{layer, tap};
  qw38::Status status = qw38::internal::validate_trace_filter(filter);
  qw38::internal::ModelInfo info;
  if (status.is_ok()) status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::internal::ScalarModelParameters parameters;
  if (status.is_ok()) {
    status = qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  }
  qw38::internal::ScalarSessionState state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(1, &state);
  }
  qw38::internal::ScalarWorkspace workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(1, &workspace);
  }
  std::array<std::string, 4> before;
  if (status.is_ok() &&
      (!vector_digest(state.gdn_convolution, &before[0]) ||
       !vector_digest(state.gdn_recurrent, &before[1]) ||
       !vector_digest(state.attention_key, &before[2]) ||
       !vector_digest(state.attention_value, &before[3]))) {
    status = {qw38::StatusCode::kInternal,
              "cannot hash scalar state before tracing"};
  }
  std::vector<float> logits(qw38::internal::kVocabularySize, NAN);
  RawTraceCapture capture{output_path};
  if (status.is_ok()) {
    status = qw38::internal::execute_scalar_token_traced(
        weights, parameters, token, &state, &workspace, logits.data(),
        logits.size(), filter, write_raw_trace_tensor, &capture);
  }
  std::array<std::string, 4> after;
  if (status.is_ok() &&
      (!vector_digest(state.gdn_convolution, &after[0]) ||
       !vector_digest(state.gdn_recurrent, &after[1]) ||
       !vector_digest(state.attention_key, &after[2]) ||
       !vector_digest(state.attention_value, &after[3]))) {
    status = {qw38::StatusCode::kInternal,
              "cannot hash scalar state after tracing"};
  }
  if (status.is_ok() && capture.count == 0) {
    status = {qw38::StatusCode::kInvalidArgument,
              "scalar trace filter matched no real tensor"};
  }
  if (!status.is_ok()) {
    std::remove(output_path);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "tap_name=" << capture.name << "\ntap_layer=" << capture.layer
            << "\nshape=";
  for (std::size_t dimension = 0; dimension < capture.rank; ++dimension) {
    if (dimension != 0) std::cout << ',';
    std::cout << capture.shape[dimension];
  }
  std::cout << "\ncount=" << capture.count << "\ntoken=" << token
            << "\nposition=0\nfrontier_before=0\nfrontier_after="
            << state.frontier << '\n';
  constexpr const char* kStateNames[] = {"gdn_convolution", "gdn_recurrent",
                                          "attention_key", "attention_value"};
  for (std::size_t index = 0; index < before.size(); ++index) {
    std::cout << "state_before_" << kStateNames[index] << "=" << before[index]
              << "\nstate_after_" << kStateNames[index] << "=" << after[index]
              << '\n';
  }
  return 0;
}

int capture_real_scalar_bundle(const char* model_path, const char* output_path,
                               const char* first_token_text,
                               const char* second_token_text) {
  std::array<std::size_t, 2> tokens{};
  if (!parse_size(first_token_text, &tokens[0]) ||
      !parse_size(second_token_text, &tokens[1])) {
    std::cerr << "invalid_argument: malformed scalar bundle token\n";
    return 1;
  }
  std::error_code error;
  if (std::filesystem::exists(output_path, error) || error) {
    std::cerr << "invalid_argument: scalar bundle output already exists\n";
    return 1;
  }
  const qw38::internal::TraceFilter filter{
      qw38::internal::kTraceAllLayers, "*"};
  qw38::Status status = qw38::internal::validate_trace_filter(filter);
  qw38::internal::ModelInfo info;
  if (status.is_ok()) status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::internal::ScalarModelParameters parameters;
  if (status.is_ok()) {
    status =
        qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  }
  qw38::internal::ScalarSessionState state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(tokens.size(), &state);
  }
  qw38::internal::ScalarWorkspace workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(tokens.size(), &workspace);
  }
  std::array<std::string, 4> before;
  if (status.is_ok() &&
      (!vector_digest(state.gdn_convolution, &before[0]) ||
       !vector_digest(state.gdn_recurrent, &before[1]) ||
       !vector_digest(state.attention_key, &before[2]) ||
       !vector_digest(state.attention_value, &before[3]))) {
    status = {qw38::StatusCode::kInternal,
              "cannot hash scalar state before bundle tracing"};
  }
  std::ofstream output;
  if (status.is_ok()) {
    output.open(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
      status = {qw38::StatusCode::kInternal,
                "cannot create scalar bundle output"};
    }
  }
  BundleTraceCapture capture{&output};
  std::vector<float> logits(qw38::internal::kVocabularySize, NAN);
  for (std::size_t position = 0;
       status.is_ok() && position < tokens.size(); ++position) {
    capture.position = position;
    status = qw38::internal::execute_scalar_token_traced(
        weights, parameters, tokens[position], &state, &workspace,
        logits.data(), logits.size(), filter, append_raw_trace_tensor, &capture);
  }
  output.close();
  if (status.is_ok() && !output) {
    status = {qw38::StatusCode::kInternal,
              "cannot close scalar bundle output"};
  }
  std::array<std::string, 4> after;
  if (status.is_ok() &&
      (!vector_digest(state.gdn_convolution, &after[0]) ||
       !vector_digest(state.gdn_recurrent, &after[1]) ||
       !vector_digest(state.attention_key, &after[2]) ||
       !vector_digest(state.attention_value, &after[3]))) {
    status = {qw38::StatusCode::kInternal,
              "cannot hash scalar state after bundle tracing"};
  }
  if (status.is_ok() && capture.record_count == 0) {
    status = {qw38::StatusCode::kInternal,
              "scalar bundle trace captured no tensors"};
  }
  if (!status.is_ok()) {
    std::remove(output_path);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::cout << "schema_version=1\ntoken_count=" << tokens.size()
            << "\ntoken_0=" << tokens[0] << "\ntoken_1=" << tokens[1]
            << "\nfrontier_before=0\nfrontier_after=" << state.frontier
            << "\ntensor_count=" << capture.record_count
            << "\nblob_bytes=" << capture.bytes << '\n';
  constexpr const char* kStateNames[] = {"gdn_convolution", "gdn_recurrent",
                                          "attention_key", "attention_value"};
  for (std::size_t index = 0; index < before.size(); ++index) {
    std::cout << "state_before_" << kStateNames[index] << '=' << before[index]
              << "\nstate_after_" << kStateNames[index] << '=' << after[index]
              << '\n';
  }
  for (std::size_t index = 0; index < capture.record_count; ++index) {
    const BundleTraceRecord& record = capture.records[index];
    std::cout << "tensor_" << index << "_name=" << record.name << "\ntensor_"
              << index << "_layer=" << record.layer << "\ntensor_" << index
              << "_position=" << record.position << "\ntensor_" << index
              << "_shape=";
    for (std::size_t dimension = 0; dimension < record.rank; ++dimension) {
      if (dimension != 0) std::cout << ',';
      std::cout << record.shape[dimension];
    }
    std::cout << "\ntensor_" << index << "_count=" << record.count
              << "\ntensor_" << index << "_offset=" << record.offset << '\n';
  }
  return 0;
}
#endif

bool parse_float_hex(const std::string& hex, std::vector<float>* values) {
  std::vector<std::uint8_t> bytes;
  if (!parse_hex(hex, &bytes) || bytes.size() % 4 != 0) return false;
  values->resize(bytes.size() / 4);
  for (std::size_t index = 0; index < values->size(); ++index) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[index * 4]) |
        (static_cast<std::uint32_t>(bytes[index * 4 + 1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[index * 4 + 2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[index * 4 + 3]) << 24U);
    std::memcpy(values->data() + index, &bits, sizeof(bits));
  }
  return true;
}

void write_float_hex(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr char kHex[] = "0123456789abcdef";
  for (unsigned int byte = 0; byte < 4; ++byte) {
    const unsigned int value_byte = (bits >> (byte * 8U)) & 0xFFU;
    std::cout << kHex[value_byte >> 4U] << kHex[value_byte & 15U];
  }
}

int check_quant(const std::string& kind, const std::string& hex) {
  std::vector<std::uint8_t> block;
  if (!parse_hex(hex, &block)) {
    std::cerr << "invalid_argument: quant block is not even-length hexadecimal\n";
    return 1;
  }
  const std::size_t value_count = kind == "q8_0"
                                      ? qw38::internal::kQ80BlockValues
                                      : qw38::internal::kQuantBlockValues;
  std::vector<float> decoded(value_count);
  std::vector<float> activation(value_count);
  for (std::size_t index = 0; index < activation.size(); ++index) {
    const int numerator = static_cast<int>((index * 37) % 101) - 50;
    activation[index] = static_cast<float>(numerator) / 32.0F;
  }
  float dot = 0.0F;
  qw38::Status status;
  if (kind == "q4_k") {
    status = qw38::internal::decode_q4_k(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q4_k(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else if (kind == "q6_k") {
    status = qw38::internal::decode_q6_k(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q6_k(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else if (kind == "q5_k") {
    status = qw38::internal::decode_q5_k(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q5_k(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else if (kind == "q8_0") {
    status = qw38::internal::decode_q8_0(block.data(), block.size(),
                                         decoded.data(), decoded.size());
    if (status.is_ok()) {
      status = qw38::internal::dot_q8_0(block.data(), block.size(),
                                        activation.data(), activation.size(),
                                        &dot);
    }
  } else {
    std::cerr << "invalid_argument: quant kind must be q4_k, q5_k, q6_k, or q8_0\n";
    return 1;
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "decoded_f32_le_hex=";
  for (float value : decoded) write_float_hex(value);
  std::cout << "\ndot_f32_le_hex=";
  write_float_hex(dot);
  std::cout << '\n';
  return 0;
}

void write_float_vector(const char* name, const std::vector<float>& values) {
  std::cout << name << '=';
  for (float value : values) write_float_hex(value);
  std::cout << '\n';
}

void write_float_vector(const std::string& name,
                        const std::vector<float>& values) {
  write_float_vector(name.c_str(), values);
}

template <std::size_t Size>
void write_float_array(const char* name, const std::array<float, Size>& values) {
  std::cout << name << '=';
  for (float value : values) write_float_hex(value);
  std::cout << '\n';
}

int check_conversion(const std::string& component) {
  if (component == "permutation") {
    constexpr std::array<float, 12> kGrouped{
        0.0F, 1.0F, 10.0F, 11.0F, 20.0F, 21.0F,
        100.0F, 101.0F, 110.0F, 111.0F, 120.0F, 121.0F};
    std::array<float, kGrouped.size()> tiled{};
    std::array<float, kGrouped.size()> roundtrip{};
    qw38::Status status = qw38::internal::gdn_grouped_to_tiled(
        kGrouped.data(), 2, 3, 2, tiled.data(), tiled.size());
    if (status.is_ok()) {
      status = qw38::internal::gdn_tiled_to_grouped(
          tiled.data(), 2, 3, 2, roundtrip.data(), roundtrip.size());
    }
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("grouped_f32_le_hex", kGrouped);
    write_float_array("tiled_f32_le_hex", tiled);
    write_float_array("roundtrip_f32_le_hex", roundtrip);
    return 0;
  }
  if (component == "gates") {
    constexpr std::array<float, 4> kA{-0.75F, -0.125F, 0.5F, 1.25F};
    constexpr std::array<float, 4> kB{-1.0F, -0.25F, 0.25F, 1.0F};
    constexpr std::array<float, 4> kALog{-1.5F, -0.5F, 0.0F, 0.75F};
    constexpr std::array<float, 4> kDt{-0.25F, 0.0F, 0.25F, 0.5F};
    std::array<float, 4> folded_a{};
    std::array<float, 4> source_decay{};
    std::array<float, 4> source_beta{};
    std::array<float, 4> gguf_decay{};
    std::array<float, 4> gguf_beta{};
    for (std::size_t index = 0; index < folded_a.size(); ++index) {
      folded_a[index] = -std::exp(kALog[index]);
    }
    qw38::Status status = qw38::internal::gdn_gates_from_source(
        kA.data(), kB.data(), kALog.data(), kDt.data(), kA.size(),
        source_decay.data(), source_beta.data());
    if (status.is_ok()) {
      status = qw38::internal::gdn_gates_from_gguf(
          kA.data(), kB.data(), folded_a.data(), kDt.data(), kA.size(),
          gguf_decay.data(), gguf_beta.data());
    }
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("folded_a_f32_le_hex", folded_a);
    write_float_array("source_decay_f32_le_hex", source_decay);
    write_float_array("gguf_decay_f32_le_hex", gguf_decay);
    write_float_array("source_beta_f32_le_hex", source_beta);
    write_float_array("gguf_beta_f32_le_hex", gguf_beta);
    return 0;
  }
  if (component == "norm") {
    constexpr std::array<float, 4> kInput{-2.0F, -0.5F, 1.0F, 3.0F};
    constexpr std::array<float, 4> kSourceWeight{-0.125F, 0.0F, 0.25F,
                                                 0.5F};
    std::array<float, 4> gguf_scale{};
    std::array<float, 4> source_output{};
    std::array<float, 4> gguf_output{};
    for (std::size_t index = 0; index < gguf_scale.size(); ++index) {
      gguf_scale[index] = 1.0F + kSourceWeight[index];
    }
    qw38::Status status = qw38::internal::rms_norm(
        kInput.data(), kSourceWeight.data(), kInput.size(),
        source_output.data());
    if (status.is_ok()) {
      status = qw38::internal::rms_norm_scale(
          kInput.data(), gguf_scale.data(), kInput.size(), gguf_output.data());
    }
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("gguf_scale_f32_le_hex", gguf_scale);
    write_float_array("source_output_f32_le_hex", source_output);
    write_float_array("gguf_output_f32_le_hex", gguf_output);
    return 0;
  }
  if (component == "invalid_folded_a") {
    float value = 0.0F;
    float result = 0.0F;
    const qw38::Status status = qw38::internal::gdn_gates_from_gguf(
        &value, &value, &value, &value, 1, &result, &result);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  std::cerr << "invalid_argument: unknown conversion component\n";
  return 1;
}

int check_weight_binding(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = admit_gguf(model_path, &info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  auto find = [&info](const std::string& name) {
    return std::find_if(info.tensors.begin(), info.tensors.end(),
                        [&name](const qw38::internal::TensorInfo& tensor) {
                          return tensor.name == name;
                        });
  };
  if (mode == "missing") {
    find("blk.0.ssm_a")->name = "blk.0.missing_ssm_a";
  } else if (mode == "role") {
    find("blk.0.ssm_a")->semantic_role = "gdn_dt_bias";
  } else if (mode == "shape") {
    find("blk.0.ssm_a")->dimensions[0] = 47;
  } else if (mode == "range") {
    find("output.weight")->offset = mapping.size();
  } else if (mode != "valid") {
    std::cerr << "invalid_argument: unknown weight-binding mode\n";
    return 1;
  }
  qw38::internal::ModelWeights weights;
  status = qw38::internal::bind_model_weights(info, mapping, &weights);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::size_t gdn_layers = 0;
  std::size_t attention_layers = 0;
  for (std::size_t layer = 0; layer < weights.geometry.layer_count; ++layer) {
    if (weights.layers[layer].kind == qw38::internal::LayerKind::kGdn) {
      ++gdn_layers;
    } else {
      ++attention_layers;
    }
  }
  std::cout << "bound_tensors=" << weights.bound_tensor_count << '\n';
  std::cout << "gdn_layers=" << gdn_layers << '\n';
  std::cout << "attention_layers=" << attention_layers << '\n';
  std::cout << "embedding_columns=" << weights.token_embedding.columns << '\n';
  std::cout << "embedding_rows=" << weights.token_embedding.rows << '\n';
  std::cout << (std::string("final") + "_norm_values=")
            << weights.output_norm.count << '\n';
  std::cout << "logit_rows=" << weights.output.rows << '\n';
  if (weights.geometry.tied_embeddings) {
    std::cout << "tied_output=1\noutput_shares_embedding="
              << (weights.output.data == weights.token_embedding.data ? 1 : 0)
              << '\n';
  }
  std::vector<float> final_norm(weights.output_norm.count);
  status = qw38::internal::vector_decode(weights.output_norm, final_norm.data(),
                                         final_norm.size());
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  write_float_vector(std::string("final") + "_norm_endpoints_f32_le_hex",
                     {final_norm.front(), final_norm.back()});
  return 0;
}

int check_projection_layout(const std::string& component) {
  if (component == "gdn") {
    constexpr std::array<float, 12> kPacked{0.0F,  1.0F,  2.0F,  3.0F,
                                            10.0F, 11.0F, 12.0F, 13.0F,
                                            20.0F, 21.0F, 22.0F, 23.0F};
    std::array<float, 4> query{};
    std::array<float, 4> key{};
    std::array<float, 4> value{};
    const qw38::Status status = qw38::internal::split_gdn_qkv(
        kPacked.data(), kPacked.size(), 2, 2, 2, 2, query.data(), query.size(),
        key.data(), key.size(), value.data(), value.size());
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("query_f32_le_hex", query);
    write_float_array("key_f32_le_hex", key);
    write_float_array("value_f32_le_hex", value);
    return 0;
  }
  if (component == "attention") {
    constexpr std::array<float, 12> kPacked{
        0.0F, 1.0F, 10.0F, 11.0F, 100.0F, 101.0F,
        110.0F, 111.0F, 200.0F, 201.0F, 210.0F, 211.0F};
    std::array<float, 6> query{};
    std::array<float, 6> gate{};
    const qw38::Status status = qw38::internal::split_attention_query_gate(
        kPacked.data(), kPacked.size(), 3, 2, query.data(), query.size(),
        gate.data(), gate.size());
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    write_float_array("query_f32_le_hex", query);
    write_float_array("gate_f32_le_hex", gate);
    return 0;
  }
  if (component == "invalid_alias") {
    std::array<float, 4> packed{};
    std::array<float, 2> gate{};
    const qw38::Status status = qw38::internal::split_attention_query_gate(
        packed.data(), packed.size(), 1, 2, packed.data(), 2, gate.data(),
        gate.size());
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  if (component == "invalid_count") {
    std::array<float, 4> packed{};
    std::array<float, 2> query{};
    std::array<float, 2> gate{};
    const qw38::Status status = qw38::internal::split_attention_query_gate(
        packed.data(), packed.size() - 1, 1, 2, query.data(), query.size(),
        gate.data(), gate.size());
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  std::cerr << "invalid_argument: unknown projection-layout component\n";
  return 1;
}

int check_mixer_projections(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> activation(qw38::internal::kResidualWidth);
  for (std::size_t index = 0; index < activation.size(); ++index) {
    activation[index] = static_cast<float>(
                            static_cast<int>((index * 37) % 101) - 50) /
                        32.0F;
  }
  std::vector<float> gdn_packed(qw38::internal::kGdnPackedQkvWidth, NAN);
  std::vector<float> gdn_value_gate(qw38::internal::kGdnValueWidth, NAN);
  std::vector<float> gdn_alpha(qw38::internal::kGdnGateCount, NAN);
  std::vector<float> gdn_beta(qw38::internal::kGdnGateCount, NAN);
  qw38::internal::GdnProjectionWorkspace gdn_workspace{
      gdn_packed.data(), gdn_packed.size(), gdn_value_gate.data(),
      gdn_value_gate.size(), gdn_alpha.data(), gdn_alpha.size(),
      gdn_beta.data(), gdn_beta.size()};
  if (mode == "invalid_workspace") {
    --gdn_workspace.packed_qkv_count;
    status = qw38::internal::project_gdn_mixer(
        weights.layers[0].gdn, activation.data(), activation.size(),
        gdn_workspace);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  if (mode != "valid") {
    std::cerr << "invalid_argument: unknown mixer-projection mode\n";
    return 1;
  }
  status = qw38::internal::project_gdn_mixer(
      weights.layers[0].gdn, activation.data(), activation.size(),
      gdn_workspace);

  std::vector<float> attention_packed(
      qw38::internal::kAttentionPackedQueryGateWidth, NAN);
  std::vector<float> attention_query(qw38::internal::kAttentionQueryWidth, NAN);
  std::vector<float> attention_gate(qw38::internal::kAttentionQueryWidth, NAN);
  std::vector<float> attention_key(qw38::internal::kAttentionKvWidth, NAN);
  std::vector<float> attention_value(qw38::internal::kAttentionKvWidth, NAN);
  const qw38::internal::AttentionProjectionWorkspace attention_workspace{
      attention_packed.data(), attention_packed.size(), attention_query.data(),
      attention_query.size(), attention_gate.data(), attention_gate.size(),
      attention_key.data(), attention_key.size(), attention_value.data(),
      attention_value.size()};
  if (status.is_ok()) {
    status = qw38::internal::project_attention_mixer(
        weights.layers[3].attention, activation.data(), activation.size(),
        attention_workspace);
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (status.is_ok() &&
      (!finite(gdn_packed) || !finite(gdn_value_gate) || !finite(gdn_alpha) ||
       !finite(gdn_beta) || !finite(attention_packed) ||
       !finite(attention_query) || !finite(attention_gate) ||
       !finite(attention_key) || !finite(attention_value))) {
    status = {qw38::StatusCode::kInternal,
              "mixer projection left a nonfinite workspace value"};
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  write_float_vector("blk.0.attn_qkv.weight", {gdn_packed[0], gdn_packed[2047],
                                                gdn_packed[2048],
                                                gdn_packed[4095],
                                                gdn_packed[4096],
                                                gdn_packed[10239]});
  write_float_vector("blk.0.attn_gate.weight",
                     {gdn_value_gate.front(), gdn_value_gate.back()});
  write_float_vector("blk.0.ssm_alpha.weight",
                     {gdn_alpha.front(), gdn_alpha.back()});
  write_float_vector("blk.0.ssm_beta.weight",
                     {gdn_beta.front(), gdn_beta.back()});
  write_float_vector("blk.3.attn_q.weight",
                     {attention_packed[0], attention_packed[255],
                      attention_packed[256], attention_packed[511],
                      attention_packed[512], attention_packed[767],
                      attention_packed[12287]});
  write_float_vector("blk.3.attn_k.weight",
                     {attention_key.front(), attention_key.back()});
  write_float_vector("blk.3.attn_v.weight",
                     {attention_value.front(), attention_value.back()});
  write_float_vector("attention_query_split_f32_le_hex",
                     {attention_query[0], attention_query[255],
                      attention_query[256], attention_query[511]});
  write_float_vector("attention_gate_split_f32_le_hex",
                     {attention_gate[0], attention_gate[255],
                      attention_gate.back()});
  std::cout << "computed_values="
            << gdn_packed.size() + gdn_value_gate.size() + gdn_alpha.size() +
                   gdn_beta.size() + attention_packed.size() +
                   attention_query.size() + attention_gate.size() +
                   attention_key.size() + attention_value.size()
            << '\n';
  return 0;
}

int check_real_gdn_step(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> input_norm(geometry.residual_width);
  std::vector<float> convolution(geometry.gdn_conv_values());
  std::vector<float> folded_a(geometry.gdn_value_heads);
  std::vector<float> dt_bias(geometry.gdn_value_heads);
  std::vector<float> recurrent_norm(geometry.gdn_head_width);
  const qw38::internal::GdnScalarParameters parameters{
      input_norm.data(), input_norm.size(), convolution.data(),
      convolution.size(), folded_a.data(), folded_a.size(), dt_bias.data(),
      dt_bias.size(), recurrent_norm.data(), recurrent_norm.size()};
  std::vector<float> ffn_norm(geometry.residual_width);
  const qw38::internal::FfnScalarParameters ffn_parameters{ffn_norm.data(),
                                                           ffn_norm.size()};
  const bool layer_mode = mode == "layer" || mode == "layer_invalid_ffn";
  if (layer_mode) {
    const qw38::internal::GdnLayerScalarParameters layer_parameters{
        parameters, ffn_parameters};
    status = qw38::internal::prepare_gdn_layer_scalar_parameters(
        weights.layers[0], layer_parameters);
  } else {
    status = qw38::internal::prepare_gdn_scalar_parameters(weights.layers[0],
                                                           parameters);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> residual(geometry.residual_width);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = static_cast<float>(
                          static_cast<int>((index * 37) % 101) - 50) /
                      32.0F;
  }
  std::vector<float> conv_state(geometry.gdn_conv_values());
  std::vector<float> recurrent_state(
      geometry.gdn_recurrent_values());
  qw38::internal::GdnLayerStateView state{
      conv_state.data(), conv_state.size(), recurrent_state.data(),
      recurrent_state.size()};

  std::vector<float> normalized(geometry.residual_width, NAN);
  std::vector<float> packed(geometry.gdn_packed_qkv(), NAN);
  std::vector<float> projected_gate(geometry.gdn_value_width(), NAN);
  std::vector<float> projected_alpha(geometry.gdn_value_heads, NAN);
  std::vector<float> projected_beta(geometry.gdn_value_heads, NAN);
  std::vector<float> convolved(geometry.gdn_packed_qkv(), NAN);
  std::vector<float> query(geometry.gdn_key_width(), NAN);
  std::vector<float> key(geometry.gdn_key_width(), NAN);
  std::vector<float> value_tiled(geometry.gdn_value_width(), NAN);
  std::vector<float> value_grouped(geometry.gdn_value_width(), NAN);
  std::vector<float> gate_grouped(geometry.gdn_value_width(), NAN);
  std::vector<float> alpha_grouped(geometry.gdn_value_heads, NAN);
  std::vector<float> beta_grouped(geometry.gdn_value_heads, NAN);
  std::vector<float> folded_grouped(geometry.gdn_value_heads, NAN);
  std::vector<float> dt_grouped(geometry.gdn_value_heads, NAN);
  std::vector<float> log_decay(geometry.gdn_value_heads, NAN);
  std::vector<float> update_gate(geometry.gdn_value_heads, NAN);
  std::vector<float> recurrent_output(geometry.gdn_value_width(), NAN);
  std::vector<float> gated_grouped(geometry.gdn_value_width(), NAN);
  std::vector<float> gated_tiled(geometry.gdn_value_width(), NAN);
  std::vector<float> mixer_output(geometry.residual_width, NAN);
  std::vector<float> post_mixer(geometry.residual_width, NAN);
  std::vector<float> ffn_normalized(geometry.residual_width, NAN);
  std::vector<float> ffn_gate(geometry.ffn_width, NAN);
  std::vector<float> ffn_up(geometry.ffn_width, NAN);
  std::vector<float> ffn_activated(geometry.ffn_width, NAN);
  std::vector<float> ffn_correction(geometry.residual_width, NAN);
  std::vector<float> output(geometry.residual_width, NAN);
  const qw38::internal::GdnProjectionWorkspace projection_workspace{
      packed.data(), packed.size(), projected_gate.data(), projected_gate.size(),
      projected_alpha.data(), projected_alpha.size(), projected_beta.data(),
      projected_beta.size()};
  qw38::internal::GdnStepWorkspace workspace{
      normalized.data(), normalized.size(), projection_workspace,
      convolved.data(), convolved.size(), query.data(), query.size(), key.data(),
      key.size(), value_tiled.data(), value_tiled.size(), value_grouped.data(),
      value_grouped.size(), gate_grouped.data(), gate_grouped.size(),
      alpha_grouped.data(), beta_grouped.data(), folded_grouped.data(),
      dt_grouped.data(), log_decay.data(), update_gate.data(),
      update_gate.size(), recurrent_output.data(), recurrent_output.size(),
      gated_grouped.data(), gated_grouped.size(), gated_tiled.data(),
      gated_tiled.size(), mixer_output.data(), mixer_output.size()};
  qw38::internal::FfnStepWorkspace ffn_workspace{
      ffn_normalized.data(), ffn_normalized.size(), ffn_gate.data(),
      ffn_gate.size(), ffn_up.data(), ffn_up.size(), ffn_activated.data(),
      ffn_activated.size(), ffn_correction.data(), ffn_correction.size()};
  if (mode == "invalid_workspace") --workspace.gated_tiled_count;
  if (mode == "layer_invalid_ffn") --ffn_workspace.activated_count;
  if (mode != "valid" && mode != "invalid_workspace" && !layer_mode) {
    std::cerr << "invalid_argument: unknown real-GDN-step mode\n";
    return 1;
  }
  if (layer_mode) {
    const qw38::internal::GdnLayerScalarParameters layer_parameters{
        parameters, ffn_parameters};
    const qw38::internal::GdnLayerWorkspace layer_workspace{
        workspace, ffn_workspace, post_mixer.data(), post_mixer.size()};
    status = qw38::internal::execute_gdn_layer_step(
        weights.layers[0], layer_parameters, residual.data(), residual.size(),
        state, layer_workspace, output.data(), output.size());
  } else {
    status = qw38::internal::execute_gdn_mixer_step(
        weights.layers[0], parameters, residual.data(), residual.size(), state,
        workspace, output.data(), output.size());
  }
  if (!status.is_ok()) {
    if (mode == "invalid_workspace" || mode == "layer_invalid_ffn") {
      const bool unchanged =
          std::all_of(conv_state.begin(), conv_state.end(),
                      [](float value) { return value == 0.0F; }) &&
          std::all_of(recurrent_state.begin(), recurrent_state.end(),
                      [](float value) { return value == 0.0F; });
      std::cout << "state_unchanged=" << (unchanged ? 1 : 0) << '\n';
      const bool output_untouched = std::all_of(
          output.begin(), output.end(),
          [](float value) { return std::isnan(value); });
      std::cout << "output_untouched=" << (output_untouched ? 1 : 0) << '\n';
    }
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (layer_mode &&
      (!finite(post_mixer) || !finite(ffn_normalized) || !finite(ffn_gate) ||
       !finite(ffn_up) || !finite(ffn_activated) || !finite(ffn_correction) ||
       !finite(output))) {
    std::cerr << "internal: GDN layer left a nonfinite value\n";
    return 1;
  }
  const auto taps = [](const std::vector<float>& values,
                       std::initializer_list<std::size_t> indices) {
    std::vector<float> selected;
    selected.reserve(indices.size());
    for (std::size_t index : indices) selected.push_back(values[index]);
    return selected;
  };
  write_float_vector("normalized_f32_le_hex",
                     tap_values(normalized, residual_end_taps(geometry)));
  write_float_vector("convolved_f32_le_hex",
                     tap_values(convolved, gdn_convolved_taps(geometry)));
  write_float_vector("value_grouped_f32_le_hex",
                     taps(value_grouped, {0, 127, 128, 255, 384, 511}));
  write_float_vector("gate_controls_f32_le_hex",
                     tap_values(alpha_grouped, gdn_gate_taps(geometry)));
  write_float_vector("log_decay_f32_le_hex",
                     tap_values(log_decay, gdn_gate_taps(geometry)));
  write_float_vector("update_gate_f32_le_hex",
                     tap_values(update_gate, gdn_gate_taps(geometry)));
  write_float_vector("recurrent_output_f32_le_hex",
                     taps(recurrent_output, {0, 127, 128, 255, 384, 511}));
  write_float_vector("gated_grouped_f32_le_hex",
                     taps(gated_grouped, {0, 127, 128, 255, 384, 511}));
  write_float_vector("mixer_output_f32_le_hex",
                     tap_values(mixer_output, residual_taps(geometry)));
  write_float_vector("residual_output_f32_le_hex",
                     tap_values(output, residual_taps(geometry)));
  if (layer_mode) {
    write_float_vector("post_mixer_f32_le_hex",
                       tap_values(post_mixer, residual_taps(geometry)));
    write_float_vector("ffn_normalized_f32_le_hex",
                       tap_values(ffn_normalized, residual_taps(geometry)));
    write_float_vector("ffn_gate_f32_le_hex",
                       tap_values(ffn_gate, ffn_taps(geometry)));
    write_float_vector("ffn_up_f32_le_hex",
                       tap_values(ffn_up, ffn_taps(geometry)));
    write_float_vector("ffn_activated_f32_le_hex",
                       tap_values(ffn_activated, ffn_taps(geometry)));
    write_float_vector("ffn_correction_f32_le_hex",
                       tap_values(ffn_correction, residual_taps(geometry)));
  }
  write_float_vector("convolution_state_f32_le_hex",
                     taps(conv_state, {0, 1, 2, 3, 16384, 16385, 16386,
                                       16387}));
  write_float_vector("recurrent_state_f32_le_hex",
                     taps(recurrent_state, {0, 127, 16383, 16384, 32767,
                                            49152, 65535}));
  return 0;
}

int check_real_ffn_step(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> norm(geometry.residual_width);
  const qw38::internal::FfnScalarParameters parameters{norm.data(), norm.size()};
  status = qw38::internal::prepare_ffn_scalar_parameters(
      weights.layers[0].common, parameters);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::vector<float> residual(geometry.residual_width);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = static_cast<float>(
                          static_cast<int>((index * 37) % 101) - 50) /
                      32.0F;
  }
  std::vector<float> normalized(geometry.residual_width, NAN);
  std::vector<float> gate(geometry.ffn_width, NAN);
  std::vector<float> up(geometry.ffn_width, NAN);
  std::vector<float> activated(geometry.ffn_width, NAN);
  std::vector<float> correction(geometry.residual_width, NAN);
  std::vector<float> output(geometry.residual_width, NAN);
  qw38::internal::FfnStepWorkspace workspace{
      normalized.data(), normalized.size(), gate.data(), gate.size(), up.data(),
      up.size(), activated.data(), activated.size(), correction.data(),
      correction.size()};
  if (mode == "invalid_workspace") --workspace.activated_count;
  if (mode != "valid" && mode != "invalid_workspace") {
    std::cerr << "invalid_argument: unknown real-FFN-step mode\n";
    return 1;
  }
  status = qw38::internal::execute_ffn_step(
      weights.layers[0].common, parameters, residual.data(), residual.size(),
      workspace, output.data(), output.size());
  if (!status.is_ok()) {
    const bool untouched =
        std::all_of(gate.begin(), gate.end(),
                    [](float value) { return std::isnan(value); }) &&
        std::all_of(output.begin(), output.end(),
                    [](float value) { return std::isnan(value); });
    std::cout << "outputs_untouched=" << (untouched ? 1 : 0) << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(normalized) || !finite(gate) || !finite(up) ||
      !finite(activated) || !finite(correction) || !finite(output)) {
    std::cerr << "internal: FFN left a nonfinite workspace value\n";
    return 1;
  }
  write_float_vector("normalized_f32_le_hex",
                     tap_values(normalized, residual_taps(geometry)));
  write_float_vector("gate_f32_le_hex", tap_values(gate, ffn_taps(geometry)));
  write_float_vector("up_f32_le_hex", tap_values(up, ffn_taps(geometry)));
  write_float_vector("activated_f32_le_hex", tap_values(activated, ffn_taps(geometry)));
  write_float_vector("correction_f32_le_hex", tap_values(correction, residual_taps(geometry)));
  write_float_vector("residual_output_f32_le_hex", tap_values(output, residual_taps(geometry)));
  std::cout << "workspace_values="
            << normalized.size() + gate.size() + up.size() + activated.size() +
                   correction.size()
            << '\n';
  return 0;
}

int check_real_attention_step(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  constexpr std::size_t kCapacity = 2;
  std::vector<float> input_norm(geometry.residual_width);
  std::vector<float> query_norm(geometry.attention_head_width);
  std::vector<float> key_norm(geometry.attention_head_width);
  const qw38::internal::AttentionScalarParameters parameters{
      input_norm.data(), input_norm.size(), query_norm.data(),
      query_norm.size(), key_norm.data(), key_norm.size()};
  std::vector<float> ffn_norm(geometry.residual_width);
  const qw38::internal::FfnScalarParameters ffn_parameters{ffn_norm.data(),
                                                           ffn_norm.size()};
  const bool layer_mode = mode == "layer" || mode == "layer_invalid_ffn";
  if (layer_mode) {
    const qw38::internal::AttentionLayerScalarParameters layer_parameters{
        parameters, ffn_parameters};
    status = qw38::internal::prepare_attention_layer_scalar_parameters(
        weights.layers[3], layer_parameters);
  } else {
    status = qw38::internal::prepare_attention_scalar_parameters(
        weights.layers[3], parameters);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  std::vector<float> normalized(geometry.residual_width, NAN);
  std::vector<float> packed_query_gate(
      geometry.attention_packed_query_gate(), NAN);
  std::vector<float> query(geometry.attention_query_width(), NAN);
  std::vector<float> gate(geometry.attention_query_width(), NAN);
  std::vector<float> key(geometry.attention_kv_width(), NAN);
  std::vector<float> value(geometry.attention_kv_width(), NAN);
  std::vector<float> attention_output(geometry.attention_query_width(),
                                      NAN);
  std::vector<float> scores(kCapacity, NAN);
  std::vector<float> mixer_output(geometry.residual_width, NAN);
#ifdef QW38_DIAGNOSTIC_TRACE
  std::vector<float> rope_query(geometry.attention_query_width(), NAN);
  std::vector<float> rope_key(geometry.attention_kv_width(), NAN);
#endif
  std::vector<float> post_mixer(geometry.residual_width, NAN);
  std::vector<float> ffn_normalized(geometry.residual_width, NAN);
  std::vector<float> ffn_gate(geometry.ffn_width, NAN);
  std::vector<float> ffn_up(geometry.ffn_width, NAN);
  std::vector<float> ffn_activated(geometry.ffn_width, NAN);
  std::vector<float> ffn_correction(geometry.residual_width, NAN);
  std::vector<float> key_cache(kCapacity * geometry.attention_kv_width(),
                               NAN);
  std::vector<float> value_cache(
      kCapacity * geometry.attention_kv_width(), NAN);
  std::vector<float> output(geometry.residual_width, NAN);
  std::vector<float> first_output(geometry.residual_width, NAN);
  qw38::internal::AttentionStepWorkspace workspace{
      normalized.data(),
      normalized.size(),
      {packed_query_gate.data(), packed_query_gate.size(), query.data(),
       query.size(), gate.data(), gate.size(), key.data(), key.size(),
       value.data(), value.size()},
      attention_output.data(),
      attention_output.size(),
      scores.data(),
      scores.size(),
      mixer_output.data(),
      mixer_output.size()
#ifdef QW38_DIAGNOSTIC_TRACE
      , rope_query.data(), rope_query.size(), rope_key.data(), rope_key.size()
#endif
  };
  qw38::internal::FfnStepWorkspace ffn_workspace{
      ffn_normalized.data(), ffn_normalized.size(), ffn_gate.data(),
      ffn_gate.size(), ffn_up.data(), ffn_up.size(), ffn_activated.data(),
      ffn_activated.size(), ffn_correction.data(), ffn_correction.size()};
  const qw38::internal::AttentionLayerStateView state{
      key_cache.data(), key_cache.size(), value_cache.data(),
      value_cache.size(), kCapacity};
  if (mode == "invalid_workspace") --workspace.attention_output_count;
  if (mode == "layer_invalid_ffn") --ffn_workspace.activated_count;
  if (mode != "valid" && mode != "invalid_workspace" &&
      mode != "capacity" && !layer_mode) {
    std::cerr << "invalid_argument: unknown real-attention-step mode\n";
    return 1;
  }

  const auto fill_residual = [](std::vector<float>* residual,
                                std::size_t token) {
    for (std::size_t index = 0; index < residual->size(); ++index) {
      (*residual)[index] = static_cast<float>(
                               static_cast<int>((index * 37 + token * 17) %
                                                101) -
                               50) /
                           32.0F;
    }
  };
  std::vector<float> residual(geometry.residual_width);
  fill_residual(&residual, 0);
  const std::size_t first_position = mode == "capacity" ? kCapacity : 0;
  const qw38::internal::AttentionLayerScalarParameters layer_parameters{
      parameters, ffn_parameters};
  const qw38::internal::AttentionLayerWorkspace layer_workspace{
      workspace, ffn_workspace, post_mixer.data(), post_mixer.size()};
  if (layer_mode) {
    status = qw38::internal::execute_attention_layer_step(
        weights.layers[3], layer_parameters, first_position, residual.data(),
        residual.size(), state, layer_workspace, output.data(), output.size());
  } else {
    status = qw38::internal::execute_attention_mixer_step(
        weights.layers[3], parameters, first_position, residual.data(),
        residual.size(), state, workspace, output.data(), output.size());
  }
  if (!status.is_ok()) {
    const bool state_unchanged =
        std::all_of(key_cache.begin(), key_cache.end(),
                    [](float item) { return std::isnan(item); }) &&
        std::all_of(value_cache.begin(), value_cache.end(),
                    [](float item) { return std::isnan(item); });
    const bool output_untouched = std::all_of(
        output.begin(), output.end(),
        [](float item) { return std::isnan(item); });
    std::cout << "state_unchanged=" << (state_unchanged ? 1 : 0) << '\n';
    std::cout << "output_untouched=" << (output_untouched ? 1 : 0) << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  first_output = output;
  fill_residual(&residual, 1);
  if (layer_mode) {
    status = qw38::internal::execute_attention_layer_step(
        weights.layers[3], layer_parameters, 1, residual.data(),
        residual.size(), state, layer_workspace, output.data(), output.size());
  } else {
    status = qw38::internal::execute_attention_mixer_step(
        weights.layers[3], parameters, 1, residual.data(), residual.size(),
        state, workspace, output.data(), output.size());
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float item) { return std::isfinite(item); });
  };
  if (!finite(normalized) || !finite(packed_query_gate) || !finite(query) ||
      !finite(gate) || !finite(key) || !finite(value) ||
      !finite(attention_output) || !finite(mixer_output) ||
      !finite(key_cache) || !finite(value_cache) || !finite(first_output) ||
      !finite(output) ||
      (layer_mode &&
       (!finite(post_mixer) || !finite(ffn_normalized) || !finite(ffn_gate) ||
        !finite(ffn_up) || !finite(ffn_activated) ||
        !finite(ffn_correction)))) {
    std::cerr << "internal: attention left a nonfinite value\n";
    return 1;
  }
  write_float_vector("normalized_f32_le_hex",
                     tap_values(normalized, residual_taps(geometry)));
  write_float_vector("query_f32_le_hex", tap_values(query, attention_head_taps(geometry)));
  write_float_vector("gate_f32_le_hex", tap_values(gate, attention_head_taps(geometry)));
  write_float_vector("key_cache_f32_le_hex", tap_values(key_cache, attention_cache_taps(geometry)));
  write_float_vector("value_cache_f32_le_hex", tap_values(value_cache, attention_cache_taps(geometry)));
  write_float_vector("attention_output_f32_le_hex",
                     tap_values(attention_output, attention_head_taps(geometry)));
  write_float_vector("first_output_f32_le_hex",
                     tap_values(first_output, residual_taps(geometry)));
  write_float_vector("mixer_output_f32_le_hex",
                     tap_values(mixer_output, residual_taps(geometry)));
  write_float_vector("residual_output_f32_le_hex",
                     tap_values(output, residual_taps(geometry)));
  if (layer_mode) {
    write_float_vector("post_mixer_f32_le_hex",
                       tap_values(post_mixer, residual_taps(geometry)));
    write_float_vector("ffn_normalized_f32_le_hex",
                       tap_values(ffn_normalized, residual_taps(geometry)));
    write_float_vector("ffn_gate_f32_le_hex",
                       tap_values(ffn_gate, ffn_taps(geometry)));
    write_float_vector("ffn_up_f32_le_hex",
                       tap_values(ffn_up, ffn_taps(geometry)));
    write_float_vector("ffn_activated_f32_le_hex",
                       tap_values(ffn_activated, ffn_taps(geometry)));
    write_float_vector("ffn_correction_f32_le_hex",
                       tap_values(ffn_correction, residual_taps(geometry)));
  }
  std::cout << "kv_values=" << key_cache.size() + value_cache.size() << '\n';
  return 0;
}

int check_real_model_boundaries(const char* model_path,
                                const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  if (mode != "valid" && mode != "invalid_token" &&
      mode != "invalid_workspace") {
    std::cerr << "invalid_argument: unknown real-model-boundary mode\n";
    return 1;
  }

  std::vector<float> embedding_zero(geometry.residual_width, NAN);
  std::vector<float> embedding(geometry.residual_width, NAN);
  std::vector<float> embedding_last(geometry.residual_width, NAN);
  if (mode == "invalid_token") {
    status = qw38::internal::embed_token(
        weights, geometry.vocabulary, embedding.data(),
        embedding.size());
    const bool untouched = std::all_of(
        embedding.begin(), embedding.end(),
        [](float value) { return std::isnan(value); });
    std::cout << "embedding_untouched=" << (untouched ? 1 : 0) << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  status = qw38::internal::embed_token(weights, 0, embedding_zero.data(),
                                       embedding_zero.size());
  if (status.is_ok()) {
    status = qw38::internal::embed_token(weights, 42, embedding.data(),
                                         embedding.size());
  }
  if (status.is_ok()) {
    status = qw38::internal::embed_token(
        weights, geometry.vocabulary - 1, embedding_last.data(),
        embedding_last.size());
  }

  std::vector<float> output_norm(geometry.residual_width);
  const qw38::internal::OutputScalarParameters parameters{output_norm.data(),
                                                           output_norm.size()};
  if (status.is_ok()) {
    status =
        qw38::internal::prepare_output_scalar_parameters(weights, parameters);
  }
  std::vector<float> normalized(geometry.residual_width, NAN);
  std::vector<float> logits(geometry.vocabulary, NAN);
  qw38::internal::OutputWorkspace workspace{normalized.data(),
                                            normalized.size()};
  if (mode == "invalid_workspace") --workspace.normalized_count;
  if (status.is_ok()) {
    status = qw38::internal::project_logits(
        weights, parameters, embedding.data(), embedding.size(), workspace,
        logits.data(), logits.size());
  }
  if (!status.is_ok()) {
    if (mode == "invalid_workspace") {
      const bool normalized_untouched = std::all_of(
          normalized.begin(), normalized.end(),
          [](float value) { return std::isnan(value); });
      const bool logits_untouched = std::all_of(
          logits.begin(), logits.end(),
          [](float value) { return std::isnan(value); });
      std::cout << "normalized_untouched=" << (normalized_untouched ? 1 : 0)
                << '\n';
      std::cout << "logits_untouched=" << (logits_untouched ? 1 : 0) << '\n';
    }
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(embedding_zero) || !finite(embedding) ||
      !finite(embedding_last) || !finite(normalized) || !finite(logits)) {
    std::cerr << "internal: model boundary left a nonfinite value\n";
    return 1;
  }
  write_float_vector("embedding_0_f32_le_hex",
                     tap_values(embedding_zero, residual_taps(geometry)));
  write_float_vector("embedding_42_f32_le_hex",
                     tap_values(embedding, residual_taps(geometry)));
  write_float_vector("embedding_last_f32_le_hex",
                     tap_values(embedding_last, residual_taps(geometry)));
  write_float_vector("final_normalized_f32_le_hex",
                     tap_values(normalized, residual_taps(geometry)));
  write_float_vector("logits_f32_le_hex", tap_values(logits, {0, 1, 42, 1000, geometry.vocabulary - 1}));
  const auto greedy = std::max_element(logits.begin(), logits.end());
  std::cout << "greedy_token=" << std::distance(logits.begin(), greedy) << '\n';
  std::cout << "logit_count=" << logits.size() << '\n';
  return 0;
}

int check_real_scalar_token(const char* model_path, const std::string& mode) {
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  if (mode != "valid" && mode != "invalid_workspace") {
    std::cerr << "invalid_argument: unknown real-scalar-token mode\n";
    return 1;
  }

  qw38::internal::ScalarModelParameters parameters;
  status = qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  qw38::internal::ScalarSessionState state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(weights.geometry, 1, &state);
  }
  qw38::internal::ScalarWorkspace workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(weights.geometry, 1, &workspace);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::vector<float> logits(geometry.vocabulary, NAN);
  if (mode == "invalid_workspace") workspace.ffn_activated.pop_back();
  status = qw38::internal::execute_scalar_token(
      weights, parameters, 42, &state, &workspace, logits.data(), logits.size());
  if (!status.is_ok()) {
    const bool state_unchanged =
        std::all_of(state.gdn_convolution.begin(),
                    state.gdn_convolution.end(),
                    [](float value) { return value == 0.0F; }) &&
        std::all_of(state.gdn_recurrent.begin(), state.gdn_recurrent.end(),
                    [](float value) { return value == 0.0F; }) &&
        std::all_of(state.attention_key.begin(), state.attention_key.end(),
                    [](float value) { return value == 0.0F; }) &&
        std::all_of(state.attention_value.begin(), state.attention_value.end(),
                    [](float value) { return value == 0.0F; });
    const bool logits_untouched = std::all_of(
        logits.begin(), logits.end(),
        [](float value) { return std::isnan(value); });
    std::cout << "state_unchanged=" << (state_unchanged ? 1 : 0) << '\n';
    std::cout << "logits_untouched=" << (logits_untouched ? 1 : 0) << '\n';
    std::cout << "frontier=" << state.frontier << '\n';
    std::cout << "layers_completed=" << workspace.layers_completed << '\n';
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const auto finite = [](const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  if (!finite(workspace.activation_a) || !finite(workspace.final_normalized) ||
      !finite(logits) || !finite(state.gdn_convolution) ||
      !finite(state.gdn_recurrent) || !finite(state.attention_key) ||
      !finite(state.attention_value)) {
    std::cerr << "internal: scalar token execution left a nonfinite value\n";
    return 1;
  }
  std::size_t gdn_mutated = 0;
  std::size_t attention_mutated = 0;
  for (std::size_t layer = 0; layer < geometry.layer_count;
       ++layer) {
    if (layer % 4 == 3) {
      const auto& view = state.attention[layer];
      const bool changed =
          std::any_of(view.key_cache, view.key_cache + view.key_cache_count,
                      [](float value) { return value != 0.0F; }) ||
          std::any_of(view.value_cache,
                      view.value_cache + view.value_cache_count,
                      [](float value) { return value != 0.0F; });
      if (changed) ++attention_mutated;
    } else {
      const auto& view = state.gdn[layer];
      const bool changed =
          std::any_of(view.convolution,
                      view.convolution + view.convolution_count,
                      [](float value) { return value != 0.0F; }) ||
          std::any_of(view.recurrent, view.recurrent + view.recurrent_count,
                      [](float value) { return value != 0.0F; });
      if (changed) ++gdn_mutated;
    }
  }
  write_float_vector("final_hidden_f32_le_hex",
                     tap_values(workspace.activation_a, residual_taps(geometry)));
  write_float_vector("final_normalized_f32_le_hex",
                     tap_values(workspace.final_normalized, residual_taps(geometry)));
  write_float_vector("logits_f32_le_hex",
                     tap_values(logits, {0, 1, 42, 1000, geometry.vocabulary - 1}));
  write_float_vector(
      "gdn_state_f32_le_hex",
      {state.gdn[0].convolution[3], state.gdn[0].recurrent[0],
       state.gdn[1].convolution[3], state.gdn[1].recurrent[0],
       state.gdn[geometry.layer_count - 2].convolution[3], state.gdn[geometry.layer_count - 2].recurrent[0]});
  write_float_vector(
      "attention_state_f32_le_hex",
      {state.attention[3].key_cache[0], state.attention[3].value_cache[0],
       state.attention[7].key_cache[0], state.attention[7].value_cache[0],
       state.attention[geometry.layer_count - 1].key_cache[0], state.attention[geometry.layer_count - 1].value_cache[0]});
  const auto greedy = std::max_element(logits.begin(), logits.end());
  std::cout << "greedy_token=" << std::distance(logits.begin(), greedy) << '\n';
  std::cout << "gdn_slots_mutated=" << gdn_mutated << '\n';
  std::cout << "attention_slots_mutated=" << attention_mutated << '\n';
  std::cout << "frontier=" << state.frontier << '\n';
  std::cout << "layers_completed=" << workspace.layers_completed << '\n';
  std::cout << "logit_count=" << logits.size() << '\n';
  std::cout << "prepared_values="
            << parameters.input_norms.size() + parameters.ffn_norms.size() +
                   parameters.gdn_convolution.size() +
                   parameters.gdn_folded_a.size() +
                   parameters.gdn_dt_bias.size() +
                   parameters.gdn_recurrent_norm.size() +
                   parameters.attention_query_norm.size() +
                   parameters.attention_key_norm.size() +
                   parameters.output_norm.size()
            << '\n';
  std::cout << "state_values="
            << state.gdn_convolution.size() + state.gdn_recurrent.size() +
                   state.attention_key.size() + state.attention_value.size()
            << '\n';
  std::cout << "workspace_values="
            << workspace.activation_a.size() + workspace.activation_b.size() +
                   workspace.post_mixer.size() +
                   workspace.gdn_normalized.size() +
                   workspace.gdn_packed.size() +
                   workspace.gdn_projected_gate.size() +
                   workspace.gdn_projected_alpha.size() +
                   workspace.gdn_projected_beta.size() +
                   workspace.gdn_convolved.size() +
                   workspace.gdn_query.size() + workspace.gdn_key.size() +
                   workspace.gdn_value_tiled.size() +
                   workspace.gdn_value_grouped.size() +
                   workspace.gdn_gate_grouped.size() +
                   workspace.gdn_alpha_grouped.size() +
                   workspace.gdn_beta_grouped.size() +
                   workspace.gdn_folded_grouped.size() +
                   workspace.gdn_dt_grouped.size() +
                   workspace.gdn_log_decay.size() +
                   workspace.gdn_update_gate.size() +
                   workspace.gdn_recurrent_output.size() +
                   workspace.gdn_gated_grouped.size() +
                   workspace.gdn_gated_tiled.size() +
                   workspace.gdn_mixer_output.size() +
                   workspace.attention_normalized.size() +
                   workspace.attention_packed.size() +
                   workspace.attention_query.size() +
                   workspace.attention_gate.size() +
                   workspace.attention_key.size() +
                   workspace.attention_value.size() +
                   workspace.attention_output.size() +
                   workspace.attention_scores.size() +
                   workspace.attention_mixer_output.size() +
                   workspace.ffn_normalized.size() +
                   workspace.ffn_gate.size() + workspace.ffn_up.size() +
                   workspace.ffn_activated.size() +
                   workspace.ffn_correction.size() +
                   workspace.final_normalized.size()
            << '\n';
  return 0;
}

int check_real_scalar_chunk(const char* model_path, const std::string& mode) {
  if (mode != "valid" && mode != "invalid_token" &&
      mode != "insufficient_capacity" && mode != "invalid_logits") {
    std::cerr << "invalid_argument: unknown real-scalar-chunk mode\n";
    return 1;
  }
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  qw38::internal::ScalarModelParameters parameters;
  if (status.is_ok()) {
    status = qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  }
  const std::array<std::size_t, 2> tokens{42, 3649};
  const std::size_t capacity = mode == "insufficient_capacity" ? 1 : 2;
  qw38::internal::ScalarSessionState chunk_state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(weights.geometry, capacity, &chunk_state);
  }
  qw38::internal::ScalarWorkspace chunk_workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(weights.geometry, capacity, &chunk_workspace);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::vector<float> chunk_logits(tokens.size() * geometry.vocabulary,
                                  NAN);
  std::array<std::size_t, 2> attempted = tokens;
  if (mode == "invalid_token") attempted[1] = geometry.vocabulary;
  std::size_t logits_count = chunk_logits.size();
  if (mode == "invalid_logits") --logits_count;
  status = qw38::internal::execute_scalar_chunk(
      weights, parameters, attempted.data(), attempted.size(), &chunk_state,
      &chunk_workspace, chunk_logits.data(), logits_count);
  if (mode != "valid") {
    const auto zeros = [](const std::vector<float>& values) {
      return std::all_of(values.begin(), values.end(),
                         [](float value) { return value == 0.0F; });
    };
    const bool state_unchanged =
        zeros(chunk_state.gdn_convolution) && zeros(chunk_state.gdn_recurrent) &&
        zeros(chunk_state.attention_key) && zeros(chunk_state.attention_value);
    const bool logits_untouched = std::all_of(
        chunk_logits.begin(), chunk_logits.end(),
        [](float value) { return std::isnan(value); });
    std::cout << "state_unchanged=" << (state_unchanged ? 1 : 0)
              << "\nlogits_untouched=" << (logits_untouched ? 1 : 0)
              << "\nfrontier=" << chunk_state.frontier
              << "\nlayers_completed=" << chunk_workspace.layers_completed
              << '\n';
    if (status.is_ok()) {
      std::cerr << "internal: invalid scalar chunk unexpectedly succeeded\n";
      return 1;
    }
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  qw38::internal::ScalarSessionState token_state;
  status = qw38::internal::create_scalar_session_state(weights.geometry, 2, &token_state);
  qw38::internal::ScalarWorkspace token_workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(weights.geometry, 2, &token_workspace);
  }
  std::vector<float> token_logits(chunk_logits.size(), NAN);
  for (std::size_t index = 0; status.is_ok() && index < tokens.size(); ++index) {
    status = qw38::internal::execute_scalar_token(
        weights, parameters, tokens[index], &token_state, &token_workspace,
        token_logits.data() + index * geometry.vocabulary,
        geometry.vocabulary);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  const bool equivalent =
      chunk_logits == token_logits &&
      chunk_state.gdn_convolution == token_state.gdn_convolution &&
      chunk_state.gdn_recurrent == token_state.gdn_recurrent &&
      chunk_state.attention_key == token_state.attention_key &&
      chunk_state.attention_value == token_state.attention_value &&
      chunk_state.frontier == token_state.frontier &&
      chunk_workspace.layers_completed == token_workspace.layers_completed;
  if (!equivalent) {
    std::cerr << "internal: scalar chunk differs from repeated token execution\n";
    return 1;
  }
  const auto taps = [&geometry](const std::vector<float>& values, std::size_t row) {
    const std::size_t base = row * geometry.vocabulary;
    return std::vector<float>{values[base], values[base + 1],
                              values[base + 42], values[base + 1000],
                              values[base + geometry.vocabulary - 1]};
  };
  write_float_vector("token0_logits_f32_le_hex", taps(chunk_logits, 0));
  write_float_vector("token1_logits_f32_le_hex", taps(chunk_logits, 1));
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    const auto begin = chunk_logits.begin() +
                       static_cast<std::ptrdiff_t>(
                           row * geometry.vocabulary);
    const auto greedy = std::max_element(
        begin, begin + geometry.vocabulary);
    std::cout << "token" << row << "_greedy="
              << std::distance(begin, greedy) << '\n';
  }
  std::cout << "equivalent=1\ntoken_count=2\nfrontier=" << chunk_state.frontier
            << "\nlayers_completed=" << chunk_workspace.layers_completed
            << "\nlogit_rows=2\nlogit_stride="
            << geometry.vocabulary << '\n';
  return 0;
}

int dump_real_scalar_logits(const char* model_path, const char* output_path) {
  qw38::internal::ModelInfo info;
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelWeights weights;
  qw38::Status status =
      load_admitted_weights(model_path, &info, &mapping, &weights);
  const qw38::internal::ModelGeometry& geometry = weights.geometry;
  qw38::internal::ScalarModelParameters parameters;
  if (status.is_ok()) {
    status = qw38::internal::prepare_scalar_model_parameters(weights, &parameters);
  }
  qw38::internal::ScalarSessionState state;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_session_state(weights.geometry, 2, &state);
  }
  qw38::internal::ScalarWorkspace workspace;
  if (status.is_ok()) {
    status = qw38::internal::create_scalar_workspace(weights.geometry, 2, &workspace);
  }
  const std::array<std::size_t, 2> tokens{42, 3649};
  std::vector<float> logits(tokens.size() * geometry.vocabulary);
  if (status.is_ok()) {
    status = qw38::internal::execute_scalar_chunk(
        weights, parameters, tokens.data(), tokens.size(), &state, &workspace,
        logits.data(), logits.size());
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }

  const std::string temporary =
      std::string(output_path) + ".tmp." + std::to_string(getpid());
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(logits.data()),
               static_cast<std::streamsize>(logits.size() * sizeof(float)));
  output.close();
  if (!output || std::rename(temporary.c_str(), output_path) != 0) {
    std::remove(temporary.c_str());
    std::cerr << "cannot commit scalar logits output\n";
    return 1;
  }
  std::cout << "schema_version=1\ntoken_count=" << tokens.size()
            << "\nvocabulary_size=" << geometry.vocabulary
            << "\ntoken_0=" << tokens[0] << "\ntoken_1=" << tokens[1];
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    const float* begin = logits.data() + row * geometry.vocabulary;
    const float* greedy =
        std::max_element(begin, begin + geometry.vocabulary);
    std::cout << "\ngreedy_" << row << '=' << greedy - begin;
  }
  std::cout << "\nlogits_bytes=" << logits.size() * sizeof(float) << '\n';
  return 0;
}

float fixture_query(std::size_t token, std::size_t head, std::size_t lane) {
  const int numerator =
      static_cast<int>((token * 11 + head * 7 + lane * 3) % 19) - 9;
  return static_cast<float>(numerator) / 8.0F;
}

float fixture_key(std::size_t token, std::size_t head, std::size_t lane) {
  const int numerator =
      static_cast<int>((token * 13 + head * 5 + lane * 7) % 23) - 11;
  return static_cast<float>(numerator) / 8.0F;
}

float fixture_value(std::size_t token, std::size_t head, std::size_t lane) {
  const int numerator =
      static_cast<int>((token * 17 + head * 3 + lane * 5) % 29) - 14;
  return static_cast<float>(numerator) / 8.0F;
}

int check_gdn_recurrent(const std::string& chunking) {
  constexpr qw38::internal::GdnShape kShape{2, 6, 3, 2};
  constexpr std::size_t kTokens = 5;
  std::vector<std::size_t> chunks;
  if (chunking == "whole") {
    chunks = {5};
  } else if (chunking == "mixed") {
    chunks = {2, 1, 2};
  } else if (chunking == "token") {
    chunks = {1, 1, 1, 1, 1};
  } else {
    std::cerr << "invalid_argument: unknown GDN chunking\n";
    return 1;
  }
  std::vector<float> state(kShape.value_heads * kShape.key_width *
                           kShape.value_width);
  for (std::size_t head = 0; head < kShape.value_heads; ++head) {
    for (std::size_t key_lane = 0; key_lane < kShape.key_width; ++key_lane) {
      for (std::size_t value_lane = 0; value_lane < kShape.value_width;
           ++value_lane) {
        const int numerator = static_cast<int>(
                                  (head * 17 + key_lane * 5 + value_lane * 3) %
                                  13) -
                              6;
        state[head * kShape.key_width * kShape.value_width +
              key_lane * kShape.value_width + value_lane] =
            static_cast<float>(numerator) / 32.0F;
      }
    }
  }
  std::vector<float> all_output;
  std::vector<float> all_log_decay;
  std::vector<float> all_beta;
  std::size_t token = 0;
  for (std::size_t chunk : chunks) {
    for (std::size_t offset = 0; offset < chunk; ++offset, ++token) {
      std::array<float, 6> query{};
      std::array<float, 6> key{};
      std::array<float, 12> value{};
      std::array<float, 6> a{};
      std::array<float, 6> b{};
      std::array<float, 6> a_log{};
      std::array<float, 6> dt_bias{};
      std::array<float, 12> output{};
      std::array<float, 6> log_decay{};
      std::array<float, 6> beta{};
      for (std::size_t head = 0; head < kShape.key_heads; ++head) {
        for (std::size_t lane = 0; lane < kShape.key_width; ++lane) {
          query[head * kShape.key_width + lane] =
              fixture_query(token, head, lane);
          key[head * kShape.key_width + lane] =
              fixture_key(token, head, lane);
        }
      }
      for (std::size_t head = 0; head < kShape.value_heads; ++head) {
        for (std::size_t lane = 0; lane < kShape.value_width; ++lane) {
          value[head * kShape.value_width + lane] =
              fixture_value(token, head, lane);
        }
        a[head] = static_cast<float>(
                      static_cast<int>((token * 5 + head * 3) % 13) - 6) /
                  8.0F;
        b[head] = static_cast<float>(
                      static_cast<int>((token * 7 + head * 2) % 11) - 5) /
                  8.0F;
        a_log[head] = std::log(0.25F * static_cast<float>(head + 1));
        dt_bias[head] =
            static_cast<float>(static_cast<int>(head) - 2) / 8.0F;
      }
      const qw38::Status status = qw38::internal::gdn_recurrent_step(
          kShape, query.data(), query.size(), key.data(), key.size(),
          value.data(), value.size(), a.data(), b.data(), a_log.data(),
          dt_bias.data(), a.size(), state.data(), state.size(), output.data(),
          output.size(), log_decay.data(), beta.data());
      if (!status.is_ok()) {
        std::cerr << qw38::status_code_name(status.code()) << ": "
                  << status.message() << '\n';
        return 1;
      }
      all_output.insert(all_output.end(), output.begin(), output.end());
      all_log_decay.insert(all_log_decay.end(), log_decay.begin(),
                           log_decay.end());
      all_beta.insert(all_beta.end(), beta.begin(), beta.end());
    }
  }
  if (token != kTokens) {
    std::cerr << "internal: GDN fixture chunking has the wrong token count\n";
    return 1;
  }
  write_float_vector("output_f32_le_hex", all_output);
  write_float_vector("final_state_f32_le_hex", state);
  write_float_vector("log_decay_f32_le_hex", all_log_decay);
  write_float_vector("beta_f32_le_hex", all_beta);
  return 0;
}

int check_gdn_convolution(const std::string& chunking) {
  constexpr std::size_t kChannels = 3;
  constexpr std::size_t kWidth = 4;
  std::vector<std::size_t> chunks;
  if (chunking == "whole") {
    chunks = {6};
  } else if (chunking == "mixed") {
    chunks = {1, 2, 3};
  } else if (chunking == "token") {
    chunks = {1, 1, 1, 1, 1, 1};
  } else {
    std::cerr << "invalid_argument: unknown GDN chunking\n";
    return 1;
  }
  std::array<float, kChannels * kWidth> weights{};
  std::array<float, kChannels * kWidth> state{};
  for (std::size_t channel = 0; channel < kChannels; ++channel) {
    for (std::size_t index = 0; index < kWidth; ++index) {
      const int numerator =
          static_cast<int>((channel * 11 + index * 3) % 13) - 6;
      weights[channel * kWidth + index] =
          static_cast<float>(numerator) / 8.0F;
    }
  }
  std::vector<float> all_output;
  std::size_t token = 0;
  for (std::size_t chunk : chunks) {
    for (std::size_t offset = 0; offset < chunk; ++offset, ++token) {
      std::array<float, kChannels> input{};
      std::array<float, kChannels> output{};
      for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const int numerator =
            static_cast<int>((token * 7 + channel * 5) % 17) - 8;
        input[channel] = static_cast<float>(numerator) / 8.0F;
      }
      const qw38::Status status = qw38::internal::causal_depthwise_conv_step(
          kChannels, kWidth, input.data(), input.size(), weights.data(),
          weights.size(), state.data(), state.size(), output.data(),
          output.size());
      if (!status.is_ok()) {
        std::cerr << qw38::status_code_name(status.code()) << ": "
                  << status.message() << '\n';
        return 1;
      }
      all_output.insert(all_output.end(), output.begin(), output.end());
    }
  }
  write_float_vector("output_f32_le_hex", all_output);
  write_float_vector("final_state_f32_le_hex", {state.begin(), state.end()});
  return 0;
}

int check_gdn(const std::string& component, const std::string& chunking) {
  if (component == "recurrent") return check_gdn_recurrent(chunking);
  if (component == "convolution") return check_gdn_convolution(chunking);
  if (component == "invalid_shape") {
    float value = 0.0F;
    const qw38::internal::GdnShape shape{2, 5, 3, 2};
    const qw38::Status status = qw38::internal::gdn_recurrent_step(
        shape, &value, 1, &value, 1, &value, 1, &value, &value, &value,
        &value, 1, &value, 1, &value, 1, &value, &value);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  if (component == "invalid_buffers") {
    float value = 0.0F;
    const qw38::internal::GdnShape shape{1, 1, 1, 1};
    const qw38::Status status = qw38::internal::gdn_recurrent_step(
        shape, &value, 0, &value, 1, &value, 1, &value, &value, &value,
        &value, 1, &value, 1, &value, 1, &value, &value);
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return status.is_ok() ? 0 : 1;
  }
  std::cerr << "invalid_argument: GDN component must be recurrent or convolution\n";
  return 1;
}

float attention_fixture_value(int layer, std::size_t token, std::size_t head,
                              std::size_t lane, int salt) {
  const int value =
      (layer * 3 + static_cast<int>(token * 11 + head * 7 + lane * 5) + salt) %
          31 -
      15;
  return static_cast<float>(value) / 8.0F;
}

bool supported_attention_layer(int layer) {
  return layer == 3 || layer == 7 || layer == 63;
}

int check_attention(int layer) {
  if (!supported_attention_layer(layer)) {
    std::cerr << "invalid_argument: attention fixture layer must be 3, 7, or 63\n";
    return 1;
  }
  constexpr qw38::internal::AttentionShape kShape{6, 2, 8, 4, 4};
  constexpr std::size_t kQueryValues = kShape.query_heads * kShape.head_width;
  constexpr std::size_t kKvValues = kShape.kv_heads * kShape.head_width;
  constexpr std::size_t kCacheValues = kShape.capacity * kKvValues;
  std::array<float, kShape.head_width> query_weight{};
  std::array<float, kShape.head_width> key_weight{};
  for (std::size_t lane = 0; lane < kShape.head_width; ++lane) {
    query_weight[lane] =
        static_cast<float>(static_cast<int>(lane) - 3) / 64.0F;
    key_weight[lane] =
        static_cast<float>(3 - static_cast<int>(lane)) / 64.0F;
  }
  std::array<float, kCacheValues> key_cache{};
  std::array<float, kCacheValues> value_cache{};
  for (std::size_t index = 0; index < kCacheValues; ++index) {
    key_cache[index] = 1000.0F + static_cast<float>(index);
    value_cache[index] = -1000.0F - static_cast<float>(index);
  }
  std::vector<float> all_output;
  for (std::size_t token = 0; token < kShape.capacity; ++token) {
    std::array<float, kQueryValues> query{};
    std::array<float, kQueryValues> gate{};
    std::array<float, kKvValues> key{};
    std::array<float, kKvValues> value{};
    std::array<float, kShape.capacity> scores{};
    std::array<float, kQueryValues> output{};
    for (std::size_t head = 0; head < kShape.query_heads; ++head) {
      for (std::size_t lane = 0; lane < kShape.head_width; ++lane) {
        const std::size_t index = head * kShape.head_width + lane;
        query[index] = attention_fixture_value(layer, token, head, lane, 1);
        gate[index] = attention_fixture_value(layer, token, head, lane, 9);
      }
    }
    for (std::size_t head = 0; head < kShape.kv_heads; ++head) {
      for (std::size_t lane = 0; lane < kShape.head_width; ++lane) {
        const std::size_t index = head * kShape.head_width + lane;
        key[index] = attention_fixture_value(layer, token, head, lane, 4);
        value[index] = attention_fixture_value(layer, token, head, lane, 13);
      }
    }
    const qw38::Status status = qw38::internal::attention_decode_step(
        kShape, token, query.data(), query.size(), key.data(), key.size(),
        value.data(), value.size(), query_weight.data(), key_weight.data(),
        gate.data(), gate.size(), key_cache.data(), key_cache.size(),
        value_cache.data(), value_cache.size(), scores.data(), scores.size(),
        output.data(), output.size());
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    all_output.insert(all_output.end(), output.begin(), output.end());
  }
  write_float_vector("output_f32_le_hex", all_output);
  write_float_vector("key_cache_f32_le_hex",
                     {key_cache.begin(), key_cache.end()});
  write_float_vector("value_cache_f32_le_hex",
                     {value_cache.begin(), value_cache.end()});
  return 0;
}

int check_ffn(int layer) {
  if (!supported_attention_layer(layer)) {
    std::cerr << "invalid_argument: FFN fixture layer must be 3, 7, or 63\n";
    return 1;
  }
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kIntermediate = 6;
  std::array<float, kHidden> input{};
  std::array<float, kIntermediate * kHidden> gate_weights{};
  std::array<float, kIntermediate * kHidden> up_weights{};
  std::array<float, kHidden * kIntermediate> down_weights{};
  for (std::size_t lane = 0; lane < kHidden; ++lane) {
    input[lane] = attention_fixture_value(layer, 0, 0, lane, 2);
  }
  for (std::size_t row = 0; row < kIntermediate; ++row) {
    for (std::size_t column = 0; column < kHidden; ++column) {
      gate_weights[row * kHidden + column] =
          attention_fixture_value(layer, row, 0, column, 3) / 4.0F;
      up_weights[row * kHidden + column] =
          attention_fixture_value(layer, row, 0, column, 7) / 4.0F;
    }
  }
  for (std::size_t row = 0; row < kHidden; ++row) {
    for (std::size_t column = 0; column < kIntermediate; ++column) {
      down_weights[row * kIntermediate + column] =
          attention_fixture_value(layer, row, 0, column, 11) / 4.0F;
    }
  }
  std::array<float, kIntermediate> gate{};
  std::array<float, kIntermediate> up{};
  std::array<float, kIntermediate> activated{};
  std::array<float, kHidden> output{};
  const qw38::Status status = qw38::internal::swiglu_ffn(
      input.data(), input.size(), gate_weights.data(), up_weights.data(),
      kIntermediate, down_weights.data(), gate.data(), up.data(),
      activated.data(), output.data());
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  write_float_vector("gate_f32_le_hex", {gate.begin(), gate.end()});
  write_float_vector("up_f32_le_hex", {up.begin(), up.end()});
  write_float_vector("activated_f32_le_hex",
                     {activated.begin(), activated.end()});
  write_float_vector("output_f32_le_hex", {output.begin(), output.end()});
  return 0;
}

int check_matvec(const std::string& kind, const char* columns_text,
                 const char* rows_text, const std::string& payload_hex,
                 const std::string& activation_hex) {
  std::size_t columns = 0;
  std::size_t rows = 0;
  std::uint32_t type = 0;
  if (kind == "f32") {
    type = 0;
  } else if (kind == "q8_0") {
    type = 8;
  } else if (kind == "q4_k") {
    type = 12;
  } else if (kind == "q5_k") {
    type = 13;
  } else if (kind == "q6_k") {
    type = 14;
  } else {
    std::cerr << "invalid_argument: unknown matrix tensor type\n";
    return 1;
  }
  std::vector<std::uint8_t> payload;
  std::vector<float> activation;
  const bool payload_ok = load_bytes_argument(payload_hex, &payload);
  bool activation_ok = false;
  if (!activation_hex.empty() && activation_hex.front() == '@') {
    std::vector<std::uint8_t> raw;
    activation_ok = load_bytes_argument(activation_hex, &raw) &&
                    raw.size() % 4 == 0;
    if (activation_ok) {
      activation.resize(raw.size() / 4);
      std::memcpy(activation.data(), raw.data(), raw.size());
    }
  } else {
    activation_ok = parse_float_hex(activation_hex, &activation);
  }
  if (!parse_size(columns_text, &columns) || !parse_size(rows_text, &rows) ||
      !payload_ok || !activation_ok) {
    std::cerr << "invalid_argument: malformed matrix dimensions or hex\n";
    return 1;
  }
  qw38::internal::TensorView view;
  qw38::Status status = qw38::internal::make_tensor_view(
      payload.data(), payload.size(), type, columns, rows, &view);
  std::vector<float> output(rows);
  std::vector<float> decoded(columns);
  if (status.is_ok()) {
    status = qw38::internal::tensor_matvec(
        view, activation.data(), activation.size(), output.data(), output.size());
  }
  if (status.is_ok()) {
    status = qw38::internal::tensor_row_decode(
        view, 0, decoded.data(), decoded.size());
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "columns=" << view.columns << "\nrows=" << view.rows
            << "\nrow_bytes=" << view.row_bytes << '\n';
  write_float_vector("output_f32_le_hex", output);
  write_float_vector("row0_f32_le_hex", decoded);
  return 0;
}

int check_tensor_row(const char* model_path, const std::string& name,
                     const char* row_text) {
  std::size_t row = 0;
  if (!parse_size(row_text, &row)) {
    std::cerr << "invalid_argument: malformed tensor row index\n";
    return 1;
  }
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) {
    qw38::internal::ModelGeometry geometry;
    status = qw38::internal::admit_pinned_geometry(&info, &geometry);
  }
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::TensorView view;
  if (status.is_ok()) {
    status = qw38::internal::bind_tensor_view(info, mapping, name, &view);
  }
  std::vector<float> activation;
  if (status.is_ok()) {
    activation.resize(view.columns);
    for (std::size_t index = 0; index < activation.size(); ++index) {
      const int numerator = static_cast<int>((index * 37) % 101) - 50;
      activation[index] = static_cast<float>(numerator) / 32.0F;
    }
  }
  float dot = 0.0F;
  if (status.is_ok()) {
    status = qw38::internal::tensor_row_dot(
        view, row, activation.data(), activation.size(), &dot);
  }
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::cout << "dtype=" << qw38::internal::ggml_type_name(view.type)
            << "\ncolumns=" << view.columns << "\nrows=" << view.rows
            << "\nrow_bytes=" << view.row_bytes << "\ndot_f32_le_hex=";
  write_float_hex(dot);
  std::cout << '\n';
  return 0;
}

int tokenize_hex(const char* model_path, const std::string& hex) {
  if (hex.size() % 2 != 0) {
    std::cerr << "invalid_argument: input hex has odd length\n";
    return 1;
  }
  std::string input;
  input.reserve(hex.size() / 2);
  for (std::size_t index = 0; index < hex.size(); index += 2) {
    const int high = hex_value(hex[index]);
    const int low = hex_value(hex[index + 1]);
    if (high < 0 || low < 0) {
      std::cerr << "invalid_argument: input is not hexadecimal\n";
      return 1;
    }
    input.push_back(static_cast<char>((high << 4) | low));
  }
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) {
    qw38::internal::ModelGeometry geometry;
    status = qw38::internal::admit_pinned_geometry(&info, &geometry);
  }
  qw38::internal::Tokenizer tokenizer;
  if (status.is_ok()) status = tokenizer.build(info);
  std::vector<std::uint32_t> ids;
  if (status.is_ok()) status = tokenizer.encode(input, &ids);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << ids[index];
  }
  std::cout << '\n';
  return 0;
}

int render_template_case(const std::string& name) {
  using qw38::internal::Message;
  using qw38::internal::MessageRole;
  if (name == "user_turn_no_thinking" || name == "user_turn_thinking" ||
      name == "empty_user_turn") {
    std::string rendered;
    const std::string content = name == "empty_user_turn" ? "  \n" : " Next ";
    const qw38::Status status = qw38::internal::render_user_turn(
        content, name == "user_turn_thinking", &rendered);
    if (!status.is_ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    std::cout << rendered;
    return 0;
  }
  if (name == "followup_tool_results" || name == "followup_mixed_invalid") {
    std::vector<Message> messages = {
        {MessageRole::kTool, "18 C"}, {MessageRole::kTool, "sunny"}};
    if (name == "followup_mixed_invalid") {
      messages.push_back({MessageRole::kUser, "Thanks"});
    }
    std::string rendered;
    const qw38::Status status =
        qw38::internal::render_followup(messages, false, &rendered);
    if (!status.is_ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    std::cout << rendered;
    return 0;
  }
  qw38::internal::TemplateInput input;
  if (name == "user_no_thinking") {
    input.messages = {{MessageRole::kUser, "Hello"}};
    input.options.enable_thinking = false;
  } else if (name == "user_default_xhigh") {
    input.messages = {{MessageRole::kUser, "Explain quartz."}};
  } else if (name == "system_low") {
    input.messages = {{MessageRole::kSystem, "Be concise."},
                      {MessageRole::kUser, "Why a watch?"}};
    input.options.reasoning_effort = "low";
  } else if (name == "assistant_history") {
    Message assistant{MessageRole::kAssistant, "4"};
    assistant.reasoning_content = "Two plus two is four.";
    input.messages = {{MessageRole::kUser, "2+2?"}, assistant,
                      {MessageRole::kUser, "And plus 3?"}};
    input.options.enable_thinking = false;
  } else if (name == "tools_and_result") {
    Message assistant{MessageRole::kAssistant, ""};
    assistant.reasoning_content = "I should check.";
    assistant.tool_calls = {
        {"weather", {{"city", "Bern"}, {"unit", "C"}}}};
    input.messages = {{MessageRole::kUser, "Weather in Bern?"}, assistant,
                      {MessageRole::kTool, "{\"temperature\":18}"}};
    input.canonical_tool_json = {
        R"({"function": {"description": "Get current weather", "name": "weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"], "type": "object"}}, "type": "function"})"};
  } else if (name == "no_messages") {
  } else if (name == "late_system") {
    input.messages = {{MessageRole::kUser, "Hi"},
                      {MessageRole::kSystem, "Too late"}};
  } else if (name == "invalid_reasoning_effort") {
    input.messages = {{MessageRole::kUser, "Hi"}};
    input.options.reasoning_effort = "maximum";
  } else if (name == "image_rejected_in_v1") {
    Message message{MessageRole::kUser, ""};
    message.has_unsupported_content = true;
    input.messages = {message};
  } else if (name == "leading_developer_mapped_to_system") {
    input.messages = {{MessageRole::kDeveloper, "Follow API policy."},
                      {MessageRole::kUser, "Hi"}};
    input.options.enable_thinking = false;
  } else {
    std::cerr << "unknown template fixture\n";
    return 2;
  }
  std::string rendered;
  const qw38::Status status = qw38::internal::render_chat(input, &rendered);
  if (!status.is_ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }
  std::cout << rendered;
  return 0;
}

int measure_host_decode(const char* model_path, std::size_t prompt_tokens,
                        std::size_t decode_tokens) {
  if (prompt_tokens == 0 || decode_tokens == 0) {
    std::cerr << "prompt and decode token counts must be positive\n";
    return 1;
  }
  qw38::Engine engine;
  const auto load_begin = std::chrono::steady_clock::now();
  qw38::Status status = qw38::Engine::open(model_path, 512, &engine);
  const auto load_end = std::chrono::steady_clock::now();
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::unique_ptr<qw38::Session> session;
  status = engine.create_session(&session);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  std::string prompt;
  prompt.reserve(prompt_tokens * 6);
  for (std::size_t index = 0; index < prompt_tokens; ++index) {
    prompt += "hello ";
  }
  std::vector<qw38::Token> encoded;
  status = engine.encode(prompt, &encoded);
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  if (encoded.size() > prompt_tokens) encoded.resize(prompt_tokens);
  while (encoded.size() < prompt_tokens) {
    encoded.push_back(encoded.empty() ? 1U : encoded.back());
  }
  const auto prefill_begin = std::chrono::steady_clock::now();
  status = session->sync(encoded);
  const auto prefill_end = std::chrono::steady_clock::now();
  if (!status.is_ok()) {
    std::cerr << qw38::status_code_name(status.code()) << ": "
              << status.message() << '\n';
    return 1;
  }
  qw38::SamplerConfig sampler;
  sampler.temperature = 0.0F;
  const auto decode_begin = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < decode_tokens; ++index) {
    qw38::Token token = 0;
    status = session->sample(sampler, &token);
    if (status.is_ok()) status = session->eval(token);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
  }
  const auto decode_end = std::chrono::steady_clock::now();
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  const std::uint64_t rss_bytes = static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  const std::uint64_t rss_bytes =
      static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
  const double load_seconds =
      std::chrono::duration<double>(load_end - load_begin).count();
  const double ttft_seconds =
      std::chrono::duration<double>(prefill_end - prefill_begin).count();
  const double decode_seconds =
      std::chrono::duration<double>(decode_end - decode_begin).count();
  std::cout << "load_seconds=" << load_seconds << '\n';
  std::cout << "ttft_seconds=" << ttft_seconds << '\n';
  std::cout << "prompt_tokens=" << prompt_tokens << '\n';
  std::cout << "decode_tokens=" << decode_tokens << '\n';
  std::cout << "decode_seconds=" << decode_seconds << '\n';
  std::cout << "decode_tok_s=" << (decode_seconds > 0.0
                                       ? decode_tokens / decode_seconds
                                       : 0.0)
            << '\n';
  std::cout << "rss_bytes=" << rss_bytes << '\n';
  std::cout << "threads=" << qw38::internal::matvec_worker_count() << '\n';
  std::cout << "sha256_backend=" << qw38::internal::sha256_backend_name()
            << '\n';
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
#ifdef QW38_DIAGNOSTIC_TRACE
  if (argc == 2 && std::string(argv[1]) == "--help") {
    return run_high_level(argc, argv, true);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-trace-filter") {
    return check_trace_filter(argv[2], argv[3]);
  }
  if (argc == 7 && std::string(argv[1]) == "--capture-real-scalar-trace") {
    return capture_real_scalar_trace(argv[2], argv[3], argv[4], argv[5],
                                     argv[6]);
  }
  if (argc == 6 && std::string(argv[1]) == "--capture-real-scalar-bundle") {
    return capture_real_scalar_bundle(argv[2], argv[3], argv[4], argv[5]);
  }
#endif
  if (argc == 2 && std::string(argv[1]) == "--help") {
    return run_high_level(argc, argv, false);
  }
  if (argc > 1 && argv[1][0] != '-') {
    return run_high_level(argc, argv,
#ifdef QW38_DIAGNOSTIC_TRACE
                          true
#else
                          false
#endif
    );
  }
  if (argc == 2 && std::string(argv[1]) == "--build-info") {
    std::cout << "brand=" << kBrand << '\n';
    std::cout << "cxx=17\n";
    std::cout << "cuda_target=sm_120\n";
    std::cout << "model_revision=" << kModelRevision << '\n';
    std::cout << "model_sha256=" << kModelSha256 << '\n';
    return 0;
  }
  if (argc >= 3 && std::string(argv[1]) == "--measure-host-decode") {
    std::size_t prompt_tokens = 64;
    std::size_t decode_tokens = 16;
    if (argc >= 4) prompt_tokens = static_cast<std::size_t>(std::atoi(argv[3]));
    if (argc >= 5) decode_tokens = static_cast<std::size_t>(std::atoi(argv[4]));
    return measure_host_decode(argv[2], prompt_tokens, decode_tokens);
  }
  if (argc == 3 && std::string(argv[1]) == "--inspect-gguf") {
    qw38::internal::ModelInfo info;
    const qw38::Status status = qw38::internal::inspect_gguf(argv[2], &info);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "version=" << info.gguf_version << '\n';
    std::cout << "architecture=" << info.architecture << '\n';
    std::cout << "name=" << info.name << '\n';
    std::cout << "metadata=" << info.metadata_count << '\n';
    std::cout << "tensors=" << info.tensors.size() << '\n';
    std::cout << "data_offset=" << info.data_offset << '\n';
    std::cout << "block_count=" << info.block_count << '\n';
    std::cout << "context_length=" << info.context_length << '\n';
    std::cout << "embedding_length=" << info.embedding_length << '\n';
    std::cout << "query_heads=" << info.query_heads << '\n';
    std::cout << "kv_heads=" << info.kv_heads << '\n';
    std::cout << "rope_dimensions=" << info.rope_dimensions << '\n';
    std::cout << "tokenizer_model=" << info.tokenizer_model << '\n';
    std::cout << "tokenizer_tokens=" << info.tokenizer_tokens.size() << '\n';
    std::cout << "tokenizer_merges=" << info.tokenizer_merges.size() << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--sha256") {
    std::string digest;
    const qw38::Status status = qw38::internal::sha256_file(argv[2], &digest);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << digest << '\n';
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--sha256-backend") {
    std::cout << qw38::internal::sha256_backend_name() << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--verify-model") {
    qw38::Engine engine;
    qw38::Status status = qw38::Engine::open(argv[2], &engine);
    std::string identity;
    if (status.is_ok()) status = engine.admitted_model(&identity);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "verified=" << identity << '\n';
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--check-contract") {
    qw38::internal::ModelInfo info;
    qw38::Status status = qw38::internal::inspect_gguf(argv[2], &info);
    qw38::internal::ModelGeometry geometry;
    if (status.is_ok()) status = qw38::internal::admit_pinned_geometry(&info, &geometry);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "contract="
              << (geometry.identity == qw38::internal::kGeometryQwen35_2B
                      ? "qwen3.5-2b-q4_k_m"
                      : "qwen3.8-27b-q4_k_m")
              << '\n';
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "--inventory-gguf") {
    return write_inventory(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--tokenize-hex") {
    return tokenize_hex(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--render-template-case") {
    return render_template_case(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-quant") {
    return check_quant(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-gdn") {
    return check_gdn(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-conversion") {
    return check_conversion(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-weight-binding") {
    return check_weight_binding(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-projection-layout") {
    return check_projection_layout(argv[2]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-mixer-projections") {
    return check_mixer_projections(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-gdn-step") {
    return check_real_gdn_step(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-ffn-step") {
    return check_real_ffn_step(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-attention-step") {
    return check_real_attention_step(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-model-boundaries") {
    return check_real_model_boundaries(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-scalar-token") {
    return check_real_scalar_token(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--check-real-scalar-chunk") {
    return check_real_scalar_chunk(argv[2], argv[3]);
  }
  if (argc == 4 && std::string(argv[1]) == "--dump-real-scalar-logits") {
    return dump_real_scalar_logits(argv[2], argv[3]);
  }
  if (argc == 3 && std::string(argv[1]) == "--check-attention") {
    return check_attention(std::atoi(argv[2]));
  }
  if (argc == 3 && std::string(argv[1]) == "--check-ffn") {
    return check_ffn(std::atoi(argv[2]));
  }
  if (argc == 7 && std::string(argv[1]) == "--check-matvec") {
    return check_matvec(argv[2], argv[3], argv[4], argv[5], argv[6]);
  }
  if (argc == 5 && std::string(argv[1]) == "--check-tensor-row") {
    return check_tensor_row(argv[2], argv[3], argv[4]);
  }
  std::cout << kBrand << '\n';
  std::cerr << "qw38-eval: use --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "--verify-model PATH, --check-contract PATH, or "
               "--inventory-gguf PATH OUTPUT, --tokenize-hex PATH HEX, or "
               "--render-template-case NAME, --check-quant KIND HEX, or "
               "--check-gdn COMPONENT CHUNKING, --check-attention LAYER, or "
               "--check-conversion COMPONENT, "
               "--check-weight-binding MODEL MODE, "
               "--check-projection-layout COMPONENT, "
               "--check-mixer-projections MODEL MODE, "
               "--check-real-gdn-step MODEL MODE, "
               "--check-real-ffn-step MODEL MODE, "
               "--check-real-attention-step MODEL MODE, "
               "--check-real-model-boundaries MODEL MODE, "
               "--check-real-scalar-token MODEL MODE, "
               "--check-real-scalar-chunk MODEL MODE, "
               "--dump-real-scalar-logits MODEL OUTPUT, "
               "--check-ffn LAYER, --check-matvec KIND COLUMNS ROWS PAYLOAD "
               "ACTIVATION, or --check-tensor-row MODEL NAME ROW\n";
  return 2;
}
