#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

struct TraceRecord {
  std::string name;
  std::array<std::int64_t, GGML_MAX_DIMS> shape{};
  std::size_t position = 0;
  std::size_t count = 0;
  std::size_t offset = 0;
};

struct TraceCapture {
  std::ofstream output;
  std::vector<std::uint8_t> staging;
  std::vector<TraceRecord> records;
  std::size_t position = 0;
  std::size_t bytes = 0;
  bool failed = false;
};

bool selected_trace_name(const char* name) {
  static const std::set<std::string> globals = {
      "model.input_embed", "result_norm", "result_output"};
  static const std::set<std::string> layers = {
      "attn_norm",       "attn_residual",       "attn_post_norm",
      "ffn_out",         "post_ffn",            "l_out",
      "linear_attn_qkv_mixed", "z",             "beta_sigmoid",
      "gate",            "conv_output_silu",    "q_conv",
      "k_conv",          "v_conv",              "q_conv_predelta",
      "k_conv_predelta", "v_conv_predelta",     "final_output",
      "linear_attn_out", "Qcur_full",           "Qcur_reshaped",
      "Qcur_normed",
      "Kcur_normed",     "Vcur",                "Qcur",
      "Kcur",            "attn_pregate",        "attn_gated",
      "attn_output"};
  const std::string full(name == nullptr ? "" : name);
  if (globals.count(full) != 0) return true;
  const std::size_t separator = full.rfind('-');
  if (separator == std::string::npos ||
      layers.count(full.substr(0, separator)) == 0) {
    return false;
  }
  const std::string layer = full.substr(separator + 1);
  return layer == "0" || layer == "3" || layer == "7" || layer == "62" ||
         layer == "63";
}

float trace_value(const std::uint8_t* data, enum ggml_type type,
                  std::size_t offset) {
  if (type == GGML_TYPE_F32) {
    float value = 0.0F;
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
  }
  if (type == GGML_TYPE_F16) {
    ggml_fp16_t value = 0;
    std::memcpy(&value, data + offset, sizeof(value));
    return ggml_fp16_to_fp32(value);
  }
  ggml_bf16_t value{};
  std::memcpy(&value, data + offset, sizeof(value));
  return ggml_bf16_to_fp32(value);
}

bool trace_callback(struct ggml_tensor* tensor, bool ask, void* user_data) {
  auto* capture = static_cast<TraceCapture*>(user_data);
  if (ask) return selected_trace_name(tensor->name);
  if (capture == nullptr || capture->failed ||
      (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 &&
       tensor->type != GGML_TYPE_BF16)) {
    if (capture != nullptr) capture->failed = true;
    return false;
  }
  const std::size_t source_bytes = ggml_nbytes(tensor);
  capture->staging.resize(source_bytes);
  const std::uint8_t* source = nullptr;
  if (ggml_backend_buffer_is_host(tensor->buffer)) {
    source = static_cast<const std::uint8_t*>(tensor->data);
  } else {
    ggml_backend_tensor_get(tensor, capture->staging.data(), 0, source_bytes);
    source = capture->staging.data();
  }
  const std::size_t count = ggml_nelements(tensor);
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::size_t remaining = index;
    std::size_t offset = 0;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
      const std::size_t coordinate =
          remaining % static_cast<std::size_t>(tensor->ne[dimension]);
      remaining /= static_cast<std::size_t>(tensor->ne[dimension]);
      offset += coordinate * tensor->nb[dimension];
    }
    if (offset + ggml_type_size(tensor->type) > source_bytes) {
      capture->failed = true;
      return false;
    }
    values[index] = trace_value(source, tensor->type, offset);
  }
  capture->output.write(reinterpret_cast<const char*>(values.data()),
                        static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!capture->output) {
    capture->failed = true;
    return false;
  }
  TraceRecord record;
  record.name = tensor->name;
  std::copy(tensor->ne, tensor->ne + GGML_MAX_DIMS, record.shape.begin());
  record.position = capture->position;
  record.count = count;
  record.offset = capture->bytes;
  capture->bytes += count * sizeof(float);
  capture->records.push_back(std::move(record));
  return true;
}

bool parse_token(const char* text, llama_token* token) {
  if (text == nullptr || token == nullptr || text[0] == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 ||
      value > INT32_MAX) {
    return false;
  }
  *token = static_cast<llama_token>(value);
  return true;
}

bool is_little_endian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

void print_usage(const char* program) {
  std::fprintf(stderr,
               "usage: %s MODEL LOGITS_F32_LE TOKEN [TOKEN ...]\n"
               "       %s --trace MODEL TAPS_F32_LE TOKEN [TOKEN ...]\n"
               "       %s --tokenize-file MODEL INPUT_BYTES\n",
               program,
               program,
               program);
}

int tokenize_file(const char* model_path, const char* input_path) {
  std::ifstream input(input_path, std::ios::binary);
  if (!input.is_open()) {
    std::fprintf(stderr, "could not open tokenizer input\n");
    return 1;
  }
  const std::string bytes((std::istreambuf_iterator<char>(input)),
                          std::istreambuf_iterator<char>());
  if (input.bad()) {
    std::fprintf(stderr, "could not read tokenizer input\n");
    return 1;
  }

  ggml_backend_load_all();
  llama_backend_init();
  llama_model_params model_params = llama_model_default_params();
  model_params.vocab_only = true;
  llama_model* model = llama_model_load_from_file(model_path, model_params);
  if (model == nullptr) {
    std::fprintf(stderr, "could not load model vocabulary\n");
    llama_backend_free();
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int32_t required =
      -llama_tokenize(vocab, bytes.data(), bytes.size(), nullptr, 0, false, true);
  if (required < 0) {
    std::fprintf(stderr, "could not size tokenized input\n");
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  std::vector<llama_token> tokens(static_cast<std::size_t>(required));
  const int32_t count = llama_tokenize(vocab, bytes.data(), bytes.size(),
                                       tokens.data(), tokens.size(), false, true);
  if (count != required) {
    std::fprintf(stderr, "could not tokenize input\n");
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }
  std::printf("schema_version=1\nllama_version=%s\ntemplate_byte_count=%zu\n"
              "token_count=%zu\ntoken_ids=",
              llama_version(), bytes.size(), tokens.size());
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) std::printf(",");
    std::printf("%d", tokens[index]);
  }
  std::printf("\n");
  llama_model_free(model);
  llama_backend_free();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "--tokenize-file") == 0) {
    return tokenize_file(argv[2], argv[3]);
  }
  const bool trace_mode = argc >= 5 && std::strcmp(argv[1], "--trace") == 0;
  if ((!trace_mode && argc < 4) || (trace_mode && argc < 5)) {
    print_usage(argv[0]);
    return 2;
  }
  if (!is_little_endian() || sizeof(float) != 4 ||
      !std::numeric_limits<float>::is_iec559) {
    std::fprintf(stderr, "unsupported host float or byte order\n");
    return 2;
  }

  std::vector<llama_token> tokens;
  const int token_start = trace_mode ? 4 : 3;
  tokens.reserve(static_cast<std::size_t>(argc - token_start));
  for (int index = token_start; index < argc; ++index) {
    llama_token token = 0;
    if (!parse_token(argv[index], &token)) {
      std::fprintf(stderr, "invalid token: %s\n", argv[index]);
      return 2;
    }
    tokens.push_back(token);
  }

  ggml_backend_load_all();
  llama_backend_init();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = -1;
  model_params.check_tensors = true;
  const char* model_path = argv[trace_mode ? 2 : 1];
  const char* output_path = argv[trace_mode ? 3 : 2];
  llama_model* model = llama_model_load_from_file(model_path, model_params);
  if (model == nullptr) {
    std::fprintf(stderr, "could not load model\n");
    llama_backend_free();
    return 1;
  }

  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
  for (const llama_token token : tokens) {
    if (token < 0 || token >= vocabulary_size) {
      std::fprintf(stderr, "token %d is outside vocabulary size %d\n", token,
                   vocabulary_size);
      llama_model_free(model);
      llama_backend_free();
      return 2;
    }
  }

  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 32;
  context_params.n_batch = 1;
  context_params.n_ubatch = 1;
  context_params.n_seq_max = 1;
  context_params.no_perf = false;
  TraceCapture trace;
  if (trace_mode) {
    trace.output.open(output_path, std::ios::binary | std::ios::trunc);
    if (!trace.output) {
      std::fprintf(stderr, "could not open trace output\n");
      llama_model_free(model);
      llama_backend_free();
      return 1;
    }
    context_params.cb_eval = trace_callback;
    context_params.cb_eval_user_data = &trace;
  }
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    std::fprintf(stderr, "could not create context\n");
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  std::FILE* output = trace_mode ? nullptr : std::fopen(output_path, "wb");
  if (!trace_mode && output == nullptr) {
    std::fprintf(stderr, "could not open logits output: %s\n",
                 std::strerror(errno));
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  bool ok = true;
  std::printf("schema_version=1\nllama_version=%s\nvocabulary_size=%d\n",
              llama_version(), vocabulary_size);
  std::printf("token_count=%zu\n", tokens.size());
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    trace.position = index;
    llama_token token = tokens[index];
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(context, batch) != 0) {
      std::fprintf(stderr, "decode failed at token index %zu\n", index);
      ok = false;
      break;
    }
    const float* logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) {
      std::fprintf(stderr, "missing logits at token index %zu\n", index);
      ok = false;
      break;
    }
    const auto greedy = std::max_element(logits, logits + vocabulary_size);
    if (!trace_mode &&
        std::fwrite(logits, sizeof(float), vocabulary_size, output) !=
        static_cast<std::size_t>(vocabulary_size)) {
      std::fprintf(stderr, "could not write logits at token index %zu\n",
                   index);
      ok = false;
      break;
    }
    std::printf("token_%zu=%d\ngreedy_%zu=%td\n", index, token, index,
                greedy - logits);
  }

  if (!trace_mode && std::fclose(output) != 0) {
    std::fprintf(stderr, "could not close logits output\n");
    ok = false;
  }
  if (trace_mode) {
    trace.output.close();
    if (!trace.output || trace.failed) {
      std::fprintf(stderr, "could not capture selected trace tensors\n");
      ok = false;
    }
    std::printf("trace_tensor_count=%zu\ntrace_blob_bytes=%zu\n",
                trace.records.size(), trace.bytes);
    for (std::size_t index = 0; index < trace.records.size(); ++index) {
      const TraceRecord& record = trace.records[index];
      std::printf("trace_%zu_name=%s\ntrace_%zu_position=%zu\n", index,
                  record.name.c_str(), index, record.position);
      std::printf("trace_%zu_shape=", index);
      for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (dimension != 0) std::printf(",");
        std::printf("%lld", static_cast<long long>(record.shape[dimension]));
      }
      std::printf("\ntrace_%zu_count=%zu\ntrace_%zu_offset=%zu\n", index,
                  record.count, index, record.offset);
    }
  }
  llama_perf_context_print(context);
  llama_free(context);
  llama_model_free(model);
  llama_backend_free();
  return ok ? 0 : 1;
}
