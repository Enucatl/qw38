#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

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
               "       %s --tokenize-file MODEL INPUT_BYTES\n",
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
  if (argc < 4) {
    print_usage(argv[0]);
    return 2;
  }
  if (!is_little_endian() || sizeof(float) != 4 ||
      !std::numeric_limits<float>::is_iec559) {
    std::fprintf(stderr, "unsupported host float or byte order\n");
    return 2;
  }

  std::vector<llama_token> tokens;
  tokens.reserve(static_cast<std::size_t>(argc - 3));
  for (int index = 3; index < argc; ++index) {
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
  llama_model* model = llama_model_load_from_file(argv[1], model_params);
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
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    std::fprintf(stderr, "could not create context\n");
    llama_model_free(model);
    llama_backend_free();
    return 1;
  }

  std::FILE* output = std::fopen(argv[2], "wb");
  if (output == nullptr) {
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
    if (std::fwrite(logits, sizeof(float), vocabulary_size, output) !=
        static_cast<std::size_t>(vocabulary_size)) {
      std::fprintf(stderr, "could not write logits at token index %zu\n",
                   index);
      ok = false;
      break;
    }
    std::printf("token_%zu=%d\ngreedy_%zu=%td\n", index, token, index,
                greedy - logits);
  }

  if (std::fclose(output) != 0) {
    std::fprintf(stderr, "could not close logits output\n");
    ok = false;
  }
  llama_perf_context_print(context);
  llama_free(context);
  llama_model_free(model);
  llama_backend_free();
  return ok ? 0 : 1;
}
