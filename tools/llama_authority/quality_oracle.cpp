#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {
struct Case { std::string name; std::vector<llama_token> context, target; };
constexpr int kPrefillBatch = 512;
bool get_u32(std::ifstream & f, std::uint32_t * v) { std::uint8_t b[4]; if (!f.read(reinterpret_cast<char *>(b), 4)) return false; *v = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24); return true; }
bool get_u16(std::ifstream & f, std::uint16_t * v) { std::uint8_t b[2]; if (!f.read(reinterpret_cast<char *>(b), 2)) return false; *v = b[0] | (b[1] << 8); return true; }
bool read_bundle(const char * path, std::vector<Case> * cases) {
  std::ifstream f(path, std::ios::binary); char magic[8]{}; const char expected[8] = {'Q', 'W', '3', '8', 'Q', 1, 0, 0}; if (!f.read(magic, 8) || std::memcmp(magic, expected, 8) != 0) return false;
  std::uint32_t count = 0; if (!get_u32(f, &count) || count == 0 || count > 64) return false;
  for (std::uint32_t i = 0; i < count; ++i) { std::uint16_t n = 0; std::uint32_t nc = 0, nt = 0; if (!get_u16(f, &n) || n == 0 || n > 128) return false; Case c; c.name.resize(n); if (!f.read(c.name.data(), n) || !get_u32(f, &nc) || !get_u32(f, &nt) || nc == 0 || nt == 0 || nc + nt > 131072) return false; c.context.resize(nc); c.target.resize(nt); for (auto * v : {&c.context, &c.target}) for (auto & t : *v) { std::uint32_t x = 0; if (!get_u32(f, &x) || x > INT32_MAX) return false; t = static_cast<llama_token>(x); } cases->push_back(std::move(c)); }
  return f.peek() == std::ifstream::traits_type::eof();
}
void top_two(const float * logits, int n_vocab, std::size_t * first, std::size_t * second) {
  *first = 0; *second = 1; for (int i = 1; i < n_vocab; ++i) { if (logits[i] > logits[*first] || (logits[i] == logits[*first] && i < static_cast<int>(*first))) { *second = *first; *first = static_cast<std::size_t>(i); } else if (i != static_cast<int>(*first) && (logits[i] > logits[*second] || (logits[i] == logits[*second] && i < static_cast<int>(*second)))) *second = static_cast<std::size_t>(i); }
}
void emit_step(const char * name, std::size_t pos, llama_token target, const float * logits, int n_vocab) {
  std::size_t first = 0, second = 1; top_two(logits, n_vocab, &first, &second);
  const float maxv = *std::max_element(logits, logits + n_vocab); double sum = 0; for (int i = 0; i < n_vocab; ++i) { if (!std::isfinite(logits[i])) std::exit(1); sum += std::exp(static_cast<double>(logits[i]) - maxv); }
  const double lp = static_cast<double>(logits[target]) - maxv - std::log(sum);
  std::printf("step\t%s\t%zu\t%d\t%.17g\t%zu\t%.9g\t%zu\t%.9g\t%.9g\n", name, pos, target, lp, first, logits[first], second, logits[second], static_cast<double>(logits[first] - logits[second]));
}
bool decode_chunked(llama_context * ctx, llama_token * tokens, std::size_t count) {
  for (std::size_t offset = 0; offset < count; ) {
    const int n = static_cast<int>(std::min(static_cast<std::size_t>(kPrefillBatch), count - offset));
    llama_batch batch = llama_batch_get_one(tokens + offset, n);
    if (llama_decode(ctx, batch) != 0) return false;
    offset += static_cast<std::size_t>(n);
  }
  return true;
}
}

int main(int argc, char ** argv) {
  int generate_count = 0;
  const char * model_path = nullptr;
  const char * bundle_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
      generate_count = std::atoi(argv[++i]);
      if (generate_count < 0 || generate_count > 16) return 2;
    } else if (model_path == nullptr) {
      model_path = argv[i];
    } else if (bundle_path == nullptr) {
      bundle_path = argv[i];
    } else {
      return 2;
    }
  }
  if (model_path == nullptr || bundle_path == nullptr) {
    std::fprintf(stderr, "usage: %s MODEL REQUEST_BUNDLE [--generate N]\n", argv[0]);
    return 2;
  }
  std::vector<Case> cases; if (!read_bundle(bundle_path, &cases)) return 2;
  llama_backend_init(); auto mp = llama_model_default_params(); mp.n_gpu_layers = -1; mp.check_tensors = true; llama_model * model = llama_model_load_from_file(model_path, mp); if (!model) return 1;
  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  auto cp = llama_context_default_params(); std::uint32_t max_tokens = 0; for (const auto & c : cases) max_tokens = std::max(max_tokens, static_cast<std::uint32_t>(c.context.size() + std::max(c.target.size(), static_cast<std::size_t>(generate_count)))); cp.n_ctx = std::max<std::uint32_t>(max_tokens, 32); cp.n_batch = kPrefillBatch; cp.n_ubatch = kPrefillBatch; cp.n_seq_max = 1;
  std::printf("schema\tqw38.quality-llama\t1\n");
  for (const auto & c : cases) {
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) { llama_model_free(model); return 1; }
    std::vector<llama_token> context = c.context;
    if (!decode_chunked(ctx, context.data(), context.size())) return 1;
    if (generate_count > 0) {
      for (int step = 0; step < generate_count; ++step) {
        const float * row = llama_get_logits_ith(ctx, -1); if (!row) return 1;
        for (int i = 0; i < vocab; ++i) if (!std::isfinite(row[i])) return 1;
        std::size_t first = 0, second = 1; top_two(row, vocab, &first, &second);
        std::printf("gen\t%s\t%zu\t%zu\t%.9g\t%zu\t%.9g\n", c.name.c_str(), c.context.size() + static_cast<std::size_t>(step), first, row[first], second, row[second]);
        llama_token next = static_cast<llama_token>(first);
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) return 1;
      }
      std::printf("end\t%s\t%d\n", c.name.c_str(), generate_count);
    } else {
      for (std::size_t i = 0; i < c.target.size(); ++i) {
        const float * row = llama_get_logits_ith(ctx, -1); if (!row) return 1;
        emit_step(c.name.c_str(), c.context.size() + i, c.target[i], row, vocab);
        llama_token next = c.target[i];
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) return 1;
      }
      std::printf("end\t%s\t%zu\n", c.name.c_str(), c.target.size());
    }
    llama_free(ctx);
  }
  llama_model_free(model); llama_backend_free(); return 0;
}
