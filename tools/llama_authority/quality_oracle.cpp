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
bool get_u32(std::ifstream & f, std::uint32_t * v) { std::uint8_t b[4]; if (!f.read(reinterpret_cast<char *>(b), 4)) return false; *v = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24); return true; }
bool get_u16(std::ifstream & f, std::uint16_t * v) { std::uint8_t b[2]; if (!f.read(reinterpret_cast<char *>(b), 2)) return false; *v = b[0] | (b[1] << 8); return true; }
bool read_bundle(const char * path, std::vector<Case> * cases) {
  std::ifstream f(path, std::ios::binary); char magic[8]{}; const char expected[8] = {'Q', 'W', '3', '8', 'Q', 1, 0, 0}; if (!f.read(magic, 8) || std::memcmp(magic, expected, 8) != 0) return false;
  std::uint32_t count = 0; if (!get_u32(f, &count) || count == 0 || count > 32) return false;
  for (std::uint32_t i = 0; i < count; ++i) { std::uint16_t n = 0; std::uint32_t nc = 0, nt = 0; if (!get_u16(f, &n) || n == 0 || n > 128) return false; Case c; c.name.resize(n); if (!f.read(c.name.data(), n) || !get_u32(f, &nc) || !get_u32(f, &nt) || nc == 0 || nt == 0 || nc + nt > 131072) return false; c.context.resize(nc); c.target.resize(nt); for (auto * v : {&c.context, &c.target}) for (auto & t : *v) { std::uint32_t x = 0; if (!get_u32(f, &x) || x > INT32_MAX) return false; t = static_cast<llama_token>(x); } cases->push_back(std::move(c)); }
  return f.peek() == std::ifstream::traits_type::eof();
}
void emit_step(const char * name, std::size_t pos, llama_token target, const float * logits, int n_vocab) {
  std::size_t first = 0, second = 1; for (int i = 1; i < n_vocab; ++i) { if (logits[i] > logits[first] || (logits[i] == logits[first] && i < static_cast<int>(first))) { second = first; first = static_cast<std::size_t>(i); } else if (i != static_cast<int>(first) && (logits[i] > logits[second] || (logits[i] == logits[second] && i < static_cast<int>(second)))) second = static_cast<std::size_t>(i); }
  const float maxv = *std::max_element(logits, logits + n_vocab); double sum = 0; for (int i = 0; i < n_vocab; ++i) { if (!std::isfinite(logits[i])) std::exit(1); sum += std::exp(static_cast<double>(logits[i]) - maxv); }
  const double lp = static_cast<double>(logits[target]) - maxv - std::log(sum);
  std::printf("step\t%s\t%zu\t%d\t%.17g\t%zu\t%.9g\t%zu\t%.9g\t%.9g\n", name, pos, target, lp, first, logits[first], second, logits[second], static_cast<double>(logits[first] - logits[second]));
}
}
int main(int argc, char ** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: %s MODEL REQUEST_BUNDLE\n", argv[0]); return 2; }
  std::vector<Case> cases; if (!read_bundle(argv[2], &cases)) return 2;
  llama_backend_init(); auto mp = llama_model_default_params(); mp.n_gpu_layers = -1; mp.check_tensors = true; llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) return 1;
  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  auto cp = llama_context_default_params(); std::uint32_t max_tokens = 0; for (const auto & c : cases) max_tokens = std::max(max_tokens, static_cast<std::uint32_t>(c.context.size() + c.target.size())); cp.n_ctx = std::max<std::uint32_t>(max_tokens, 32); cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
  std::printf("schema\tqw38.quality-llama\t1\n");
  for (const auto & c : cases) {
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) { llama_model_free(model); return 1; }
    auto decode = [&](llama_token token) { llama_batch b = llama_batch_get_one(&token, 1); return llama_decode(ctx, b); };
    for (const auto token : c.context) if (decode(token) != 0) return 1;
    for (std::size_t i = 0; i < c.target.size(); ++i) { const float * row = llama_get_logits_ith(ctx, -1); if (!row) return 1; emit_step(c.name.c_str(), c.context.size() + i, c.target[i], row, vocab); if (decode(c.target[i]) != 0) return 1; }
    std::printf("end\t%s\t%zu\n", c.name.c_str(), c.target.size());
    llama_free(ctx);
  }
  llama_model_free(model); llama_backend_free(); return 0;
}
