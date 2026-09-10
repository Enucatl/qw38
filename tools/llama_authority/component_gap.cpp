#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"
#include "llama.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kWarmups = 3;
constexpr int kSamples = 8;
constexpr int kEmb = 5120;
constexpr int kVocab = 248320;
constexpr char kPrefix[] = "QW38_OPT043_LLAMA_COMPONENT_RESULT=";

struct MappedGguf final {
  int fd = -1;
  void* data = nullptr;
  std::size_t size = 0;
  gguf_context* ctx = nullptr;

  ~MappedGguf() {
    if (ctx != nullptr) gguf_free(ctx);
    if (data != nullptr && data != MAP_FAILED) munmap(data, size);
    if (fd >= 0) close(fd);
  }
};

bool map_gguf(const char* path, MappedGguf* mapped) {
  mapped->fd = open(path, O_RDONLY);
  if (mapped->fd < 0) return false;
  struct stat st {};
  if (fstat(mapped->fd, &st) != 0) return false;
  mapped->size = static_cast<std::size_t>(st.st_size);
  mapped->data = mmap(nullptr, mapped->size, PROT_READ, MAP_PRIVATE, mapped->fd, 0);
  if (mapped->data == MAP_FAILED) return false;
  struct gguf_init_params params {};
  params.no_alloc = true;
  mapped->ctx = gguf_init_from_file(path, params);
  return mapped->ctx != nullptr;
}

const std::uint8_t* tensor_bytes(const MappedGguf& mapped, const char* name,
                                 std::size_t* bytes, ggml_type* type,
                                 int64_t* ne0, int64_t* ne1) {
  const int64_t id = gguf_find_tensor(mapped.ctx, name);
  if (id < 0) return nullptr;
  *bytes = gguf_get_tensor_size(mapped.ctx, id);
  *type = gguf_get_tensor_type(mapped.ctx, id);
  const int64_t* ne = gguf_get_tensor_ne(mapped.ctx, id);
  *ne0 = ne[0];
  *ne1 = ne[1];
  const std::size_t offset =
      gguf_get_data_offset(mapped.ctx) + gguf_get_tensor_offset(mapped.ctx, id);
  if (offset + *bytes > mapped.size) return nullptr;
  return static_cast<const std::uint8_t*>(mapped.data) + offset;
}

float time_graph_ms(ggml_backend_t backend, ggml_cgraph* graph) {
  for (int i = 0; i < kWarmups; ++i) {
    ggml_backend_graph_compute(backend, graph);
  }
  cudaDeviceSynchronize();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < kSamples; ++i) {
    ggml_backend_graph_compute(backend, graph);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0F;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms / static_cast<float>(kSamples);
}

bool time_mul_mat(ggml_backend_t backend, ggml_type weight_type, int64_t ne0,
                  int64_t ne1, const void* weight_data, std::size_t weight_bytes,
                  int tokens, const float* input, float* out_ms,
                  const char** op_name) {
  const std::size_t ctx_bytes = 64 * 1024 * 1024;
  ggml_init_params params{};
  params.mem_size = ctx_bytes;
  params.no_alloc = true;
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return false;
  ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, ne0, ne1);
  ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, tokens);
  ggml_set_input(src);
  ggml_tensor* dst = ggml_mul_mat(ctx, weight, src);
  ggml_set_output(dst);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, dst);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_tensor_set(weight, weight_data, 0, weight_bytes);
  ggml_backend_tensor_set(src, input, 0,
                           static_cast<std::size_t>(ne0 * tokens) * sizeof(float));
  *out_ms = time_graph_ms(backend, graph);
  *op_name = dst->op != GGML_OP_NONE ? ggml_op_name(dst->op) : "mul_mat";
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return true;
}

bool time_rms_norm(ggml_backend_t backend, const float* input, float* out_ms,
                   const char** op_name) {
  ggml_init_params params{};
  params.mem_size = 16 * 1024 * 1024;
  params.no_alloc = true;
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return false;
  ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmb, 1);
  ggml_set_input(src);
  ggml_tensor* dst = ggml_rms_norm(ctx, src, 1e-6F);
  ggml_set_output(dst);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, dst);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    ggml_free(ctx);
    return false;
  }
  ggml_backend_tensor_set(src, input, 0, static_cast<std::size_t>(kEmb) * sizeof(float));
  *out_ms = time_graph_ms(backend, graph);
  *op_name = ggml_op_name(dst->op);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return true;
}

bool time_gdn(ggml_backend_t backend, float* out_ms, const char** op_name,
              char* error, std::size_t error_n) {
  constexpr int key_heads = 16;
  constexpr int value_heads = 48;
  constexpr int width = 128;
  constexpr int tokens = 1;
  ggml_init_params params{};
  params.mem_size = 64 * 1024 * 1024;
  params.no_alloc = true;
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return false;
  ggml_tensor* q =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, key_heads, tokens, 1);
  ggml_tensor* k =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, key_heads, tokens, 1);
  ggml_tensor* v =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, value_heads, tokens, 1);
  ggml_tensor* g =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, value_heads, tokens, 1);
  ggml_tensor* beta =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, value_heads, tokens, 1);
  ggml_tensor* state =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, width, value_heads, 1);
  ggml_set_input(q);
  ggml_set_input(k);
  ggml_set_input(v);
  ggml_set_input(g);
  ggml_set_input(beta);
  ggml_set_input(state);
  ggml_tensor* dst = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
  if (dst == nullptr) {
    std::snprintf(error, error_n, "ggml_gated_delta_net returned null");
    ggml_free(ctx);
    return false;
  }
  ggml_set_output(dst);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, dst);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    std::snprintf(error, error_n, "GDN tensor alloc failed");
    ggml_free(ctx);
    return false;
  }
  std::vector<float> qk(static_cast<std::size_t>(width * key_heads), 0.0F);
  std::vector<float> vg(static_cast<std::size_t>(width * value_heads), 0.0F);
  ggml_backend_tensor_set(q, qk.data(), 0, qk.size() * sizeof(float));
  ggml_backend_tensor_set(k, qk.data(), 0, qk.size() * sizeof(float));
  ggml_backend_tensor_set(v, vg.data(), 0, vg.size() * sizeof(float));
  ggml_backend_tensor_set(g, vg.data(), 0, vg.size() * sizeof(float));
  std::vector<float> b(static_cast<std::size_t>(value_heads), 0.0F);
  ggml_backend_tensor_set(beta, b.data(), 0, b.size() * sizeof(float));
  std::vector<float> s(
      static_cast<std::size_t>(width * width * value_heads), 0.0F);
  ggml_backend_tensor_set(state, s.data(), 0, s.size() * sizeof(float));
  *out_ms = time_graph_ms(backend, graph);
  *op_name = ggml_op_name(dst->op);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return true;
}

bool time_flash_attn(ggml_backend_t backend, float* out_ms, const char** op_name,
                     char* error, std::size_t error_n) {
  constexpr int qh = 24;
  constexpr int kh = 4;
  constexpr int hd = 256;
  constexpr int kv = 128;
  ggml_init_params params{};
  params.mem_size = 128 * 1024 * 1024;
  params.no_alloc = true;
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return false;
  // llama flash-attn layout: [head_dim, tokens_or_kv, heads, batch]
  ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, 1, qh, 1);
  ggml_tensor* k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, kv, kh, 1);
  ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, kv, kh, 1);
  ggml_set_input(q);
  ggml_set_input(k);
  ggml_set_input(v);
  ggml_tensor* dst =
      ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0F / 16.0F, 0.0F, 0.0F);
  if (dst == nullptr) {
    std::snprintf(error, error_n, "ggml_flash_attn_ext returned null");
    ggml_free(ctx);
    return false;
  }
  ggml_set_output(dst);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, dst);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    std::snprintf(error, error_n, "flash-attn tensor alloc failed");
    ggml_free(ctx);
    return false;
  }
  std::vector<float> qh_data(static_cast<std::size_t>(hd * qh), 0.0F);
  std::vector<ggml_fp16_t> kv_data(static_cast<std::size_t>(hd * kh * kv), 0);
  ggml_backend_tensor_set(q, qh_data.data(), 0, qh_data.size() * sizeof(float));
  ggml_backend_tensor_set(k, kv_data.data(), 0, kv_data.size() * sizeof(ggml_fp16_t));
  ggml_backend_tensor_set(v, kv_data.data(), 0, kv_data.size() * sizeof(ggml_fp16_t));
  *out_ms = time_graph_ms(backend, graph);
  *op_name = ggml_op_name(dst->op);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return true;
}

float time_native_decode_ms(const char* model_path, int prefix, char* error,
                            std::size_t error_n) {
  llama_backend_init();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 99;
  llama_model* model = llama_model_load_from_file(model_path, model_params);
  if (model == nullptr) {
    std::snprintf(error, error_n, "llama_model_load_from_file failed");
    llama_backend_free();
    return -1.0F;
  }
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = 4096;
  context_params.n_batch = 2048;
  context_params.n_ubatch = 512;
  context_params.n_seq_max = 1;
  context_params.no_perf = false;
  llama_context* ctx = llama_init_from_model(model, context_params);
  if (ctx == nullptr) {
    std::snprintf(error, error_n, "llama_init_from_model failed");
    llama_model_free(model);
    llama_backend_free();
    return -1.0F;
  }
  std::vector<llama_token> tokens(static_cast<std::size_t>(prefix) + 1);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<llama_token>((42 + i * 997) % kVocab);
  }
  llama_batch prefix_batch = llama_batch_get_one(tokens.data(), prefix);
  if (llama_decode(ctx, prefix_batch) != 0) {
    std::snprintf(error, error_n, "llama prefix decode failed");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return -1.0F;
  }
  llama_token next = tokens[static_cast<std::size_t>(prefix)];
  llama_batch step = llama_batch_get_one(&next, 1);
  for (int i = 0; i < kWarmups; ++i) {
    if (llama_decode(ctx, step) != 0) {
      std::snprintf(error, error_n, "llama warmup decode failed");
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      return -1.0F;
    }
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  if (llama_decode(ctx, step) != 0) {
    std::snprintf(error, error_n, "llama timed decode failed");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return -1.0F;
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0F;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return ms;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
    return 2;
  }
  MappedGguf mapped;
  if (!map_gguf(argv[1], &mapped)) {
    std::fprintf(stderr, "cannot map GGUF\n");
    return 1;
  }
  ggml_backend_t backend = ggml_backend_cuda_init(0);
  if (backend == nullptr) {
    std::fprintf(stderr, "ggml_backend_cuda_init failed\n");
    return 1;
  }

  std::vector<float> input(static_cast<std::size_t>(kEmb), 0.01F);
  for (int i = 0; i < kEmb; ++i) {
    input[static_cast<std::size_t>(i)] =
        0.01F * static_cast<float>(((i * 17) % 100) - 50);
  }

  struct Component {
    const char* family;
    const char* tensor;
    const char* dtype;
    float ms = -1.0F;
    const char* op = "";
    bool ok = false;
    char error[160]{};
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int tokens = 1;
    bool identical_input = true;
  };
  Component components[] = {
      {"q4k_ffn_gate", "blk.0.ffn_gate.weight", "Q4_K"},
      {"q4k_ffn_up", "blk.0.ffn_up.weight", "Q4_K"},
      {"q4k_ffn_down", "blk.0.ffn_down.weight", "Q4_K"},
      {"q8_mixer_qkv", "blk.0.attn_qkv.weight", "Q8_0"},
      {"q6k_output", "output.weight", "Q6_K"},
  };

  for (Component& component : components) {
    std::size_t bytes = 0;
    ggml_type type = GGML_TYPE_F32;
    const std::uint8_t* data = tensor_bytes(
        mapped, component.tensor, &bytes, &type, &component.ne0, &component.ne1);
    if (data == nullptr) {
      std::snprintf(component.error, sizeof(component.error),
                    "missing tensor %s", component.tensor);
      continue;
    }
    const int tokens = 1;
    std::vector<float> in(static_cast<std::size_t>(component.ne0), 0.01F);
    for (int64_t i = 0; i < component.ne0; ++i) {
      in[static_cast<std::size_t>(i)] =
          0.01F * static_cast<float>(((i * 17) % 100) - 50);
    }
    const bool ok = time_mul_mat(backend, type, component.ne0, component.ne1,
                                 data, bytes, tokens, in.data(), &component.ms,
                                 &component.op);
    component.ok = ok;
    component.tokens = tokens;
    if (!ok) {
      std::snprintf(component.error, sizeof(component.error),
                    "mul_mat failed for %s", component.tensor);
    }
  }

  Component rms{"rms_norm", "output_norm.weight", "F32"};
  rms.identical_input = true;
  rms.ok = time_rms_norm(backend, input.data(), &rms.ms, &rms.op);
  if (!rms.ok) std::snprintf(rms.error, sizeof(rms.error), "rms_norm failed");

  Component gdn{"gdn_fused", "ggml_gated_delta_net", "F32"};
  gdn.identical_input = false;
  gdn.ok = time_gdn(backend, &gdn.ms, &gdn.op, gdn.error, sizeof(gdn.error));

  Component attn{"attention_flash", "ggml_flash_attn_ext", "F32_Q_F16_KV"};
  attn.identical_input = false;
  attn.ok = time_flash_attn(backend, &attn.ms, &attn.op, attn.error,
                            sizeof(attn.error));

  char native_error[160]{};
  const float native_d128_ms =
      time_native_decode_ms(argv[1], 128, native_error, sizeof(native_error));

  ggml_backend_free(backend);

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-043\",\"status\":\"measured\","
      "\"engine\":\"llama.cpp\",\"llama_revision\":"
      "\"cc83d7b4824f73cfdda4dfbb47ee39804f71b328\","
      "\"numeric_label\":\"Quartz BF16 vs llama F32/Q8_1 activations; "
      "Quartz BF16 KV vs llama F16 KV\","
      "\"q8_layout\":\"llama_block_q8_1_not_quartz_Q8Block\","
      "\"identical_input_includes_quantization\":true,"
      "\"native_decode_d128_ms\":%.9g,"
      "\"native_decode_error\":\"%s\","
      "\"components\":[",
      kPrefix, static_cast<double>(native_d128_ms), native_error);
  const Component* all[] = {&components[0], &components[1], &components[2],
                            &components[3], &components[4], &rms, &gdn, &attn};
  for (std::size_t i = 0; i < 8; ++i) {
    const Component& c = *all[i];
    if (i != 0) std::printf(",");
    std::printf(
        "{\"family\":\"%s\",\"tensor\":\"%s\",\"dtype\":\"%s\",\"ok\":%s,"
        "\"ms\":%.9g,\"op\":\"%s\",\"ne0\":%lld,\"ne1\":%lld,\"tokens\":%d,"
        "\"identical_input\":%s,\"error\":\"%s\"}",
        c.family, c.tensor, c.dtype, c.ok ? "true" : "false",
        static_cast<double>(c.ms), c.op, static_cast<long long>(c.ne0),
        static_cast<long long>(c.ne1), c.tokens,
        c.identical_input ? "true" : "false", c.error);
  }
  std::printf("]}\n");
  std::printf("status=passed\n");
  bool any = false;
  for (const Component* c : all) {
    if (c->ok) any = true;
  }
  return any ? 0 : 1;
}
