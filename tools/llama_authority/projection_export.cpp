#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>

#include <cmath>
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

constexpr char kPrefix059[] = "QW38_OPT059_LLAMA_GPU_EXPORT=";
constexpr char kPrefix074[] = "QW38_OPT074_LLAMA_GPU_EXPORT=";
constexpr char kRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr char kProducer[] = "qw38-llama-projection-export/opt074";
constexpr int kQ81 = 32;

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
  gguf_init_params params {};
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

bool load_f32_file(const char* path, std::vector<float>* values) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) return false;
  if (std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    return false;
  }
  const long bytes = std::ftell(file);
  if (bytes < 0 || bytes % static_cast<long>(sizeof(float)) != 0) {
    std::fclose(file);
    return false;
  }
  std::rewind(file);
  values->assign(static_cast<std::size_t>(bytes) / sizeof(float), 0.0F);
  const bool ok =
      std::fread(values->data(), 1, static_cast<std::size_t>(bytes), file) ==
      static_cast<std::size_t>(bytes);
  std::fclose(file);
  return ok;
}

bool load_bf16_as_f32(const char* path, std::vector<float>* values,
                      std::vector<std::uint16_t>* original) {
  FILE* file = std::fopen(path, "rb");
  if (file == nullptr) return false;
  if (std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    return false;
  }
  const long bytes = std::ftell(file);
  if (bytes < 0 || bytes % 2 != 0) {
    std::fclose(file);
    return false;
  }
  std::rewind(file);
  original->assign(static_cast<std::size_t>(bytes) / 2, 0);
  const bool ok =
      std::fread(original->data(), 1, static_cast<std::size_t>(bytes), file) ==
      static_cast<std::size_t>(bytes);
  std::fclose(file);
  if (!ok) return false;
  values->resize(original->size());
  for (std::size_t index = 0; index < original->size(); ++index) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>((*original)[index]) << 16U;
    std::memcpy(&(*values)[index], &bits, sizeof(float));
  }
  return true;
}

bool write_f32_file(const char* path, const float* values, std::size_t count) {
  FILE* file = std::fopen(path, "wb");
  if (file == nullptr) return false;
  const bool ok =
      std::fwrite(values, sizeof(float), count, file) == count;
  std::fclose(file);
  return ok;
}

bool write_u16_file(const char* path, const std::uint16_t* values,
                    std::size_t count) {
  FILE* file = std::fopen(path, "wb");
  if (file == nullptr) return false;
  const bool ok =
      std::fwrite(values, sizeof(std::uint16_t), count, file) == count;
  std::fclose(file);
  return ok;
}

bool write_bytes(const char* path, const std::uint8_t* values,
                 std::size_t count) {
  FILE* file = std::fopen(path, "wb");
  if (file == nullptr) return false;
  const bool ok = std::fwrite(values, 1, count, file) == count;
  std::fclose(file);
  return ok;
}

ggml_type parse_type(const char* name) {
  if (std::strcmp(name, "Q4_K") == 0) return GGML_TYPE_Q4_K;
  if (std::strcmp(name, "Q6_K") == 0) return GGML_TYPE_Q6_K;
  if (std::strcmp(name, "Q8_0") == 0) return GGML_TYPE_Q8_0;
  return GGML_TYPE_F32;
}

std::uint16_t float_to_half_bits(float value) {
  std::uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(raw));
  const std::uint16_t sign =
      static_cast<std::uint16_t>((raw >> 16U) & 0x8000U);
  int exponent = static_cast<int>((raw >> 23U) & 0xFFU) - 127;
  std::uint32_t mantissa = raw & 0x7FFFFFU;
  if (((raw >> 23U) & 0xFFU) == 0xFFU) {
    return static_cast<std::uint16_t>(
        sign | 0x7C00U | (mantissa ? 0x200U : 0U));
  }
  if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);
  if (exponent > -15) {
    std::uint32_t rounded = mantissa + 0x1000U + ((mantissa >> 13U) & 1U);
    if (rounded & 0x800000U) {
      exponent += 1;
      mantissa = 0;
    } else {
      mantissa = rounded;
    }
    if (exponent > 15) return static_cast<std::uint16_t>(sign | 0x7C00U);
    return static_cast<std::uint16_t>(
        sign | ((exponent + 15) << 10) | (mantissa >> 13U));
  }
  return sign;
}

void quantize_q8_1_sum_x(const std::vector<float>& input,
                         std::vector<std::uint8_t>* out) {
  const std::size_t padded =
      ((input.size() + static_cast<std::size_t>(kQ81) - 1U) /
       static_cast<std::size_t>(kQ81)) *
      static_cast<std::size_t>(kQ81);
  const std::size_t blocks = padded / static_cast<std::size_t>(kQ81);
  out->assign(blocks * 36U, 0);
  for (std::size_t block = 0; block < blocks; ++block) {
    float amax = 0.0F;
    float sum = 0.0F;
    float values[kQ81];
    for (int lane = 0; lane < kQ81; ++lane) {
      const std::size_t index =
          block * static_cast<std::size_t>(kQ81) + static_cast<std::size_t>(lane);
      const float xi = index < input.size() ? input[index] : 0.0F;
      values[lane] = xi;
      amax = std::max(amax, std::fabs(xi));
      sum += xi;
    }
    const float scale = amax / 127.0F;
    std::uint8_t* dest = out->data() + block * 36U;
    const std::uint16_t d_bits = float_to_half_bits(scale);
    const std::uint16_t s_bits = float_to_half_bits(sum);
    dest[0] = static_cast<std::uint8_t>(d_bits & 0xFFU);
    dest[1] = static_cast<std::uint8_t>(d_bits >> 8U);
    dest[2] = static_cast<std::uint8_t>(s_bits & 0xFFU);
    dest[3] = static_cast<std::uint8_t>(s_bits >> 8U);
    for (int lane = 0; lane < kQ81; ++lane) {
      const std::int8_t q =
          amax == 0.0F
              ? 0
              : static_cast<std::int8_t>(std::round(values[lane] / scale));
      dest[4 + lane] = static_cast<std::uint8_t>(q);
    }
  }
}

const char* dispatch_family(ggml_type type, int64_t n) {
  if (n != 1) return "unsupported_n_not_1";
  if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K ||
      type == GGML_TYPE_Q8_0) {
    return "mmvq";
  }
  return "other";
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model GGUF --tensor NAME --input in.f32 "
               "--output out.f32 [--input-bf16 in.bf16] [--retain-bf16 out.bf16] "
               "[--staging q8_1.bin] [--q8-1 q8_1.bin] [--evidence] "
               "[--fuse-swiglu --up-tensor NAME] [--row-limit N]\n"
               "   or: %s --weights bytes --type Q4_K --rows R --columns K "
               "--input in.f32 --output out.f32\n",
               argv0, argv0);
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = nullptr;
  const char* tensor = nullptr;
  const char* up_tensor = nullptr;
  const char* input_path = nullptr;
  const char* input_bf16_path = nullptr;
  const char* retain_bf16_path = nullptr;
  const char* output_path = nullptr;
  const char* staging_path = nullptr;
  const char* q81_path = nullptr;
  const char* type_name = nullptr;
  const char* weight_path = nullptr;
  const char* task = "OPT-059";
  const char* producer = kProducer;
  const char* gguf_sha = "";
  int64_t rows = 0;
  int64_t columns = 0;
  int row_limit = 0;
  bool evidence = false;
  bool fuse_swiglu = false;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--model") == 0 && index + 1 < argc) {
      model = argv[++index];
    } else if (std::strcmp(arg, "--tensor") == 0 && index + 1 < argc) {
      tensor = argv[++index];
    } else if (std::strcmp(arg, "--up-tensor") == 0 && index + 1 < argc) {
      up_tensor = argv[++index];
    } else if (std::strcmp(arg, "--input") == 0 && index + 1 < argc) {
      input_path = argv[++index];
    } else if (std::strcmp(arg, "--input-bf16") == 0 && index + 1 < argc) {
      input_bf16_path = argv[++index];
    } else if (std::strcmp(arg, "--retain-bf16") == 0 && index + 1 < argc) {
      retain_bf16_path = argv[++index];
    } else if (std::strcmp(arg, "--output") == 0 && index + 1 < argc) {
      output_path = argv[++index];
    } else if (std::strcmp(arg, "--staging") == 0 && index + 1 < argc) {
      staging_path = argv[++index];
    } else if (std::strcmp(arg, "--q8-1") == 0 && index + 1 < argc) {
      q81_path = argv[++index];
    } else if (std::strcmp(arg, "--type") == 0 && index + 1 < argc) {
      type_name = argv[++index];
    } else if (std::strcmp(arg, "--weights") == 0 && index + 1 < argc) {
      weight_path = argv[++index];
    } else if (std::strcmp(arg, "--task") == 0 && index + 1 < argc) {
      task = argv[++index];
    } else if (std::strcmp(arg, "--producer") == 0 && index + 1 < argc) {
      producer = argv[++index];
    } else if (std::strcmp(arg, "--gguf-sha") == 0 && index + 1 < argc) {
      gguf_sha = argv[++index];
    } else if (std::strcmp(arg, "--rows") == 0 && index + 1 < argc) {
      rows = std::strtoll(argv[++index], nullptr, 10);
    } else if (std::strcmp(arg, "--columns") == 0 && index + 1 < argc) {
      columns = std::strtoll(argv[++index], nullptr, 10);
    } else if (std::strcmp(arg, "--row-limit") == 0 && index + 1 < argc) {
      row_limit = std::atoi(argv[++index]);
    } else if (std::strcmp(arg, "--evidence") == 0) {
      evidence = true;
    } else if (std::strcmp(arg, "--fuse-swiglu") == 0) {
      fuse_swiglu = true;
    } else if (arg[0] != '-' && model == nullptr) {
      model = arg;
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (output_path == nullptr ||
      (input_path == nullptr && input_bf16_path == nullptr)) {
    std::fprintf(stderr, "input and output paths are required\n");
    return 2;
  }
  if (evidence && row_limit > 0) {
    std::fprintf(stderr,
                 "evidence forbids --row-limit; full M N=1 is required\n");
    return 2;
  }
  if (evidence && weight_path != nullptr) {
    std::fprintf(stderr,
                 "evidence requires a real GGUF tensor, not synthetic weights\n");
    return 2;
  }
  if (fuse_swiglu && (tensor == nullptr || up_tensor == nullptr ||
                      model == nullptr)) {
    std::fprintf(stderr, "fused SwiGLU needs --model --tensor --up-tensor\n");
    return 2;
  }

  std::vector<float> input;
  std::vector<std::uint16_t> original_bf16;
  bool used_bf16 = false;
  if (input_bf16_path != nullptr) {
    if (!load_bf16_as_f32(input_bf16_path, &input, &original_bf16)) {
      std::fprintf(stderr, "cannot read input BF16 file\n");
      return 1;
    }
    used_bf16 = true;
  } else if (!load_f32_file(input_path, &input)) {
    std::fprintf(stderr, "cannot read input FP32 file\n");
    return 1;
  }
  if (retain_bf16_path != nullptr) {
    if (!used_bf16) {
      std::fprintf(stderr, "retain-bf16 requires --input-bf16 originals\n");
      return 2;
    }
    if (!write_u16_file(retain_bf16_path, original_bf16.data(),
                        original_bf16.size())) {
      std::fprintf(stderr, "cannot write retained BF16\n");
      return 1;
    }
  }

  MappedGguf mapped;
  const std::uint8_t* weight_data = nullptr;
  const std::uint8_t* up_data = nullptr;
  std::size_t weight_bytes = 0;
  std::size_t up_bytes = 0;
  ggml_type weight_type = GGML_TYPE_F32;
  ggml_type up_type = GGML_TYPE_F32;
  int64_t ne0 = columns;
  int64_t ne1 = rows;
  int64_t up_ne0 = 0;
  int64_t up_ne1 = 0;
  std::vector<std::uint8_t> weight_store;
  if (weight_path != nullptr) {
    FILE* file = std::fopen(weight_path, "rb");
    if (file == nullptr) return 1;
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::rewind(file);
    weight_store.resize(static_cast<std::size_t>(bytes));
    const bool ok = std::fread(weight_store.data(), 1,
                               weight_store.size(), file) == weight_store.size();
    std::fclose(file);
    if (!ok || type_name == nullptr || rows <= 0 || columns <= 0) return 1;
    weight_data = weight_store.data();
    weight_bytes = weight_store.size();
    weight_type = parse_type(type_name);
    ne0 = columns;
    ne1 = rows;
  } else {
    if (model == nullptr || tensor == nullptr || !map_gguf(model, &mapped)) {
      std::fprintf(stderr, "cannot map GGUF/tensor\n");
      return 1;
    }
    weight_data =
        tensor_bytes(mapped, tensor, &weight_bytes, &weight_type, &ne0, &ne1);
    if (weight_data == nullptr) {
      std::fprintf(stderr, "missing tensor %s\n", tensor);
      return 1;
    }
    if (fuse_swiglu) {
      up_data = tensor_bytes(mapped, up_tensor, &up_bytes, &up_type, &up_ne0,
                             &up_ne1);
      if (up_data == nullptr) {
        std::fprintf(stderr, "missing up tensor %s\n", up_tensor);
        return 1;
      }
      if (up_ne0 != ne0 || up_ne1 != ne1 || up_type != weight_type) {
        std::fprintf(stderr, "gate/up shape or type mismatch\n");
        return 1;
      }
    }
  }
  if (static_cast<int64_t>(input.size()) != ne0) {
    std::fprintf(stderr, "input length %zu != K %lld\n", input.size(),
                 static_cast<long long>(ne0));
    return 1;
  }
  const int64_t full_m = ne1;
  if (row_limit > 0 && static_cast<int64_t>(row_limit) < ne1) {
    const std::size_t row_bytes =
        weight_bytes / static_cast<std::size_t>(ne1);
    weight_bytes = row_bytes * static_cast<std::size_t>(row_limit);
    ne1 = row_limit;
  }
  const char* family = dispatch_family(weight_type, 1);
  if (evidence && std::strcmp(family, "mmvq") != 0) {
    std::fprintf(stderr, "unsupported dispatch %s\n", family);
    return 1;
  }
  if (evidence && ne1 != full_m) {
    std::fprintf(stderr, "evidence dispatch changed M\n");
    return 1;
  }

  ggml_backend_t backend = ggml_backend_cuda_init(0);
  if (backend == nullptr) {
    std::fprintf(stderr, "ggml_backend_cuda_init failed\n");
    return 1;
  }

  ggml_init_params params{};
  params.mem_size = 256 * 1024 * 1024;
  params.no_alloc = true;
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) {
    ggml_backend_free(backend);
    return 1;
  }
  ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, ne0, ne1);
  ggml_tensor* weight_up = nullptr;
  ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
  ggml_set_input(src);
  ggml_tensor* dst = nullptr;
  if (fuse_swiglu) {
    weight_up = ggml_new_tensor_2d(ctx, up_type, ne0, ne1);
    ggml_tensor* gate_out = ggml_mul_mat(ctx, weight, src);
    ggml_tensor* up_out = ggml_mul_mat(ctx, weight_up, src);
    dst = ggml_swiglu_split(ctx, gate_out, up_out);
  } else {
    dst = ggml_mul_mat(ctx, weight, src);
  }
  ggml_set_output(dst);
  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, dst);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (buf == nullptr) {
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::fprintf(stderr, "backend alloc failed\n");
    return 1;
  }
  ggml_backend_tensor_set(weight, weight_data, 0, weight_bytes);
  if (weight_up != nullptr) {
    ggml_backend_tensor_set(weight_up, up_data, 0, up_bytes);
  }
  ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
  if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
    std::fprintf(stderr, "ggml_backend_graph_compute failed\n");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 1;
  }
  ggml_backend_synchronize(backend);
  cudaDeviceSynchronize();

  const int64_t out_rows = fuse_swiglu ? ne1 : ne1;
  std::vector<float> output(static_cast<std::size_t>(dst->ne[0] * dst->ne[1]),
                            0.0F);
  ggml_backend_tensor_get(dst, output.data(), 0,
                           output.size() * sizeof(float));
  if (!write_f32_file(output_path, output.data(), output.size())) {
    std::fprintf(stderr, "cannot write output\n");
    return 1;
  }
  bool staging_ok = staging_path == nullptr;
  if (staging_path != nullptr) {
    std::fprintf(stderr,
                 "skipping Q8_1 ggml_cpy staging export; CUDA f32->q8_1 is "
                 "unsupported in this revision\n");
    staging_ok = false;
  }
  const char* q81_status = "not_requested";
  if (q81_path != nullptr) {
    std::vector<std::uint8_t> q81;
    quantize_q8_1_sum_x(input, &q81);
    if (!write_bytes(q81_path, q81.data(), q81.size())) {
      std::fprintf(stderr, "cannot write Q8_1 adapter export\n");
      return 1;
    }
    q81_status = "private_adapter_quantize_q8_1_sum_x";
  }

  const char* prefix =
      std::strcmp(task, "OPT-074") == 0 || evidence ? kPrefix074 : kPrefix059;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"%s\","
      "\"engine\":\"llama.cpp_gpu\",\"llama_revision\":\"%s\","
      "\"producer\":\"%s\",\"producer_revision\":\"%s\","
      "\"gguf_sha256\":\"%s\",\"not_cpu_replica\":true,"
      "\"tensor\":\"%s\",\"up_tensor\":\"%s\",\"type\":\"%s\","
      "\"ne0\":%lld,\"ne1\":%lld,\"full_m\":%lld,\"n\":1,"
      "\"rows_exported\":%lld,\"row_limit\":%d,"
      "\"dispatch_family\":\"%s\",\"fused_gate_up_swiglu\":%s,"
      "\"input_was_bf16\":%s,\"cuda_synchronized\":true,"
      "\"output\":\"%s\",\"staging\":\"%s\",\"q8_1\":\"%s\","
      "\"reference_flags\":[\"full_m\",\"n=1\",\"no_row_limit_evidence\"],"
      "\"status\":\"ok\"}\n",
      prefix, evidence ? "OPT-074" : task, kRevision, producer, kRevision,
      gguf_sha, tensor != nullptr ? tensor : "supplied_bytes",
      up_tensor != nullptr ? up_tensor : "",
      ggml_type_name(weight_type), static_cast<long long>(ne0),
      static_cast<long long>(ne1), static_cast<long long>(full_m),
      static_cast<long long>(output.size()), row_limit, family,
      fuse_swiglu ? "true" : "false", used_bf16 ? "true" : "false",
      output_path,
      staging_path == nullptr ? "not_requested"
                              : (staging_ok ? "exported" : "unsupported_cuda_f32_q8_1"),
      q81_status);
  (void)out_rows;
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(backend);
  return 0;
}
