#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

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

constexpr char kPrefix[] = "QW38_OPT059_LLAMA_GPU_EXPORT=";
constexpr char kRevision[] = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";

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

bool write_f32_file(const char* path, const float* values, std::size_t count) {
  FILE* file = std::fopen(path, "wb");
  if (file == nullptr) return false;
  const bool ok =
      std::fwrite(values, sizeof(float), count, file) == count;
  std::fclose(file);
  return ok;
}

ggml_type parse_type(const char* name) {
  if (std::strcmp(name, "Q4_K") == 0) return GGML_TYPE_Q4_K;
  if (std::strcmp(name, "Q6_K") == 0) return GGML_TYPE_Q6_K;
  if (std::strcmp(name, "Q8_0") == 0) return GGML_TYPE_Q8_0;
  return GGML_TYPE_F32;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model = nullptr;
  const char* tensor = nullptr;
  const char* input_path = nullptr;
  const char* output_path = nullptr;
  const char* staging_path = nullptr;
  const char* type_name = nullptr;
  const char* weight_path = nullptr;
  int64_t rows = 0;
  int64_t columns = 0;
  int row_limit = 0;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--model") == 0 && index + 1 < argc) {
      model = argv[++index];
    } else if (std::strcmp(arg, "--tensor") == 0 && index + 1 < argc) {
      tensor = argv[++index];
    } else if (std::strcmp(arg, "--input") == 0 && index + 1 < argc) {
      input_path = argv[++index];
    } else if (std::strcmp(arg, "--output") == 0 && index + 1 < argc) {
      output_path = argv[++index];
    } else if (std::strcmp(arg, "--staging") == 0 && index + 1 < argc) {
      staging_path = argv[++index];
    } else if (std::strcmp(arg, "--type") == 0 && index + 1 < argc) {
      type_name = argv[++index];
    } else if (std::strcmp(arg, "--weights") == 0 && index + 1 < argc) {
      weight_path = argv[++index];
    } else if (std::strcmp(arg, "--rows") == 0 && index + 1 < argc) {
      rows = std::strtoll(argv[++index], nullptr, 10);
    } else if (std::strcmp(arg, "--columns") == 0 && index + 1 < argc) {
      columns = std::strtoll(argv[++index], nullptr, 10);
    } else if (std::strcmp(arg, "--row-limit") == 0 && index + 1 < argc) {
      row_limit = std::atoi(argv[++index]);
    } else if (arg[0] != '-' && model == nullptr) {
      model = arg;
    } else {
      std::fprintf(stderr,
                   "usage: %s --model GGUF --tensor NAME --input in.f32 "
                   "--output out.f32 [--staging q8_1.bin] [--row-limit N]\n"
                   "   or: %s --weights bytes --type Q4_K --rows R --columns K "
                   "--input in.f32 --output out.f32\n",
                   argv[0], argv[0]);
      return 2;
    }
  }
  if (input_path == nullptr || output_path == nullptr) {
    std::fprintf(stderr, "input and output paths are required\n");
    return 2;
  }
  std::vector<float> input;
  if (!load_f32_file(input_path, &input)) {
    std::fprintf(stderr, "cannot read input FP32 file\n");
    return 1;
  }

  MappedGguf mapped;
  const std::uint8_t* weight_data = nullptr;
  std::size_t weight_bytes = 0;
  ggml_type weight_type = GGML_TYPE_F32;
  int64_t ne0 = columns;
  int64_t ne1 = rows;
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
  }
  if (static_cast<int64_t>(input.size()) != ne0) {
    std::fprintf(stderr, "input length %zu != K %lld\n", input.size(),
                 static_cast<long long>(ne0));
    return 1;
  }
  if (row_limit > 0 && static_cast<int64_t>(row_limit) < ne1) {
    const std::size_t row_bytes =
        weight_bytes / static_cast<std::size_t>(ne1);
    weight_bytes = row_bytes * static_cast<std::size_t>(row_limit);
    ne1 = row_limit;
  }

  ggml_backend_t backend = ggml_backend_cuda_init(0);
  if (backend == nullptr) {
    std::fprintf(stderr, "ggml_backend_cuda_init failed\n");
    return 1;
  }

  ggml_init_params params{};
  params.mem_size = 64 * 1024 * 1024;
  params.no_alloc = true;
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) {
    ggml_backend_free(backend);
    return 1;
  }
  ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, ne0, ne1);
  ggml_tensor* src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
  ggml_set_input(src);
  ggml_tensor* dst = ggml_mul_mat(ctx, weight, src);
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

  std::vector<float> output(static_cast<std::size_t>(ne1), 0.0F);
  ggml_backend_tensor_get(dst, output.data(), 0,
                           output.size() * sizeof(float));
  if (!write_f32_file(output_path, output.data(), output.size())) {
    std::fprintf(stderr, "cannot write output\n");
    return 1;
  }
  bool staging_ok = staging_path == nullptr;
  if (staging_path != nullptr) {
    // This llama revision's CUDA backend aborts on ggml_cpy F32->Q8_1
    // (cpy.cu unsupported type combination). Do not call it. Q8_1 sum
    // semantics are audited from pinned quantize.cu versus Quartz kernels.
    std::fprintf(stderr,
                 "skipping Q8_1 ggml_cpy staging export; CUDA f32->q8_1 is "
                 "unsupported in this revision\n");
    staging_ok = false;
  }

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-059\","
      "\"engine\":\"llama.cpp_gpu\",\"llama_revision\":\"%s\","
      "\"not_cpu_replica\":true,\"tensor\":\"%s\",\"type\":\"%s\","
      "\"ne0\":%lld,\"ne1\":%lld,\"rows_exported\":%lld,"
      "\"output\":\"%s\",\"staging\":\"%s\",\"status\":\"ok\"}\n",
      kPrefix, kRevision, tensor != nullptr ? tensor : "supplied_bytes",
      ggml_type_name(weight_type), static_cast<long long>(ne0),
      static_cast<long long>(ne1),
      static_cast<long long>(ne1), output_path,
      staging_path == nullptr ? "not_requested"
                              : (staging_ok ? "exported" : "unsupported_cuda_f32_q8_1"));
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(backend);
  return 0;
}
