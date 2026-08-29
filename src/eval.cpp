#include <iostream>
#include <string>

#include "qw38/engine.h"
#include "model.h"
#include "sha256.h"

namespace {
constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr const char* kModelRevision =
    "0669b98607d47046c7c2b3f801011d54a08cfccf";
constexpr const char* kModelSha256 =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--build-info") {
    std::cout << "brand=" << kBrand << '\n';
    std::cout << "cxx=17\n";
    std::cout << "cuda_target=sm_120\n";
    std::cout << "model_revision=" << kModelRevision << '\n';
    std::cout << "model_sha256=" << kModelSha256 << '\n';
    return 0;
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
  if (argc == 3 && std::string(argv[1]) == "--verify-model") {
    qw38::Engine engine;
    const qw38::Status status = qw38::Engine::open(argv[2], &engine);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "verified=pinned-qwen3.8-27b-q4_k_m\n";
    return 0;
  }
  std::cout << kBrand << '\n';
  std::cerr << "qw38-eval: use --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "or --verify-model PATH\n";
  return 2;
}
