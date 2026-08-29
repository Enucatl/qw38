#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>

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
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
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
  output << "{\n  \"schema_version\": 1,\n";
  output << "  \"model_sha256\": \"" << kModelSha256 << "\",\n";
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
  if (argc == 3 && std::string(argv[1]) == "--check-contract") {
    qw38::internal::ModelInfo info;
    qw38::Status status = qw38::internal::inspect_gguf(argv[2], &info);
    if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
    if (!status.is_ok()) {
      std::cerr << qw38::status_code_name(status.code()) << ": "
                << status.message() << '\n';
      return 1;
    }
    std::cout << "contract=qwen3.8-27b-q4_k_m\n";
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "--inventory-gguf") {
    return write_inventory(argv[2], argv[3]);
  }
  std::cout << kBrand << '\n';
  std::cerr << "qw38-eval: use --build-info, --inspect-gguf PATH, --sha256 PATH, "
               "--verify-model PATH, --check-contract PATH, or "
               "--inventory-gguf PATH OUTPUT\n";
  return 2;
}
