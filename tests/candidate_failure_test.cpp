#include "compiler/compiler.hpp"
#include "cuda/alloc.hpp"
#include "format/reader.hpp"
#include "format_reader_support.hpp"
#include "runtime/runtime.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <string_view>
#include <vector>

namespace {

std::atomic<std::uint64_t> g_fail_after_malloc_count{
    std::numeric_limits<std::uint64_t>::max()};
int g_failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
  }
}

qw38::compiler::ExpectedTensor expected(qw38::compiler::TensorFamily family,
                                         std::uint64_t n, std::uint64_t k) {
  for (auto tensor : qw38::compiler::expand_identity_table()) {
    if (tensor.family == family) {
      tensor.shape.rank = 2;
      tensor.shape.dims = {n, k, 0};
      return tensor;
    }
  }
  return {};
}

qw38::compiler::SyntheticTensor tensor(qw38::compiler::TensorFamily family,
                                        std::uint64_t n, std::uint64_t k) {
  qw38::compiler::SyntheticTensor result{};
  result.expected = expected(family, n, k);
  result.bytes.resize(static_cast<std::size_t>(n * k * 2));
  for (std::size_t i = 0; i < result.bytes.size(); i += 2) {
    result.bytes[i] = std::byte{0x80};
    result.bytes[i + 1] = std::byte{0x3F}; // BF16 1.0
  }
  return result;
}

qw38::format::Hash256 seed_hash(std::uint8_t seed) {
  qw38::format::Hash256 hash{};
  hash.bytes[0] = seed;
  return hash;
}

} // namespace

void* operator new(std::size_t bytes) {
  auto threshold = g_fail_after_malloc_count.load(std::memory_order_relaxed);
  if (qw38::cuda::malloc_count() > threshold &&
      g_fail_after_malloc_count.compare_exchange_strong(
          threshold, std::numeric_limits<std::uint64_t>::max(),
          std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (auto* pointer = std::malloc(bytes == 0 ? 1 : bytes)) return pointer;
  throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

int main() {
  using qw38::compiler::WeightFormatPolicy;
  using qw38::format::Artifact;
  using qw38::runtime::ErrorCode;

  qw38::format::test::ScratchDir dir{"qw38-candidate-failure"};
  auto const path = dir.file("candidate.qw38");
  std::vector<qw38::compiler::SyntheticTensor> tensors;
  tensors.push_back(tensor(qw38::compiler::TensorFamily::Embed, 8, 8));
  tensors.back().expected.layout =
      qw38::format::PhysicalLayoutId::CudaBf16RowMajorV0;
  tensors.push_back(tensor(qw38::compiler::TensorFamily::LmHead, 8, 256));
  tensors.push_back(tensor(qw38::compiler::TensorFamily::MlpDownProj, 8, 256));
  qw38::format::CompilerRevision const revision{
      .ident = qw38::compiler::kCandidateCompilerIdent,
      .major = 0, .minor = 1, .patch = 1};
  auto compiled = qw38::compiler::compile_synthetic(
      path, seed_hash(1), seed_hash(2), seed_hash(3), revision,
      std::move(tensors), WeightFormatPolicy::CandidateV1);
  expect(static_cast<bool>(compiled), "candidate fixture compiles");
  if (!compiled) return 1;
  auto artifact = Artifact::open(path);
  expect(static_cast<bool>(artifact), "candidate fixture opens");
  if (!artifact) return 1;
  auto runtime = qw38::runtime::Runtime::create();
  expect(static_cast<bool>(runtime), "runtime creates");
  if (!runtime) return 1;

  auto const mallocs_before_bad = qw38::cuda::malloc_count();
  auto const live_before_bad = qw38::cuda::live_bytes();
  auto const* mlp = artifact->find_tensor(
      "model.language_model.layers.0.mlp.down_proj.weight");
  expect(mlp != nullptr, "candidate MLP tensor is present");
  if (!mlp) return 1;
  auto corrupt = qw38::format::test::read_all(path);
  corrupt[static_cast<std::size_t>(mlp->payload.offset)] = std::byte{0x88};
  auto const corrupt_path = dir.file("invalid-candidate.qw38");
  qw38::format::test::write_all(corrupt_path, corrupt);
  auto rejected = runtime->load(corrupt_path);
  expect(!rejected && rejected.error().code == ErrorCode::Format,
         "candidate payload corruption is rejected at production load boundary");
  expect(qw38::cuda::malloc_count() == mallocs_before_bad &&
             qw38::cuda::live_bytes() == live_before_bad,
         "candidate reader failure allocates no device memory");

  auto const mallocs_before_failure = qw38::cuda::malloc_count();
  auto const live_before_failure = qw38::cuda::live_bytes();
  g_fail_after_malloc_count.store(mallocs_before_failure + 1,
                                  std::memory_order_relaxed);
  auto failed = runtime->upload(*artifact);
  g_fail_after_malloc_count.store(std::numeric_limits<std::uint64_t>::max(),
                                  std::memory_order_relaxed);
  expect(!failed && failed.error().code == ErrorCode::AllocationFailed,
         "partial candidate upload failure is typed");
  expect(qw38::cuda::malloc_count() >= mallocs_before_failure + 2,
         "failure occurs after an earlier candidate span allocation");
  expect(qw38::cuda::live_bytes() == live_before_failure,
         "partial candidate upload releases all device buffers");

  {
    auto recovered = runtime->upload(*artifact);
    expect(static_cast<bool>(recovered), "candidate upload succeeds after failure");
    if (recovered) {
      expect(recovered->device_bytes() > 0 &&
                 qw38::cuda::live_bytes() ==
                     live_before_failure + recovered->device_bytes(),
             "candidate upload device accounting is exact");
    }
  }
  expect(qw38::cuda::live_bytes() == live_before_failure,
         "candidate upload releases buffers after use");
  if (g_failures != 0) {
    std::cerr << g_failures << " candidate failure checks failed\n";
    return 1;
  }
  return 0;
}
