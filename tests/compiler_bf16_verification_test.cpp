#include "compiler/compiler.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstddef>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

using qw38::compiler::current_peak_rss_bytes;
using qw38::compiler::verify_bf16_tensor;
using qw38::format::MappingKind;
using qw38::format::TensorRecord;

namespace {

int failures = 0;

void expect(bool good, std::string_view description) {
  if (!good) {
    std::cerr << "FAIL: " << description << '\n';
    ++failures;
  }
}

TensorRecord record(MappingKind mapping, std::uint64_t n, std::uint64_t k) {
  TensorRecord out{};
  out.logical_name = "bf16.fixture";
  out.mapping.kind = mapping;
  out.shape.rank = mapping == MappingKind::TapMajorConvC1T ? 3 : 2;
  out.shape.logical = mapping == MappingKind::TapMajorConvC1T
                          ? std::array<std::uint64_t, 8>{n, 1, k}
                          : std::array<std::uint64_t, 8>{n, k};
  return out;
}

void check_corruption(MappingKind mapping, std::uint64_t n, std::uint64_t k) {
  auto rec = record(mapping, n, k);
  std::vector<std::byte> source(n * k * 2);
  std::vector<std::byte> payload(source);
  expect(static_cast<bool>(verify_bf16_tensor(rec, payload, source)),
         "valid BF16 layout verifies");
  payload[payload.size() / 2] ^= std::byte{1};
  expect(!verify_bf16_tensor(rec, payload, source),
         "BF16 layout detects one corrupt byte");
}

void check_isolated_memory(MappingKind mapping, std::uint64_t n,
                           std::uint64_t k) {
  auto const child = fork();
  if (child < 0) {
    expect(false, "fork isolated verification process");
    return;
  }
  if (child == 0) {
    auto rec = record(mapping, n, k);
    std::vector<std::byte> source(n * k * 2);
    std::vector<std::byte> payload(source);
    // Touch every page before taking the baseline. The measured delta is
    // verifier-owned scratch, not mapped input/output page residency.
    auto* source_pages = static_cast<volatile std::byte*>(source.data());
    auto* payload_pages = static_cast<volatile std::byte*>(payload.data());
    for (std::size_t i = 0; i < source.size(); i += 4096) {
      source_pages[i] = std::byte{0};
      payload_pages[i] = std::byte{0};
    }
    auto const before = current_peak_rss_bytes();
    auto verified = verify_bf16_tensor(rec, payload, source);
    auto const after = current_peak_rss_bytes();
    auto const delta = after >= before ? after - before : UINT64_MAX;
    std::cout << "bf16_verifier_owned_peak_rss_delta_bytes mapping="
              << static_cast<unsigned>(mapping) << " input_bytes="
              << source.size() << " delta=" << delta << '\n' << std::flush;
    _exit(verified && before != 0 && delta <= (4ULL << 20) ? 0 : 1);
  }
  int status = 0;
  expect(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
             WEXITSTATUS(status) == 0,
         "isolated BF16 verification stays within 4 MiB owned scratch");
}

}  // namespace

int main() {
  check_corruption(MappingKind::Identity, 8, 256);
  check_corruption(MappingKind::DenseTileNK, 8, 256);
  check_corruption(MappingKind::TapMajorConvC1T, 32, 4);
  auto invalid = record(MappingKind::DenseTileNK, 8, 256);
  std::vector<std::byte> small(16);
  expect(!verify_bf16_tensor(invalid, small, small),
         "BF16 verifier rejects geometry larger than spans");
  invalid.shape.logical[0] = UINT64_MAX;
  expect(!verify_bf16_tensor(invalid, small, small),
         "BF16 verifier rejects overflowing geometry");

  for (auto mapping : {MappingKind::Identity, MappingKind::DenseTileNK}) {
    check_isolated_memory(mapping, 512, 8192);
    check_isolated_memory(mapping, 2048, 8192);
  }
  return failures == 0 ? 0 : 1;
}
