#include "compiler/quantization/q4k.hpp"
#include "compiler/quantization/quantizer.hpp"
#include "compiler/quantization/reference.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"
#include "format/unpack.hpp"
#include "../third_party/llama.cpp-q4k/reference.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {
int failures = 0;
bool saw_subnormal_d = false;
bool saw_subnormal_min = false;
bool saw_recalculated_code = false;
void expect(bool condition, std::string_view message) {
  if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

void check_production_wrappers(std::vector<float> const& input,
                               std::uint64_t n, std::uint64_t k) {
  using namespace qw38::format;
  std::vector<std::byte> source(input.size() * 2);
  std::vector<float> bf16_source(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    auto bits = fp32_to_bf16_rne(input[i]);
    store_u16_le(source.data() + 2 * i, bits);
    bf16_source[i] = bf16_to_fp32(bits);
  }
  constexpr auto quantizer = LogicalQuantizerId::Q4KCandidateV2;
  constexpr auto layout = PhysicalLayoutId::CudaQ4KCandidateV2;
  auto logical = qw38::compiler::quantize_bf16(quantizer, n, k, source);
  expect(bool(logical), "BF16 production quantization succeeds");
  if (!logical) return;
  std::vector<pinned_llama_q4k::block_q4_K> reference(input.size() / 256);
  std::vector<float> decoded(input.size());
  pinned_llama_q4k::quantize_row_q4_K_ref(bf16_source.data(), reference.data(), input.size());
  pinned_llama_q4k::dequantize_row_q4_K(reference.data(), decoded.data(), input.size());
  for (std::size_t b = 0; b < reference.size(); ++b) {
    auto const& block = reference[b];
    auto const* meta = logical->q4k_metadata.data() + 16 * b;
    expect(meta[0] == (block.d & 255) && meta[1] == (block.d >> 8) &&
           meta[2] == (block.dmin & 255) && meta[3] == (block.dmin >> 8),
           "BF16 quantizer exact superblock scales");
    for (int j = 0; j < 12; ++j) expect(meta[j + 4] == block.scales[j], "BF16 quantizer exact six-bit metadata");
    for (int j = 0; j < 256; ++j) {
      auto byte = block.qs[(j / 64) * 32 + j % 32];
      int code = j % 64 < 32 ? byte & 15 : byte >> 4;
      expect(logical->codes[256 * b + j] == code, "BF16 quantizer exact codes");
    }
  }
  auto direct = qw38::compiler::dequantize_to_bf16(*logical);
  expect(bool(direct), "production logical decoder succeeds");
  auto packed = pack_cuda_v0(quantizer, layout, *logical);
  expect(bool(packed), "production Q4_K packing succeeds");
  if (!packed) return;
  auto unpacked = unpack_cuda_v0(quantizer, layout, n, k, packed->codes, packed->scales);
  expect(bool(unpacked), "independent production unpacker succeeds");
  if (!unpacked) return;
  expect(unpacked->codes == logical->codes && unpacked->q4k_metadata == logical->q4k_metadata,
         "physical representation preserves exact pinned codes and metadata");
  auto roundtrip = qw38::compiler::dequantize_to_bf16(*unpacked);
  expect(bool(roundtrip), "unpacked production decoder succeeds");
  if (!direct || !roundtrip) return;
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    auto expected = fp32_to_bf16_rne(decoded[i]);
    expect((*direct)[i] == expected && (*roundtrip)[i] == expected,
           "both production decoder paths equal independently pinned decoded BF16");
  }
}

void check(std::array<float, 256> const& input, std::string_view name) {
  check_production_wrappers({input.begin(), input.end()}, 1, 256);
  std::array<std::int8_t, 256> codes{};
  std::array<std::uint8_t, 16> metadata{};
  auto result = qw38::compiler::quantize_q4k_block(input, codes, metadata);
  if (!result) { expect(false, name); return; }
  pinned_llama_q4k::block_q4_K reference{};
  pinned_llama_q4k::quantize_row_q4_K_ref(input.data(), &reference, 256);
  expect(metadata[0] == (reference.d & 255) && metadata[1] == (reference.d >> 8) &&
         metadata[2] == (reference.dmin & 255) && metadata[3] == (reference.dmin >> 8), name);
  for (int i = 0; i < 12; ++i) expect(metadata[4 + i] == reference.scales[i], name);
  for (int i = 0; i < 256; ++i) {
    auto byte = reference.qs[(i / 64) * 32 + i % 32];
    auto code = (i % 64 < 32) ? byte & 15 : byte >> 4;
    expect(codes[i] == code, name);
  }
  std::array<float, 256> decoded{};
  pinned_llama_q4k::dequantize_row_q4_K(&reference, decoded.data(), 256);
  float d = qw38::format::fp16_to_fp32(metadata[0] | (metadata[1] << 8));
  float dmin = qw38::format::fp16_to_fp32(metadata[2] | (metadata[3] << 8));
  for (int i = 0; i < 256; ++i) {
    int g = i / 32;
    unsigned scale = g < 4 ? metadata[4 + g] & 63 :
        (metadata[8 + g] & 15) | ((metadata[g] >> 6) << 4);
    unsigned minimum = g < 4 ? metadata[8 + g] & 63 :
        (metadata[8 + g] >> 4) | ((metadata[4 + g] >> 6) << 4);
    float actual = (d * scale) * codes[i] - dmin * minimum;
    expect(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(decoded[i]), name);
    expect(qw38::format::fp32_to_bf16_rne(actual) ==
           qw38::format::fp32_to_bf16_rne(decoded[i]), "same decoded BF16 weight operand");
  }
  saw_subnormal_d |= reference.d > 0 && reference.d < 0x400;
  saw_subnormal_min |= reference.dmin > 0 && reference.dmin < 0x400;
  for (int g = 0; g < 8; ++g) {
    std::array<std::uint8_t, 32> initial{}, auxiliary{};
    std::array<float, 32> weights{};
    float sum = 0;
    for (int i = 0; i < 32; ++i) sum += input[32 * g + i] * input[32 * g + i];
    for (int i = 0; i < 32; ++i) weights[i] = sqrtf(sum / 32) + fabsf(input[32 * g + i]);
    float minimum;
    pinned_llama_q4k::make_qkx2_quants(32, 15, input.data() + 32 * g, weights.data(),
        initial.data(), &minimum, auxiliary.data(), -1.f, 0.1f, 20, false);
    for (int i = 0; i < 32; ++i) saw_recalculated_code |= initial[i] != codes[32 * g + i];
  }
}
}  // namespace

int main() {
  // Exhaustive finite half decoding checks catch the factor-of-two error at
  // every subnormal, including 0x0001 -> 2^-24 and both signs.
  for (unsigned bits = 0; bits < 65536; ++bits) {
    if ((bits & 0x7C00) == 0x7C00) continue;
    float expected = pinned_llama_q4k::ggml_compute_fp16_to_fp32(bits);
    expect(std::bit_cast<std::uint32_t>(qw38::format::fp16_to_fp32(bits)) ==
           std::bit_cast<std::uint32_t>(expected), "exhaustive finite FP16 decode");
  }
  std::array<float, 256> input{};
  check(input, "zero");
  for (float constant : {1.0f, -1.0f, 0.03125f, -0.03125f, 1.0e-5f, -1.0e-5f,
                         1.0e-20f, -1.0e-20f, 1.0e-30f, -1.0e-30f}) {
    input.fill(constant); check(input, "constant");
  }
  for (int i = 0; i < 256; ++i) input[i] = float(i % 32 - 7) * (1 + i / 32) / 19.f;
  check(input, "asymmetric groups");
  input.fill(0.01f); input[0] = -99.f; input[37] = 127.f; input[255] = -1.7f;
  check(input, "outliers");
  for (int i = 0; i < 256; ++i) input[i] = float(i % 32 - 9) * (1 + i / 32) * 0x1p-22f;
  check(input, "subnormal superblock scales");
  // Deterministic heterogeneous BF16 blocks cover source-checkpoint precision,
  // all six scale/min bits, and final code changes after scale compression.
  std::uint32_t state = 0x912782AC;
  for (int block = 0; block < 256; ++block) {
    for (int i = 0; i < 256; ++i) {
      state = state * 1664525u + 1013904223u;
      float value = float(int(state >> 16) - 30000) / 16000.f;
      value = std::ldexp(value, (i / 32 + block) % 16 - 12);
      input[i] = qw38::format::bf16_to_fp32(qw38::format::fp32_to_bf16_rne(value));
    }
    check(input, "original BF16 operands");
  }
  expect(saw_subnormal_d && saw_subnormal_min, "nonzero FP16 subnormal scale and minimum exercised");
  expect(saw_recalculated_code, "fixtures require code recalculation after scale compression");
  std::vector<float> tiled(9 * 512);
  for (std::size_t i = 0; i < tiled.size(); ++i) {
    tiled[i] = std::ldexp(float(int(i % 43) - 11), int(i / 32 % 13) - 19);
  }
  check_production_wrappers(tiled, 9, 512);
  std::array<std::int8_t, 256> codes{};
  std::array<std::uint8_t, 16> metadata{};
  input[0] = std::numeric_limits<float>::infinity();
  expect(!qw38::compiler::quantize_q4k_block(input, codes, metadata), "reject nonfinite input");
  input[0] = std::numeric_limits<float>::max();
  expect(!qw38::compiler::quantize_q4k_block(input, codes, metadata), "reject unrepresentable input");
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    -std::numeric_limits<float>::infinity()}) {
    input[0] = bad;
    auto result = qw38::compiler::quantize_q4k_block(input, codes, metadata);
    expect(!result && result.error().code == qw38::compiler::CompilerErrorCode::Nonfinite,
           "NaN and negative infinity return Nonfinite");
  }
  // Both fitting reciprocals and compression reciprocals can overflow for
  // tiny but finite BF16 source values. Those are outside the pinned fitter's
  // finite domain; reject instead of triggering nearest_int's assertion.
  for (float tiny : {0x1p-126f, -0x1p-126f, 0x1p-133f, -0x1p-133f}) {
    input.fill(tiny);
    auto result = qw38::compiler::quantize_q4k_block(input, codes, metadata);
    expect(!result && result.error().code == qw38::compiler::CompilerErrorCode::Unrepresentable,
           "tiny finite BF16 outside pinned reciprocal domain is rejected");
  }
  std::array<std::byte, 512> invalid_bf16{};
  for (std::uint16_t bits : {std::uint16_t{0x7FC0}, std::uint16_t{0x7F80},
                             std::uint16_t{0xFF80}, std::uint16_t{0x7F7F},
                             std::uint16_t{0x0080}, std::uint16_t{0x0001}}) {
    for (std::size_t i = 0; i < 256; ++i) {
      qw38::format::store_u16_le(invalid_bf16.data() + i * 2, bits);
    }
    auto result = qw38::compiler::quantize_bf16(
        qw38::format::LogicalQuantizerId::Q4KCandidateV2, 1, 256, invalid_bf16);
    auto expected = (bits & 0x7F80) == 0x7F80
        ? qw38::compiler::CompilerErrorCode::Nonfinite
        : qw38::compiler::CompilerErrorCode::Unrepresentable;
    expect(!result && result.error().code == expected,
           "BF16 entry point rejects nonfinite or undefined fitting domain");
  }
  if (failures) return 1;
  std::cout << "Q4_K pinned-reference exact block, reconstructed FP32/BF16, and FP16 checks passed\n";
}
