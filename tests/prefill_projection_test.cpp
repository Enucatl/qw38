#include "compiler/quantization/quantizer.hpp"
#include "compiler/quantization/reference.hpp"
#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "cuda/prefill.hpp"
#include "cuda/upload.hpp"
#include "format/floatcvt.hpp"
#include "format/pack.hpp"
#include "format/unpack.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

using qw38::cuda::PrefillEngine;
using qw38::cuda::PrefillEpilogue;
using qw38::cuda::PrefillProjection;
using qw38::cuda::PrefillWeight;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;

int failures = 0;
void expect(bool condition, char const* message) {
  if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

template <class T>
std::span<std::byte const> bytes(std::vector<T> const& v) {
  return {reinterpret_cast<std::byte const*>(v.data()), v.size() * sizeof(T)};
}

std::vector<std::uint16_t> make_bf16(std::size_t count, int seed) {
  std::vector<std::uint16_t> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    float const x = 0.07f * static_cast<float>(static_cast<int>((i * 17u + seed) % 37u) - 18);
    out[i] = qw38::format::fp32_to_bf16_rne(x);
  }
  return out;
}

void exercise(LogicalQuantizerId quantizer, PhysicalLayoutId layout,
              std::uint32_t n, std::uint32_t k, std::uint32_t m,
              PrefillEngine& engine, qw38::cuda::Stream const& stream) {
  auto source = make_bf16(static_cast<std::size_t>(n) * k, 2);
  auto logical = qw38::compiler::quantize_bf16(quantizer, n, k, bytes(source));
  expect(bool(logical), "quantize source");
  if (!logical) return;
  auto packed = qw38::format::pack_cuda_v0(quantizer, layout, *logical);
  expect(bool(packed), "pack source");
  if (!packed) return;
  auto reconstructed = qw38::compiler::dequantize_to_bf16(*logical);
  expect(bool(reconstructed), "reconstruct source");
  if (!reconstructed) return;
  auto input = make_bf16(static_cast<std::size_t>(m) * k, 8);
  auto dc = qw38::cuda::upload(packed->codes, stream);
  auto ds = qw38::cuda::upload(packed->scales, stream);
  auto dx = qw38::cuda::upload(bytes(input), stream);
  std::vector<float> output_initial(static_cast<std::size_t>(m + 1u) * n, 123.0f);
  auto dy = qw38::cuda::upload(bytes(output_initial), stream);
  expect(bool(dc) && bool(ds) && bool(dx) && bool(dy), "allocate and upload");
  if (!dc || !ds || !dx || !dy) return;
  auto w = PrefillWeight{.codes = dc->data(), .scales = ds->data(),
      .layout = static_cast<std::uint16_t>(layout),
      .quantizer = static_cast<std::uint16_t>(quantizer),
      .n = n, .k = k,
      .padded_n = static_cast<std::uint32_t>(packed->padded_n),
      .padded_k = static_cast<std::uint32_t>(packed->padded_k),
      .codes_bytes = packed->codes.size(),
      .scales_bytes = packed->scales.size()};
  auto d = PrefillProjection{.weight = w,
      .input = static_cast<std::uint16_t const*>(dx->data()),
      .output = dy->data(), .valid_tokens = m,
      .first_position = 317, .epilogue = PrefillEpilogue::StoreFp32};
  auto const alloc_before = qw38::cuda::malloc_count();
  auto st = engine.project(d);
  expect(bool(st), "projection launch");
  expect(qw38::cuda::malloc_count() == alloc_before,
         "projection reuses preallocated device buffers");
  if (!st) { std::cerr << qw38::cuda::error_message(st.error()) << '\n'; return; }
  std::vector<float> got(static_cast<std::size_t>(m) * n);
  st = qw38::cuda::copy_d2h(got.data(), dy->data(), got.size() * 4u, stream);
  expect(bool(st) && bool(stream.sync()), "download projection");
  if (!st) return;
  std::vector<float> tail(n);
  auto const* tail_device = static_cast<float const*>(dy->data()) + static_cast<std::size_t>(m) * n;
  st = qw38::cuda::copy_d2h(tail.data(), tail_device, tail.size() * 4u, stream);
  expect(bool(st) && bool(stream.sync()), "download output guard row");
  for (float value : tail) expect(value == 123.0f, "no write beyond valid tokens");
  for (std::uint32_t token = 0; token < m; ++token) {
    for (std::uint32_t row = 0; row < n; ++row) {
      float ref = 0.0f;
      for (std::uint32_t col = 0; col < k; ++col) {
        float const x = qw38::format::bf16_to_fp32(input[static_cast<std::size_t>(token) * k + col]);
        float const y = qw38::format::bf16_to_fp32((*reconstructed)[static_cast<std::size_t>(row) * k + col]);
        ref = std::fma(x, y, ref);
      }
      float const diff = std::fabs(got[static_cast<std::size_t>(token) * n + row] - ref);
      if (diff > 0.02f + 0.002f * std::fabs(ref)) {
        expect(false, "independent reconstructed contraction");
        return;
      }
    }
  }
  auto dbf16 = qw38::cuda::DeviceBuffer::allocate(static_cast<std::uint64_t>(m) * n * 2u);
  expect(bool(dbf16), "BF16 output allocation");
  if (!dbf16) return;
  d.output = dbf16->data();
  d.epilogue = PrefillEpilogue::StoreBf16;
  st = engine.project(d);
  expect(bool(st), "BF16 projection launch");
  if (!st) return;
  std::vector<std::uint16_t> got_bf16(got.size());
  st = qw38::cuda::copy_d2h(got_bf16.data(), dbf16->data(),
                             got_bf16.size() * 2u, stream);
  expect(bool(st) && bool(stream.sync()), "download BF16 projection");
  if (!st) return;
  for (std::size_t i = 0; i < got.size(); ++i)
    expect(got_bf16[i] == qw38::format::fp32_to_bf16_rne(got[i]),
           "BF16 store uses RNE on FP32 accumulator");
  d.output = dy->data();
  d.epilogue = PrefillEpilogue::StoreFp32;
  std::vector<float> residual(got.size(), 0.25f);
  auto dr = qw38::cuda::upload(bytes(residual), stream);
  expect(bool(dr), "residual upload");
  if (!dr) return;
  d.residual = static_cast<float const*>(dr->data());
  d.epilogue = PrefillEpilogue::ResidualAddFp32;
  st = engine.project(d);
  expect(bool(st), "residual launch");
  if (!st) return;
  std::vector<float> added(got.size());
  st = qw38::cuda::copy_d2h(added.data(), dy->data(), added.size() * 4u, stream);
  expect(bool(st) && bool(stream.sync()), "download residual");
  for (std::size_t i = 0; i < got.size(); ++i)
    expect(std::fabs(added[i] - (got[i] + 0.25f)) < 1.0e-4f, "residual epilogue");
  d.valid_tokens = 0;
  expect(!engine.project(d), "reject zero tokens");
  d.valid_tokens = m;
  d.output = dx->data();
  expect(!engine.project(d), "reject overlapping input and output");
  d.output = dy->data();
  d.valid_tokens = engine.token_capacity() + 1;
  expect(!engine.project(d), "reject excess tokens");
  d.valid_tokens = m;
  d.residual = nullptr;
  d.epilogue = PrefillEpilogue::StoreFp32;
  d.input = static_cast<std::uint16_t const*>(dx->data()) + 1;
  expect(!engine.project(d), "reject short input device allocation");
  d.input = static_cast<std::uint16_t const*>(dx->data());
  d.output = got.data();
  expect(!engine.project(d), "reject host output pointer");
  d.output = static_cast<std::byte*>(dy->data()) + 1;
  expect(!engine.project(d), "reject misaligned output pointer");
  d.output = dy->data();
  w.codes = static_cast<std::byte const*>(dc->data()) + 2;
  d.weight = w;
  expect(!engine.project(d), "reject short weight allocation");
  w.codes = dc->data();
  w.quantizer = 0;
  d.weight = w;
  d.valid_tokens = m;
  expect(!engine.project(d), "reject mismatched layout and quantizer");
}

void exercise_dense(PrefillEngine& engine, qw38::cuda::Stream const& stream) {
  constexpr std::uint32_t n = 8, k = 256, m = 3;
  auto weight = make_bf16(n * k, 13);
  auto input = make_bf16(m * k, 29);
  auto packed = qw38::format::pack_bf16_dense_tile_v0(bytes(weight), n, k);
  expect(bool(packed), "pack BF16 dense control");
  if (!packed) return;
  auto dw = qw38::cuda::upload(packed->codes, stream);
  auto dx = qw38::cuda::upload(bytes(input), stream);
  auto dy = qw38::cuda::DeviceBuffer::allocate(m * n * 4u);
  if (!dw || !dx || !dy) { expect(false, "allocate BF16 dense control"); return; }
  PrefillWeight const w{.codes = dw->data(),
      .layout = static_cast<std::uint16_t>(PhysicalLayoutId::CudaBf16DenseTileV0),
      .quantizer = static_cast<std::uint16_t>(LogicalQuantizerId::None),
      .n = n, .k = k,
      .padded_n = static_cast<std::uint32_t>(packed->padded_n),
      .padded_k = static_cast<std::uint32_t>(packed->padded_k),
      .codes_bytes = packed->codes.size()};
  auto st = engine.project(PrefillProjection{
      .weight = w, .input = static_cast<std::uint16_t const*>(dx->data()),
      .output = dy->data(), .valid_tokens = m,
      .epilogue = PrefillEpilogue::StoreFp32});
  expect(bool(st), "BF16 dense projection");
  if (!st) return;
  std::vector<float> got(m * n);
  st = qw38::cuda::copy_d2h(got.data(), dy->data(), got.size() * 4u, stream);
  if (!st || !stream.sync()) { expect(false, "download BF16 dense result"); return; }
  for (std::uint32_t t = 0; t < m; ++t) {
    for (std::uint32_t row = 0; row < n; ++row) {
      float ref = 0.0f;
      for (std::uint32_t col = 0; col < k; ++col)
        ref = std::fma(qw38::format::bf16_to_fp32(input[t * k + col]),
                       qw38::format::bf16_to_fp32(weight[row * k + col]), ref);
      expect(std::fabs(got[t * n + row] - ref) < 0.02f + 0.002f * std::fabs(ref),
             "BF16 dense independent contraction");
    }
  }
}

}  // namespace

int main() {
  auto stream = qw38::cuda::Stream::create();
  expect(bool(stream), "stream creation");
  if (!stream) return 1;
  auto engine = PrefillEngine::create(*stream, 128);
  expect(bool(engine), "prefill workspace creation");
  if (!engine) return 1;
  exercise(LogicalQuantizerId::Q4KCandidateV2,
           PhysicalLayoutId::CudaQ4KCandidateV2, 5, 256, 3, *engine, *stream);
  exercise(LogicalQuantizerId::Q8G32CandidateV1,
           PhysicalLayoutId::CudaQ8G32CandidateV1, 13, 512, 17, *engine, *stream);
  exercise(LogicalQuantizerId::Q8G32CandidateV1,
           PhysicalLayoutId::CudaQ8G32CandidateV1, 513, 256, 128, *engine, *stream);
  exercise_dense(*engine, *stream);
  expect(engine->workspace_bytes() < 32u * 1024u * 1024u,
         "bounded workspace below 32 MiB");
  auto large_engine = PrefillEngine::create(*stream, 1024, 512);
  expect(bool(large_engine), "1024-token workspace creation");
  if (large_engine) {
    exercise(LogicalQuantizerId::Q4KCandidateV2,
             PhysicalLayoutId::CudaQ4KCandidateV2, 5, 256, 1023, *large_engine, *stream);
    exercise(LogicalQuantizerId::Q8G32CandidateV1,
             PhysicalLayoutId::CudaQ8G32CandidateV1, 13, 256, 1024, *large_engine, *stream);
    auto moved = std::move(*large_engine);
    exercise(LogicalQuantizerId::Q4KCandidateV2,
             PhysicalLayoutId::CudaQ4KCandidateV2, 5, 256, 3, moved, *stream);
    expect(bool(stream->close()), "close borrowed stream after completed work");
    expect(!moved.project({}), "reject a closed borrowed stream");
  }
  if (failures) std::cerr << failures << " failures\n";
  return failures ? 1 : 0;
}
