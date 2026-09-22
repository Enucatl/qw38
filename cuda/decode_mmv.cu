#include "cuda/decode_mmv.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace qw38::cuda {
namespace {

enum class WeightKind { Q4, Q8, Bf16 };

__device__ __forceinline__ float bf16_to_fp32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

__device__ __forceinline__ std::uint16_t fp32_to_bf16_rne(float x) {
  std::uint32_t const bits = __float_as_uint(x);
  std::uint32_t const exp = (bits >> 23) & 0xFFu;
  if (exp == 0xFFu) {
    std::uint16_t h = static_cast<std::uint16_t>(bits >> 16);
    if ((bits & 0x7FFFFFu) != 0) {
      h = static_cast<std::uint16_t>(h | 0x0040u);
    }
    return h;
  }
  std::uint32_t const lsb = (bits >> 16) & 1u;
  std::uint32_t const add = 0x7FFFu + lsb;
  return static_cast<std::uint16_t>((bits + add) >> 16);
}

__device__ __forceinline__ float fp16_to_fp32(std::uint16_t h) {
  std::uint32_t const sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t const exp = (h >> 10) & 0x1Fu;
  std::uint32_t const man = h & 0x3FFu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      std::uint32_t m = man;
      std::uint32_t e = 127 - 15;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      bits = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  return __uint_as_float(bits);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  return v;
}

__device__ __forceinline__ float decode_scaled(int code, float scale) {
  float decoded = 0.0f;
  if (scale != 0.0f) {
    decoded = static_cast<float>(code) * scale;
  }
  return bf16_to_fp32(fp32_to_bf16_rne(decoded));
}

__device__ __forceinline__ float sigmoid_fp32(float u) {
  if (u >= 0.0f) {
    float const e = expf(-u);
    return 1.0f / (1.0f + e);
  }
  float const e = expf(u);
  return e / (1.0f + e);
}

__device__ __forceinline__ float silu_fp32(float z) {
  return z * sigmoid_fp32(z);
}

template <WeightKind Kind>
__device__ void decode8(std::byte const* codes, std::byte const* scales,
                        std::uint64_t tile_row, int lane, float out[8]) {
  if constexpr (Kind == WeightKind::Q4) {
    auto const* row = codes + tile_row * 128u + static_cast<std::uint32_t>(lane) * 4u;
    auto const word = *reinterpret_cast<std::uint32_t const*>(row);
    int const group = lane >> 3;
    std::uint16_t scale_bits = 0;
    if ((lane & 7) == 0) {
      auto const* sp = reinterpret_cast<std::uint16_t const*>(scales);
      scale_bits = sp[tile_row * 4u + static_cast<std::uint32_t>(group)];
    }
    scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);
    float const scale = fp16_to_fp32(scale_bits);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int const nib = static_cast<int>((word >> (4 * i)) & 0xFu);
      int const code = (nib << 28) >> 28;
      out[i] = decode_scaled(code, scale);
    }
  } else if constexpr (Kind == WeightKind::Q8) {
    auto const* row = codes + tile_row * 256u + static_cast<std::uint32_t>(lane) * 8u;
    auto const word = *reinterpret_cast<std::uint64_t const*>(row);
    int const group = lane >> 2;
    std::uint16_t scale_bits = 0;
    if ((lane & 3) == 0) {
      auto const* sp = reinterpret_cast<std::uint16_t const*>(scales);
      scale_bits = sp[tile_row * 8u + static_cast<std::uint32_t>(group)];
    }
    scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~3);
    float const scale = fp16_to_fp32(scale_bits);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      auto const code =
          static_cast<std::int8_t>(static_cast<std::uint8_t>((word >> (8 * i)) & 0xFFu));
      out[i] = decode_scaled(static_cast<int>(code), scale);
    }
  } else {
    auto const* row = reinterpret_cast<std::uint16_t const*>(
        codes + tile_row * 512u + static_cast<std::uint32_t>(lane) * 16u);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out[i] = bf16_to_fp32(row[i]);
    }
  }
}

__device__ void apply_epilogue(DecodeEpilogue epilogue, void* output,
                               float const* residual, std::uint32_t row,
                               float acc) {
  if (epilogue == DecodeEpilogue::StoreBf16) {
    static_cast<std::uint16_t*>(output)[row] = fp32_to_bf16_rne(acc);
  } else if (epilogue == DecodeEpilogue::StoreFp32) {
    static_cast<float*>(output)[row] = acc;
  } else {
    auto* destination = output == nullptr ? const_cast<float*>(residual)
                                          : static_cast<float*>(output);
    destination[row] = residual[row] + acc;
  }
}

template <WeightKind Kind, bool Paired>
__global__ void decode_mmv_kernel(std::byte const* codes_a, std::byte const* scales_a,
                                 std::byte const* codes_b, std::byte const* scales_b,
                                 std::uint16_t const* input, void* output_a,
                                 void* output_b, float const* residual_a,
                                 float const* residual_b,
                                 std::uint32_t n, std::uint32_t k, std::uint32_t padded_k,
                                 DecodeEpilogue epilogue) {
  std::uint32_t const tn = blockIdx.x;
  std::uint32_t const warp = threadIdx.x >> 5;
  int const lane = static_cast<int>(threadIdx.x & 31u);
  std::uint32_t const row = tn * 8u + warp;
  std::uint32_t const tiles_k = padded_k / static_cast<std::uint32_t>(kDecodeTileK);

  __shared__ alignas(16) std::uint16_t xs[kDecodeMaxK];
  for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    xs[i] = input[i];
  }
  for (std::uint32_t i = k + threadIdx.x; i < padded_k; i += blockDim.x) {
    xs[i] = 0;
  }
  __syncthreads();

  float acc_a = 0.0f;
  float acc_b = 0.0f;
  for (std::uint32_t tk = 0; tk < tiles_k; ++tk) {
    std::uint64_t const tile_row =
        (static_cast<std::uint64_t>(tn) * tiles_k + tk) * 8u + warp;
    float da[8];
    decode8<Kind>(codes_a, scales_a, tile_row, lane, da);
    float db[8];
    if constexpr (Paired) {
      decode8<Kind>(codes_b, scales_b, tile_row, lane, db);
    }
    std::uint32_t const k0 =
        tk * static_cast<std::uint32_t>(kDecodeTileK) + static_cast<std::uint32_t>(lane) * 8u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float const x = bf16_to_fp32(xs[k0 + static_cast<std::uint32_t>(i)]);
      acc_a = fmaf(da[i], x, acc_a);
      if constexpr (Paired) {
        acc_b = fmaf(db[i], x, acc_b);
      }
    }
  }

  acc_a = warp_sum(acc_a);
  if constexpr (Paired) {
    acc_b = warp_sum(acc_b);
  }
  if (lane == 0 && row < n) {
    if (epilogue == DecodeEpilogue::SwigluStoreBf16) {
      if constexpr (Paired) {
        float const prod = silu_fp32(acc_a) * acc_b;
        static_cast<std::uint16_t*>(output_a)[row] = fp32_to_bf16_rne(prod);
      }
    } else {
      apply_epilogue(epilogue, output_a, residual_a, row, acc_a);
      if constexpr (Paired) {
        apply_epilogue(epilogue, output_b, residual_b, row, acc_b);
      }
    }
  }
}

std::expected<void, Error> require_stream(Stream const& stream, std::string_view op) {
  if (stream.empty()) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "empty stream"));
  }
  return {};
}

bool layout_quantizer_ok(std::uint16_t layout, std::uint16_t quantizer) noexcept {
  if (layout == kDecodeLayoutQ4G64V0) {
    return quantizer == kDecodeQuantizerQ4G64V0;
  }
  if (layout == kDecodeLayoutQ8G32V0) {
    return quantizer == kDecodeQuantizerQ8G32V0;
  }
  if (layout == kDecodeLayoutBf16DenseTileV0) {
    return quantizer == kDecodeQuantizerNone;
  }
  return false;
}

DecodeDtype weight_dtype(std::uint16_t layout) noexcept {
  if (layout == kDecodeLayoutQ4G64V0) {
    return DecodeDtype::Q4;
  }
  if (layout == kDecodeLayoutQ8G32V0) {
    return DecodeDtype::Q8;
  }
  return DecodeDtype::Bf16;
}

bool same_shape(DecodeOperandView const& v, std::uint32_t logical_n,
                std::uint32_t logical_k, std::uint32_t padded_n,
                std::uint32_t padded_k) noexcept {
  return v.logical_n == logical_n && v.logical_k == logical_k &&
         v.padded_n == padded_n && v.padded_k == padded_k;
}

std::expected<void, Error> validate_view(
    DecodeOperandView const& v, DecodeDtype dtype, std::uint16_t layout,
    std::uint32_t logical_n, std::uint32_t logical_k,
    std::uint32_t padded_n, std::uint32_t padded_k, std::uint64_t bytes,
    std::uint32_t min_alignment, bool writable, std::string_view op,
    std::string_view name) {
  if (v.pointer == nullptr || v.space != DecodeMemorySpace::Device ||
      v.dtype != dtype || v.layout != layout ||
      !same_shape(v, logical_n, logical_k, padded_n, padded_k) ||
      v.bytes != bytes || v.alignment < min_alignment ||
      (v.alignment & (v.alignment - 1u)) != 0u ||
      (reinterpret_cast<std::uintptr_t>(v.pointer) % v.alignment) != 0u ||
      v.writable != writable) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, op,
        std::string(name) + " typed view does not match the required contract"));
  }
  return {};
}

bool empty_view(DecodeOperandView const& v) noexcept {
  return v == DecodeOperandView{};
}

bool overlaps(DecodeOperandView const& a, DecodeOperandView const& b) noexcept {
  if (a.pointer == nullptr || b.pointer == nullptr || a.bytes == 0 || b.bytes == 0) {
    return false;
  }
  auto const ab = reinterpret_cast<std::uintptr_t>(a.pointer);
  auto const bb = reinterpret_cast<std::uintptr_t>(b.pointer);
  if (a.bytes > std::numeric_limits<std::uintptr_t>::max() - ab ||
      b.bytes > std::numeric_limits<std::uintptr_t>::max() - bb) {
    return true;
  }
  return ab < bb + b.bytes && bb < ab + a.bytes;
}

std::expected<void, Error> validate_non_overlap(DecodeMmvDesc const& d,
                                                std::string_view op) {
  DecodeOperandView const* views[] = {
      &d.codes, &d.scales, &d.input, &d.output, &d.residual};
  for (std::size_t i = 0; i < std::size(views); ++i) {
    for (std::size_t j = i + 1; j < std::size(views); ++j) {
      if (overlaps(*views[i], *views[j])) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                          "operand views must not overlap"));
      }
    }
  }
  return {};
}

std::uint32_t group_size(std::uint16_t layout) noexcept {
  if (layout == kDecodeLayoutQ4G64V0) {
    return 64;
  }
  if (layout == kDecodeLayoutQ8G32V0) {
    return 32;
  }
  return 1;
}

std::expected<void, Error> validate_geometry(DecodeMmvDesc const& d,
                                             std::string_view op, bool paired_b) {
  if (!layout_quantizer_ok(d.layout, d.quantizer)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "layout/quantizer version mismatch"));
  }
  if (d.n == 0 || d.k == 0) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "N and K must be nonzero"));
  }
  if (d.n > kDecodeMaxN) {
    return std::unexpected(make_error(
        ErrorCode::Overflow, op,
        "N exceeds the largest supported row-tile-padded value"));
  }
  if (d.k > static_cast<std::uint32_t>(kDecodeMaxK) ||
      d.padded_k > static_cast<std::uint32_t>(kDecodeMaxK)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "K exceeds decode staging limit 17408"));
  }
  if (d.padded_n != decode_pad_n(d.n) || d.padded_k != decode_pad_k(d.k)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "padded N/K metadata does not match layout tiles"));
  }
  if (d.layout == kDecodeLayoutBf16DenseTileV0) {
    if (d.n % static_cast<std::uint32_t>(kDecodeTileRows) != 0 ||
        d.k % static_cast<std::uint32_t>(kDecodeTileK) != 0) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "BF16 dense tile requires N%8==0 and K%256==0"));
    }
  } else {
    auto const g = group_size(d.layout);
    if (d.k % g != 0) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "logical K must divide the quantizer group"));
    }
  }
  auto const want_codes = decode_code_bytes(d.layout, d.padded_n, d.padded_k);
  auto const want_scales = decode_scale_bytes(d.layout, d.padded_n, d.padded_k);
  auto st = validate_view(d.codes, weight_dtype(d.layout), d.layout, d.n, d.k,
                          d.padded_n, d.padded_k, want_codes, 16, false, op,
                          "codes");
  if (!st) {
    return st;
  }
  if (want_scales == 0) {
    if (!empty_view(d.scales)) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "BF16 dense tile must not supply scales"));
    }
  } else {
    st = validate_view(d.scales, DecodeDtype::Fp16, d.layout, d.padded_n,
                       d.padded_k / group_size(d.layout), d.padded_n,
                       d.padded_k / group_size(d.layout), want_scales, 2, false,
                       op, "scales");
    if (!st) {
      return st;
    }
  }
  st = validate_view(d.input, DecodeDtype::Bf16, kDecodeLayoutBf16VectorV0,
                     d.k, 1, d.k, 1,
                     static_cast<std::uint64_t>(d.k) * 2u, 2, false, op,
                     "input");
  if (!st) {
    return st;
  }
  if (d.epilogue != DecodeEpilogue::StoreBf16 &&
      d.epilogue != DecodeEpilogue::StoreFp32 &&
      d.epilogue != DecodeEpilogue::ResidualAddFp32 &&
      d.epilogue != DecodeEpilogue::SwigluStoreBf16) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "unknown epilogue"));
  }
  if (d.epilogue == DecodeEpilogue::SwigluStoreBf16 && !paired_b) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "SwiGLU epilogue requires paired gate/up launch"));
  }
  if (d.epilogue == DecodeEpilogue::ResidualAddFp32) {
    bool const separate_output = !empty_view(d.output);
    st = validate_view(d.residual, DecodeDtype::Fp32, kDecodeLayoutFp32VectorV0,
                       d.n, 1, d.n, 1,
                       static_cast<std::uint64_t>(d.n) * 4u, 4,
                       !separate_output, op,
                       "residual");
    if (!st) {
      return st;
    }
    if (separate_output) {
      st = validate_view(d.output, DecodeDtype::Fp32,
                         kDecodeLayoutFp32VectorV0, d.n, 1, d.n, 1,
                         static_cast<std::uint64_t>(d.n) * 4u, 4, true, op,
                         "output");
      if (!st) {
        return st;
      }
    }
  } else {
    if (!empty_view(d.residual)) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "store epilogue must not supply residual"));
    }
    DecodeDtype const out_dtype =
        (d.epilogue == DecodeEpilogue::StoreFp32) ? DecodeDtype::Fp32
                                                  : DecodeDtype::Bf16;
    std::uint16_t const out_layout =
        (out_dtype == DecodeDtype::Fp32) ? kDecodeLayoutFp32VectorV0
                                         : kDecodeLayoutBf16VectorV0;
    std::uint64_t const out_bytes =
        static_cast<std::uint64_t>(d.n) *
        ((out_dtype == DecodeDtype::Fp32) ? 4u : 2u);
    st = validate_view(d.output, out_dtype, out_layout, d.n, 1, d.n, 1,
                       out_bytes, (out_dtype == DecodeDtype::Fp32) ? 4u : 2u,
                       true, op, "output");
    if (!st) {
      return st;
    }
  }
  return validate_non_overlap(d, op);
}

std::expected<void, Error> validate_paired_side(DecodeMmvPairedDesc const& d,
                                                std::string_view op) {
  auto st = validate_geometry(d.a, op, true);
  if (!st) {
    return st;
  }
  DecodeMmvDesc b_for_validation = d.b;
  if (d.a.epilogue == DecodeEpilogue::SwigluStoreBf16) {
    if (!empty_view(d.b.output) || !empty_view(d.b.residual)) {
      return std::unexpected(make_error(
          ErrorCode::InvalidArgument, op,
          "SwiGLU paired B must not declare a separately materialized output"));
    }
    b_for_validation.output = d.a.output;
  }
  st = validate_geometry(b_for_validation, op, true);
  if (!st) {
    return st;
  }
  if (d.b.layout != d.a.layout || d.b.quantizer != d.a.quantizer ||
      d.b.n != d.a.n || d.b.k != d.a.k ||
      d.b.padded_n != d.a.padded_n || d.b.padded_k != d.a.padded_k ||
      d.b.epilogue != d.a.epilogue || d.b.input.pointer != d.a.input.pointer ||
      d.b.input != d.a.input) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, op,
        "paired B must match A shape, layout, input, and epilogue contract"));
  }
  DecodeOperandView const* av[] = {&d.a.codes, &d.a.scales, &d.a.output,
                                   &d.a.residual};
  DecodeOperandView const* bv[] = {&d.b.codes, &d.b.scales, &d.b.output,
                                   &d.b.residual};
  for (auto* a : av) {
    for (auto* b : bv) {
      if (overlaps(*a, *b)) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                          "paired A/B operands must not overlap"));
      }
    }
  }
  return {};
}

template <WeightKind Kind, bool Paired>
std::expected<void, Error> launch_kind(DecodeMmvDesc const& a,
                                       DecodeMmvDesc const* b,
                                       Stream const& stream,
                                       std::string_view op) {
  unsigned const blocks = a.padded_n / static_cast<unsigned>(kDecodeTileRows);
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  decode_mmv_kernel<Kind, Paired><<<blocks, kDecodeThreads, 0, stream.native()>>>(
      static_cast<std::byte const*>(a.codes.pointer),
      static_cast<std::byte const*>(a.scales.pointer),
      b == nullptr ? nullptr : static_cast<std::byte const*>(b->codes.pointer),
      b == nullptr ? nullptr : static_cast<std::byte const*>(b->scales.pointer),
      static_cast<std::uint16_t const*>(a.input.pointer), a.output.pointer,
      b == nullptr ? nullptr : b->output.pointer,
      static_cast<float const*>(a.residual.pointer),
      b == nullptr ? nullptr : static_cast<float const*>(b->residual.pointer),
      a.n, a.k, a.padded_k, a.epilogue);
  return check(cudaGetLastError(), op);
}

template <bool Paired>
std::expected<void, Error> launch_layout(DecodeMmvDesc const& a,
                                         DecodeMmvDesc const* b,
                                         Stream const& stream,
                                         std::string_view op) {
  if (a.layout == kDecodeLayoutQ4G64V0) {
    return launch_kind<WeightKind::Q4, Paired>(a, b, stream, op);
  }
  if (a.layout == kDecodeLayoutQ8G32V0) {
    return launch_kind<WeightKind::Q8, Paired>(a, b, stream, op);
  }
  return launch_kind<WeightKind::Bf16, Paired>(a, b, stream, op);
}

template <WeightKind Kind>
__global__ void decode_mmv_ranges_kernel(
    std::byte const* codes0, std::byte const* scales0, void* out0,
    std::uint32_t n0, std::uint32_t tiles0, std::byte const* codes1,
    std::byte const* scales1, void* out1, std::uint32_t n1, std::uint32_t tiles1,
    std::byte const* codes2, std::byte const* scales2, void* out2,
    std::uint32_t n2, std::uint16_t const* input, std::uint32_t k,
    std::uint32_t padded_k) {
  std::byte const* codes = codes0;
  std::byte const* scales = scales0;
  void* output = out0;
  std::uint32_t n = n0;
  std::uint32_t tn = blockIdx.x;
  if (blockIdx.x >= tiles0 + tiles1) {
    codes = codes2;
    scales = scales2;
    output = out2;
    n = n2;
    tn = blockIdx.x - tiles0 - tiles1;
  } else if (blockIdx.x >= tiles0) {
    codes = codes1;
    scales = scales1;
    output = out1;
    n = n1;
    tn = blockIdx.x - tiles0;
  }
  std::uint32_t const warp = threadIdx.x >> 5;
  int const lane = static_cast<int>(threadIdx.x & 31u);
  std::uint32_t const row = tn * 8u + warp;
  std::uint32_t const tiles_k = padded_k / static_cast<std::uint32_t>(kDecodeTileK);

  __shared__ alignas(16) std::uint16_t xs[kDecodeMaxK];
  for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    xs[i] = input[i];
  }
  for (std::uint32_t i = k + threadIdx.x; i < padded_k; i += blockDim.x) {
    xs[i] = 0;
  }
  __syncthreads();

  float acc = 0.0f;
  for (std::uint32_t tk = 0; tk < tiles_k; ++tk) {
    std::uint64_t const tile_row =
        (static_cast<std::uint64_t>(tn) * tiles_k + tk) * 8u + warp;
    float d[8];
    decode8<Kind>(codes, scales, tile_row, lane, d);
    std::uint32_t const k0 =
        tk * static_cast<std::uint32_t>(kDecodeTileK) + static_cast<std::uint32_t>(lane) * 8u;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float const x = bf16_to_fp32(xs[k0 + static_cast<std::uint32_t>(i)]);
      acc = fmaf(d[i], x, acc);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0 && row < n) {
    static_cast<std::uint16_t*>(output)[row] = fp32_to_bf16_rne(acc);
  }
}

std::expected<void, Error> validate_range_side(DecodeMmvDesc const& d,
                                               DecodeMmvDesc const& first,
                                               std::string_view field) {
  if (d.layout != first.layout || d.quantizer != first.quantizer) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "ranged projections must share one layout family"));
  }
  if (d.k != first.k || d.padded_k != first.padded_k) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "ranged projections must share K"));
  }
  if (d.input != first.input) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "ranged projections must share the input"));
  }
  if (d.epilogue != DecodeEpilogue::StoreBf16) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "ranged projections store BF16"));
  }
  return validate_geometry(d, field, false);
}

template <WeightKind Kind>
std::expected<void, Error> launch_range_kind(
    std::span<DecodeMmvDesc const> ranges, Stream const& stream) {
  auto const& d0 = ranges[0];
  auto const& d1 = ranges[1];
  DecodeMmvDesc const* d2 = ranges.size() == 3 ? &ranges[2] : nullptr;
  unsigned const tiles0 =
      d0.padded_n / static_cast<unsigned>(kDecodeTileRows);
  unsigned const tiles1 =
      d1.padded_n / static_cast<unsigned>(kDecodeTileRows);
  unsigned const tiles2 =
      d2 == nullptr ? 0u
                    : d2->padded_n / static_cast<unsigned>(kDecodeTileRows);
  unsigned const blocks = tiles0 + tiles1 + tiles2;
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  decode_mmv_ranges_kernel<Kind><<<blocks, kDecodeThreads, 0, stream.native()>>>(
      static_cast<std::byte const*>(d0.codes.pointer),
      static_cast<std::byte const*>(d0.scales.pointer), d0.output.pointer, d0.n,
      tiles0, static_cast<std::byte const*>(d1.codes.pointer),
      static_cast<std::byte const*>(d1.scales.pointer), d1.output.pointer, d1.n,
      tiles1,
      d2 == nullptr ? nullptr : static_cast<std::byte const*>(d2->codes.pointer),
      d2 == nullptr ? nullptr : static_cast<std::byte const*>(d2->scales.pointer),
      d2 == nullptr ? nullptr : d2->output.pointer, d2 == nullptr ? 0u : d2->n,
      static_cast<std::uint16_t const*>(d0.input.pointer), d0.k, d0.padded_k);
  return check(cudaGetLastError(), "decode_mmv_ranges_kernel");
}

}  // namespace

std::expected<void, Error> launch_decode_mmv(DecodeMmvDesc const& desc,
                                             Stream const& stream) {
  auto st = require_stream(stream, "decode_mmv");
  if (!st) {
    return st;
  }
  st = validate_geometry(desc, "decode_mmv", false);
  if (!st) {
    return st;
  }
  return launch_layout<false>(desc, nullptr, stream, "decode_mmv_kernel");
}

std::expected<void, Error> launch_decode_mmv_paired(DecodeMmvPairedDesc const& desc,
                                                   Stream const& stream) {
  auto st = require_stream(stream, "decode_mmv_paired");
  if (!st) {
    return st;
  }
  st = validate_paired_side(desc, "decode_mmv_paired");
  if (!st) {
    return st;
  }
  return launch_layout<true>(desc.a, &desc.b, stream,
                             "decode_mmv_paired_kernel");
}

std::expected<void, Error> launch_decode_ab_bf16(DecodeMmvPairedDesc const& desc,
                                                 Stream const& stream) {
  auto st = require_stream(stream, "decode_ab_bf16");
  if (!st) {
    return st;
  }
  if (desc.a.layout != kDecodeLayoutBf16DenseTileV0 ||
      desc.a.quantizer != kDecodeQuantizerNone) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "decode_ab_bf16",
                                      "a/b contraction requires cuda_bf16_dense_tile_v0"));
  }
  if (desc.a.epilogue != DecodeEpilogue::StoreFp32) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "decode_ab_bf16",
                                      "a/b contraction requires FP32 output"));
  }
  st = validate_paired_side(desc, "decode_ab_bf16");
  if (!st) {
    return st;
  }
  return launch_layout<true>(desc.a, &desc.b, stream, "decode_ab_bf16_kernel");
}

std::expected<void, Error> launch_decode_mmv_ranges(DecodeMmvRangeDesc const& desc,
                                                    Stream const& stream) {
  auto st = require_stream(stream, "decode_mmv_ranges");
  if (!st) {
    return st;
  }
  if (desc.ranges.size() < 2 || desc.ranges.size() > 3) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, "decode_mmv_ranges",
        "ranged launch requires two or three projections"));
  }
  auto const& first = desc.ranges.front();
  if (first.epilogue != DecodeEpilogue::StoreBf16) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "decode_mmv_ranges",
                                      "ranged projections store BF16"));
  }
  st = validate_geometry(first, "decode_mmv_ranges[0]", false);
  if (!st) {
    return st;
  }
  for (std::size_t i = 1; i < desc.ranges.size(); ++i) {
    st = validate_range_side(desc.ranges[i], first, "decode_mmv_ranges");
    if (!st) {
      return st;
    }
  }
  for (std::size_t i = 0; i < desc.ranges.size(); ++i) {
    for (std::size_t j = i + 1; j < desc.ranges.size(); ++j) {
      DecodeOperandView const* left[] = {
          &desc.ranges[i].codes, &desc.ranges[i].scales,
          &desc.ranges[i].output};
      DecodeOperandView const* right[] = {
          &desc.ranges[j].codes, &desc.ranges[j].scales,
          &desc.ranges[j].output};
      for (auto* a : left) {
        for (auto* b : right) {
          if (overlaps(*a, *b)) {
            return std::unexpected(make_error(
                ErrorCode::InvalidArgument, "decode_mmv_ranges",
                "ranged projection operand spans must not overlap"));
          }
        }
      }
    }
  }
  if (first.layout == kDecodeLayoutQ4G64V0) {
    return launch_range_kind<WeightKind::Q4>(desc.ranges, stream);
  }
  if (first.layout == kDecodeLayoutQ8G32V0) {
    return launch_range_kind<WeightKind::Q8>(desc.ranges, stream);
  }
  return launch_range_kind<WeightKind::Bf16>(desc.ranges, stream);
}

}  // namespace qw38::cuda
