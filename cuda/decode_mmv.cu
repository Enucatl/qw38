#include "cuda/decode_mmv.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

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
    std::uint32_t word;
    std::memcpy(&word, row, sizeof(word));
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
    std::uint64_t word;
    std::memcpy(&word, row, sizeof(word));
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
                               float* residual, std::uint32_t row, float acc) {
  if (epilogue == DecodeEpilogue::StoreBf16) {
    static_cast<std::uint16_t*>(output)[row] = fp32_to_bf16_rne(acc);
  } else if (epilogue == DecodeEpilogue::StoreFp32) {
    static_cast<float*>(output)[row] = acc;
  } else {
    residual[row] += acc;
  }
}

template <WeightKind Kind, bool Paired>
__global__ void decode_mmv_kernel(std::byte const* codes_a, std::byte const* scales_a,
                                 std::byte const* codes_b, std::byte const* scales_b,
                                 std::uint16_t const* input, void* output_a,
                                 void* output_b, float* residual_a, float* residual_b,
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
  if (d.codes == nullptr || d.codes_bytes != want_codes) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "code view length does not match layout"));
  }
  if (want_scales == 0) {
    if (d.scales != nullptr || d.scales_bytes != 0) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "BF16 dense tile must not supply scales"));
    }
  } else if (d.scales == nullptr || d.scales_bytes != want_scales) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "scale view length does not match layout"));
  }
  if (d.input == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "null input"));
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
    if (d.residual == nullptr) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "residual-add requires an FP32 residual"));
    }
  } else if (d.output == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "null output"));
  }
  return {};
}

std::expected<void, Error> validate_paired_side(DecodeMmvPairedDesc const& d,
                                                std::string_view op) {
  auto st = validate_geometry(d.a, op, true);
  if (!st) {
    return st;
  }
  auto const want_codes = decode_code_bytes(d.a.layout, d.a.padded_n, d.a.padded_k);
  auto const want_scales = decode_scale_bytes(d.a.layout, d.a.padded_n, d.a.padded_k);
  if (d.codes_b == nullptr || d.codes_b_bytes != want_codes) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "paired code view length does not match layout"));
  }
  if (want_scales == 0) {
    if (d.scales_b != nullptr || d.scales_b_bytes != 0) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "paired BF16 must not supply scales"));
    }
  } else if (d.scales_b == nullptr || d.scales_b_bytes != want_scales) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                      "paired scale view length does not match layout"));
  }
  if (d.a.epilogue == DecodeEpilogue::ResidualAddFp32) {
    if (d.residual_b == nullptr) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, op,
                                        "paired residual-add requires two FP32 residuals"));
    }
  } else if (d.a.epilogue != DecodeEpilogue::SwigluStoreBf16 &&
             d.output_b == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, op, "null paired output"));
  }
  return {};
}

template <WeightKind Kind, bool Paired>
std::expected<void, Error> launch_kind(DecodeMmvDesc const& a, std::byte const* codes_b,
                                       std::byte const* scales_b, void* output_b,
                                       float* residual_b, Stream const& stream,
                                       std::string_view op) {
  unsigned const blocks = a.padded_n / static_cast<unsigned>(kDecodeTileRows);
  decode_mmv_kernel<Kind, Paired><<<blocks, kDecodeThreads, 0, stream.native()>>>(
      a.codes, a.scales, codes_b, scales_b, a.input, a.output, output_b, a.residual,
      residual_b, a.n, a.k, a.padded_k, a.epilogue);
  return check(cudaGetLastError(), op);
}

template <bool Paired>
std::expected<void, Error> launch_layout(DecodeMmvDesc const& a, std::byte const* codes_b,
                                         std::byte const* scales_b, void* output_b,
                                         float* residual_b, Stream const& stream,
                                         std::string_view op) {
  if (a.layout == kDecodeLayoutQ4G64V0) {
    return launch_kind<WeightKind::Q4, Paired>(a, codes_b, scales_b, output_b, residual_b,
                                               stream, op);
  }
  if (a.layout == kDecodeLayoutQ8G32V0) {
    return launch_kind<WeightKind::Q8, Paired>(a, codes_b, scales_b, output_b, residual_b,
                                               stream, op);
  }
  return launch_kind<WeightKind::Bf16, Paired>(a, codes_b, scales_b, output_b, residual_b,
                                              stream, op);
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
                                               DecodeMmvDesc const& qg,
                                               std::string_view field) {
  if (d.layout != qg.layout || d.quantizer != qg.quantizer) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "q/g, k, and v must share one layout family"));
  }
  if (d.k != qg.k || d.padded_k != qg.padded_k) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "q/g, k, and v must share K"));
  }
  if (d.input != qg.input) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "q/g, k, and v must share the normalized input"));
  }
  if (d.epilogue != DecodeEpilogue::StoreBf16) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, field,
                                      "ranged projections store BF16"));
  }
  return validate_geometry(d, field, false);
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
  return launch_layout<false>(desc, nullptr, nullptr, nullptr, nullptr, stream,
                              "decode_mmv_kernel");
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
  return launch_layout<true>(desc.a, desc.codes_b, desc.scales_b, desc.output_b,
                             desc.residual_b, stream, "decode_mmv_paired_kernel");
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
  return launch_layout<true>(desc.a, desc.codes_b, desc.scales_b, desc.output_b,
                             desc.residual_b, stream, "decode_ab_bf16_kernel");
}

std::expected<void, Error> launch_decode_mmv_ranges(DecodeMmvRangeDesc const& desc,
                                                    Stream const& stream) {
  auto st = require_stream(stream, "decode_mmv_ranges");
  if (!st) {
    return st;
  }
  if (desc.qg.epilogue != DecodeEpilogue::StoreBf16) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "decode_mmv_ranges",
                                      "ranged projections store BF16"));
  }
  st = validate_geometry(desc.qg, "decode_mmv_ranges.qg", false);
  if (!st) {
    return st;
  }
  st = validate_range_side(desc.k, desc.qg, "decode_mmv_ranges.k");
  if (!st) {
    return st;
  }
  st = validate_range_side(desc.v, desc.qg, "decode_mmv_ranges.v");
  if (!st) {
    return st;
  }
  if (desc.qg.output == desc.k.output || desc.qg.output == desc.v.output ||
      desc.k.output == desc.v.output) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "decode_mmv_ranges",
                                      "q/g, k, and v outputs must be distinct"));
  }
  unsigned const tiles0 =
      desc.qg.padded_n / static_cast<unsigned>(kDecodeTileRows);
  unsigned const tiles1 =
      desc.k.padded_n / static_cast<unsigned>(kDecodeTileRows);
  unsigned const tiles2 =
      desc.v.padded_n / static_cast<unsigned>(kDecodeTileRows);
  unsigned const blocks = tiles0 + tiles1 + tiles2;
  if (desc.qg.layout == kDecodeLayoutQ4G64V0) {
    decode_mmv_ranges_kernel<WeightKind::Q4>
        <<<blocks, kDecodeThreads, 0, stream.native()>>>(
            desc.qg.codes, desc.qg.scales, desc.qg.output, desc.qg.n, tiles0,
            desc.k.codes, desc.k.scales, desc.k.output, desc.k.n, tiles1,
            desc.v.codes, desc.v.scales, desc.v.output, desc.v.n, desc.qg.input,
            desc.qg.k, desc.qg.padded_k);
  } else if (desc.qg.layout == kDecodeLayoutQ8G32V0) {
    decode_mmv_ranges_kernel<WeightKind::Q8>
        <<<blocks, kDecodeThreads, 0, stream.native()>>>(
            desc.qg.codes, desc.qg.scales, desc.qg.output, desc.qg.n, tiles0,
            desc.k.codes, desc.k.scales, desc.k.output, desc.k.n, tiles1,
            desc.v.codes, desc.v.scales, desc.v.output, desc.v.n, desc.qg.input,
            desc.qg.k, desc.qg.padded_k);
  } else {
    decode_mmv_ranges_kernel<WeightKind::Bf16>
        <<<blocks, kDecodeThreads, 0, stream.native()>>>(
            desc.qg.codes, desc.qg.scales, desc.qg.output, desc.qg.n, tiles0,
            desc.k.codes, desc.k.scales, desc.k.output, desc.k.n, tiles1,
            desc.v.codes, desc.v.scales, desc.v.output, desc.v.n, desc.qg.input,
            desc.qg.k, desc.qg.padded_k);
  }
  return check(cudaGetLastError(), "decode_mmv_ranges_kernel");
}

}  // namespace qw38::cuda
