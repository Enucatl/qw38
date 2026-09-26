#include "cuda/prefill.hpp"
#include "cuda/nvfp4.hpp"

#include "cuda/activation.hpp"

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <array>
#include <string_view>
#include <utility>

namespace qw38::cuda {
namespace {

constexpr std::uint64_t kCublasWorkspaceBytes = 4u * 1024u * 1024u;

std::expected<void, Error> cublas_check(cublasStatus_t status,
                                         std::string_view operation) {
  if (status == CUBLAS_STATUS_SUCCESS) return {};
  return std::unexpected(Error{.code = ErrorCode::Status,
                               .cuda_status = static_cast<int>(status),
                               .operation = std::string(operation),
                               .detail = "cuBLAS status"});
}

__device__ __forceinline__ std::uint16_t fp32_to_bf16_rne(float x) {
  std::uint32_t const bits = __float_as_uint(x);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    auto h = static_cast<std::uint16_t>(bits >> 16);
    return (bits & 0x007fffffu) ? static_cast<std::uint16_t>(h | 0x40u) : h;
  }
  return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float fp16_to_fp32(std::uint16_t bits) {
  std::uint32_t const sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  std::uint32_t const exponent = (bits >> 10) & 31u;
  std::uint32_t mantissa = bits & 1023u;
  std::uint32_t result;
  if (exponent == 0) {
    if (mantissa == 0) return __uint_as_float(sign);
    std::uint32_t e = 113;
    while ((mantissa & 1024u) == 0) { mantissa <<= 1; --e; }
    result = sign | (e << 23) | ((mantissa & 1023u) << 13);
  } else if (exponent == 31) {
    result = sign | 0x7f800000u | (mantissa << 13);
  } else {
    result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }
  return __uint_as_float(result);
}

__global__ void unpack_tile_kernel(std::uint8_t const* codes,
                                   std::uint8_t const* scales,
                                   std::uint16_t* output,
                                   std::uint16_t layout,
                                   std::uint32_t n, std::uint32_t k,
                                   std::uint32_t padded_k,
                                   std::uint32_t row_start,
                                   std::uint32_t rows) {
  std::uint64_t const idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint64_t const extent = static_cast<std::uint64_t>(rows) * k;
  if (idx >= extent) return;
  std::uint32_t const local_row = static_cast<std::uint32_t>(idx / k);
  std::uint32_t const col = static_cast<std::uint32_t>(idx % k);
  std::uint32_t const row = row_start + local_row;
  if (row >= n) return;
  std::uint32_t const tile_k = col / 256u;
  std::uint32_t const in_tile = col % 256u;
  std::uint64_t const tile_row =
      ((static_cast<std::uint64_t>(row) / 8u) * (padded_k / 256u) + tile_k) * 8u + row % 8u;
  float decoded = 0.0f;
  if (layout == kDecodeLayoutQ4KCandidateV2) {
    auto const code = (codes[tile_row * 128u + in_tile / 2u] >> (4u * (in_tile & 1u))) & 15u;
    auto const* meta = scales + tile_row * 16u;
    auto const* super = reinterpret_cast<std::uint16_t const*>(meta);
    auto const* packed = meta + 4u;
    unsigned const group = in_tile / 32u;
    unsigned const scale = group < 4 ? packed[group] & 63u
        : (packed[group + 4u] & 15u) | ((packed[group - 4u] >> 6u) << 4u);
    unsigned const minimum = group < 4 ? packed[group + 4u] & 63u
        : (packed[group + 4u] >> 4u) | ((packed[group] >> 6u) << 4u);
    float const d = __fmul_rn(fp16_to_fp32(super[0]), static_cast<float>(scale));
    float const m = __fmul_rn(fp16_to_fp32(super[1]), static_cast<float>(minimum));
    decoded = __fsub_rn(__fmul_rn(d, static_cast<float>(code)), m);
  } else if (layout == kDecodeLayoutQ8G32CandidateV1 ||
             layout == kDecodeLayoutQ8G32V0) {
    auto const code = static_cast<std::int8_t>(codes[tile_row * 256u + in_tile]);
    auto const* s = reinterpret_cast<std::uint16_t const*>(scales);
    float const scale = fp16_to_fp32(s[tile_row * 8u + in_tile / 32u]);
    decoded = scale == 0.0f ? 0.0f : static_cast<float>(code) * scale;
  } else {
    auto const* weight = reinterpret_cast<std::uint16_t const*>(codes);
    output[idx] = weight[tile_row * 256u + in_tile];
    return;
  }
  output[idx] = fp32_to_bf16_rne(decoded);
}

__global__ void epilogue_kernel(float const* accum, void* output,
                                float const* residual, std::uint32_t rows,
                                std::uint32_t n, std::uint32_t tile_rows,
                                std::uint32_t row_start,
                                PrefillEpilogue epilogue) {
  std::uint64_t const idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::uint64_t>(rows) * tile_rows) return;
  std::uint32_t const token = static_cast<std::uint32_t>(idx / tile_rows);
  std::uint32_t const col = row_start + static_cast<std::uint32_t>(idx % tile_rows);
  std::uint64_t const dst = static_cast<std::uint64_t>(token) * n + col;
  float value = accum[idx];
  if (epilogue == PrefillEpilogue::StoreBf16) {
    static_cast<std::uint16_t*>(output)[dst] = fp32_to_bf16_rne(value);
  } else if (epilogue == PrefillEpilogue::StoreFp32) {
    static_cast<float*>(output)[dst] = value;
  } else {
    static_cast<float*>(output)[dst] = residual[dst] + value;
  }
}

__device__ __forceinline__ float sigmoid(float x) {
  if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
  float const e = expf(x);
  return e / (1.0f + e);
}

__global__ void swiglu_kernel(float const* gate, float const* up,
                              std::uint16_t* output, std::uint32_t rows,
                              std::uint32_t n, std::uint32_t tile_rows,
                              std::uint32_t row_start) {
  std::uint64_t const idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::uint64_t>(rows) * tile_rows) return;
  std::uint32_t const token = static_cast<std::uint32_t>(idx / tile_rows);
  std::uint32_t const col = row_start + static_cast<std::uint32_t>(idx % tile_rows);
  std::uint64_t const dst = static_cast<std::uint64_t>(token) * n + col;
  float const g = gate[idx];
  output[dst] = fp32_to_bf16_rne((g * sigmoid(g)) * up[idx]);
}

bool layout_ok(PrefillWeight const& w) {
  if (w.layout == kDecodeLayoutNvFp4V1)
    return w.quantizer == kDecodeQuantizerNvFp4V1 && w.k <= 5120 && w.n % 8 == 0;
  if (w.layout == kDecodeLayoutQ4KCandidateV2)
    return w.quantizer == kDecodeQuantizerQ4KCandidateV2;
  if (w.layout == kDecodeLayoutQ8G32CandidateV1)
    return w.quantizer == kDecodeQuantizerQ8G32CandidateV1;
  if (w.layout == kDecodeLayoutQ8G32V0)
    return w.quantizer == kDecodeQuantizerQ8G32V0;
  if (w.layout == kDecodeLayoutBf16DenseTileV0)
    return w.quantizer == kDecodeQuantizerNone;
  return false;
}

DecodeMmvDesc nvfp4_decode(PrefillWeight const& w, std::uint16_t const* input,
                            void* output, DecodeEpilogue epilogue) {
  DecodeMmvDesc d;
  d.layout=w.layout; d.quantizer=w.quantizer; d.n=w.n; d.k=w.k;
  d.padded_n=w.padded_n; d.padded_k=w.padded_k; d.epilogue=epilogue;
  d.codes=decode_matrix_view(const_cast<void*>(w.codes),DecodeDtype::NvFp4,w.layout,
      w.n,w.k,w.padded_n,w.padded_k,w.codes_bytes,16);
  auto units=static_cast<std::uint32_t>(w.scales_bytes/2);
  d.scales=decode_matrix_view(const_cast<void*>(w.scales),DecodeDtype::Fp16,w.layout,
      1,units,1,units,w.scales_bytes,16);
  d.input=decode_vector_view(const_cast<std::uint16_t*>(input),DecodeDtype::Bf16,
      kDecodeLayoutBf16VectorV0,w.k,std::uint64_t(w.k)*2,2,false);
  bool bf16=epilogue==DecodeEpilogue::StoreBf16 || epilogue==DecodeEpilogue::SwigluStoreBf16;
  d.output=decode_vector_view(output,bf16?DecodeDtype::Bf16:DecodeDtype::Fp32,
      bf16?kDecodeLayoutBf16VectorV0:kDecodeLayoutFp32VectorV0,w.n,
      std::uint64_t(w.n)*(bf16?2:4),bf16?2:4,true);
  return d;
}

struct Region {
  std::uintptr_t begin{};
  std::uintptr_t end{};
};

std::expected<Region, Error> region(void const* pointer, std::uint64_t bytes) {
  auto const begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (!pointer || bytes == 0 ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.region",
                                      "empty or overflowing address range"));
  }
  return Region{begin, begin + static_cast<std::uintptr_t>(bytes)};
}

// Validate a borrowed device view before any kernel sees it. CUDA reports the
// containing allocation for interior pointers, so byte extents are checked
// independently of a caller's logical matrix shape.
std::expected<Region, Error> device_region(void const* pointer,
                                           std::uint64_t bytes,
                                           std::uint32_t alignment,
                                           int device) {
  auto r = region(pointer, bytes);
  if (!r) return std::unexpected(r.error());
  if (alignment == 0 || r->begin % alignment != 0) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.device_view", "misaligned pointer"));
  }
  cudaPointerAttributes attributes{};
  auto status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.device_view", "pointer is not device memory"));
  }
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.device_view", "device mismatch"));
  }
  CUdeviceptr base = 0;
  std::size_t allocation_bytes = 0;
  auto const driver_status = cuMemGetAddressRange(
      &base, &allocation_bytes,
      static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(pointer)));
  if (driver_status != CUDA_SUCCESS) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.device_view", "unknown device allocation"));
  }
  auto allocation = region(reinterpret_cast<void const*>(base), allocation_bytes);
  if (!allocation || r->begin < allocation->begin || r->end > allocation->end) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "prefill.device_view", "operand exceeds allocation"));
  }
  return r;
}

bool overlaps(Region a, Region b) {
  return a.begin < b.end && b.begin < a.end;
}

std::expected<void, Error> distinct(std::span<Region const> ranges) {
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    for (std::size_t j = i + 1; j < ranges.size(); ++j) {
      if (overlaps(ranges[i], ranges[j])) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "prefill.alias", "live buffers overlap"));
      }
    }
  }
  return {};
}

}  // namespace

std::expected<void, Error> validate_prefill_device_span(
    void const* pointer, std::uint64_t bytes, std::uint32_t alignment,
    int device) {
  auto value = device_region(pointer, bytes, alignment, device);
  if (!value) return std::unexpected(value.error());
  return {};
}

PrefillEngine::PrefillEngine(PrefillEngine&& other) noexcept { *this = std::move(other); }

PrefillEngine& PrefillEngine::operator=(PrefillEngine&& other) noexcept {
  if (this != &other) {
    release();
    stream_ = std::exchange(other.stream_, nullptr);
    handle_ = std::exchange(other.handle_, nullptr);
    workspace_ = std::move(other.workspace_);
    weight_tile_ = std::exchange(other.weight_tile_, nullptr);
    accum_a_ = std::exchange(other.accum_a_, nullptr);
    accum_b_ = std::exchange(other.accum_b_, nullptr);
    normalized_ = std::exchange(other.normalized_, nullptr);
    swiglu_ = std::exchange(other.swiglu_, nullptr);
    library_workspace_ = std::exchange(other.library_workspace_, nullptr);
    token_capacity_ = std::exchange(other.token_capacity_, 0);
    weight_rows_ = std::exchange(other.weight_rows_, 0);
    native_stream_ = std::exchange(other.native_stream_, nullptr);
    device_ = std::exchange(other.device_, -1);
  }
  return *this;
}

PrefillEngine::~PrefillEngine() { release(); }

void PrefillEngine::release() noexcept {
  if (handle_) { (void)cublasDestroy(handle_); handle_ = nullptr; }
}

std::expected<PrefillEngine, Error> PrefillEngine::create(
    Stream const& stream, std::uint32_t token_capacity,
    std::uint32_t weight_rows) {
  if (stream.empty() || token_capacity == 0 || token_capacity > kPrefillMaxTokens ||
      (weight_rows != 128 && weight_rows != 256 && weight_rows != 512)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.create",
                                      "invalid stream or token capacity"));
  }
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  std::uint64_t const weight_bytes =
      static_cast<std::uint64_t>(weight_rows) * kPrefillMaxK * 2u;
  std::uint64_t const accum_bytes =
      static_cast<std::uint64_t>(token_capacity) * weight_rows * 4u;
  std::uint64_t const normalized_bytes =
      static_cast<std::uint64_t>(token_capacity) * 5120u * 2u;
  std::uint64_t const swiglu_bytes =
      static_cast<std::uint64_t>(token_capacity) * 17408u * 2u;
  std::uint64_t const total = weight_bytes + 2u * accum_bytes +
                              normalized_bytes + swiglu_bytes +
                              kCublasWorkspaceBytes;
  auto buffer = DeviceBuffer::allocate(total, stream.device());
  if (!buffer) return std::unexpected(buffer.error());
  PrefillEngine result;
  result.stream_ = &stream;
  result.native_stream_ = stream.native();
  result.device_ = stream.device();
  result.token_capacity_ = token_capacity;
  result.weight_rows_ = weight_rows;
  result.workspace_ = std::move(*buffer);
  auto* ptr = result.workspace_.as_bytes();
  result.weight_tile_ = reinterpret_cast<std::uint16_t*>(ptr);
  ptr += weight_bytes;
  result.accum_a_ = reinterpret_cast<float*>(ptr);
  ptr += accum_bytes;
  result.accum_b_ = reinterpret_cast<float*>(ptr);
  ptr += accum_bytes;
  result.normalized_ = reinterpret_cast<std::uint16_t*>(ptr);
  ptr += normalized_bytes;
  result.swiglu_ = reinterpret_cast<std::uint16_t*>(ptr);
  ptr += swiglu_bytes;
  result.library_workspace_ = ptr;
  auto st = cublas_check(cublasCreate(&result.handle_), "cublasCreate");
  if (!st) return std::unexpected(st.error());
  st = cublas_check(cublasSetStream(result.handle_, stream.native()), "cublasSetStream");
  if (!st) return std::unexpected(st.error());
  st = cublas_check(cublasSetWorkspace(result.handle_, result.library_workspace_,
                                       kCublasWorkspaceBytes), "cublasSetWorkspace");
  if (!st) return std::unexpected(st.error());
  return result;
}

std::expected<void, Error> PrefillEngine::contract(
    PrefillWeight const& w, std::uint32_t valid_tokens) const {
  if (!stream_ || stream_->empty() || stream_->native() != native_stream_ ||
      stream_->device() != device_ || !handle_ || workspace_.empty() ||
      valid_tokens == 0 || valid_tokens > token_capacity_ ||
      !layout_ok(w) || !w.codes || w.n == 0 || w.k == 0 ||
      w.k > kPrefillMaxK || w.padded_n != decode_pad_n(w.n) ||
      w.padded_k != decode_pad_k(w.k) ||
      w.codes_bytes != decode_code_bytes(w.layout, w.padded_n, w.padded_k) ||
      w.scales_bytes != decode_scale_bytes(w.layout, w.padded_n, w.padded_k) ||
      (w.scales_bytes != 0 && !w.scales) ||
      (w.scales_bytes == 0 && w.scales) ||
      (w.layout != kDecodeLayoutBf16DenseTileV0 && w.k % 256u != 0u)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.weight",
                                      "weight, layout or token contract mismatch"));
  }
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(guard.error());
  auto codes = device_region(w.codes, w.codes_bytes, w.layout == kDecodeLayoutNvFp4V1 ? 16u : 2u, device_);
  auto workspace = device_region(workspace_.data(), workspace_.bytes(), 16u, device_);
  if (!codes || !workspace) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.weight",
                                      "invalid weight or workspace range"));
  }
  if (w.scales_bytes) {
    auto scales = device_region(w.scales, w.scales_bytes, w.layout == kDecodeLayoutNvFp4V1 ? 16u : 2u, device_);
    if (!scales) return std::unexpected(scales.error());
    std::array<Region, 3> regions{*codes, *scales, *workspace};
    return distinct(regions);
  }
  std::array<Region, 2> regions{*codes, *workspace};
  return distinct(regions);
}

std::expected<void, Error> PrefillEngine::gemm_tile(
    PrefillWeight const& w, std::uint16_t const* input,
    std::uint32_t valid_tokens, std::uint32_t row_start, float* accum) {
  std::uint32_t const tile_rows = std::min(weight_rows_, w.n - row_start);
  if (w.layout == kDecodeLayoutNvFp4V1)
    return nvfp4_gemm_tile(w.codes,w.scales,packed_input(),packed_scales(),
        valid_tokens,tile_rows,w.k,row_start,accum,library_workspace_,
        kCublasWorkspaceBytes,*stream_);
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(guard.error());
  std::uint64_t const elements = static_cast<std::uint64_t>(tile_rows) * w.k;
  unpack_tile_kernel<<<static_cast<unsigned>((elements + 255u) / 256u), 256, 0,
                       stream_->native()>>>(
      static_cast<std::uint8_t const*>(w.codes),
      static_cast<std::uint8_t const*>(w.scales), weight_tile_, w.layout,
      w.n, w.k, w.padded_k, row_start, tile_rows);
  if (auto st = check(cudaGetLastError(), "prefill.unpack_tile"); !st) return st;
  float const alpha = 1.0f;
  float const beta = 0.0f;
  return cublas_check(cublasGemmEx(
      handle_, CUBLAS_OP_T, CUBLAS_OP_N,
      static_cast<int>(tile_rows), static_cast<int>(valid_tokens),
      static_cast<int>(w.k), &alpha, weight_tile_, CUDA_R_16BF,
      static_cast<int>(w.k), input, CUDA_R_16BF, static_cast<int>(w.k),
      &beta, accum, CUDA_R_32F, static_cast<int>(tile_rows),
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "prefill.cublasGemmEx");
}

std::expected<void, Error> PrefillEngine::project(PrefillProjection const& d) {
  if (!stream_ || stream_->empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.project",
                                      "stream is closed"));
  }
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(guard.error());
  if (auto st = contract(d.weight, d.valid_tokens); !st) return st;
  if (!d.input || !d.output ||
      d.first_position > std::numeric_limits<std::uint64_t>::max() - d.valid_tokens ||
      d.dispatch != PrefillDispatch::BoundedUnpackBf16Cublas ||
      (d.epilogue == PrefillEpilogue::ResidualAddFp32 && !d.residual) ||
      (d.epilogue != PrefillEpilogue::ResidualAddFp32 && d.residual) ||
      (d.epilogue != PrefillEpilogue::StoreBf16 &&
       d.epilogue != PrefillEpilogue::StoreFp32 &&
       d.epilogue != PrefillEpilogue::ResidualAddFp32)) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.project",
                                      "input, output, position or epilogue contract mismatch"));
  }
  auto input = device_region(d.input,
      static_cast<std::uint64_t>(d.valid_tokens) * d.weight.k * 2u, 2u, device_);
  auto output = device_region(d.output,
      static_cast<std::uint64_t>(d.valid_tokens) * d.weight.n *
          (d.epilogue == PrefillEpilogue::StoreBf16 ? 2u : 4u),
      d.epilogue == PrefillEpilogue::StoreBf16 ? 2u : 4u, device_);
  auto codes = region(d.weight.codes, d.weight.codes_bytes);
  auto workspace = region(workspace_.data(),
      static_cast<std::uint64_t>(weight_rows_) * kPrefillMaxK * 2u +
      2u * static_cast<std::uint64_t>(token_capacity_) * weight_rows_ * 4u);
  if (!input || !output || !codes || !workspace) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.project",
                                      "invalid live range"));
  }
  std::array<Region, 7> ranges{};
  std::size_t count = 0;
  ranges[count++] = *input;
  ranges[count++] = *output;
  ranges[count++] = *codes;
  ranges[count++] = *workspace;
  auto library = region(library_workspace_, kCublasWorkspaceBytes);
  if (!library) return std::unexpected(library.error());
  ranges[count++] = *library;
  if (d.weight.scales_bytes) {
    auto scales = region(d.weight.scales, d.weight.scales_bytes);
    if (!scales) return std::unexpected(scales.error());
    ranges[count++] = *scales;
  }
  if (d.residual) {
    auto residual = device_region(d.residual,
        static_cast<std::uint64_t>(d.valid_tokens) * d.weight.n * 4u, 4u, device_);
    if (!residual) return std::unexpected(residual.error());
    ranges[count++] = *residual;
  }
  if (auto st = distinct(std::span<Region const>{ranges.data(), count}); !st) return st;
  if (d.weight.layout == kDecodeLayoutNvFp4V1) {
    if (d.valid_tokens == 1) {
      auto desc = nvfp4_decode(d.weight,d.input,d.output,static_cast<DecodeEpilogue>(d.epilogue));
      if (d.residual) desc.residual=decode_vector_view(const_cast<float*>(d.residual),
          DecodeDtype::Fp32,kDecodeLayoutFp32VectorV0,d.weight.n,std::uint64_t(d.weight.n)*4,4,false);
      return launch_decode_mmv(desc,*stream_);
    }
    if (auto st=launch_pack_nvfp4(d.input,packed_input(),packed_scales(),d.valid_tokens,d.weight.k,*stream_); !st) return st;
  }
  for (std::uint32_t start = 0; start < d.weight.n; start += weight_rows_) {
    std::uint32_t const rows = std::min(weight_rows_, d.weight.n - start);
    if (auto st = gemm_tile(d.weight, d.input, d.valid_tokens, start, accum_a_); !st)
      return st;
    std::uint64_t const elems = static_cast<std::uint64_t>(rows) * d.valid_tokens;
    epilogue_kernel<<<static_cast<unsigned>((elems + 255u) / 256u), 256, 0,
                      stream_->native()>>>(accum_a_, d.output, d.residual,
                                           d.valid_tokens, d.weight.n, rows,
                                           start, d.epilogue);
    if (auto st = check(cudaGetLastError(), "prefill.epilogue"); !st) return st;
  }
  return {};
}

std::expected<void, Error> PrefillEngine::paired_swiglu(
    PrefillWeight const& gate, PrefillWeight const& up,
    std::uint16_t const* normalized, std::uint16_t* swiglu,
    std::uint32_t valid_tokens, std::uint64_t first_position) {
  return paired_swiglu_impl(gate,up,normalized,swiglu,valid_tokens,first_position,false);
}

std::expected<void, Error> PrefillEngine::paired_swiglu_impl(
    PrefillWeight const& gate, PrefillWeight const& up,
    std::uint16_t const* normalized, std::uint16_t* swiglu,
    std::uint32_t valid_tokens, std::uint64_t first_position, bool packed) {
  if (!stream_ || stream_->empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.swiglu",
                                      "stream is closed"));
  }
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(guard.error());
  if (auto st = contract(gate, valid_tokens); !st) return st;
  if (auto st = contract(up, valid_tokens); !st) return st;
  if (!normalized || !swiglu || gate.n != up.n || gate.k != up.k ||
      gate.layout != up.layout || gate.quantizer != up.quantizer ||
      first_position > std::numeric_limits<std::uint64_t>::max() - valid_tokens) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.swiglu",
                                      "paired weight or activation contract mismatch"));
  }
  auto x = device_region(normalized,
      static_cast<std::uint64_t>(valid_tokens) * gate.k * 2u, 2u, device_);
  auto y = device_region(swiglu,
      static_cast<std::uint64_t>(valid_tokens) * gate.n * 2u, 2u, device_);
  auto gate_codes = region(gate.codes, gate.codes_bytes);
  auto up_codes = region(up.codes, up.codes_bytes);
  auto workspace = region(workspace_.data(),
      static_cast<std::uint64_t>(weight_rows_) * kPrefillMaxK * 2u +
      2u * static_cast<std::uint64_t>(token_capacity_) * weight_rows_ * 4u);
  if (!x || !y || !gate_codes || !up_codes || !workspace) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.swiglu",
                                      "invalid live range"));
  }
  std::array<Region, 8> ranges{};
  std::size_t count = 0;
  ranges[count++] = *x; ranges[count++] = *y;
  ranges[count++] = *gate_codes; ranges[count++] = *up_codes;
  ranges[count++] = *workspace;
  auto library = region(library_workspace_, kCublasWorkspaceBytes);
  if (!library) return std::unexpected(library.error());
  ranges[count++] = *library;
  if (gate.scales_bytes) {
    auto scales = region(gate.scales, gate.scales_bytes);
    if (!scales) return std::unexpected(scales.error());
    ranges[count++] = *scales;
  }
  if (up.scales_bytes) {
    auto scales = region(up.scales, up.scales_bytes);
    if (!scales) return std::unexpected(scales.error());
    ranges[count++] = *scales;
  }
  if (auto st = distinct(std::span<Region const>{ranges.data(), count}); !st) return st;
  if (gate.layout == kDecodeLayoutNvFp4V1) {
    if (valid_tokens == 1) {
      DecodeMmvPairedDesc d;
      d.a=nvfp4_decode(gate,normalized,swiglu,DecodeEpilogue::SwigluStoreBf16);
      d.b=nvfp4_decode(up,normalized,swiglu,DecodeEpilogue::SwigluStoreBf16);
      d.b.output={};
      return launch_decode_mmv_paired(d,*stream_);
    }
    if (!packed)
      if (auto st=launch_pack_nvfp4(normalized,packed_input(),packed_scales(),valid_tokens,gate.k,*stream_); !st) return st;
  }
  for (std::uint32_t start = 0; start < gate.n; start += weight_rows_) {
    std::uint32_t const rows = std::min(weight_rows_, gate.n - start);
    if (auto st = gemm_tile(gate, normalized, valid_tokens, start, accum_a_); !st)
      return st;
    if (auto st = gemm_tile(up, normalized, valid_tokens, start, accum_b_); !st)
      return st;
    std::uint64_t const elems = static_cast<std::uint64_t>(rows) * valid_tokens;
    swiglu_kernel<<<static_cast<unsigned>((elems + 255u) / 256u), 256, 0,
                    stream_->native()>>>(accum_a_, accum_b_, swiglu,
                                         valid_tokens, gate.n, rows, start);
    if (auto st = check(cudaGetLastError(), "prefill.swiglu_epilogue"); !st)
      return st;
  }
  return {};
}

std::expected<void, Error> PrefillEngine::mlp(
    PrefillWeight const& gate, PrefillWeight const& up,
    PrefillWeight const& down, float const* h_mid,
    std::uint16_t const* gamma, float eps, float* next_h,
    std::uint32_t valid_tokens, std::uint64_t first_position) {
  if (!stream_ || stream_->empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.mlp",
                                      "stream is closed"));
  }
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(guard.error());
  if (!h_mid || !gamma || !next_h || !std::isfinite(eps) || eps <= 0.0f ||
      gate.n != 17408 || gate.k != 5120 || down.n != 5120 || down.k != 17408 ||
      up.n != gate.n || up.k != gate.k || up.layout != gate.layout ||
      up.quantizer != gate.quantizer ||
      first_position > std::numeric_limits<std::uint64_t>::max() - valid_tokens) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.mlp",
                                      "MLP shape or pointer contract mismatch"));
  }
  auto residual = device_region(h_mid,
      static_cast<std::uint64_t>(valid_tokens) * 5120u * 4u, 4u, device_);
  auto output = device_region(next_h,
      static_cast<std::uint64_t>(valid_tokens) * 5120u * 4u, 4u, device_);
  auto norm = device_region(gamma, 5120u * 2u, 2u, device_);
  auto workspace = region(workspace_.data(), workspace_.bytes());
  if (!residual || !output || !norm || !workspace) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.mlp",
                                      "invalid live range"));
  }
  std::array<Region, 4> ranges{*residual, *output, *norm, *workspace};
  if (auto st = distinct(ranges); !st) return st;
  if (auto st = contract(gate, valid_tokens); !st) return st;
  if (auto st = contract(up, valid_tokens); !st) return st;
  if (auto st = contract(down, valid_tokens); !st) return st;
  std::array<PrefillWeight const*, 3> weights{&gate, &up, &down};
  for (auto const* weight : weights) {
    auto codes = region(weight->codes, weight->codes_bytes);
    if (!codes || overlaps(*output, *codes)) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.mlp",
                                        "output overlaps model weight"));
    }
    if (weight->scales_bytes) {
      auto scales = region(weight->scales, weight->scales_bytes);
      if (!scales || overlaps(*output, *scales)) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.mlp",
                                          "output overlaps model scale"));
      }
    }
  }
  bool const packed = gate.layout == kDecodeLayoutNvFp4V1 && valid_tokens >= 2;
  if (packed) {
    if (auto st=launch_hidden_rms_nvfp4(h_mid,gamma,eps,valid_tokens,
                                       packed_input(),packed_scales(),*stream_); !st) return st;
  } else {
    if (auto st=launch_hidden_rms(h_mid,gamma,eps,valid_tokens,normalized_,*stream_); !st) return st;
  }
  if (auto st=paired_swiglu_impl(gate,up,normalized_,swiglu_,valid_tokens,first_position,packed); !st) return st;
  return project(PrefillProjection{.weight = down, .input = swiglu_,
                                   .output = next_h, .residual = h_mid,
                                   .valid_tokens = valid_tokens,
                                   .first_position = first_position,
                                   .epilogue = PrefillEpilogue::ResidualAddFp32});
}

std::expected<void, Error> PrefillEngine::head_contract(
    PrefillWeight const& head, std::uint16_t const* normalized,
    std::uint32_t valid_tokens, float* logits, std::uint32_t output_rows) {
  if (!stream_ || stream_->empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.head",
                                      "stream is closed"));
  }
  auto guard = stream_->activate();
  if (!guard) return std::unexpected(guard.error());
  if (auto st = contract(head, 1); !st) return st;
  auto input = device_region(normalized,
      static_cast<std::uint64_t>(valid_tokens) * head.k * 2u, 2u, device_);
  auto output = device_region(logits,
      static_cast<std::uint64_t>(output_rows) * head.n * 4u, 4u, device_);
  auto codes = region(head.codes, head.codes_bytes);
  auto workspace = region(workspace_.data(), workspace_.bytes());
  if (!input || !output || !codes || !workspace) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.head",
                                      "invalid head operand range"));
  }
  std::array<Region, 5> ranges{};
  std::size_t count = 0;
  ranges[count++] = *input;
  ranges[count++] = *output;
  ranges[count++] = *codes;
  ranges[count++] = *workspace;
  if (head.scales_bytes) {
    auto scales = region(head.scales, head.scales_bytes);
    if (!scales) return std::unexpected(scales.error());
    ranges[count++] = *scales;
  }
  return distinct(std::span<Region const>{ranges.data(), count});
}

std::expected<void, Error> PrefillEngine::head_generation(
    PrefillWeight const& head, std::uint16_t const* normalized,
    std::uint32_t valid_tokens, std::uint64_t first_position, float* logits) {
  if (!normalized || !logits || valid_tokens == 0 ||
      valid_tokens > token_capacity_ || head.n != 248320 ||
      head.k != 5120 ||
      first_position > std::numeric_limits<std::uint64_t>::max() - valid_tokens) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.head",
                                      "generation head contract mismatch"));
  }
  if (auto st = head_contract(head, normalized, valid_tokens, logits, 1); !st)
    return st;
  return project(PrefillProjection{
      .weight = head,
      .input = normalized + static_cast<std::uint64_t>(valid_tokens - 1u) * head.k,
      .output = logits, .valid_tokens = 1,
      .first_position = first_position + valid_tokens - 1u,
      .epilogue = PrefillEpilogue::StoreFp32});
}

std::expected<void, Error> PrefillEngine::head_evaluation(
    PrefillWeight const& head, std::uint16_t const* normalized,
    std::uint32_t valid_tokens, std::uint64_t first_position,
    std::span<std::uint32_t const> requested_rows, float* logits) {
  if (!normalized || !logits || requested_rows.empty() ||
      requested_rows.size() > kPrefillHeadRows || head.n != 248320 ||
      head.k != 5120 || valid_tokens == 0 || valid_tokens > token_capacity_ ||
      first_position > std::numeric_limits<std::uint64_t>::max() - valid_tokens) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.head",
                                      "evaluation head contract mismatch"));
  }
  for (std::size_t i = 0; i < requested_rows.size(); ++i) {
    auto row = requested_rows[i];
    if (row >= valid_tokens) {
      return std::unexpected(make_error(ErrorCode::InvalidArgument, "prefill.head.rows",
                                        "requested row is out of range"));
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (row == requested_rows[j]) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                          "prefill.head.rows", "duplicate requested row"));
      }
    }
  }
  if (auto st = head_contract(head, normalized, valid_tokens, logits,
                              static_cast<std::uint32_t>(requested_rows.size())); !st)
    return st;
  for (std::size_t i = 0; i < requested_rows.size(); ++i) {
    auto row = requested_rows[i];
    if (auto st = project(PrefillProjection{
        .weight = head,
        .input = normalized + static_cast<std::uint64_t>(row) * head.k,
        .output = logits + static_cast<std::uint64_t>(i) * head.n,
        .valid_tokens = 1, .first_position = first_position + row,
        .epilogue = PrefillEpilogue::StoreFp32}); !st) return st;
  }
  return {};
}

}  // namespace qw38::cuda
