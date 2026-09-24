#include "compiler/quantization/quantizer.hpp"

#include "format/floatcvt.hpp"
#include "format/layout.hpp"

#include <algorithm>
#include <cmath>

namespace qw38::compiler {
namespace {

using qw38::format::bf16_to_fp32;
using qw38::format::ceil_pos_fp16;
using qw38::format::fp16_to_fp32;
using qw38::format::fp32_is_finite;
using qw38::format::kBf16Size;
using qw38::format::kFp16MinNormal;
using qw38::format::kFp16Zero;
using qw38::format::load_u16_le;
using qw38::format::LogicalQuantizerId;
using qw38::format::rne_to_int;

CompilerError qerr(CompilerErrorCode code, std::string_view detail) {
  return make_error(code, "quantizer", detail);
}

}  // namespace

std::uint32_t quantizer_group_size(LogicalQuantizerId id) noexcept {
  switch (id) {
    case LogicalQuantizerId::Q4G64V0:
    case LogicalQuantizerId::Q4G64CandidateV1:
      return qw38::format::kQ4GroupSize;
    case LogicalQuantizerId::Q8G32V0:
    case LogicalQuantizerId::Q8G32CandidateV1:
      return qw38::format::kQ8GroupSize;
    case LogicalQuantizerId::None:
      return 0;
  }
  return 0;
}

int quantizer_qmax(LogicalQuantizerId id) noexcept {
  switch (id) {
    case LogicalQuantizerId::Q4G64V0:
    case LogicalQuantizerId::Q4G64CandidateV1:
      return 7;
    case LogicalQuantizerId::Q8G32V0:
    case LogicalQuantizerId::Q8G32CandidateV1:
      return 127;
    case LogicalQuantizerId::None:
      return 0;
  }
  return 0;
}

std::expected<QuantizedGroup, CompilerError> quantize_group(
    LogicalQuantizerId quantizer, std::span<float const> weights) {
  QuantizedGroup out;
  out.codes.resize(weights.size());
  auto scale = quantize_group_into(quantizer, weights, out.codes);
  if (!scale) {
    return std::unexpected(scale.error());
  }
  out.scale_bits = *scale;
  return out;
}

std::expected<std::uint16_t, CompilerError> quantize_group_into(
    LogicalQuantizerId quantizer, std::span<float const> weights,
    std::span<std::int8_t> codes) {
  auto const group = quantizer_group_size(quantizer);
  auto const qmax = quantizer_qmax(quantizer);
  if (group == 0 || qmax <= 0) {
    return std::unexpected(qerr(CompilerErrorCode::Internal,
                                "logical quantizer id is not Q4G64 or Q8G32"));
  }
  if (weights.size() != group || codes.size() != group) {
    return std::unexpected(
        qerr(CompilerErrorCode::ShapeMismatch,
             "group and code lengths must equal the quantizer group"));
  }
  for (float w : weights) {
    if (!fp32_is_finite(w)) {
      return std::unexpected(qerr(CompilerErrorCode::Nonfinite,
                                  "nonfinite source weight"));
    }
  }

  float a = 0.0f;
  for (float w : weights) {
    a = std::max(a, std::fabs(w));
  }

  if (a == 0.0f) {
    std::fill(codes.begin(), codes.end(), 0);
    return kFp16Zero;
  }

  float const need = a / static_cast<float>(qmax);
  if (!fp32_is_finite(need) || need < 0.0f) {
    return std::unexpected(qerr(CompilerErrorCode::Unrepresentable,
                                "scale is nonfinite"));
  }
  float const min_normal = fp16_to_fp32(kFp16MinNormal);
  std::uint16_t scale_bits = 0;
  if (need <= min_normal) {
    scale_bits = kFp16MinNormal;
  } else {
    scale_bits = ceil_pos_fp16(need);
    if (scale_bits == 0xFFFF) {
      return std::unexpected(qerr(CompilerErrorCode::Unrepresentable,
                                  "scale exceeds finite FP16"));
    }
  }
  float const scale = fp16_to_fp32(scale_bits);
  float const hi = static_cast<float>(qmax) + 0.5f;
  float const lo = static_cast<float>(-qmax) - 0.5f;
  for (std::uint32_t i = 0; i < group; ++i) {
    float const qf = weights[i] / scale;
    int q = 0;
    if (qf >= hi) {
      q = qmax;
    } else if (qf <= lo) {
      q = -qmax;
    } else {
      q = rne_to_int(qf);
      if (q > qmax) {
        q = qmax;
      } else if (q < -qmax) {
        q = -qmax;
      }
    }
    codes[i] = static_cast<std::int8_t>(q);
  }
  return scale_bits;
}

std::expected<LogicalWeightCodes, CompilerError> quantize_fp32(
    LogicalQuantizerId quantizer, std::uint64_t n, std::uint64_t k,
    std::span<float const> weights) {
  auto const group = quantizer_group_size(quantizer);
  auto const qmax = quantizer_qmax(quantizer);
  if (group == 0 || qmax <= 0) {
    return std::unexpected(qerr(CompilerErrorCode::Internal,
                                "logical quantizer id is not Q4G64 or Q8G32"));
  }
  if (n == 0 || k == 0 || k % group != 0) {
    return std::unexpected(qerr(CompilerErrorCode::ShapeMismatch,
                                "K must be positive and divide the group size"));
  }
  auto const need = n * k;
  if (need / k != n || weights.size() != need) {
    return std::unexpected(qerr(CompilerErrorCode::ShapeMismatch,
                                "weight count must equal N*K"));
  }
  LogicalWeightCodes out;
  out.quantizer = quantizer;
  out.n = n;
  out.k = k;
  out.group_size = group;
  out.qmax = qmax;
  out.codes.resize(static_cast<std::size_t>(need));
  auto const groups_per_row = k / group;
  out.scales.resize(static_cast<std::size_t>(n * groups_per_row));
  for (std::uint64_t row = 0; row < n; ++row) {
    for (std::uint64_t g = 0; g < groups_per_row; ++g) {
      auto const* src = weights.data() + row * k + g * group;
      auto qg = quantize_group(quantizer, std::span<float const>{src, group});
      if (!qg) {
        return std::unexpected(qg.error());
      }
      out.scales[static_cast<std::size_t>(row * groups_per_row + g)] =
          qg->scale_bits;
      std::copy(qg->codes.begin(), qg->codes.end(),
                out.codes.begin() +
                    static_cast<std::ptrdiff_t>(row * k + g * group));
    }
  }
  return out;
}

std::expected<LogicalWeightCodes, CompilerError> quantize_bf16(
    LogicalQuantizerId quantizer, std::uint64_t n, std::uint64_t k,
    std::span<std::byte const> weights) {
  if (n == 0 || k == 0) {
    return std::unexpected(qerr(CompilerErrorCode::ShapeMismatch,
                                "N and K must be positive"));
  }
  auto need = qw38::format::checked_mul(n, k, 0, "quantizer.elements");
  if (!need) {
    return std::unexpected(from_format(need.error()));
  }
  auto bytes =
      qw38::format::checked_mul(*need, kBf16Size, 0, "quantizer.bytes");
  if (!bytes) {
    return std::unexpected(from_format(bytes.error()));
  }
  if (weights.size() != *bytes) {
    return std::unexpected(qerr(CompilerErrorCode::ShapeMismatch,
                                "BF16 byte length must equal 2*N*K"));
  }
  std::vector<float> fp32(static_cast<std::size_t>(*need));
  for (std::uint64_t i = 0; i < *need; ++i) {
    auto const bits = load_u16_le(weights.data() + i * kBf16Size);
    float const v = bf16_to_fp32(bits);
    if (!fp32_is_finite(v)) {
      return std::unexpected(qerr(CompilerErrorCode::Nonfinite,
                                  "nonfinite BF16 source weight"));
    }
    fp32[static_cast<std::size_t>(i)] = v;
  }
  return quantize_fp32(quantizer, n, k, fp32);
}

}  // namespace qw38::compiler
