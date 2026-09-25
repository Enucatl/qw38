// Q4_K fitting port from llama.cpp e6ab7c1a4, ggml/src/ggml-quants.c
// and portable half conversions from ggml/src/ggml-impl.h.
// Copyright (c) 2023-2026 The ggml authors. MIT license:
// third_party/llama.cpp-q4k/LICENSE.
#include "compiler/quantization/q4k.hpp"
#include "format/floatcvt.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace qw38::compiler {
namespace {
using ggml_fp16_t = uint16_t;
#define GGML_RESTRICT __restrict
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#undef GGML_RESTRICT
#undef MIN
#undef MAX
}  // namespace

std::expected<void, CompilerError> quantize_q4k_block(
    std::span<float const, 256> x, std::span<std::int8_t, 256> codes,
    std::span<std::uint8_t, 16> metadata) {
  for (float const value : x) {
    if (!std::isfinite(value)) {
      return std::unexpected(make_error(CompilerErrorCode::Nonfinite,
                                       "weight", "Q4_K input is nonfinite"));
    }
    // Squared/cubic fitting moments must remain finite. Check before invoking
    // the reference's integer rounding helper (undefined on nonfinite input).
    if (std::abs(value) > 1.0e10f) {
      return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                       "weight", "Q4_K fitting range exceeded"));
    }
  }
  uint8_t levels[256];
  uint8_t auxiliary[32];
  float weights[32];
  float minima[8];
  float scales[8];
  float max_scale = 0;
  float max_min = 0;
  for (int group = 0; group < 8; ++group) {
    float minimum = 0;
    float maximum = x[32 * group];
    for (int i = 0; i < 32; ++i) {
      minimum = std::min(minimum, x[32 * group + i]);
      maximum = std::max(maximum, x[32 * group + i]);
    }
    // The reference assumes finite reciprocal fitting scales. Very small
    // finite BF16 inputs can violate that before any FP16 conversion occurs.
    // Reject that undefined fitting domain instead of feeding NaN to its
    // nearest_int helper; FP16 subnormal output scales remain supported.
    if (maximum != minimum && !std::isfinite(16.f / (maximum - minimum))) {
      return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                       "weight", "Q4_K fitting reciprocal overflow"));
    }
    float sum_x2 = 0;
    for (int i = 0; i < 32; ++i) sum_x2 += x[32 * group + i] * x[32 * group + i];
    float average = sqrtf(sum_x2 / 32);
    for (int i = 0; i < 32; ++i) weights[i] = average + fabsf(x[32 * group + i]);
    scales[group] = make_qkx2_quants(32, 15, x.data() + 32 * group, weights,
        levels + 32 * group, &minima[group], auxiliary, -1.f, 0.1f, 20, false);
    if (scales[group] > max_scale) max_scale = scales[group];
    if (minima[group] > max_min) max_min = minima[group];
  }
  float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
  float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
  if (!std::isfinite(inv_scale) || !std::isfinite(inv_min)) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                     "scale", "Q4_K compression reciprocal overflow"));
  }
  auto* packed = metadata.data() + 4;
  for (int group = 0; group < 8; ++group) {
    uint8_t scale = nearest_int(inv_scale * scales[group]);
    uint8_t minimum = nearest_int(inv_min * minima[group]);
    scale = std::min(uint8_t{63}, scale);
    minimum = std::min(uint8_t{63}, minimum);
    if (group < 4) {
      packed[group] = scale;
      packed[group + 4] = minimum;
    } else {
      packed[group + 4] = (scale & 0xF) | ((minimum & 0xF) << 4);
      packed[group - 4] |= ((scale >> 4) << 6);
      packed[group] |= ((minimum >> 4) << 6);
    }
  }
  uint16_t d = ggml_compute_fp32_to_fp16(max_scale / 63.f);
  uint16_t dmin = ggml_compute_fp32_to_fp16(max_min / 63.f);
  if ((d & 0x7C00u) == 0x7C00u || (dmin & 0x7C00u) == 0x7C00u) {
    return std::unexpected(make_error(CompilerErrorCode::Unrepresentable,
                                     "scale", "Q4_K FP16 scale overflow"));
  }
  metadata[0] = d & 0xFF;
  metadata[1] = d >> 8;
  metadata[2] = dmin & 0xFF;
  metadata[3] = dmin >> 8;
  // Recalculate codes using the compressed six-bit scales and rounded FP16
  // superblock scales. Keeping the first fit's codes changes the algorithm.
  for (int group = 0; group < 8; ++group) {
    uint8_t scale, minimum;
    get_scale_min_k4(group, packed, &scale, &minimum);
    float const step = ggml_compute_fp16_to_fp32(d) * scale;
    if (!step) continue;
    float const offset = ggml_compute_fp16_to_fp32(dmin) * minimum;
    for (int i = 0; i < 32; ++i) {
      int level = nearest_int((x[32 * group + i] + offset) / step);
      levels[32 * group + i] = std::clamp(level, 0, 15);
    }
  }
  for (int i = 0; i < 256; ++i) codes[i] = static_cast<std::int8_t>(levels[i]);
  return {};
}
}  // namespace qw38::compiler
