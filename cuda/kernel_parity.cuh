#pragma once

// OPT-081 canonical kernel-parity checker. GPU candidates are compared to an
// independent CPU/dequant of the same quantized weights and staged/input
// activations. Do not compare against original unquantized FP64 weights, llama
// GPU, or a production-layer error budget. New work must call this helper;
// do not copy a third tolerance implementation.

#include <cmath>
#include <cstddef>
#include <cstring>

#ifndef __CUDA_ARCH__
#include <limits>
#include <vector>
#endif

#ifdef __CUDACC__
#define QW38_KERNEL_PARITY_HD __host__ __device__
#else
#define QW38_KERNEL_PARITY_HD
#endif

namespace qw38::cuda::kernel_parity {

constexpr float kQ80AbsScale = 0.05F;
constexpr float kQ80RelTol = 0.05F;
constexpr float kQ4KAbsScale = 0.20F;
constexpr float kQ4KRelTol = 0.05F;
constexpr float kQ6KAbsScale = 0.20F;
constexpr float kQ6KRelTol = 0.05F;
constexpr float kSameMathAbsTol = 0.0F;
constexpr float kSameMathRelTol = 0.0F;
constexpr float kCud001HistoricalAbs = 3.0e-4F;
constexpr bool kOpt074FamilyAdmissionRequired = false;

enum class Family {
  Q8_0,
  Q4_K,
  Q6_K,
  Q2_K,
  IQ2,
};

enum class Class {
  QuantizedOperationAssociation,
  SameMathEquivalence,
};

struct Envelope {
  bool applicable;
  float abs_scale;
  float rel_tol;
};

struct Diagnostics {
  float max_abs;
  float max_rel;
  float rms;
  float cosine;
  std::size_t failing_count;
  std::size_t nonfinite_count;
  std::size_t compared;
  bool pass;
  bool applicable;
  Class cls;
  Family family;
  float abs_tol;
  float rel_tol;
};

QW38_KERNEL_PARITY_HD inline bool is_finite_f(float value) {
#if defined(__CUDA_ARCH__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

QW38_KERNEL_PARITY_HD inline float sqrt_f(float value) {
#if defined(__CUDA_ARCH__)
  return sqrtf(value);
#else
  return std::sqrt(value);
#endif
}

QW38_KERNEL_PARITY_HD inline float abs_f(float value) {
#if defined(__CUDA_ARCH__)
  return fabsf(value);
#else
  return std::fabs(value);
#endif
}

QW38_KERNEL_PARITY_HD inline float nan_f() {
#if defined(__CUDA_ARCH__)
  return nanf("");
#else
  return std::numeric_limits<float>::quiet_NaN();
#endif
}

QW38_KERNEL_PARITY_HD inline float inf_f() {
#if defined(__CUDA_ARCH__)
  return __int_as_float(0x7f800000);
#else
  return std::numeric_limits<float>::infinity();
#endif
}

QW38_KERNEL_PARITY_HD inline float sqrt_d_as_f(double value) {
#if defined(__CUDA_ARCH__)
  return sqrtf(static_cast<float>(value));
#else
  return static_cast<float>(std::sqrt(value));
#endif
}

QW38_KERNEL_PARITY_HD inline float abs_tolerance(float abs_scale,
                                                 std::size_t columns) {
  return abs_scale * sqrt_f(static_cast<float>(columns));
}

QW38_KERNEL_PARITY_HD inline Envelope family_envelope(Family family) {
  Envelope envelope;
  envelope.applicable = false;
  envelope.abs_scale = 0.0F;
  envelope.rel_tol = 0.0F;
  if (family == Family::Q8_0) {
    envelope.applicable = true;
    envelope.abs_scale = kQ80AbsScale;
    envelope.rel_tol = kQ80RelTol;
  } else if (family == Family::Q4_K) {
    envelope.applicable = true;
    envelope.abs_scale = kQ4KAbsScale;
    envelope.rel_tol = kQ4KRelTol;
  } else if (family == Family::Q6_K) {
    envelope.applicable = true;
    envelope.abs_scale = kQ6KAbsScale;
    envelope.rel_tol = kQ6KRelTol;
  }
  return envelope;
}

#ifndef __CUDA_ARCH__
inline Family family_from_name(const char* name) {
  if (name != nullptr && std::strcmp(name, "Q8_0") == 0) return Family::Q8_0;
  if (name != nullptr && std::strcmp(name, "Q4_K") == 0) return Family::Q4_K;
  if (name != nullptr && std::strcmp(name, "Q6_K") == 0) return Family::Q6_K;
  if (name != nullptr && std::strcmp(name, "Q2_K") == 0) return Family::Q2_K;
  if (name != nullptr && std::strcmp(name, "IQ2") == 0) return Family::IQ2;
  return Family::IQ2;
}

inline Class class_from_name(const char* name) {
  if (name != nullptr && std::strcmp(name, "same_math_equivalence") == 0) {
    return Class::SameMathEquivalence;
  }
  return Class::QuantizedOperationAssociation;
}
#endif

QW38_KERNEL_PARITY_HD inline Diagnostics empty_diagnostics(Family family,
                                                           Class cls) {
  Diagnostics diagnostics{};
  diagnostics.max_abs = 0.0F;
  diagnostics.max_rel = 0.0F;
  diagnostics.rms = 0.0F;
  diagnostics.cosine = nan_f();
  diagnostics.failing_count = 0;
  diagnostics.nonfinite_count = 0;
  diagnostics.compared = 0;
  diagnostics.pass = false;
  diagnostics.applicable = false;
  diagnostics.cls = cls;
  diagnostics.family = family;
  diagnostics.abs_tol = 0.0F;
  diagnostics.rel_tol = 0.0F;
  return diagnostics;
}

// Association: an element fails only when abs > abs_tol AND rel > rel_tol.
// Passing either test is sufficient. Nonfinite count must be 0.
template <typename T>
QW38_KERNEL_PARITY_HD inline Diagnostics check_close(
    const T* got, const T* ref, std::size_t count, Family family,
    std::size_t columns, Class cls, bool compute_cosine) {
  Diagnostics diagnostics = empty_diagnostics(family, cls);
  if (got == nullptr || ref == nullptr) {
    diagnostics.nonfinite_count = 1;
    return diagnostics;
  }
  float abs_scale = 0.0F;
  float rel_tol = 0.0F;
  if (cls == Class::SameMathEquivalence) {
    diagnostics.applicable = true;
    abs_scale = 0.0F;
    rel_tol = kSameMathRelTol;
    diagnostics.abs_tol = kSameMathAbsTol;
    diagnostics.rel_tol = kSameMathRelTol;
  } else {
    const Envelope envelope = family_envelope(family);
    diagnostics.applicable = envelope.applicable;
    if (!envelope.applicable) {
      return diagnostics;
    }
    abs_scale = envelope.abs_scale;
    rel_tol = envelope.rel_tol;
    diagnostics.abs_tol = abs_tolerance(abs_scale, columns);
    diagnostics.rel_tol = rel_tol;
  }
  const float abs_tol = diagnostics.abs_tol;
  double squared = 0.0;
  double dot = 0.0;
  double got_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const float candidate = static_cast<float>(got[index]);
    const float reference = static_cast<float>(ref[index]);
    if (!is_finite_f(candidate) || !is_finite_f(reference)) {
      ++diagnostics.nonfinite_count;
      continue;
    }
    ++diagnostics.compared;
    const float absolute = abs_f(candidate - reference);
    const float relative =
        reference != 0.0F ? absolute / abs_f(reference)
                          : (absolute > 0.0F ? inf_f() : 0.0F);
    if (absolute > diagnostics.max_abs) diagnostics.max_abs = absolute;
    if (is_finite_f(relative) && relative > diagnostics.max_rel) {
      diagnostics.max_rel = relative;
    }
    squared += static_cast<double>(absolute) * static_cast<double>(absolute);
    if (compute_cosine) {
      dot += static_cast<double>(candidate) * static_cast<double>(reference);
      got_sq += static_cast<double>(candidate) * static_cast<double>(candidate);
      ref_sq += static_cast<double>(reference) * static_cast<double>(reference);
    }
    if (absolute > abs_tol && relative > rel_tol) {
      ++diagnostics.failing_count;
    }
  }
  diagnostics.rms =
      diagnostics.compared == 0
          ? 0.0F
          : sqrt_d_as_f(squared / static_cast<double>(diagnostics.compared));
  if (compute_cosine) {
    const float got_norm = sqrt_d_as_f(got_sq);
    const float ref_norm = sqrt_d_as_f(ref_sq);
    const float denom = got_norm * ref_norm;
    if (denom > 0.0F) {
      diagnostics.cosine = static_cast<float>(dot / static_cast<double>(denom));
    }
  }
  diagnostics.pass =
      diagnostics.applicable && diagnostics.nonfinite_count == 0 &&
      diagnostics.failing_count == 0;
  return diagnostics;
}

#ifndef __CUDA_ARCH__
inline Diagnostics check_close(const std::vector<float>& got,
                               const std::vector<float>& ref, Family family,
                               std::size_t columns, Class cls,
                               bool compute_cosine = true) {
  if (got.size() != ref.size()) {
    Diagnostics diagnostics = empty_diagnostics(family, cls);
    diagnostics.nonfinite_count = 1;
    return diagnostics;
  }
  return check_close(got.data(), ref.data(), got.size(), family, columns, cls,
                     compute_cosine);
}
#endif

}  // namespace qw38::cuda::kernel_parity
