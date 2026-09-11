"""Canonical kernel-parity checker (Python; matches cuda/kernel_parity.cuh).

GPU candidates are compared to an independent CPU/dequant of the same quantized
weights and staged/input activations. Admission uses quantized-operation
association (OR-rule) or same-math bit identity. OPT-074 family admission is
not required.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "cuda/kernel_parity.cuh"
CONTRACT = ROOT / "pins/kernel_parity_v1_contract.json"

Q80_ABS_SCALE = 0.05
Q80_REL_TOL = 0.05
Q4K_ABS_SCALE = 0.20
Q4K_REL_TOL = 0.05
Q6K_ABS_SCALE = 0.20
Q6K_REL_TOL = 0.05
SAME_MATH_ABS_TOL = 0.0
SAME_MATH_REL_TOL = 0.0
CUD001_HISTORICAL_ABS = 3.0e-4
OPT074_FAMILY_ADMISSION_REQUIRED = False

ASSOCIATION = "quantized_operation_association"
SAME_MATH = "same_math_equivalence"
STAGING_SUM_Q = "quartz_q8_1_sum_q"
STAGING_SUM_X = "llama_q8_1_sum_x"

FAMILIES: dict[str, dict[str, Any]] = {
    "Q8_0": {
        "applies": True,
        "abs_scale": Q80_ABS_SCALE,
        "rel_tol": Q80_REL_TOL,
    },
    "Q4_K": {
        "applies": True,
        "abs_scale": Q4K_ABS_SCALE,
        "rel_tol": Q4K_REL_TOL,
    },
    "Q6_K": {
        "applies": True,
        "abs_scale": Q6K_ABS_SCALE,
        "rel_tol": Q6K_REL_TOL,
    },
    "Q2_K": {"applies": False, "status": "not_applicable"},
    "IQ2": {"applies": False, "status": "not_applicable"},
}

POLICY_STATEMENT = (
    "Kernel admission proves implementation of the intended quantized "
    "operation. It does not prove full-model quality and does not depend on "
    "llama GPU versus FP64 consistency."
)


class KernelParityError(AssertionError):
    """Inadmissible kernel-parity comparison or envelope."""


def family_envelope(family: str) -> dict[str, Any]:
    try:
        return dict(FAMILIES[family])
    except KeyError as exc:
        raise KernelParityError(f"unknown family {family}") from exc


def abs_tolerance(abs_scale: float, columns: int) -> float:
    if columns < 0:
        raise KernelParityError("K (weight columns) must be non-negative")
    return float(abs_scale) * math.sqrt(float(columns))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def check_staging(
    candidate_field: str | None,
    reference_field: str | None,
) -> dict[str, Any]:
    """Hard invariant: a tolerance cannot equate Q8_1 sum_q and sum_x."""
    candidate = str(candidate_field or "")
    reference = str(reference_field or "")
    fields = {candidate, reference}
    equated = STAGING_SUM_Q in fields and STAGING_SUM_X in fields
    record = {
        "candidate_field": candidate or None,
        "reference_field": reference or None,
        "equated_sum_q_and_sum_x": equated,
        "pass": not equated,
    }
    if equated:
        raise KernelParityError(
            "Q8_1 sum_q and sum_x are distinct typed fields; a kernel-parity "
            "tolerance cannot equate them"
        )
    return record


def check_close(
    got: Sequence[float],
    ref: Sequence[float],
    *,
    family: str,
    columns: int,
    class_name: str = ASSOCIATION,
    compute_cosine: bool = True,
    candidate_field: str | None = None,
    reference_field: str | None = None,
) -> dict[str, Any]:
    """Identical semantics to cuda/kernel_parity.cuh check_close."""
    if candidate_field or reference_field:
        check_staging(candidate_field, reference_field)
    diagnostics: dict[str, Any] = {
        "class": class_name,
        "family": family,
        "columns": int(columns),
        "max_abs": 0.0,
        "max_rel": 0.0,
        "rms": 0.0,
        "cosine": None,
        "failing_count": 0,
        "nonfinite_count": 0,
        "compared": 0,
        "pass": False,
        "applicable": False,
        "abs_tol": 0.0,
        "rel_tol": 0.0,
        "reason": "",
    }
    if len(got) != len(ref):
        diagnostics["nonfinite_count"] = 1
        diagnostics["reason"] = "shape_mismatch"
        return diagnostics
    if class_name == SAME_MATH:
        diagnostics["applicable"] = True
        abs_tol = SAME_MATH_ABS_TOL
        rel_tol = SAME_MATH_REL_TOL
    elif class_name == ASSOCIATION:
        envelope = family_envelope(family)
        if not envelope.get("applies"):
            diagnostics["reason"] = "not_applicable"
            return diagnostics
        diagnostics["applicable"] = True
        abs_tol = abs_tolerance(float(envelope["abs_scale"]), int(columns))
        rel_tol = float(envelope["rel_tol"])
    else:
        raise KernelParityError(f"unknown parity class {class_name}")
    diagnostics["abs_tol"] = abs_tol
    diagnostics["rel_tol"] = rel_tol
    squared = 0.0
    dot = 0.0
    got_sq = 0.0
    ref_sq = 0.0
    for candidate, reference in zip(got, ref, strict=True):
        cand = float(candidate)
        reference_f = float(reference)
        if not _finite(cand) or not _finite(reference_f):
            diagnostics["nonfinite_count"] += 1
            continue
        diagnostics["compared"] += 1
        absolute = abs(cand - reference_f)
        if reference_f != 0.0:
            relative = absolute / abs(reference_f)
        else:
            relative = math.inf if absolute > 0.0 else 0.0
        diagnostics["max_abs"] = max(float(diagnostics["max_abs"]), absolute)
        if math.isfinite(relative):
            diagnostics["max_rel"] = max(float(diagnostics["max_rel"]), relative)
        squared += absolute * absolute
        if compute_cosine:
            dot += cand * reference_f
            got_sq += cand * cand
            ref_sq += reference_f * reference_f
        if absolute > abs_tol and relative > rel_tol:
            diagnostics["failing_count"] += 1
    compared = int(diagnostics["compared"])
    diagnostics["rms"] = 0.0 if compared == 0 else math.sqrt(squared / compared)
    if compute_cosine:
        denom = math.sqrt(got_sq) * math.sqrt(ref_sq)
        if denom > 0.0:
            diagnostics["cosine"] = dot / denom
    diagnostics["pass"] = (
        bool(diagnostics["applicable"])
        and int(diagnostics["nonfinite_count"]) == 0
        and int(diagnostics["failing_count"]) == 0
    )
    if diagnostics["pass"]:
        diagnostics["reason"] = "pass"
    elif int(diagnostics["nonfinite_count"]) > 0:
        diagnostics["reason"] = "nonfinite"
    else:
        diagnostics["reason"] = "association_fail"
    return diagnostics


def header_constants(text: str | None = None) -> dict[str, float | bool]:
    source = HEADER.read_text(encoding="utf-8") if text is None else text
    names = {
        "kQ80AbsScale": "q8_0_abs_scale",
        "kQ80RelTol": "q8_0_rel_tol",
        "kQ4KAbsScale": "q4_k_abs_scale",
        "kQ4KRelTol": "q4_k_rel_tol",
        "kQ6KAbsScale": "q6_k_abs_scale",
        "kQ6KRelTol": "q6_k_rel_tol",
        "kSameMathAbsTol": "same_math_abs_tol",
        "kSameMathRelTol": "same_math_rel_tol",
        "kCud001HistoricalAbs": "cud001_historical_abs",
    }
    parsed: dict[str, float | bool] = {}
    for cpp_name, key in names.items():
        match = re.search(rf"constexpr float {cpp_name} = ([0-9eE.+-]+)F;", source)
        if not match:
            raise KernelParityError(f"missing {cpp_name} in {HEADER}")
        parsed[key] = float(match.group(1))
    flag = re.search(
        r"constexpr bool kOpt074FamilyAdmissionRequired = (true|false);",
        source,
    )
    if not flag:
        raise KernelParityError("missing kOpt074FamilyAdmissionRequired")
    parsed["opt074_family_admission_required"] = flag.group(1) == "true"
    compact = source.replace(" ", "")
    if "abs_scale*sqrt_f" not in compact and "abs_scale*sqrt(" not in compact:
        raise KernelParityError("C++ abs_tolerance must be abs_scale * sqrt(K)")
    if "absolute > abs_tol && relative > rel_tol" not in source:
        raise KernelParityError("C++ checker must implement the association AND-fail")
    return parsed


def assert_cpp_matches_python() -> dict[str, float | bool]:
    parsed = header_constants()
    expected = {
        "q8_0_abs_scale": Q80_ABS_SCALE,
        "q8_0_rel_tol": Q80_REL_TOL,
        "q4_k_abs_scale": Q4K_ABS_SCALE,
        "q4_k_rel_tol": Q4K_REL_TOL,
        "q6_k_abs_scale": Q6K_ABS_SCALE,
        "q6_k_rel_tol": Q6K_REL_TOL,
        "same_math_abs_tol": SAME_MATH_ABS_TOL,
        "same_math_rel_tol": SAME_MATH_REL_TOL,
        "cud001_historical_abs": CUD001_HISTORICAL_ABS,
        "opt074_family_admission_required": OPT074_FAMILY_ADMISSION_REQUIRED,
    }
    for key, value in expected.items():
        if parsed[key] != value:
            raise KernelParityError(f"C++ {key}={parsed[key]!r} != Python {value!r}")
    return parsed


def _cpp_driver_source() -> str:
    return r"""
#include "cuda/kernel_parity.cuh"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
  using qw38::cuda::kernel_parity::check_close;
  using qw38::cuda::kernel_parity::class_from_name;
  using qw38::cuda::kernel_parity::family_from_name;
  char class_name[64];
  char family_name[64];
  int columns = 0;
  int count = 0;
  int cosine = 1;
  if (std::scanf("%63s %63s %d %d %d", class_name, family_name, &columns, &count,
                 &cosine) != 5) {
    std::fprintf(stderr, "bad header\n");
    return 2;
  }
  std::vector<float> got(static_cast<std::size_t>(count));
  std::vector<float> ref(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    if (std::scanf("%f", &got[static_cast<std::size_t>(index)]) != 1) return 2;
  }
  for (int index = 0; index < count; ++index) {
    if (std::scanf("%f", &ref[static_cast<std::size_t>(index)]) != 1) return 2;
  }
  const auto diagnostics = check_close(
      got, ref, family_from_name(family_name),
      static_cast<std::size_t>(columns), class_from_name(class_name),
      cosine != 0);
  std::printf(
      "pass=%d applicable=%d max_abs=%.9g max_rel=%.9g rms=%.9g "
      "failing=%zu nonfinite=%zu compared=%zu cosine=%.9g abs_tol=%.9g "
      "rel_tol=%.9g\n",
      diagnostics.pass ? 1 : 0, diagnostics.applicable ? 1 : 0,
      diagnostics.max_abs, diagnostics.max_rel, diagnostics.rms,
      diagnostics.failing_count, diagnostics.nonfinite_count,
      diagnostics.compared,
      diagnostics.cosine, diagnostics.abs_tol, diagnostics.rel_tol);
  return 0;
}
"""


_HOST_BINARY: Path | None = None


def host_cpp_binary() -> Path:
    global _HOST_BINARY
    if _HOST_BINARY is not None and _HOST_BINARY.is_file():
        return _HOST_BINARY
    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)
    source = build / "opt081_kernel_parity_host_check.cpp"
    binary = build / "opt081_kernel_parity_host_check"
    source.write_text(_cpp_driver_source(), encoding="utf-8")
    compile_cmd = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O0",
        "-I",
        str(ROOT),
        str(source),
        "-o",
        str(binary),
    ]
    completed = subprocess.run(
        compile_cmd, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise KernelParityError(
            "host C++ kernel-parity checker failed to compile:\n" + completed.stderr
        )
    _HOST_BINARY = binary
    return binary


def cpp_check_close(
    got: Sequence[float],
    ref: Sequence[float],
    *,
    family: str,
    columns: int,
    class_name: str = ASSOCIATION,
    compute_cosine: bool = True,
) -> dict[str, Any]:
    payload = (
        f"{class_name} {family} {int(columns)} {len(got)} "
        f"{1 if compute_cosine else 0}\n"
        + " ".join(str(float(value)) for value in got)
        + "\n"
        + " ".join(str(float(value)) for value in ref)
        + "\n"
    )
    completed = subprocess.run(
        [str(host_cpp_binary())],
        input=payload,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise KernelParityError(
            "host C++ kernel-parity checker failed:\n" + completed.stderr
        )
    line = completed.stdout.strip().splitlines()[-1]
    parsed: dict[str, Any] = {}
    for token in line.split():
        key, _, raw = token.partition("=")
        if raw in {"nan", "NaN", "-nan"}:
            parsed[key] = None
            continue
        if key in {"pass", "applicable"}:
            parsed[key] = raw == "1"
        elif key in {"failing", "nonfinite", "compared"}:
            parsed[key] = int(raw)
        else:
            parsed[key] = float(raw)
    parsed["failing_count"] = int(parsed.pop("failing"))
    parsed["nonfinite_count"] = int(parsed.pop("nonfinite"))
    parsed["class"] = class_name
    parsed["family"] = family
    return parsed


def diagnostics_agree(
    python: Mapping[str, Any], cpp: Mapping[str, Any], *, tol: float = 1e-6
) -> None:
    for key in ("pass", "applicable", "failing_count", "nonfinite_count"):
        if python[key] != cpp[key]:
            raise KernelParityError(f"{key}: python={python[key]!r} cpp={cpp[key]!r}")
    for key in ("max_abs", "max_rel", "rms", "abs_tol", "rel_tol"):
        left = float(python[key])
        right = float(cpp[key])
        if abs(left - right) > tol and not (left == 0.0 and right == 0.0):
            raise KernelParityError(f"{key}: python={left!r} cpp={right!r}")
    left_cos = python.get("cosine")
    right_cos = cpp.get("cosine")
    if left_cos is None or (isinstance(right_cos, float) and math.isnan(right_cos)):
        return
    if right_cos is None or (
        isinstance(left_cos, float) and math.isnan(float(left_cos))
    ):
        return
    if abs(float(left_cos) - float(right_cos)) > 1e-5:
        raise KernelParityError(f"cosine: python={left_cos!r} cpp={right_cos!r}")


def _reset_host_binary_cache() -> None:
    global _HOST_BINARY
    _HOST_BINARY = None
