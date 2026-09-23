# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.5.3"]
# ///
"""Compare TASK-017 BF16 and V0 decode against pinned source outputs."""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Read one FP32 checkpoint and enforce its declared shape."""
    values = np.fromfile(path, dtype="<f4")
    if values.size != int(np.prod(shape)) or not np.isfinite(values).all():
        raise ValueError(f"invalid FP32 checkpoint: {path}")
    return values.reshape(shape)


def read_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """Widen an engine BF16 checkpoint without changing its bits."""
    values = np.fromfile(path, dtype="<u2")
    if values.size != int(np.prod(shape)):
        raise ValueError(f"invalid BF16 checkpoint: {path}")
    return (values.astype("<u4") << 16).view("<f4").reshape(shape)


def digest(values: np.ndarray) -> str:
    """Hash activation bytes for reproducible checkpoint identity."""
    return hashlib.sha256(values.tobytes()).hexdigest()


def compare(
    name: str, got: np.ndarray, want: np.ndarray, mean: float, maximum: float
) -> bool:
    """Print numerical error and test declared absolute tolerances."""
    difference = np.abs(got - want)
    observed_mean = float(difference.mean())
    observed_max = float(difference.max())
    passed = observed_mean <= mean and observed_max <= maximum
    print(
        name,
        "mean_abs",
        observed_mean,
        "max_abs",
        observed_max,
        "got_sha256",
        digest(got),
        "source_sha256",
        digest(want),
        "PASS" if passed else "FAIL",
    )
    return passed


def main() -> int:
    """Check logits, residuals, recurrent/conv state, and attention KV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    base = args.directory
    passed = True
    for position in (0, 1):
        source = read_f32(base / f"source-logits-{position}.f32", (248320,))
        bf16 = read_f32(base / f"bf16-logits-{position}.f32", (248320,))
        v0 = read_f32(base / f"engine-logits-{position}.f32", (248320,))
        passed &= compare(f"bf16_logits_{position}", bf16, source, 0.035, 0.16)
        v0_error = np.abs(v0 - source)
        print(
            f"v0_logits_{position}",
            "mean_abs",
            float(v0_error.mean()),
            "max_abs",
            float(v0_error.max()),
            "argmax",
            int(v0.argmax()),
            "sha256",
            digest(v0),
        )

        source_residuals = read_f32(
            base / f"source-residuals-{position}.f32", (65, 5120)
        )
        engine_residuals = read_f32(
            base / f"bf16-logits-residuals-{position}.f32", (64, 5120)
        )
        for layer in (0, 28, 60):
            passed &= compare(
                f"residual_{position}_{layer}",
                engine_residuals[layer],
                source_residuals[layer + 1],
                0.075,
                1.5,
            )
            source_s = read_f32(
                base / f"source-recurrent-{position}-{layer}.f32", (48, 128, 128)
            )
            engine_s = read_f32(
                base / f"bf16-logits-recurrent-{position}-{layer}.bin", (48, 128, 128)
            )
            passed &= compare(
                f"recurrent_{position}_{layer}",
                engine_s,
                source_s.transpose(0, 2, 1),
                0.0002,
                0.05,
            )
            source_conv = read_f32(
                base / f"source-conv-{position}-{layer}.f32", (10240, 4)
            )
            engine_conv = read_bf16(
                base / f"bf16-logits-conv-{position}-{layer}.bin", (3, 10240)
            )
            oldest_first = [(position + 1 + i) % 3 for i in range(3)]
            passed &= compare(
                f"conv_{position}_{layer}",
                engine_conv[oldest_first],
                source_conv[:, -3:].T,
                0.02,
                0.26,
            )

        engine_kv = read_bf16(base / f"bf16-logits-kv-{position}-3.bin", (2, 4, 2, 256))
        for component, kind in enumerate(("key", "value")):
            source_kv = read_f32(
                base / f"source-{kind}-{position}-3.f32", (4, position + 1, 256)
            )
            passed &= compare(
                f"kv_{kind}_{position}_3",
                engine_kv[component, :, : position + 1, :],
                source_kv,
                0.06,
                1.2,
            )

    source_continued = read_f32(base / "source-logits-1.f32", (248320,))
    source_two_token = read_f32(base / "source-prefill-logits-1.f32", (248320,))
    passed &= compare(
        "source_cached_vs_two_token",
        source_continued,
        source_two_token,
        0.025,
        0.15,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
