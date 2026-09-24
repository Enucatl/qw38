"""Small CPU reference for logical NVFP4 and MXFP4 block quantization."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Literal, Sequence

Format = Literal["nvfp4", "mxfp4"]

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_BLOCK_SIZE = {"nvfp4": 16, "mxfp4": 32}


@dataclass(frozen=True)
class PackedFp4:
    """A row-major packed matrix with one logical scale per row and K block.

    Attributes:
        format: Scale family, either ``nvfp4`` or ``mxfp4``.
        rows: Number of logical matrix rows.
        columns: Number of logical columns before K padding.
        padded_columns: K rounded up to the format's scale block size.
        payload: Two E2M1 codes per byte, even flattened index in low nibble.
        scale_codes: One UE4M3 or UE8M0 code per row and padded-K block.
        tensor_scale: Explicit FP32 multiplier used by the NVFP4 reference.
    """

    format: Format
    rows: int
    columns: int
    padded_columns: int
    payload: bytes
    scale_codes: tuple[int, ...]
    tensor_scale: float


def _f32(value: float) -> float:
    """Round a value to IEEE binary32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _e2m1_decode(code: int) -> float:
    """Decode one sign-magnitude E2M1 code."""
    magnitude = _E2M1_MAGNITUDES[code & 0x7]
    return -magnitude if code & 0x8 else magnitude


def _e2m1_encode(value: float) -> int:
    """Round to nearest E2M1 value, ties to an even code."""
    sign = 0x8 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    candidates = range(8)
    code = min(
        candidates,
        key=lambda item: (abs(magnitude - _E2M1_MAGNITUDES[item]), item & 1),
    )
    return sign | code


def _ue4m3_decode(code: int) -> float:
    """Decode a finite unsigned E4M3 scale code."""
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if code == 0x7F:
        raise ValueError("UE4M3 NaN is not a scale")
    if exponent == 0:
        return math.ldexp(float(mantissa), -9)
    return math.ldexp(1.0 + mantissa / 8.0, exponent - 7)


def _ue4m3_encode(value: float) -> int:
    """Round a nonnegative value to the nearest finite UE4M3 code."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("UE4M3 scale must be finite and nonnegative")
    codes = range(0x7F)
    return min(codes, key=lambda code: (abs(value - _ue4m3_decode(code)), code & 1))


def _ue8m0_decode(code: int) -> float:
    """Decode an unsigned E8M0 scale code."""
    if code == 0xFF:
        raise ValueError("UE8M0 NaN is not a scale")
    if code == 0:
        return math.ldexp(1.0, -127)
    return math.ldexp(1.0, code - 127)


def _ue8m0_ceil(value: float) -> int:
    """Encode a nonnegative scale by rounding up to a power of two.

    Values outside the finite UE8M0 range saturate to its nearest endpoint.
    """
    if not math.isfinite(value) or value < 0:
        raise ValueError("UE8M0 scale must be finite and nonnegative")
    if value == 0:
        return 0
    fraction, next_exponent = math.frexp(value)
    exponent = next_exponent - 1 if fraction == 0.5 else next_exponent
    return min(254, max(0, exponent + 127))


def quantize_pack(
    matrix: Sequence[Sequence[float]],
    format: Format,
    *,
    tensor_scale: float = 1.0,
) -> PackedFp4:
    """Quantize, scale, and pack a row-major matrix into logical FP4 blocks.

    NVFP4 reconstructs as ``E2M1 * UE4M3_block_scale * tensor_scale``.
    Nonzero blocks clamp an underflowed UE4M3 scale to its minimum positive code.
    MXFP4 reconstructs as ``E2M1 * UE8M0_block_scale``. Zero padding is added
    only after the logical row extent, and the final partial block is scaled
    using its logical values.

    Args:
        matrix: Rectangular matrix in row-major logical order.
        format: ``nvfp4`` or ``mxfp4``.
        tensor_scale: Positive binary32 multiplier, used for NVFP4.

    Returns:
        Packed codes, logical scale codes, shape, and the explicit tensor scale.

    Raises:
        ValueError: If the format, dimensions, tensor scale, or values are invalid.
    """
    if format not in _BLOCK_SIZE:
        raise ValueError(f"unsupported FP4 format: {format}")
    rows = len(matrix)
    if rows == 0 or not matrix[0]:
        raise ValueError("matrix must have at least one row and column")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix rows must have equal length")
    if not math.isfinite(tensor_scale) or tensor_scale <= 0:
        raise ValueError("tensor_scale must be finite and positive")
    tensor_scale = _f32(tensor_scale)
    if not math.isfinite(tensor_scale) or tensor_scale <= 0:
        raise ValueError("tensor_scale must remain positive and finite in binary32")

    block_size = _BLOCK_SIZE[format]
    padded_columns = math.ceil(columns / block_size) * block_size
    codes: list[int] = []
    scales: list[int] = []
    for row in matrix:
        if any(not math.isfinite(value) for value in row):
            raise ValueError("matrix values must be finite")
        for start in range(0, padded_columns, block_size):
            values = list(row[start : start + block_size])
            peak = max((abs(value) for value in values), default=0.0)
            if format == "nvfp4":
                desired_scale = peak / (6.0 * tensor_scale)
                scale_code = _ue4m3_encode(desired_scale)
                if peak > 0.0:
                    scale_code = max(1, scale_code)
                block_scale = _ue4m3_decode(scale_code)
            else:
                desired_scale = peak / 6.0
                scale_code = _ue8m0_ceil(desired_scale)
                block_scale = _ue8m0_decode(scale_code)
            scales.append(scale_code)
            effective_scale = (
                block_scale * tensor_scale if format == "nvfp4" else block_scale
            )
            for offset in range(block_size):
                value = values[offset] if offset < len(values) else 0.0
                q = 0.0 if peak == 0.0 else value / effective_scale
                codes.append(_e2m1_encode(q))

    packed = bytearray((len(codes) + 1) // 2)
    for index, code in enumerate(codes):
        if index % 2 == 0:
            packed[index // 2] = code
        else:
            packed[index // 2] |= code << 4
    return PackedFp4(
        format=format,
        rows=rows,
        columns=columns,
        padded_columns=padded_columns,
        payload=bytes(packed),
        scale_codes=tuple(scales),
        tensor_scale=tensor_scale,
    )


def reconstruct(packed: PackedFp4) -> list[list[float]]:
    """Unpack and reconstruct only the logical matrix extent."""
    block_size = _BLOCK_SIZE[packed.format]
    expected_codes = packed.rows * packed.padded_columns
    if len(packed.payload) * 2 != expected_codes or expected_codes % 2:
        raise ValueError("packed payload length does not match padded geometry")
    expected_scales = packed.rows * (packed.padded_columns // block_size)
    if len(packed.scale_codes) != expected_scales:
        raise ValueError("scale count does not match row/block geometry")

    result: list[list[float]] = []
    codes_per_row = packed.padded_columns
    blocks_per_row = packed.padded_columns // block_size
    for row in range(packed.rows):
        values: list[float] = []
        for column in range(packed.columns):
            flat_index = row * codes_per_row + column
            byte = packed.payload[flat_index // 2]
            code = (byte >> 4) & 0xF if flat_index % 2 else byte & 0xF
            scale_index = row * blocks_per_row + column // block_size
            scale_code = packed.scale_codes[scale_index]
            if packed.format == "nvfp4":
                scale = _ue4m3_decode(scale_code) * packed.tensor_scale
            else:
                scale = _ue8m0_decode(scale_code)
            values.append(_f32(_e2m1_decode(code) * scale))
        result.append(values)
    return result


def contract_fp32(
    a: Sequence[Sequence[float]], b_rows: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Compute ``A[M,K] @ B[N,K].T`` with FP32 product and sum rounding."""
    if not a or not b_rows or not a[0] or not b_rows[0]:
        raise ValueError("contraction operands must be nonempty")
    k = len(a[0])
    if any(len(row) != k for row in a) or any(len(row) != k for row in b_rows):
        raise ValueError("contraction K dimensions must match")
    out: list[list[float]] = []
    for a_row in a:
        row_out: list[float] = []
        for b_row in b_rows:
            acc = 0.0
            for lhs, rhs in zip(a_row, b_row, strict=True):
                acc = _f32(acc + _f32(_f32(lhs) * _f32(rhs)))
            row_out.append(acc)
        out.append(row_out)
    return out
