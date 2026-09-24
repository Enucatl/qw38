"""Focused CPU checks for the TASK-019 logical FP4 reference."""

from __future__ import annotations

import math
import struct
import unittest
from dataclasses import replace

from scripts.task019_fp4_reference import (
    contract_bf16,
    contract_fp32,
    quantize_pack,
    reconstruct,
    round_bf16,
)


def _bf16(value: float) -> float:
    """Round an FP32 value to BF16 with round-to-nearest, ties-to-even."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def _packed_code(payload: bytes, index: int) -> int:
    """Read one low-first packed nibble by flattened element index."""
    value = payload[index // 2]
    return (value >> 4) & 0xF if index % 2 else value & 0xF


class Task019Fp4ReferenceTest(unittest.TestCase):
    """Check the logical FP4 reference using only the Python standard library."""

    def test_known_scale_and_code_bytes_for_both_formats(self) -> None:
        """Known E2M1 and scale encodings stay independent of reconstruction."""
        cases = (
            ("nvfp4", [6.0, -3.0, 0.0], 0.5, 64, 16),
            ("mxfp4", [12.0, -6.0, 0.0], 1.0, 128, 32),
        )
        for format_name, row, tensor_scale, scale_code, block_size in cases:
            with self.subTest(format=format_name):
                packed = quantize_pack([row, [0.0] * 3], format_name, tensor_scale=tensor_scale)
                self.assertEqual(packed.padded_columns, block_size)
                self.assertEqual(packed.scale_codes, (scale_code, 0))
                self.assertEqual(packed.payload[0], 0xD7)  # +6, -3, low nibble first
                self.assertEqual(packed.payload[1:], bytes(len(packed.payload) - 1))
                self.assertEqual(reconstruct(packed), [row, [0.0] * 3])

    def test_reconstruct_rejects_malformed_geometry(self) -> None:
        """A corrupted descriptor cannot imply a shorter physical K extent."""
        packed = quantize_pack([[1.0]], "nvfp4")
        for invalid in (
            replace(packed, rows=0),
            replace(packed, columns=0),
            replace(packed, columns=17),
            replace(packed, padded_columns=32),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    reconstruct(invalid)

    def test_bf16_output_rounds_after_fp32_contraction(self) -> None:
        """The example's BF16 D store uses nearest-even output rounding."""
        self.assertEqual(round_bf16(1.0 + 1.0 / 256.0), 1.0)
        self.assertEqual(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 1.0 / 64.0)
        self.assertEqual(
            contract_bf16([[1.0, 0.0]], [[1.0 + 3.0 / 256.0, 9.0]]),
            [[1.0 + 1.0 / 64.0]],
        )

    def test_fp4_formats_keep_row_scale_orientation_and_zero_blocks(self) -> None:
        """Per-row scales stay attached to their own blocks; zero rows stay zero."""
        matrix = [
            [0.0] * 32 + [12.0, 0.0, 0.0],
            [0.0] * 32 + [0.0, 0.0, 0.0],
        ]
        packed = quantize_pack(matrix, "mxfp4")

        self.assertEqual(packed.rows, 2)
        self.assertEqual(packed.columns, 35)
        self.assertEqual(packed.padded_columns, 64)
        self.assertEqual(len(packed.scale_codes), 4)
        self.assertEqual(packed.scale_codes[0], 0)  # minimum E8M0 scale code
        self.assertGreater(packed.scale_codes[1], packed.scale_codes[0])
        self.assertEqual(packed.scale_codes[2:], (0, 0))
        restored = reconstruct(packed)
        self.assertEqual(len(restored), 2)
        self.assertTrue(all(value == 0.0 for value in restored[1]))
        self.assertEqual(
            len(restored[0]), 35
        )  # padded K stays outside logical view
        self.assertEqual(restored[0][-3], 12.0)
        self.assertTrue(
            all(_packed_code(packed.payload, index) == 0 for index in range(35, 64))
        )

    def test_nvfp4_tail_and_tensor_scale_reconstruction(self) -> None:
        """NVFP4 uses E2M1 × UE4M3 scale × explicit tensor scale."""
        matrix = [[0.0] * 16 + [6.0, -3.0, 0.0]]
        packed = quantize_pack(matrix, "nvfp4", tensor_scale=0.5)

        self.assertEqual(packed.padded_columns, 32)
        self.assertEqual(len(packed.scale_codes), 2)
        restored = reconstruct(packed)[0]
        self.assertEqual(restored[:16], [0.0] * 16)
        self.assertEqual(restored[16:], [6.0, -3.0, 0.0])
        self.assertTrue(
            all(_packed_code(packed.payload, index) == 0 for index in range(19, 32))
        )

    def test_nvfp4_nonzero_block_clamps_underflowed_scale(self) -> None:
        """A tiny nonzero block uses the smallest positive UE4M3 scale."""
        packed = quantize_pack([[0.001]], "nvfp4")

        self.assertEqual(packed.scale_codes, (1,))
        self.assertEqual(len(reconstruct(packed)[0]), 1)
        self.assertTrue(math.isfinite(reconstruct(packed)[0][0]))

    def test_mxfp4_scale_ceil_just_above_power_of_two(self) -> None:
        """An E8M0 scale never rounds below an in-range block maximum."""
        just_above = math.nextafter(1024.0, math.inf)
        packed = quantize_pack([[6.0 * just_above]], "mxfp4")

        self.assertEqual(packed.scale_codes, (138,))
        self.assertEqual(reconstruct(packed)[0], [6144.0])

    def test_reconstructed_contraction_uses_a_times_transposed_b_orientation(
        self,
    ) -> None:
        """The FP32 contraction returns an asymmetric M-by-N result."""
        a = [[1.0, -2.0, 3.0, -4.0, 2.0], [2.0, 1.0, -1.0, 3.0, -2.0]]
        b_rows = [
            [1.0, 0.0, -2.0, 1.0, 3.0],
            [-1.0, 2.0, 1.0, 0.0, -2.0],
            [3.0, -1.0, 0.0, 2.0, 1.0],
        ]
        for format_name in ("nvfp4", "mxfp4"):
            a_rec = reconstruct(quantize_pack(a, format_name, tensor_scale=0.25))
            b_rec = reconstruct(quantize_pack(b_rows, format_name, tensor_scale=0.5))
            output = contract_fp32(a_rec, b_rec)
            self.assertEqual(len(output), 2)
            self.assertTrue(all(len(row) == 3 for row in output))
            manual = [
                [
                    sum(lhs * rhs for lhs, rhs in zip(a_row, b_row, strict=True))
                    for b_row in b_rec
                ]
                for a_row in a_rec
            ]
            for output_row, manual_row in zip(output, manual, strict=True):
                for actual, expected in zip(output_row, manual_row, strict=True):
                    self.assertTrue(
                        math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
                    )

    def test_quantization_error_is_separate_from_reconstructed_arithmetic(self) -> None:
        """Quantization changes operands; arithmetic is checked after decoding."""
        bf16_a = [[_bf16(v) for v in [0.3, -1.2, 2.7, 0.8]]]
        bf16_b = [[_bf16(v) for v in [1.7, 0.2, -0.9, 2.1]]]
        bf16_baseline = sum(x * y for x, y in zip(bf16_a[0], bf16_b[0], strict=True))
        for format_name in ("nvfp4", "mxfp4"):
            with self.subTest(format=format_name):
                decoded_a = reconstruct(quantize_pack(bf16_a, format_name))
                decoded_b = reconstruct(quantize_pack(bf16_b, format_name))
                ref = contract_fp32(decoded_a, decoded_b)
                arithmetic_from_decoded = sum(
                    x * y for x, y in zip(decoded_a[0], decoded_b[0], strict=True)
                )
                self.assertTrue(math.isclose(ref[0][0], arithmetic_from_decoded, rel_tol=1e-6))
                self.assertNotEqual(ref[0][0], bf16_baseline)


if __name__ == "__main__":
    unittest.main()
