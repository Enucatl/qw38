# TASK-021 candidate artifact contract

The TASK-020 [policy](task020-policy.json) selects one resident packed view:
Q4G64 for primary-language MLP gate/up/down, Q8G32 for primary-language
attention/GDN qkv/z/out projections and `lm_head`, and BF16 for the remaining
primary-language weights. Inactive MTP matrices retain their V0 representation;
the MTP embedding and head are aliases of the primary-language owners. The
candidate uses BF16 projection operands and FP32 accumulation. Calibration
selected the family map; it does not fit or transform any weight scale.

## Identity and compatibility

| Meaning | ID | Wire value |
| --- | --- | ---: |
| Candidate Q4G64 logical quantizer | `Q4G64CandidateV1` | `0x0103` |
| Candidate Q8G32 logical quantizer | `Q8G32CandidateV1` | `0x0104` |
| Candidate Q4 physical layout | `CudaQ4G64CandidateV1` | `0x020B` |
| Candidate Q8 physical layout | `CudaQ8G32CandidateV1` | `0x020C` |
| Candidate precision/family policy | `CandidateV1` | `0x0402` |

Container and manifest versions remain 1: their record grammar is unchanged.
The candidate uses new IDs so an older reader rejects it as unsupported. A
reader supporting these IDs accepts V0 artifacts and rejects mixed V0/candidate
quantizer-layout pairs. V0 precision policy rejects candidate quantizer IDs.
Reader validation checks geometry, span lengths, 256-byte offsets, nonoverlap,
forbidden codes, finite normal scales, zero padded coordinates, graph aliases,
and the manifest SHA-256 before binding. It never hashes payload or scale spans.

The source config metadata SHA-256 is
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`;
the safetensors index metadata SHA-256 is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.
The TASK-020 policy JSON SHA-256 is
`2ef01bff2b40095f08aedad3b84c50317ffec66e940019342bd12abcda50110e`.
The train calibration manifest SHA-256 is
`8ac4a9cab7181c7f7008f95b52420f0d76775524ac64868d0351411c2c2021a4`.
The compiler revision identifier embeds the last two metadata hashes. The
calibration manifest identifies 128 train windows from Wikitext revision
`b08601e04326c79dfdd32d625aee71d232d685c3`; it is not read by the
packer because the selected scale recipe uses only the source weights.

## Reconstruction and physical order

Logical coordinates are `[N,K]`; a group is 64 consecutive K positions in one
row for Q4, or 32 for Q8. Let `a=max(abs(w))` over the group and `qmax=7` or
`127`. A zero group has all zero codes and positive-zero FP16 scale. Otherwise,
the stored FP16 scale `s` is the smallest finite normal FP16 number at least
`max(a/qmax, 2^-14)`. A nonrepresentable or nonfinite input is rejected. Each
code is `clamp(round_to_nearest_even(w/s), -qmax, qmax)`; `-8` and `-128` are
forbidden. The decoded FP32 value is `code * fp16_to_fp32(s)` and the weight
operand is then rounded once to BF16 for contraction. There are no zero points,
second-level scale factors, clipping parameters, activation scales, or other
side arrays. The one scale array stores one little-endian FP16 value per group.

The physical view pads N to 8 and K to 256. Tiles are ordered
`[N/8, K/256]`, followed by row within tile and K within row. Q4 puts the
earlier K code in the low nibble; Q8 stores one signed two's-complement byte
per coordinate. The separate scale span follows
`[N/8, K/256, row, group-within-256]`. Padded codes and scales are positive
zero; logical lengths exclude padding. Q4 rows use 128 payload bytes and four
FP16 scales per tile, Q8 rows use 256 payload bytes and eight FP16 scales.
Payload and scale spans each begin at a 256-byte aligned file offset. The
candidate's swizzle is this eight-row tiled order; no second view or runtime
whole-tensor repack is declared. Decode and planned GEMM consumers bind the
same payload and scale spans through the tensor's distinct quantizer and
layout IDs. Production kernel support and speed are TASK-022/023 work.

Compilation reads mapped BF16 shards one tensor at a time, quantizes in
bounded eight-row chunks, writes payload chunks directly to the final view,
and writes a bounded per-tensor scale array. The largest planned scale buffer
is the head's grouped scale span. Verification independently decodes tile
coordinates and recomputes source groups in bounded tile scratch, without
expanding a full BF16 model. Actual compile, verification, load and upload
memory and file-size measurements are recorded in the TASK-021 completion
report.
