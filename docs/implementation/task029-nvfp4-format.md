# TASK-029 native gate/up recipe (frozen before development scoring)

Policy `precision_nvfp4_mlp_v1` (0x0404) changes only MLP gate/up to storage
`nvfp4` (0x0005), quantizer `nvfp4_v1` (0x0106), layout `cuda_nvfp4_v1`
(0x020E), mapping `NvFp4TN` (0x0704). Down retains CandidateV2 Q4_K;
attention/GDN/head and BF16 tensors retain the accepted policy. Source
transformations, tokenizer/model identities and manifest-only digests are unchanged.

For each BF16 weight tensor choose the FP32 power-of-two factor
`2^ceil(log2(maxabs / (6*448)))`, clamped below to the smallest normal FP32;
an all-zero tensor uses factor 1. For each contiguous 16-value K block divide
weights by the tensor factor, encode `maxabs/6` with pinned CUTLASS nonnegative
E4M3 RNE/saturation, and raise a nonzero block's zero scale code to 1 (2^-9).
Encode each normalized value divided by that **stored** scale as E2M1 using
CUTLASS RNE with ties to even and saturation at ±6. E2M1 positive codes are
`0,.5,1,1.5,2,3,4,6`; bit 3 is the sign. Exact zero is canonical +0. Nonfinite
weights are rejected. Reconstruction is `factor * scale * code` in FP32.
No BF16 weight reconstruction round is introduced in the native consumers.

Codes are `[padded_N,padded_K]` row-major, lower nibble first; N rounds to 8,
K to 256, with zero padding. The separate 256-aligned scale span begins with
one little-endian FP32 tensor factor and 252 zero reserved bytes. Then follow
`ceil(padded_N/128)*128*padded_K/16` E4M3 bytes in CUTLASS's SM120 K-major
scale swizzle. For row r and block b (K coordinate /16), byte index is
`(r/128)*128*(padded_K/16) + (b/4)*512 + (r%32)*16 + ((r%128)/32)*4 + b%4`.
Unused scales are zero; valid scale codes are 0..126. One resident code/scale
view serves both phases, with no fallback weights resident alongside it.

GPU activation factor is fixed at 1. Each row independently applies the same
block recipe to BF16-rounded RMS output. Finite outliers saturate at scale
448 and code ±6; a nonfinite block uses scale NaN (127), propagating a nonfinite
contraction instead of silently replacing invalid input with zero. Packing
never depends on other rows. M=1 consumes BF16 directly through same-view
GEMV; M>=2 packs once and shares the codes/scales across gate and up.
The fused producer uses the existing robust RMS reduction and BF16 rounding.
FP32 gate/up output tiles feed one FP32 SwiGLU epilogue and BF16 down input.

Activation codes/scales reuse the existing bounded unpack workspace. FP32
output tiles and library workspace also reuse existing allocations. The down
projection reclaims the unpack buffer after both native consumers finish on
the session stream. Persistent weights, KV/history and recurrent state retain
their existing lifetimes and precision.

CUTLASS revision: `098de2a652cf8f00fd70b2df54051c7eccbb855a` (TASK-019 pin).
Its native SM120 TN block-scaled operation is compiled at `sm_120a`, with
FP32 output/accumulation and a 128×128×128 tile. The CUDA boundary owns this
conditional dependency; no inference framework is added. To restore the shared
checkout before configuring the pinned container build:

```sh
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git .cache/dependencies/cutlass
git -C .cache/dependencies/cutlass checkout --detach 098de2a652cf8f00fd70b2df54051c7eccbb855a
```

`QW38_CUTLASS_DIR` may name another checkout of that exact revision. CMake
rejects a different revision. Final acceptance, timings and disposition belong
in TASK-029's Completion Report; this recipe is not a quality claim.
