# Immediate full-model decode profile — 2026-09-23

`qw38-decode --profile` reports separate wall times for model load/upload,
session creation and plan binding, slow repeated-decode prompt ingestion, and
decode after the prompt has populated session state. `--tokens-file` accepts a
nonempty little-endian U32 token file. For this CLI, `--generate 9` emits nine
tokens and executes eight populated decode steps; the first emitted token comes
from the final prompt step.

The runtime emits NVTX ranges for those phases, each decode step, the language
layers, GDN/attention mixers, MLP, embedding, and vocabulary readout. Use
`scripts/analyze_decode_profile.py` on an Nsight Systems SQLite export to
measure CUDA stream synchronization and kernel launch API duration, then group
GPU kernels into projection, GDN recurrence, attention, vocabulary readout,
and other work. Vocabulary kernels are identified by their launch inside the
readout NVTX range. Projection includes all other `decode_mmv*` kernels;
attention includes prepare, scan, and merge; recurrence is the
`gdn_recurrence_kernel`. CPU API and GPU kernel durations overlap and must not
be added.

## Reproduce

```bash
docker run --rm --gpus all -v "$PWD:/repo" -w /repo qw38-dev:cuda13.4.1-pinned \
  cmake --build .cache/task018-build --target qw38_decode -j 8
docker run --rm --gpus all -v "$PWD:/repo" -w /repo qw38-dev:cuda13.4.1-pinned \
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/repo/.cache/decode-profile-512-current \
  .cache/task018-build/src/qw38-decode \
  --artifact build/pinned-debug/qwen-v0.qw38 \
  --tokens-file .cache/evaluation/qw38-language-v1/tokens/R-512-s0-d0.1.prompt.u32le \
  --generate 9 --profile
docker run --rm -v "$PWD:/repo" -w /repo qw38-dev:cuda13.4.1-pinned \
  nsys export --type sqlite --force-overwrite true \
  --output /repo/.cache/decode-profile-512-current.sqlite \
  /repo/.cache/decode-profile-512-current.nsys-rep
uv run --script scripts/analyze_decode_profile.py .cache/decode-profile-512-current.sqlite
```

## Initial measured case

RTX 5090, driver 590.48.01, pinned CUDA 13.4.1 container image
`sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
Source HEAD was `741681b` with local profiling edits. Executable SHA-256:
`9054479cd0861c035c21fc38beead939c563c1ba7315c4f6d4dca1aa3b534cad`.
Artifact SHA-256:
`044020b5d020307dc6e5e159f75cfaeacf8e8daed641a57290d440b3bf123fd9`.
The selected R-512 prompt file contains 507 tokens. The eight measured steps
start at populated length 507 and advance through length 514. The complete
trace is `.cache/decode-profile-512-full.nsys-rep`; its SQLite export is
`.cache/decode-profile-512-full.sqlite`.

| Phase | Wall time |
| --- | ---: |
| Model load and upload | 3,800.07 ms |
| Session creation and plan binding | 2.56 ms |
| Prompt ingestion, 507 tokens | 16,487.12 ms |
| Populated decode, eight steps | 265.38 ms, or 33.17 ms/step |

| Decode component | Count per step | Time per step |
| --- | ---: | ---: |
| Host `cudaStreamSynchronize` | 65 | 28.28 ms API duration |
| Host `cudaLaunchKernel` | 675 | 1.82 ms API duration |
| Host `cudaMemcpyAsync` | 1 | 2.08 ms API duration |
| GPU projection kernels | 304 | 25.50 ms |
| GPU GDN recurrence kernels | 48 | 0.165 ms |
| GPU attention kernels | 48 | 3.29 ms |
| GPU vocabulary readout kernels | 2 | 1.52 ms |
| GPU other kernels | 273 | 1.64 ms |

The readout NVTX range, including host copy, synchronization, and argmax,
took 2.20 ms per step. Stream synchronization API duration largely waits for
GPU work already counted in the kernel rows. The prompt uses the current slow
repeated-decode path; this profile is a development diagnostic, not the later
production prefill or performance acceptance baseline.

## Q4 MLP projection change and re-measurement

Nsight Compute sampled the first real-weight MLP gate/up and down kernels.
Initially both reserved 35.84 KB of shared memory per block because the
decode kernel staged the maximum 17,408-element input for every shape. Both
achieved about 31% occupancy with no local-memory spills. The gate/up input
is only 5,120 elements, so the launch now allocates shared memory for its
validated padded K: 10.24 KB of dynamic shared memory. Its measured achieved
occupancy rose to 83%. The full-width Q4 down kernel reads its BF16 input
directly from device memory instead of reserving a 34 KB shared buffer;
achieved occupancy rose to 55%. Packed weights, reductions, and epilogues
did not change. The final Nsight Compute report is
`.cache/q4-mlp-direct-down.ncu-repz`. The measured memory throughput rose
from 500 to 615 GB/s for paired gate/up and from 363 to 582 GB/s for down.

```bash
docker run --rm --gpus all -v "$PWD:/repo" -w /repo qw38-dev:cuda13.4.1-pinned \
  ncu --nvtx --nvtx-include 'mlp]' \
  --kernel-name regex:decode_mmv_kernel --launch-count 2 \
  --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section WarpStateStats --force-overwrite \
  --export /repo/.cache/q4-mlp-direct-down \
  .cache/task018-build/src/qw38-decode \
  --artifact build/pinned-debug/qwen-v0.qw38 \
  --tokens 248045,846 --generate 3 --profile
```

The same artifact, 507-token prompt, and eight populated decode steps were
captured again in `.cache/decode-profile-512-direct-down.nsys-rep` and
`.cache/decode-profile-512-direct-down.sqlite`. The final executable SHA-256
was `67fb62a0aef66ba23c8b558498bc5c0665afc5fcead5e632dcf342a02b3e157e`.

| Measurement per decode step | Initial | Final | Change |
| --- | ---: | ---: | ---: |
| Full decode wall time, traced | 33.17 ms | 28.61 ms | −13.7% |
| Q4 MLP paired gate/up GPU time | 10.41 ms | 7.69 ms | −26.1% |
| Q4 MLP down GPU time | 6.47 ms | 4.91 ms | −24.2% |
| All projection GPU time | 25.50 ms | 21.03 ms | −17.5% |
| Prompt ingestion wall time | 16.49 s | 14.17 s | −14.1% |

A second final-build run without Nsight tracing measured 27.17 ms per
populated decode step. The prompt argmax and all nine emitted token IDs matched
the initial run. These timings cover a populated length of 507–514; longer
attention contexts and the future production prefill schedule need separate
measurement.
