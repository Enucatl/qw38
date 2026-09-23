# TASK-016 prerequisite repair evidence

This records the review repairs made before the one-layer checkpoint. TASK-016
remains `TODO`; its own composition and continuation criteria have not been
accepted by these component checks.

| Review ID | Production boundary | Discriminating check |
| --- | --- | --- |
| AR-02 | `verify_compiled_artifact` calls metadata policy/schema comparison before opening source tensor payloads | `compiler_integration`: identity/production policy mismatch, missing and swapped graph bindings, extra tensor, format and same-count shape change, reordered directory with remapped tensor IDs |
| AR-03 | Session execution health poisons after mutation/failure; reset or restore recovers | `runtime_session_integration`, `gdn_integration`, `attention_integration`: reset/restore synchronization failures, GDN failure after mutation, attention core failure after preparation, snapshot/execute rejection, exact recovered state versus clean control |
| AR-04 | Plan metadata lives at a stable address across Session movement | `gdn_integration`, `attention_integration`: execute already-bound plans after move construction and move assignment |
| AR-05 | Session binders require the owning stream | `gdn_integration`, `attention_integration`: second same-device stream rejected by GDN front/recurrence/mixer, MLP, and attention preparation/mixer |
| AR-06 | Checked view and transform geometry | `runtime_plan`, `runtime_session_integration`, `compiler_transform`: invalid counts/indices, overflow, zero dimensions, short/long payloads, valid round trips |
| AR-07 | BF16 verifier compares identity and mapped spans with bounded scratch | `compiler_bf16_verification`: one-byte corruption in row-major, dense-tile and tap-major layouts; isolated 8 MiB and 32 MiB input checks |
| AR-08 | Closed Runtime rejects fallible operations | `runtime_session_integration`: repeated shutdown and rejected upload/session creation |
| AR-09 | Attention output reads the original residual and writes the other buffer | `attention_integration` and `attention_unit`: Q4 and BF16-control input residuals preserved exactly, output agrees with reference; isolated trace below |
| AR-10 | GDN and MLP excluded benchmark consumers match current APIs | Both benchmark targets build and smoke; Nsight Systems GDN trace below |
| AR-12 | Session binders resolve canonical V0 names to IDs and check graph role, node, and layer before execution | `runtime_plan`: same-shaped Q/K names, distinct layers, reordered directories, missing/duplicate/wrong-layer associations; uploaded-model GDN/attention/MLP integration fixtures |

The BF16 memory test measures **additional verifier-owned peak RSS after both
input spans are allocated and touched**, not total RSS including inputs or
mapped artifact pages. In isolated processes, both identity and dense-tile
verification increased peak RSS by less than 0.5 MiB for each of 8 MiB and
32 MiB inputs. The test gate permits up to 4 MiB to accommodate allocator
variation. No authority checkpoint was read for this measurement.

For AR-10, the RTX 5090 (driver 590.48.01, 32607 MiB) trace used CUDA 13.4.1
in development image
`sha256:be0903b40ab2e14ec1b5ba285655219cc9e521cd1455b070a6dfdb68ed4e4bfb`.
`nsys profile --trace cuda` recorded the GDN mixer benchmark binary with SHA-256
`7f156f8424c8740eca273467347eb9c3c37d01bc883b8b5b040199f49be01009`.
The trace contains eight kernel types, each with 18 instances: 2 warmup,
8 untimed, and 8 timed executions, or exactly **eight kernels per execution**.
Its GPU memory-operation report contains 19 setup memsets and no memcpy node.
The local trace is `build/release/gdn-ar10.nsys-rep` (SHA-256
`b209b5254333c396e6b17a9394b3da20b9d6fd4530c192b0aa57607569424736`);
the summary is `build/release/gdn-ar10.stats`.

For AR-09, the isolated Release command
`build/release/tests/qw38_attention_unit_test --bf16-mixer-only` was profiled
with `nsys profile --trace cuda`. Its binary SHA-256 is
`a880ddb526f09b53dbdb100f9daea16e90fc555846b13d8e1e07939a07ed9b75`.
The trace shows exactly six kernels, one each for RMS, ranged projections,
preparation, scan, merge, and output projection. Its memory report has nine
fixture H2D copies, two output-check D2H copies, two setup memsets, and **no
D2D copy**. The local trace is `build/release/attention-bf16-ar09.nsys-rep`
(SHA-256 `133a1e2b9fe7939bebbd93008d6f8ebf0c04520ddeb632177489744556ea40d7`);
the summary is `build/release/attention-bf16-ar09.stats`.

The small metadata and component fixtures do not replace TASK-016's
one-GDN-layer plus one-attention-layer composition test. They also do not
replace the separate full authoritative-checkpoint reconstruction gate when
an artifact is promoted or compiler semantics change.
