# OPT-128 — Remove measured host submission and synchronization stalls

Status: **no_material_opportunity**. Parent is OPT-127 kept `decode_segments8` (`opt127_kept_decode_segments8`). Production graph selector is unchanged. Host-stall pins stay false unless this sitting kept a candidate.

`claims_throughput: False`.

## Sitting identity

- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- parent: `opt127_kept_decode_segments8` / `decode_segments8`
- measured_at: `2026-09-14T00:13:51Z`

## Before/after critical-path attribution

Complete Session sample/eval/output loop on decode_segments8. Server eval always passes a cancellation poller; bench does not. Overlapping GPU waits are listed separately from host work that delays the next submission. A shifted wait is not a removed wait.

| Arm | wall ms/token | delaying host ms/token | overlapping wait ms/token | poll DeviceSynchronize count |
|---|---:|---:|---:|---:|
| parent bench (no poll) | 16.31103325 | 0.037732496875 | 16.2069893125 | 0 |
| parent server poll | 16.426921875 | 0.06265054625 | 16.2953663125 | 128 |
| poll host-only | 16.363834375 | 0.0391575619375 | 16.2608165625 | 0 |
| defer elapsed sync | 16.3600654375 | 0.040705941625 | 0.047295685875 | 0 |

Poll wall save `0.06308750000000174` ms/token. Defer wall save `-0.049032187499999935` ms/token. Freeze threshold `0.2` ms/token.

## Ranked host stalls

| Rank | Name | kind | delaying ms/token | overlapping ms/token |
|---:|---|---|---:|---:|
| 1 | non_greedy_host_alloc (off greedy production path; not frozen) | hot_path_allocation | 0.894850016 | 0.0 |
| 2 | poll_without_device_sync | stream_device_synchronization | 0.06308750000000174 | 15.6179275625 |
| 3 | finish_output_commit_greedy_d2h | blocking_copy | 0.008891312375 | 0.0 |
| 4 | hot_path_event_allocation | hot_path_allocation | 0.0026066250875 | 0.0 |
| 5 | defer_elapsed_event_sync | stream_device_synchronization | 0.0 | 16.1674213125 |

### non_greedy_host_alloc

Session::sample temperature>0 allocates 248320 candidates.

### poll_without_device_sync

execute_token poll after each decode_segments8 launch: cudaDeviceSynchronize then host atomic. Server Session::eval always passes cancelled, so this is the production sample/eval path.

### finish_output_commit_greedy_d2h

finish_output_commit cudaStreamSynchronize(copy) + blocking 4-byte cudaMemcpy of greedy index. OPT-118 lazy path.

### hot_path_event_allocation

cudaEventCreate/Destroy start/stop every execute_token.

### defer_elapsed_event_sync

execute_token cudaEventCreate/Record/Synchronize(stop) before commit_outputs+scatter. The synchronize overlaps GPU compute and delays copy-stream argmax / KV scatter submission.

## Lifetime / logits / cancellation

lifetime=`True` cancel=`True` logits_consumer=`True`. Lazy greedy D2H remains 4 bytes. Full-logit `copy_last_outputs` still materializes the vocab. Buffer/event lifetimes cover cancel, restore, and a subsequent request.

## Frozen candidates (at most two)

Admitted: `[]`. Frozen: `none`. A shifted wait is not a removed wait. Overlapping EventSynchronize and DeviceSynchronize that cover useful GPU work are not attributed as removable. Candidates freeze only when the sample/eval wall save clears 0.2000 ms/token (max of 0.2 ms and 1% of parent server-poll wall).

**Verdict: `no_material_opportunity`.** poll pin `False`; defer pin `False`. D128 tok/s delta `0.0`; speedup `0.0`; baseline_unchanged `True`.

