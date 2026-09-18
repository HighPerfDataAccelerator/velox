# FINRA host-spill pipeline follow-up (2026-09-18)

## Stack and review boundary

This is a Draft follow-up to [Velox #39](https://github.com/HighPerfDataAccelerator/velox/pull/39),
pinned at `53b835a1b3d5bd467d99d36055bab1560e00d68f`.
The comparison base is an exact-SHA dependency branch, not a new production
integration branch. Merge the parent first, then rebase/retarget this PR to
`dev`; do not merge only into the dependency branch.

The paired Gluten change is a narrow follow-up to
[NVIDIA/spark-gluten #98](https://github.com/NVIDIA/spark-gluten/pull/98)
at `38e3ca91ccf492c5568978280e3e3056d37846e7`.
That parent already contains ORC *write* support. The follow-up admits only
proven whole-file ORC *scan* splits to the GPU connector.

Existing PRs #117 and NVIDIA/spark-gluten #176 are separate histories and
were not overwritten. Unrelated local OrderBy changes are excluded.

## Code groups

1. GPU ORC reader: whole-file checks on both planning and runtime sides,
   chunked decoding, filter evaluation before projection, and scan tests.
   BufferedInput uses a bounded reusable pinned transfer pool (4 x 32 MiB)
   with event-fenced reuse, not a claim that all host storage becomes pinned.
2. Grace join: bounded pinned transfers and restore waves, asynchronous host
   preparation/demotion with explicit ownership, incremental probe output,
   blocking-future integration, and corresponding lifecycle/shape tests.
   A scheduler-visible wait is not by itself evidence of recovered overlap.
3. Writer: bounded eager file submission and independently progressing
   consumer, ready-input coalescing, error/drain ownership, metrics and tests.
   Coalescing never waits for a batch to fill. Extra concatenation workspace
   requires non-blocking admission; failure falls back to individual writes.
4. UCX: sender/receiver ownership and metadata-buffer fixes, bounded local
   transfer leases, refreshable statistics, and budgeted pageable backing reuse.
   Reuse preserves CPU memcpy but amortizes allocation and first-touch cost.

Rejected arena/prefault/early pinned-preparation/parallel build-restore-copy
experiments are not included. The bounded sink does not enable experimental
out-of-order retirement. Diagnostic and opt-in controls retained in the code
are not claims that every setting improves performance.

Important opt-ins (executor environment):

- `GLUTEN_CUDF_GRACE_STREAM_PROBE_OUTPUT=1`
- `GLUTEN_CUDF_ASYNC_TABLE_WRITE=1`
- `GLUTEN_CUDF_ASYNC_TABLE_WRITE_COALESCE=1`
- `GLUTEN_UCX_PAGEABLE_CACHE=1`

Stream probe output, writer coalescing and pageable caching remain opt-in.
They require the rest of the qualified runner configuration, not just these
four settings.

## Budgets and ownership to review

- Pageable cache: best-fit whole blocks, at most 512 MiB / 64 idle entries.
  Active plus idle backing retains the **full original** sender-budget token;
  the idle cache is not additional host capacity. Pressure evicts idle blocks.
- Cache memory cannot be reused until the final asynchronous/local H2D/remote
  replay owner releases it. Free backing before releasing its budget token.
- The existing single-oversized-packet progress exception remains. A 1-MiB
  pressure test observed a 2,000,064-byte sender peak; this is not a strict-cap
  result. Formal FINRA sender peaks stayed within the unchanged 2-GiB limit.
- Writer's 1-GiB / 64-item queue includes active plus queued inputs, not only
  waiting entries. Input owners and original workspace credits survive until
  GPU use completes. Upstream can still hold its capacity-blocked next input.
- Grace build/restore/device admission and pinned-pool bounds are not replaced
  by an unbounded host cache. Join data cannot simply be discarded/recomputed.
- Failure, cancellation, shutdown and cross-stream ownership still need full
  exact-head GPU qualification; CPU helper tests do not establish CUDA safety.

## Historical performance evidence, NOT exact-PR-head qualification

These are predecessor-runtime results retained locally. This PR curates and
formats source, removes disabled experiments, and adds in-repo tests. It has
not yet produced a clean linked runtime or a new end-to-end measurement.

Scope: local FINRA **160M-row primary join/write**, two GPUs, 19 output columns,
62,315,629,480 input bytes, original ORC/Parquet layout, Snappy. Host spill
budget 32 GiB/executor; build threshold 8 GiB; resident build 2 GiB; restore
4 GiB; depth 4; UCX local queue cap 1; 16 demote workers. This is **not**
full 640M FINRA, AWS/EFA, or a new comparison against cuDF-Spark.

Qualified native binary SHA256:
`d50e535238075d6e29223100e05d246409707bbb284fee2e539906d1a17dcaa5`.

Same-binary cache-only A/B; stream probe output and writer coalescing enabled
in both modes:

| Shot | Body (s) | NVML GPU duty (%) | True DCGM SM Active (%) |
| --- | ---: | ---: | ---: |
| OFF control-r01 | 11.446119443 | 44.429 | 19.116 |
| ON r02 | 10.795132487 | 46.583 | 20.845 |
| OFF control-r02 | 11.038273613 | 43.914 | 20.358 |
| ON r03 | 10.673542531 | 44.170 | 20.720 |
| OFF mean | 11.242196528 | 44.1715 | 19.7370 |
| ON mean | 10.734337509 | 45.3765 | 20.7825 |

Mean elapsed reduction 4.5174%, duty +1.205 percentage points, SM +1.0455
points. Small repeated A/B, not statistical proof of a stable effect size.
Native-plan gates, whole-body telemetry and full-scan probabilistic output
fingerprints passed. The failed ON-r01 disk-preflight receipt remains failed;
no benchmark body ran and it is excluded, not silently relabeled.

The earlier stream-output + writer-coalescing two-feature A/B reduced mean
11.427689498 -> 10.888268343 s (4.7203%); it does not isolate either feature
and its gain must not be added to the cache gain.

The **80% duty / 40% true SM goal remains unachieved**.

## CPU/GPU attribution and remaining gaps

Separate same-shot perf/Nsys diagnostic, body 11.436722707 s, not in formal
timing means. Mean per-GPU disjoint body intervals:

| Observed state | Seconds | Body share |
| --- | ---: | ---: |
| Kernel only | 2.358612108 | 20.6231% |
| Copy/memset only | 2.016037467 | 17.6278% |
| Kernel plus copy | 0.499770880 | 4.3699% |
| No observed activity | 6.562302253 | 57.3792% |

Remaining scope/idle intersections on workers 0 / 1: Grace host materialization
and initial pinning 1.706 / 1.932 s; restore preparation/wait 1.422 / 1.472 s;
UCX pageable memcpy 0.192 / 0.274 s. **These overlap and are not additive
wall-clock attribution or guaranteed recoverable time.**

Perf corroborates first-touch stacks during Grace demotion (3,245 / 3,353
fault-stack samples out of 3,654 / 3,759 samples) and memcpy in build-host
staging. Samples are not fault counts or exclusive elapsed time. Off-CPU
coverage is unavailable; perf's interrupted/partial status and Nsys warnings
are retained. Thus "no observed activity", not certified capture completeness.
No new NCU qualification: low whole-body SM neither proves nor disproves
active-kernel efficiency.

Next optimization remains the dependency/ownership schedule during Grace
restore and partition drain, not larger memory allowances or moving startup
cost outside the measurement window.

## Tests and reproduction

Host-only regression suite, no CUDA runtime required:

```sh
cmake -S velox/experimental/cudf/tests/host -B /tmp/finra-host-tests -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/finra-host-tests -j 4
ctest --test-dir /tmp/finra-host-tests --output-on-failure --repeat until-fail:5
```

The two tests cover cache budget/last-owner/concurrent reuse and writer
progress/backpressure/error/drain. Both passed five repeats on the curated
source. Both also passed a host GCC ASan/UBSan invocation after formatting.
The Docker build image lacks libasan, so its sanitizer configure attempt
failed before testing; sanitizer validation used the host compiler instead.

Syntax-only compilation passed for CudfHashJoin, CudfHiveDataSink,
CudfSplitReader, UcxExchangeServer and UcxExchangeProtocol, using this source
with existing read-only generated build headers/dependencies. The paired
Gluten plan converter and new admission test source also passed syntax checks.
These are not a full rebuild, link, GPU execution or performance test.

Predecessor runtime additionally retained 276 UCX GPU test executions across
cache ON/OFF, one/four slots and pressure modes, with actual cache hits;
60 writer GPU executions included actual coalescing and nested/null Snappy
roundtrip. These results **do not certify this PR head**.

Before promotion: clean build/link both repos, run join/ORC/UCX/writer GPU
tests including cancellation/failure, rerun full-content correctness and
same-binary repeated A/B with unchanged budgets and measured-window coverage.

Local provenance (not publicly downloadable artifacts):

- `industry-workloads/analysis/finra-ucx-pageable-cache-20260918.md`
- `industry-workloads/analysis/finra-writer-coalesce-20260918.md`
- `/disk2/ferdinandx/finra-capacity-20260917/ucx-pageable-cache-build-r01`
- `flux-primary-160m-ucx-pageable-cache-cgroup-nsys-r01` retained diagnostic

Raw traces and full runner/configuration bundle are not included in this
source PR; an independently reproducible public benchmark requires publishing
that bundle separately. No current AWS benefit or exact baseline speedup is
claimed here.
