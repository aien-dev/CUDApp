# Native benchmark receipts

Schema `cudapp.benchmark.v1`, contract `cudapp.numeric.v1`.
All collection is opt-in. These tools never load a model, send a model request,
start a server, switch a driver, or enable an optimization. Native collection
runs only when an approved workload calls the adapter.

## Build and use

```
make OUT_DIR=build-local native-benchmark-adapter test-native-benchmark
node tools/native-benchmark.mjs prepare manifest.json plan.json
node tools/native-benchmark.mjs import manifest.json capture.jsonl correctness.json run.json
node tools/native-benchmark.mjs report run.json
node tools/native-benchmark.mjs report run.json --text
node tools/native-benchmark.mjs compare baseline.json candidate.json
node tools/native-benchmark.mjs compare baseline.json candidate.json --text
```

`capture` is an alias of `prepare`: it writes a NOT_RUN collection plan and
command manifest without executing the workload. Import/prepare refuse to
replace an existing output. Node 18+ and its standard library are needed only
for host receipt tooling. The native adapter links against the existing
native C library and uses no third-party profiler or inference library.

A manifest supplies `schema`, `contract`, `kind` (`fixture` or `model`),
`source_sha` (40 lowercase hex digits), SHA256 strings `model_sha256`,
`prompt_sha256`, `config_sha256`, `binary_sha256`, `driver_sha256`,
`policy_sha256`, `precision` (`fp8-e4m3-bf16-f32`), `chip`, `backend`
(`native-graph` or `graph-cuda`), `policy`, `supported_context_limit`,
`workloads`, and `conditions`.
`policy_sha256` hashes UTF-8 `JSON.stringify(policy)` without a newline.
`conditions` contains `thermal_clock_protocol_sha256` and a plain description.
The thermal/clock protocol must include run order, clock/thermal observations,
quiet-window identity, matching criteria and explicit rejection criteria.
The tool compares protocol identities; it cannot verify a physical setup.

`workloads` is a nonempty sweep of `{batch,context,input_tokens,output_tokens,
seed}` objects. v1 native graph collection admits batch one; larger batch
paths require their own adapter and versioned admission rather than relabeling
single-token graph stats. Context is the final measured token's KV context,
including its input. Every generated output context needs a matching row in
the correctness trace. Workloads outside the declared supported context and
traced output coverage are refused. The tool does not infer new supported
shapes from fixture samples.

`correctness.json` contains `reference` and `candidate` traces in
[the numerical schema](NUMERICAL_CONTRACT.md). Both must match model, prompt,
config, precision, contract, seed and declared contexts; candidate source and
binary pins must match the manifest. Exact tokens, tie choices and routed IDs
are checked even for an experimental float tolerance. A comparison also checks
baseline and candidate model outputs against each other. Missing correctness
artifacts refuse import/comparison. Absence of real model artifacts is NOT_RUN;
fixture traces cannot establish model validity or quality.

## Native graph adapter

Link `libomega_native_benchmark.a` before `libomega_gpu.a`. In the existing
model workload, create an `OmegaBenchmarkSample` with a unique independent
session ID, seed, ordinal, input token count, cold/warm label and stamps setting.
Replace an opt-in call to `omega_gpu_decode_graph_token` with
`omega_benchmark_graph_token` using the already loaded graph, embedding,
position and output token pointer. It calls the real graph token API, reads
session allocation counters before/after, optionally retrieves the existing
graph timeline, and writes one JSONL event. It never builds a fictional graph.

For actual serving-request measurements, supply monotonic host-clock readings
for request start, prefill complete, first token and request end to
`omega_benchmark_write_request`. This records TTFT, prefill, decode and total
request latency with raw readings. Decode spans prefill completion through
request end, including the first decode token. Graph single-token events have
TTFT/prefill unknown. Do not combine request and per-token observations into
one distribution. Choose independent request/session samples for comparison.
A cold token label does not claim that setup/model ingestion was timed:
`setup_ns` is unknown unless separately observed.

The pure `omega_benchmark_write_graph` formatter accepts actual stats for
integrations that already call the graph API. It refuses incomplete launch
semaphores, reversed/poisoned timestamps and bad sample metadata. A separate
`omega_benchmark_write_cmdlist` diagnostic bridge consumes the existing
`OmegaGpuCmdListStats` without resubmission, preserving method-word count,
host method construction and completion observations. Its diagnostic events
are separate from whole-model samples; they do not become token latency.

The host tests call the production formatter with labeled fixtures and the
production collection wrapper with fixture device callbacks. One test imports
the compiled C formatter's JSONL through the host CLI. Real graph/workload
execution remains NOT_RUN in this branch's receipt.

## Measurement meanings

Each run stores raw samples and correctness traces. Each sample records
`session`, `ordinal`, `phase` (`cold` or `warm`), `stamps`, `workload`,
`metrics`, `timestamps`, `kinds`, and `correctness_sha256` (the SHA256 of
`encodeJSON(candidate_trace)` (the compact serializer exported by the tool, preserving negative zero)). Request events also store `request_clock`.
Timestamps are decimal uint64 strings, so JSON does not lose their precision.
Differences and counters must be exactly representable safe integers.

Every metric has `category` (`measured`, `estimated`, `unknown`). Measured or
estimated observations have a nonnegative `value` and a `method`. Unknown
observations have `value:null` and a nonempty `reason`. Missing fields fail;
unknown is never converted to zero. The required metric names are:

- Durations: `setup_ns`, `ttft_ns`, `prefill_ns`, `decode_ns`, `total_ns`,
  `host_record_ns`, `host_submit_ns`, `native_completion_ns`, `gpu_ns`.
- Counters: `launches`, `submits`, `barriers`, `allocations`, `allocation_bytes`.
- Process memory: `rss_bytes`, `pss_bytes`, `driver_process_bytes`.
- Traffic estimates: `weight_bytes`, `kv_bytes`, `intermediate_bytes`.

The graph adapter exposes total token time, record time, completion time,
launches/submits and successful session allocation deltas. `chip_ns` in the
existing graph stats is host-observed submission-to-marker completion, so it
is named `native_completion_ns` here. It is not GPU execution time. With
stamps on, `gpu_ns` is the last-first device timestamp span; per-kind stage
spans include barrier/report cost. With stamps off it is unknown. The public
graph stats do not expose host submit duration or an executed barrier counter;
these stay unknown rather than being inferred by subtracting unrelated times.

Reuse #14's process-memory snapshots as measured RSS, PSS and driver process
bytes, with their sampling phase and method. The inspected canonical code has
no callable #14 snapshot API, so this adapter does not duplicate one or claim
its completion. Import accepts these observations from that producer in each
event's `measured` object, with required per-field scope/phase descriptions
in the event's `methods` object. Reports keep RSS, proportional PSS and driver bytes
separate. They never add shared-weight RSS across processes, add driver memory
to host mappings, or subtract an estimated weight size to claim measured
overhead. Existing allocation counters are session-layer allocation traffic,
not total process residence. Scope and phase must travel in the method text.

Traffic estimates require an explicit derivation in `method`; they are not
measured memory transactions. The importer accepts `estimated` entries of `{value,method}` for these
three byte fields and labels them estimated. Absent estimates stay unknown. No unsupported
counter is promoted to an estimate. Any bandwidth probe stays an
access-pattern-specific reference. No read-only probe rate is applied to total
mixed traffic as a proven physical ceiling.

## Pairing and report decisions

Run complete baseline/candidate pairs in the parent-coordinated quiet window,
with identical workload, seed, source/model/prompt/config/binary identities
recorded for each side, and the same driver/chip and thermal/clock protocol.
Source/binary may differ between implementations; model, config, prompt,
contract, precision, policy, driver and chip must agree. Preserve unfavorable
runs and all individual pair ratios. The tool never enables a candidate.

Record stamps-off and stamps-on observations for each independent session,
phase and shape. Their paired total-time delta and fraction quantify observed
instrumentation cost; negative deltas are retained. Unpaired instrumentation
cost is unknown. Token distributions and ratios remain separated by cold/warm,
stamps setting and exact shape/seed. Multiple observations from the same
session/group are refused to prevent treating correlated tokens as independent
sessions. Session provenance and actual independence still need coordinator
review. The report does not invent confidence intervals.

Percentiles use nearest rank: sort raw independent session observations and
select max(0,ceil(q*n)-1). p50/p95, min, max, session counts and raw inputs are
preserved. Mixed missing/measured groups report unknown, with known sample
count. Throughput uses output tokens/decode duration when decode is measured.
Fixture reports always say `NONE_FIXTURE` and hardware NOT_RUN. Imported model
reports say `RAW_IMPORTED_ONLY`; pair decisions remain REVIEW_REQUIRED (or
INSUFFICIENT_SESSIONS below three), including slower candidates. Complete-model
quality stays NOT_RUN. Matching artifacts do not prove a hardware run occurred.

A graph-enabled CUDA baseline may be imported with backend `graph-cuda` and
identical complete-model correctness/workload gates. Without such data the
CUDA comparison is NOT_RUN. The deployed native runtime gains no CUDA userspace
dependency. Default or publication claims require independent review and the
final #5/#16 coordinated GB10 receipts.
