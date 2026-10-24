# LocalRouter with an Omega text server

This is a text-only candidate setup, tracked by [CUDApp #27](https://github.com/aien-dev/CUDApp/issues/27).
It generates configuration and checks HTTP behavior. It installs no service, downloads no weights,
and changes no GPU driver. It does not port LocalRouter's image/video kernels to CUDA++.

## What connects

An OpenAI client or LocalRouter's MCP `chat` tool reaches LocalRouter on loopback port 8192.
LocalRouter's existing `openai_server` worker launches a separate `aien-serve-http` candidate
on loopback port 18092. That server selects the Omega engine. Its actual linked GPU backend
must be qualified through the canonical CUDApp consumption work in #5.

Inspected sources:

| Repository | Revision | Contract |
| --- | --- | --- |
| [LocalRouter](https://github.com/jayleaton/localrouter/tree/70ad4fa198dfc0c8175d02fff746e95759c42d59) | `70ad4fa198dfc0c8175d02fff746e95759c42d59` | `src/engines/openai_server.zig`, `src/config.zig`, `src/tool/process.zig` |
| [sovereign-core](https://github.com/aien-dev/aien-sovereign-core-mv/tree/754c07b118f41ad899607b60329bd851b1ccb13e) | `754c07b118f41ad899607b60329bd851b1ccb13e` | `crates/aien-serve-http/src/{main,server,omega_engine}.rs` |
| CUDA++ | `232332422bfdc31c8918085b50f13c1e18ab702e` | baseline before this integration |

Recheck these contracts if consuming different revisions. Do not overwrite the live motor's binary,
configuration or CUDApp pin to make this example run.

## Two integration limits

**Process ownership:** `openai_server` starts a child and terminates it on unload or daemon exit.
It cannot attach to the existing motor. Use a separate immutable candidate binary, candidate
model ID, data directory and ports. The generator reserves port 18091 for the live motor.
Other live-service ports must also be excluded by the operator. `check` refuses occupied ports,
but checking a free port does not reserve it against another process.

**GPU locking:** LocalRouter normally holds its configured host lock shared while any model
is loaded. The inspected Omega engine takes `/tmp/aien-gb10.lock` exclusively during load and
generation. Using that file for both would block Omega. This profile therefore leaves
LocalRouter's `gpu_lock` empty, keeps Omega's lock unchanged, and permits exactly one text tool.
Do not add LocalRouter image/video/speech GPU workers to this profile. Their execution would
need an explicitly qualified lock/admission handoff. The memory budget is admission accounting,
not a global reservation against other processes. Coordinate candidate loading with the GPU
owner and keep the live motor's memory in the budget.

## Before starting

You need Python 3.10+, a LocalRouter binary and a separately qualified `aien-serve-http` candidate.
The inspected LocalRouter build requires **exactly Zig 0.17.0** and pins TensorFold in
`build.zig.zon`. Its documented host build is `zig build -Doptimize=ReleaseSafe`; omit `-Dnvcc`
for this text-only experiment. That avoids embedding its own CUDA kernels. It does not prove
that every LocalRouter process is free of CUDA library mappings. Record and inspect the actual
binaries/processes before making that claim. Run upstream's `zig build test` on the build host.

Reuse the server build and complete source/dependency closure from the #5 integration owner.
The inspected server accepts `--engine omega`, `--model-dir`, `--bind`, `--port`, `--model-id`
and `--max-context`. This profile limits context to 4096 and uses the server's existing default
FP8 weights. Use its supported Qwen checkpoint and tokenizer, not an image/video checkpoint.
The candidate needs enough free memory for its own load even when the live motor is resident.

Measure and record `resident_bytes` and incremental per-request `working_bytes` for the
chosen engine, checkpoint and context. Do not copy another model's memory numbers or enter
zero for an unmeasured Omega resident size. Zero incremental working memory is valid only
when measured. The generator requires these values explicitly.

## Generate and check a candidate config

Run from the CUDA++ checkout on the target host. Supply real absolute paths and measured byte
counts. These shell parameters deliberately refuse missing values:

```sh
python3 tools/localrouter/setup.py config \
  --server "${OMEGA_CANDIDATE_BIN:?absolute path to a separate aien-serve-http}" \
  --model-dir "${OMEGA_MODEL_DIR:?absolute path to the supported checkpoint}" \
  --data-dir "${LOCALROUTER_CANDIDATE_DATA:?absolute path to a separate data directory}" \
  --resident-bytes "${OMEGA_RESIDENT_BYTES:?measured resident bytes}" \
  --working-bytes "${OMEGA_WORKING_BYTES:?measured incremental working bytes}" \
  --router-port 8192 --backend-port 18092 \
  > localrouter-omega.candidate.json

python3 tools/localrouter/setup.py check --config localrouter-omega.candidate.json
```

Keep generated host-specific config outside version control. Generation only inspects paths.
`check` additionally tests whether the ports are available. Neither starts a server or opens a GPU.
The 8 GiB OS reserve is a starting configuration, not a measured model budget or an OOM guarantee.
Increase it to account for the other resident workloads as needed with `--reserve-bytes`.

Use a foreground daemon first, on the same host as the candidate binary and model files:

```sh
/absolute/path/to/localrouter serve --config localrouter-omega.candidate.json
```

There is no boot service, container, tailnet exposure or full Jarvis installation in this setup.
The generated `keep_loaded: false` prevents eager model loading at daemon startup. A generation
request loads the candidate. It can then remain warm until its 300-second idle timeout or an
explicit unload. Logs are under the configured data directory.

## Read-only inspection, then opt-in generation

In a second terminal:

```sh
# Lists the candidate without asking it to load or generate.
python3 tools/localrouter/setup.py probe --config localrouter-omega.candidate.json

# Coordinated GPU window only: this sends several short requests and loads the model.
python3 tools/localrouter/setup.py probe --config localrouter-omega.candidate.json \
  --exercise --timeout 180 > localrouter-omega.http-receipt.json
```

The exercised probe checks router model listing, a whole completion, backend health/model/engine,
a second turn, a complete SSE response, rejected tool calling and an unknown-model error.
SSE errors, empty replies, wrong models and missing finish/[DONE] events fail. A local mock test
of the client is not a test of the real Zig daemon or Rust model server.

The HTTP receipt always leaves GPU correctness, process lifecycle and CUDA-free provenance
`NOT_RUN`. It establishes only protocol behavior. Qualify the actual model with golden/numerical
checks, cancellation, unload/reload, process cleanup, memory measurements, source/binary hashes,
and `/proc/<pid>/maps` observations in a separate receipt before promoting this configuration.
Confirm the live motor is still healthy before and after the window. Do not run deliberate
faults against the live service.

At the inspected sovereign-core revision, nonempty `tools` and `tool` message roles are rejected.
The generated voice settings disable tools, model management, speculative chat and priming.
LocalRouter MCP can still expose a plain-text `chat` tool to an external agent, but this does
not give Omega native tool calling. Validate MCP separately if enabling it in a client.

## Stop and rollback

Stop only the foreground candidate daemon with Ctrl-C. LocalRouter is expected to stop its
worker and owned server. Verify port 18092 is closed and its child PID has exited; record a
lifecycle failure if either survives. Keep the live 18091 motor untouched. Keep model files and
qualified binaries; remove only the candidate configuration/data when no longer needed.
Do not use a broad `pkill`, restart the production service, unload GPU modules, or install the
full upstream Jarvis profile as a shortcut for this experiment.

## Host smoke test without a model

Use the server's explicit `--engine stub` mode with an empty temporary model directory and
`--resident-bytes 0 --working-bytes 0`. Follow the same generate/check/start/probe sequence.
Label every result `stub`, not GPU-qualified. The setup script itself has no compiler or model
dependency:

```sh
python3 -m unittest discover -s tests/localrouter -v
```

## After text routing works

1. Qualify shared-lock ownership and capability reporting before adding other GPU workers.
2. Benchmark one matched RMSNorm/SwiGLU operation with identical shape, datatype and numerical
   contract. Compare full outputs, warm latency, submission cost and memory on the same GB10.
3. Extend to a short native chain using #18 and #19. Reusable graph replay additionally needs
   persistent resources, safe parameter updates and lifecycle tests; command lists alone do
   not establish CUDA Graph equivalence.
4. Treat NVFP4/image/video parity as separate numerical and kernel-coverage work. Existing CUDA
   fatbins cannot simply be passed to CUDA++'s native kernel builder.

No speedup or full Jarvis capability is claimed by this integration.
