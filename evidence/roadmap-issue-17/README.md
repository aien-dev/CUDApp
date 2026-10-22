# Native benchmark host receipt

Executable source pin: `source-sha.txt`. Numerical source prerequisite:
3565287250e4ebd40e79a23d978edd3c936c94f2. Numerical evidence base:
104b9a968dec7dbb3642b303470a00876928a1f2. Canonical carried base:
349913f17c083996680c72b32f767b225005155d. Child branch was created from the
pushed numerical source and fast-forwarded through its evidence-only commit.
This evidence-only follow-up does not name itself as executable source.

`input-sha256.txt` pins source/test/tool/build/doc inputs. `binary-sha256.txt`
pins the existing native runtime, new opt-in collection adapter archive and
both C host-test executables. `host-tools.txt` records actual host, compiler
and tool identities and hashes. The native adapter archive is linked by an
approved workload explicitly; it changes no serving default or graph ABI.

## Actual commands and results

- `quiet-build timeout 30 node --test tests/native_benchmark_test.mjs`: the initial red run (`red.log`, exit 1) exposed uncovered context, unsupported batch and unbound seed acceptance. All three now refuse. These were actual behavioral gaps in the first implementation.
- Target progression is preserved in `target*.log`. Final target command:
  `quiet-build timeout 120 make OUT_DIR=build-roadmap-contracts-bench test-native-benchmark native-benchmark-adapter`, exit 0 (`target-ready.log`).
- Final full command: `quiet-build timeout 900 make -j2 OUT_DIR=build-roadmap-contracts-bench test native-benchmark-adapter`, exit and raw output in `host-suite.log`. This includes all 33 existing host targets and both new issue targets. No target, case, threshold or refusal was removed. This run began with no other roadmap make running.
- `tools/zero-cuda-gate.sh build-roadmap-contracts-bench/native_benchmark_adapter_test build-roadmap-contracts-bench/native_benchmark_collect_test`, exit 0 (`zero-cuda.log`). These binaries need only the ordinary host runtime libraries shown there.
- `./build-roadmap-contracts-bench/native_benchmark_adapter_test --fixture` writes `fixture-capture.jsonl` (exit 0 in `fixture-adapter.log`). This executes the production formatter with deliberately fictional fixture observations. It does not run the graph or a model.
- CLI `prepare`, `import`, `report`, `report --text` and `compare` were run in that order on the committed fixture paths; every exit code is 0 in `fixture-cli.log`. Reproduce with `node tools/native-benchmark.mjs COMMAND ...` and the file paths indicated by the command names below. Prepare/import require fresh output paths because they refuse overwrites.

```
node tools/native-benchmark.mjs prepare evidence/roadmap-issue-17/fixture-manifest.json fresh-plan.json
node tools/native-benchmark.mjs import evidence/roadmap-issue-17/fixture-manifest.json evidence/roadmap-issue-17/fixture-capture.jsonl evidence/roadmap-issue-17/fixture-correctness.json fresh-run.json
node tools/native-benchmark.mjs report fresh-run.json
node tools/native-benchmark.mjs report fresh-run.json --text
node tools/native-benchmark.mjs compare fresh-run.json fresh-run.json
```

`fixture-model.txt`, `fixture-prompt.txt`, `fixture-config.txt` and
`fixture-driver.txt` are the exact fixture bytes whose hashes appear in the
manifest. No real model weights, prompt, config or driver were sampled.
Fixture source/binary pins refer to the actual compiled host formatter source
and executable, with `kind:fixture` throughout. Reports explicitly state
NONE_FIXTURE, NOT_RUN hardware and NOT_RUN model quality. Fixture latency
values are test inputs, never a synthetic performance result or speedup claim.
No actual baseline or candidate whole-model benchmark was performed.

## Coverage and limits

The C formatter has 18 checks. The production graph collection wrapper has
12 checks with fixture device callbacks, including failing token/stamp paths.
The JavaScript target has 62 tests, including 19 reused numerical trace tests
and 43 benchmark tests. It imports JSONL from the compiled C formatter through
the CLI, checks request timestamps, nearest-rank percentiles, session/phase/
shape grouping, unknown-vs-zero, stamps-off/on pairing, unfavorable candidates,
fabricated fixture speedups, model/prompt/config/driver/policy/source/binary
identity, correctness coverage and seed binding, poison/reversed/nonfinite
or malformed values, and exact signed-zero artifact serialization.

Memory snapshots from #14 are accepted with explicit sampling scope/method;
RSS/PSS/driver bytes are never added together or double-counted as shared
weights. The inspected canonical code has no callable #14 snapshot collector,
so that producer and its actual process-tax result remain NOT_RUN. Traffic
estimates keep derivations and estimated labels; unknown counters stay null.
Read-only bandwidth references are never promoted to mixed-traffic ceilings.

The existing graph's chip_ns is host-observed submission completion time,
reported as native_completion_ns. Only actual device stamp spans are gpu_ns.
Host submit duration and barrier count are unknown where the ABI does not
expose them. Request TTFT/prefill/decode capture is a callable timestamp bridge;
no request was sent. Actual adapter-on-GPU collection, cold ingestion cost,
model correctness/quality, shared quiet timing, instrumentation cost and
CUDA graph-baseline comparison all remain NOT_RUN.

The parent must integrate #5/#6/#15 and run final coordinated GB10 gates with
real traces, source/model/prompt/config/binary/driver identities and independent
quality/timing review. The original baseline remains selectable. No issue was
closed and no PR, issue comment, default-enable or publication claim was made.

Final verification: `sha256sum -c input-sha256.txt` and `sha256sum -c
binary-sha256.txt` both exit 0 (`final-*-verification.log`). The earlier
numerical binary hashes still match after the complete child suite
(`numerical-binaries-still-identical.log`, exit 0). Source/document whitespace
check exits 0 (`source-whitespace-check.log`); raw tool output retains original
trailing whitespace because these are unedited receipts.
