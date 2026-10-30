# MAX interop Gate 2: does a CUDApp cubin load and run?

Goal: get the smallest Omega kernel (`matmul` 16x16x8, sm_121) to load and run
through MAX's `DeviceExternalFunction` path, with a known-good control and the
C/libcuda probe as an independent oracle.

Status: host preparation is IMPLEMENTED and the host refusals are EXERCISED.
Every device row is **NOT_RUN** because no authorized window has been opened
for this gate. The device results will be added under `G2/<window-id>/`.

## Earlier device runs (2026-10-24, transcript only, no receipt)

These runs exist only in an omp session transcript
(`~/.omp/agent/sessions/-tmp/2026-10-24T03-09-36-465Z_01a16be2-*.jsonl`).
No receipt was written and no cubin sha256 was recorded. Driver 580.173.02,
CUDA 13.0, old `cubin_probe` (`cuModuleLoadData`, no JIT log).

| window | image | output | transcript line |
|---|---|---|---|
| 20261024T043818Z-aa5c86 (shared) | `68/matmul-16x16x8.cubin` (no program headers) | `cuModuleLoadData -> 200 device kernel image is invalid`, rc=4 | 2268 |
| same | `/tmp/refcubin/k.cubin` (ptxas, a different kernel) | loaded; `worst_abs=3.25 bad=127`. The logged `ref_rc=0` is the exit code of `tail`, not of the probe | 2324 |
| 20261024T044814Z-4dc098 | `/tmp/phdr-mat.cubin` (built 04:47:56Z from the uncommitted program-header change; PHDR at 0xd60, the same layout as main) | `cuModuleLoadData -> 200`, rc=4 | 2370 |

Consequences:
- Program headers are **not sufficient** for a load. Commit 130a101
  (landed via #77) and the old `put_phdr` comment both say the loader rejects
  the image without them, "confirmed on GB10". No run shows that the headers
  are necessary, and the run with headers still failed. Both comments in
  `src/omega_cubin.{c,h}` are corrected in this change.
- `68/matmul-16x16x8.cubin` (sha256 `7d130693…4b34eb`) predates the
  program-header commit. Do not use it as the current writer's output.
  `68/RESULT.md` is left unchanged.

## Load matrix (fixtures outside this repo)

`~/handoffs/max-g2/cubin_variants.py` derives every image from two inputs.
The first is the current writer's output: `artifact-export` at main b5746b94,
artifact `acfba3aa…3b1e`. The second is a ptxas 13.0.88 control built from
`control.cu` (same symbol and argument ABI, `-fmad=false`). The ptxas images
and E2 contain NVIDIA-tool bytes, so they stay out of this repo
(PROVENANCE.md). Only their sha256 values are recorded here.

| name | fn | expect | sha256 | isolates |
|---|---|---|---|---|
| A-ctl | omega_matmul_bf16 | PASS | `d95a489211830f58b93f092bab94ed4174c56cc4ae49779da3091aea2bf581d2` | probe+driver+ABI known-good |
| B-ctl-nomerc | omega_matmul_bf16 | PASS if capmerc not required | `2d2516305d8bcba5273af73e72e135473ec20f73ef83dd12f74d832406cb414a` | capmerc/merc sections |
| C-omega-stale-nophdr | omega_matmul_bf16 | REJECT 200 (seen 10-24, no receipt) | `7d1306933b67fa24a20dba59d39bf48fbd4aa5cec37e046e3b664e5e0c4b34eb` | program headers absent |
| D-omega-main | omega_matmul_bf16 | REJECT 200 expected (10-24 phdr-mat) | `c4de09ece9f92d762c91eac0ca7af635618538697f90654dc76da2497e492e48` | current CUDApp writer |
| E1-omega-symfix | omega_matmul_bf16 | UNKNOWN | `7721f9566b8246437b3dc3f55c8ada057a49f3cc9b16fc6d395faa73bee5caed` | symtab sh_info |
| E2-omega-tkinfo-ptxas | omega_matmul_bf16 | UNKNOWN | `62acfe37b66a7f58db311b5ba3dcf2ccdd42220b56386a8bbcfc8e4178a992a2` | .note.nv.tkinfo (ptxas bytes) |
| E3-omega-tkinfo-own | omega_matmul_bf16 | UNKNOWN | `67bd97401061daf1d798dfc1032066c3ef9f3579c9bc49883200a1d48f16a99e` | .note.nv.tkinfo (own strings) |
| E4-omega-tkinfo-cg-symfix | omega_matmul_bf16 | UNKNOWN | `e4dad818f5fa68452e90a9a6807cf02ae055c6c2284889fd803c645789778c3e` | all three metadata gaps |
| N1-ctl-truncated | omega_matmul_bf16 | REJECT | `e059683df43d9a2a6e8fc5fbcc831378dbd5d7d75d2e32e4accf9ca79d43387d` | truncated image |
| N2-ctl-wrong-entry | omega_matmul_bf16_missing | REJECT (500 NOT_FOUND) | `d95a489211830f58b93f092bab94ed4174c56cc4ae49779da3091aea2bf581d2` | entry name |
| N3-ctl-eflags-sm80 | omega_matmul_bf16 | REJECT | `194a48dce16bcef34faadb233a46b9d9370a80d43326d62c2cdfd9819e5f84be` | e_flags arch field |
| N4-max-vecadd-sm80 | vec_add | REJECT (209 NO_BINARY_FOR_GPU) | `b0a171991dc9a2c456b70d3221b6353d2e671159f94e5eca292986ff5bcf21e8` | wrong-arch real cubin |

The same table is `~/handoffs/max-g2/variants/MATRIX.tsv`. All images
pass `readelf`. All except N1 pass `cuobjdump`.

## Probes

- `tools/interop/cubin_probe` (this change): uses `cuModuleLoadDataEx` with
  8 KiB error and info JIT logs (`CU_JIT_LOG_VERBOSE`), resolves the entry,
  fills the output with the sentinel -12345, and checks against an fp32 oracle
  at tolerance 1e-4. Modes: default run, `--load-only`, `--expect-reject`.
  Exit codes: 0 pass (or expected reject), 2 usage, 3 no window, 4 load/CUDA
  failure, 5 numeric, 6 reject expected but image accepted.
- `~/handoffs/max-g2/max_cubin_probe.mojo` (mojo 1.2.0, max 26.7.0): uses
  `DeviceExternalFunction` with MAX-owned buffers and stream. Same sentinel,
  oracle and exit codes. Modes: `run|loadonly|reject`.
- `~/handoffs/max-g2/run_matrix.sh`: loads each image with both probes first,
  and launches an image only if both loaders accepted it. Writes `raw.log` and
  `results.tsv` to `$SPARK_WINDOW_DIR/max-g2/`. Disables MAX's pooled memory
  manager, so a shared window does not reserve chip memory next to the
  resident servers.

## Host results (EXERCISED)

```
build/cubin_probe                       -> usage, rc=2
build/cubin_probe --bogus               -> usage, rc=2
build/cubin_probe --load-only X fn 16 16 8 (no window)
  -> refused: SPARK_WINDOW is not set; no authorized GPU window, rc=3
max_cubin_probe (no window)             -> refused, rc=3
run_matrix.sh (no window)               -> refused, rc=3
run_matrix.sh with stub probes          -> 12 rows; launch rows skipped when a load fails
```

## Device run (NOT_RUN; needs a ledger grant)

```
spark-window open maxinterop-g2 --ledger <ID> --gb10 shared --quiet none --lease 10
spark-window run ~/handoffs/max-g2/run_matrix.sh
spark-window close <window-id>
```
