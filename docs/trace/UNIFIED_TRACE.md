# Unified trace: one timeline from Omega source to launch

Status: **PROPOSED** schema `aien.unified-trace/1`. The host recorder, the `omtrace` tool and their
host checks are IMPLEMENTED and EXERCISED on the host. Device-side timing is **NOT_RUN**: nothing
here reads a GPU clock, and no GPU was opened to produce any number in this file.

## What it gives you

One trace that follows a program from the Omega compiler through the CUDA++ runtime:

```
omtrace                                   (host process: compile and run)
├─ source.read  compose  omc.compile  diag.decode     compile phases
├─ omc.error    file.om:3:13, error 3                 compiler diagnostic, in the unit that caused it
└─ program.run                                         the traced program
   ├─ gpu.open        launch.om:gbopen                 bytes allocated, code bytes, status
   │  └─ mem.alloc    launch.om:gbmem   x8             bytes, buffer slot
   ├─ mem.copy        host.upload:a.b                  bytes, direction
   ├─ kernel.launch   kernel:VECTOR_1D/vector.om:gbvec elements, last stage reached, status
   │  └─ mem.moved    launch.om:gbcmd                  command bytes written
   └─ trap            range, candidates t1.om:3        Omega trap code and the source lines that raise it
device (NOT_RUN: no device clock recorded)
```

Names are semantic (Omega unit, function, kernel), not driver handles. The same file opens in
Perfetto and Chrome's trace viewer, and the same events export as OpenTelemetry spans, so one
schema serves a desktop viewer and a fleet collector.

It uses no CUPTI, NVTX, Nsight, libcuda or other NVIDIA interface. The runtime records what its own
code does; every timestamp is the host's `CLOCK_MONOTONIC`.

## Pieces

| File | Role |
|---|---|
| `native/trace/htrec.om` | The recorder. Fixed tables (4096 events, 256 names, 16 KiB of name bytes), off until `htinit`, writes the binary OTR1 file with `htdump`. |
| `native/trace/htlaunch.om` | `htopen` / `htlaunch`: traced entry points for `native/blackwell/launch.om`. They behave exactly like `gbopen` / `gblaunch`. `launch.om` is not changed. |
| `tools/omtrace/omtrace.om` | Wraps the pinned `omc`: times the compile phases, decodes diagnostics, optionally runs the program, merges the OTR1 file it writes, validates it, and writes Chrome JSON and OTLP JSON. |
| `docs/trace/unified-trace.chrome.schema.json` | JSON Schema (2020-12) for the Chrome export. |
| `tests/host-trace/` | Host checks with negative controls and the overhead driver (below). |

All three are handwritten seeded Omega. A unit that does not concatenate `htrec.om` and
`htlaunch.om` contains no trace code at all.

## Using it

```sh
O=/path/to/omega-mv/seed/out/omc
cat native/trace/htrec.om tools/omtrace/omtrace.om | $O > omtrace && chmod +x omtrace

# compile only:              ./omtrace c $O out/prog -  unit1.om unit2.om ...
# compile and run:           ./omtrace r $O out/prog -  unit1.om unit2.om ...
# compile, run, merge OTR1:  ./omtrace f $O out/prog out/prog.otr unit1.om ...
#   (the program is run as `out/prog.bin out/prog.otr` and must write its recording there)
# validate a recording:      ./omtrace v out/prog.otr
```

Outputs: `PREFIX.unit.om` (the composed unit), `PREFIX.bin`, `PREFIX.omc.err`,
`PREFIX.trace.json` (Chrome/Perfetto), `PREFIX.otlp.json` (OTLP/JSON).

Exit codes: 0 traced; 2 usage; 3 compile failed (the trace is still written and shows where);
4 output too large; 5 cannot write; 6 fragment unreadable; 20..31 fragment rejected, reason on
stderr as `OMTRACE_REJECT <reason>`. A traced program's exit status 100+N means it hit trap N.

In a program: compose `htrec.om` (and `htlaunch.om` for launches), call `htinit` to start, use
`htopen` / `htlaunch` in place of `gbopen` / `gblaunch`, add your own spans with
`s = htbeg 9 ID` … `htend s a0 a1 rc` and marks with `htmark 10 ID a0 a1 rc`, register names with
`htnbeg ID` / `htnbyte b` / `htnend`, and call `htdump fd` at the end.

### Viewing

1. Open <https://ui.perfetto.dev>, choose **Open trace file**, pick `PREFIX.trace.json`. The UI
   parses the file in the browser. Chrome's `chrome://tracing` also opens it.
2. SQL over the same file with Perfetto's trace processor:
   `trace_processor -q q.sql PREFIX.trace.json` with
   `select name, category, ts, dur from slice order by ts;`
3. OpenTelemetry: `PREFIX.otlp.json` is an OTLP/JSON `TracesData` body. A collector with an OTLP/HTTP
   receiver accepts it at `/v1/traces` with `Content-Type: application/json`. (Posting to a collector
   was not exercised; see NOT_RUN.)

A sample from the host launch test is in
`evidence/host-trace-2026-10-26/sample/`.

## Event kinds

Every event has a kind, a name id, a start and end time, two unsigned arguments `a0`, `a1` and a
signed status `rc`. Exports label `a0`/`a1` per kind:

| id | kind | form | a0 | a1 | status |
|---|---|---|---|---|---|
| 1 | compile.phase | span | bytes | count | phase result |
| 2 | diag | instant | line | col | omc error number |
| 3 | gpu.open | span | bytes_allocated | code_bytes | nverr |
| 4 | kernel.launch | span | elements | stage (+ `stage_name`) | nverr |
| 5 | mem.alloc | instant | bytes | slot | handle sign |
| 6 | mem.copy | span | bytes | direction (1 to device, 2 to host) | result |
| 7 | host.wait | span | a0 | a1 | result |
| 8 | trap | instant | code (+ `trap_name`) | count | program exit |
| 9 | user.span | span | a0 | a1 | rc |
| 10 | user.mark | instant | a0 | a1 | rc |
| 11 | mem.moved | instant | bytes | direction | rc |

Arguments are counts, sizes, stages and status codes only, never addresses, payload bytes or user
data. Launch stages: 40 build.upload, 41 submit, 42 host.wait, 43 complete.check, 0 refused.
Because `launch.om` has no hook between stages, a host wait inside `gblaunch` appears as a
`kernel.launch` span whose last stage is `host.wait` (42), not as a separate `host.wait` span.

Trap names: 1 overflow, 2 divzero, 3 range, 4 loopbound, 6 stack, 7 shift, 9 requires,
10 ensures (also invariant and seal), others `codeN`.

## OTR1 recording (binary, little-endian)

Header, 64 bytes:

| offset | type | field |
|---|---|---|
| 0 | u32 | magic 827479119 (`OTR1`) |
| 4 | u16 | version 1 |
| 6 | u16 | record size 48 |
| 8 | u32 | event count |
| 12 | u32 | dropped (past capacity) |
| 16 | u32 | refused (malformed calls) |
| 20 | u32 | clock domain, 1 = host CLOCK_MONOTONIC ns |
| 24 | u64 | monotonic time at `htinit` |
| 32 | u64 | realtime at `htinit` (ns since the Unix epoch) |
| 40 | u32 | process id |
| 44 | u32 | name count |
| 48 | u32 | spans still open at dump |
| 52 | 12 bytes | zero |

Then `count` records of 48 bytes: u16 kind, u16 flags (bit 0 closed), u32 name id, u64 t0, u64 t1,
u64 a0, u64 a1, i64 rc. Then `name count` names: u32 id, u16 length (1..63), bytes from
`[A-Za-z0-9._:/-]`. The largest recording is 214592 bytes.

The recorder refuses, and counts, a call with an unknown kind, a kind in the wrong form, an unknown
name, a bad span slot, a second end or an end before its begin. Events past capacity are counted as
dropped. Both counts, and the number of spans left open, are in the header; the exports carry the
totals (see Exports) so a trace that lost events never looks complete.

`omtrace v` and `omtrace f` reject a recording, with a distinct exit, when the magic (20), version
(21), record size (22), clock domain (23), file size (24), count (25), kind (26), form (27), times
(28), a name (29), a name reference (30) or the open-span count (31) is wrong.

## Clocks

Times are host monotonic nanoseconds. `omtrace` and the program each record their monotonic and
real time at start, so the exporter places both on one axis: Chrome `ts`/`dur` are microseconds from
the first event; OTLP times are Unix nanoseconds (`real0 + t - mono0`). No device clock exists in
this schema version; a future clock domain 2 (device) needs GPU-side evidence first.

## Exports

**Chrome trace JSON** (`displayTimeUnit` ns). Process `omtrace` for compile phases, process `program`
for runtime events, and a `device` row carrying the instant `device.timing.NOT_RUN`. Spans are `X`
events, instants `i` events with process scope; `cat` is the kind; `args` hold the labels above.
`otherData` records the schema, the clock, `device_timing: NOT_RUN`, and `dropped`, `refused` and
`open_spans`: dropped and refused add omtrace's own recorder to the program's header counts;
`open_spans` is the program's header count (omtrace's own root span is always closed by export).
Names are pooled once per name, not per event, so a full 4096-event recording exports.

**OTLP/JSON** (`TracesData`). One resource (`service.name=omtrace`, `aien.trace.schema`,
`aien.device_timing=NOT_RUN`, and integer `aien.dropped`, `aien.refused`, `aien.open_spans` with the
same values as `otherData`), scope `aien.unified-trace` version 1. `traceId` is the hex of
omtrace's real start time and pid; `spanId` is the event index + 1; the parent is the smallest span
that contains the event (so program events sit under `program.run`). Instants become zero-length
spans. Attributes: `aien.kind`, `aien.source`, the per-kind labels as `aien.<label>`, and for
diagnostics `code.file.path`, `code.line.number`, `code.column.number` (OpenTelemetry semantic
conventions, code registry). Status code 2 (error) when the status is nonzero. Trace and span ids
are hex, as OTLP/JSON requires.

## Source locations

- **Compiler diagnostics** are exact: omc reports a line of the composed unit (and a column for
  `om2 diag` units, whose header must be on line 1 of the composed unit); `omtrace` maps it back to
  the unit file and line it came from.
- **Runtime traps** are not exact. The pinned omc emits one trap stub per code per function, so the
  exit status says which trap code fired but not which statement. `omtrace` lists every explicit
  `trap N` line for that code as `candidates`; implicit traps (bounds, overflow checks) have no
  candidate line. Exact per-site identity needs a compiler change (an opt-in per-site stub mode, or
  a site id in omc's diagnostic records); it is proposed separately and not part of this change.

## Cost

Tracing composed but off costs one load and one branch per wrapped call. Measured with
`tests/host-trace/overhead*.om` (1000 launches of 64 elements on the host stand-in per sample, 300
samples per arm in 10 interleaved rounds, one process per arm and round pinned to one core with
`taskset`, load average 17.8-19.6):

| core | absent (median ns / launch) | off minus absent | on minus absent |
|---|---|---|---|
| cpu19 (Cortex-X925) | 6915.2 | +10.4 ns (+0.15%) | +415.0 ns (+6.0%) |
| cpu14 (Cortex-A725) | 14245.5 | -100.7 ns (-0.71%) | +538.2 ns (+3.8%) |

The off-versus-absent difference changes sign between core classes and is within 1%. It is the
difference between two different binaries (code layout), not a measurable cost of the off flag.
Recording costs about 0.4-0.5 microseconds per launch on this host path (two clock reads and two
events). Raw samples and the Wolfram statistics are in
`evidence/host-trace-2026-10-26/overhead/`.

## Checks

`OMEGA_COMPILER=/path/to/seed/out/omc sh tests/host-trace/run-host-trace.sh [OUTDIR]` compiles and
runs every check with the seeded compiler and prints `HOST TRACE CHECKS PASS`:

1. `recorder-test.om`: 68 checks on the recorder, including off by default, refusals, capacity,
   name rules and the dump layout.
2. `launch-test.om` traced through `omtrace f` against the host stand-in for the device
   (`fakenv.om`, `fakedev.om`): 45 checks, then the Chrome and OTLP files are checked for the
   launch, copy, stage and NOT_RUN events and the recording is revalidated.
3. A trap and a compiler diagnostic are traced and located in the unit that caused them.
4. Malformed recordings: `badfrag.om` writes 13 mutants of a good recording; each is rejected
   with its own reason; a missing file exits 6.
5. Source mutations: removing the second-end refusal, the name charset check or the off-by-default
   check makes `recorder-test` fail; removing the end-before-start check stops the time rejection.

The runner is POSIX sh, like `tests/device-trace/run-emitter-checks.sh`: it only composes, runs and
compares exit codes and lines; every check is a seeded Omega program.

## NOT_RUN

- Device-side timing on GB10 (kernel start/end on the device clock): no GPU window was opened.
- Running `htopen`/`htlaunch` against the real `context.om` on a GB10: host stand-in only.
- Posting the OTLP file to a running OpenTelemetry collector.
- A screenshot of the trace in the Perfetto UI (headless browser screenshots failed on the build
  host, including on a blank page).

## Relation to other work

- CUDApp #82 / Omega #778 (OMGT device trace) and CUDApp #93 (trace emitter) cover device-side
  trace records. This schema is the host timeline those records can join once a device clock
  domain exists.
- Omega trace issues #607-#612 (#610 offline CLI, #612 viewer): `omtrace` is an offline CLI and
  uses existing viewers instead of a new one.
- Promoting `aien.unified-trace/1` to a shared protocol repository is a decision for the owner.
