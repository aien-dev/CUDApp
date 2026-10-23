# CUL1 public native-host task contract

Scope: the bounded CUDA++ domain adapter for #53. **NOT full #53 acceptance, a new benchmark service, or a GPU execution path.** All newly delivered executable logic/tests are seeded `.om`. Existing Linux/shell commands still assemble/build the modules; replacing that inherited host glue belongs to Omega #635/#636 and CUDA++ #15. No retirement claim.

## Authority and interface

`task.om` exports `kvalid(n:ulong)->int` and `keval(n:ulong,c:ulong)->int`, using single-threaded fixed scratch arrays `kt:uchar[4096]` (trusted frozen task) and `ko:uchar[1032]` (observed candidate output). It never writes either array. `wire.om` exposes `kframe(n:ulong)->int` over `packet:uchar[5137]`, copying a complete public developer frame into those arrays before checking. Buffers must not be used concurrently.

The eventual Omega #653 caller must bind the manifest, compiler, evaluator, actual candidate/launch and output acquisition to immutable identities and keep the task inaccessible to candidate modification. No such framework attachment is qualified here. The data-only stdin entry can be given any public task by its caller; it does **not authenticate** that task and is not a sandbox or promotion authority. Its `HOST_ONLY 0` text is a diagnostic, not a signed receipt. Candidate bytes are never interpreted as code, claimed timing, policy, tolerances or a PASS flag. Tests of fake result bytes do not qualify protection against substituting the evaluator executable or recorder.

## Canonical serialization

One manifest is 24 little-endian unsigned 32-bit header words (96 bytes), followed by A's complete physical footprint and then B's, in little-endian u32 words. Exact length only, no optional/trailing fields, whitespace or alternate serialization. Serialization is this fixed byte layout, not JSON. Headers and all unused flags are validated before any payload access.

| Word | Field | Accepted values |
|---:|---|---|
| 0 | magic | ASCII `CUL1`, LE value827086147 |
| 1 | schema | 1 |
| 2 | operation | 1 add; 2 left-to-right sum; 3 matmul |
| 3 | scalar profile | 1 unsigned32 modular |
| 4 | numerical mode | 0 exact integer, no floating tolerance |
| 5,6,7 | M,N,K | registered shapes below |
| 8,9,10 | A,B,output element/row strides | bounded0..32 and checked footprints; zero B stride only for sum |
| 11,12,13 | A,B,output physical word counts | each <=256; A/output nonempty by shape; B zero only for sum |
| 14 | pool | 1 public development; 2 validation and3 final refused with77 |
| 15 | normalized shape/operation family | derived from actual dimensions, not a caller-selected family |
| 16 | display task ID | nonzero u32, **not** a deduplication or content identity |
| 17 | reference semantics revision | 1 |
| 18 | alias rights | 0 disjoint inputs/output only |
| 19 | effects | 1 output-only writes, inputs unchanged |
| 20 | target/evidence class | 1 native-host reference; 121/other targets unsupported |
| 21,22 | absolute/relative tolerances | both exactly0, changing either is invalid |
| 23 | reserved | exactly0 |

A manifest's immutable identity is its exact bytes, externally hashed by the framework. Semantic fields/input bytes, provenance/task label/pool and performance context must remain separately identified in that framework. This module does not invent an abbreviated hash or a second identity registry. A renamed task is not declared novel. Family tags are necessary metadata, not a completed family-split/deduplication service.

Registered families:
- Add: `(M,N,K)=(3,1,1),(8,1,1),(17,1,1)`, families101/102/103. Each operand/output footprint is `(M-1)*stride+1`, positive strides.
- Sum: same shapes, families201/202/203. A has that footprint; B count/stride0, output count/stride1.
- Matmul: `(2,3,2),(3,2,3)`, families301/302. A is MxK, B is KxN, output MxN. Respective row strides >=K, >=N, >=N. Footprint is `(rows-1)*stride+columns` with no unused trailing row padding.
- All products/offsets are bounded after shape/stride/count admission; unsupported shape is never a GPU capability claim.

The scalar reference is independent of CUDA++'s generator, encoder and candidate implementations. Add/multiply/sum retain the low32 bits. Sum and matmul traverse the reduction dimension in increasing order. For this modular integer domain different summation orders are mathematically equivalent; these tests **cannot detect or prove floating-point reduction-order preservation**. NaN/Inf/signed-zero/subnormal semantics and FP conversion/approximation are unsupported, not covered as floats by integer bit patterns. This implements the exact-integer branch required by #16's addendum, not a replacement for its finite-FP comparator, model-quality authority or trace format.

## Observed output layout

Candidate output is `(output_count+2)*4` bytes: leading canary, complete physical output footprint, trailing canary. Canaries and internal padding equal `0xA5A5A5A5`. Every logical word is independently recomputed from the trusted task, every padding/canary checked. Exact output size is mandatory. Correct initial bytes cannot prove a write occurred; skipped-write controls poison the output and demonstrate detection of omitted observable results, not arbitrary memory-effect tracing. The future acquisition owner must also observe protected inputs and actual device memory.

Statuses:0 checked match in this public native-host domain;1 mismatch/canary corruption;65 malformed record, counts, policy or output length;69 unsupported version/profile/op/shape/effects/target;77 non-public pool access refused;74 stream I/O failure. Input-length bounds precede field reads; structural checks precede semantic support, pool, shape/stride and output checks. No cached PASS survives another call.

## Native developer replay

Build only with the explicit, source-reconstructed compiler SHA256
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
Qualified source import: omega-bridge `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`, seed origin omega-mv `29b12760f4c962606411d6bf25032aedef063301`.

Existing development commands on Linux AArch64, with a private fresh output directory and `OMC` pointing to that verified compiler:

```sh
cat native/learn53/task.om native/learn53/wire.om native/learn53/stream.om > "$OUT/stream.om"
"$OMC" < "$OUT/stream.om" > "$OUT/stream"
# Require successful compilation, nonempty static AArch64 EXEC, then chmod700.
timeout 10 "$OUT/stream" < public-frame.bin
```

A frame is `u32le(task_length) || manifest || u32le(output_length) || observed_output`. EOF is required. The reader handles fragmented reads, refuses truncation/trailing bytes and reads at most5137 bytes. It never opens a file or launches a process. An open pipe can stall: the caller's existing deadline/isolation is required; no native wall-clock/process supervisor is claimed.

Native test construction follows the same concatenation with `tests/learn53/fixtures.om` plus `unit.om`. The frame tests add `wire.om`, `stream-fixtures.om`, `stream-unit.om`. The public emitter substitutes `stream-input.om` and reads one mode byte:0 valid,1 wrong result,2 truncated,3 extra byte,4 forged length,5 final-pool refusal. `stream-check.om` independently checks the exact success diagnostic. No test runner or compiler wrapper in another implementation language was added.

## Acceptance still pending

Omega653 framework attachment and complete attempt/budget record; independent split custody/deduplication/sealing; #15 seeded host qualification; #16 FP/numerical policy migration; #17 real native full-cost collector/paired baselines; one measured hot operator; #10/#35 independent binary/launch admission; #24 runtime memory/session integration; GB10 and full consuming graph; independent judge and production release. No speed reward, candidate generation, training, proof or default selection occurs in this slice. Hardware/consumer tests are NOT_RUN, not NO-GO experiments.
