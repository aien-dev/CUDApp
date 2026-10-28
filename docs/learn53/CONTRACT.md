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

## Joint resident MoE operation-instance binding v1

`native/learn53/moe-closure.om` extends this existing task/evaluator custody
boundary for #35/#53/#159/#26/#17. It is not another benchmark, task catalog,
artifact registry, receipt validator, signer or source of qualification authority.
Its immutable dependencies are in `moe-closure.lock`. The unchanged
`omega653.lock` and `omega-adapter.om` still own public CUL1 observation and
attempt accounting. No MoE numerical oracle or full-cost measurement is
invented here. The caller obtains those outcomes from the existing independent
#16/#17/#53 owners and retains their actual result provenance.

### Subject versus live instance

There are two deliberately distinct identities:

1. The independently frozen **approved joint subject**, already present in
   selector `q18..21` and row37..40. One owner-resolved subject binds the approved
   complete graph, its alternatives and exact allowed GU/DN partners. It remains
   invariant across incumbent/alternative comparison. Equal digests are necessary
   metadata, not proof of approved custody.
2. The **operation-instance SHA256** returned by `mc_bind`/`mc_bindctx`, binding
   one exact selected pair and one retained live allocation/session snapshot.
   It is consumption-time metadata, never a replacement for q18, workload,
   policy, applicability or qualification identity. New allocations can change
   it without rewriting immutable qualification records.

All embedded identity groups are independently frozen inputs. None is defined
as a hash of this instance wire or of a receipt containing its instance digest.
Terminal result custody is outside the preimage. In particular applicability
and qualification-context groups must not be made recursively self-referential.

The expected wire belongs to the existing immutable owner-resolved bundle.
The observed wire is independently assembled from actual selected artifacts,
actual canonical Quant/Composer metadata and launch, and retained allocation
owner state under the existing graph/session lock. Copying expected metadata
into observed metadata does not establish observation. Allocation ownership,
directory/tensor coherence, immutable template validation, protected evaluator
execution and authentication remain their existing owners' obligations.

### One canonical wire

Capacity is `uchar16384`. Exact active length is11264 bytes. Integers are
unsigned64, explicitly little-endian. There are no native structs or alternate
encodings. Byte0..1023 contains scalar128. Byte1024..3071 contains64 raw SHA256
groups. Byte3072..11263 contains128 ordered window rows of eight LE64 cells.
Unused window cells, reserved scalars and bytes11264..16383 must be zero.
Both expected and observed descriptors are validated before exact comparison.
Even two byte-equal malformed descriptors refuse.

| Scalar cell | Meaning and required value |
| --- | --- |
|0,1,2|Version1, domain159, exact active length11264|
|3,4|Complete window count1..128, selected zero-based ordinal less than count|
|5,6,7,8|Total tokens1..128; selected window token0, token count and R|
|9,10|H2048, EI512|
|11,12,13,14,15|Routed expert count256, routed picks8, contributions9, shared expert256, incumbent window capacity128|
|16|Actual router byte stride, u32 range, multiple16, at least144|
|17,18,19,20|Actual session epoch (zero valid), exact admitted profile121, live buffer-owner generation, frozen lifecycle identity (nonzero)|
|21,22|Layer index, tensor/directory owner generation|
|23|Alias policy0; exact permitted lifetimes/aliases also bound by group58|
|24..31|Effects, router, stable ordering, permutation, SwiGLU, combine, numerical, completion revisions: each1|
|32..63|Resources0..7, four cells each as below|
|64..72|Explicit original contribution order0,1,2,3,4,5,6,7,8. Shared slot8 is mandatory, including zero-weight contributions|
|73,74|Full map entry capacity exactly total_tokens*9+8; segment capacity128 including trailing zero descriptors|
|75,76|flush_each0; exact original head8 mode0 or1|
|77,78,79|Layer count positive; last-layer flag exactly `(layer+1==layer_count)`; incumbent combine variant1|
|80..111|Resources8..15, four cells each|
|112..115|GU input mapX1/output mapY2; DN input mapY2/output mapY2|
|116..121|Map padding8; duplicate-preserving sort1, mandatory shared1, barrier-reuse policy1, window-local maps1, original weight-slot policy1|
|122,123|Exact GU and DN final code byte lengths, positive/multiple16/at most16384|
|124..127|Reserved0|

Each resource is `{baseVA, actual_allocation_extent_bytes, owner_generation,
row_or_element_stride_bytes}`. All VAs are nonzero/eight-byte aligned; extents
must cover their required full-operation footprint and not wrap the address
domain; resource generation matches scalar19 except tensor resources14/15
which match scalar22. Ownership is not inferred from alignment or a claimed VA.

|Resource|Role|Stride|
|---|---|---|
|0,1,2,3|Router IDs, original weights, GU directory, DN directory|Actual scalar16|
|4,5|GU/DN Seg32 descriptors|32|
|6,7|X/Y maps|4|
|8|XP activation|H*4|
|9|GU output|9*EI*8|
|10,11|SwiGLU activation and permuted activation|9*EI*4|
|12|DN contribution output|9*H*4|
|13|Original combine output|H*4|
|14,15|Corresponding GU/DN tensor allocations|Actual positive physical stride, interpreted by canonical Quant/directory owner|

The existing `moe_windows` is the only window planner. `mc_windows` compares the
entire table against its exact output, including ragged final rows and byte
offsets; arbitrary contiguous repartitions are not accepted. Each row is
`{token0,tokens,R,XPbytes,GUbytes,ACTbytes,DNbytes,routerbytes}`. Selected scalar6..8
must match that exact ordinal. GROUP uses window-local X=t,Y=t*9+slot; global
token0 is represented by shifted bases, never added twice to map values.
Both GU/DN use M=R and R descriptors including zero-count trailing descriptors.
The existing `moe_alias` independently checks GROUP read/write and write/write
bounding spans at the selected offsets. Broader graph alias/lifetime rules
remain the owner-validated policy bound in group58 and the retained snapshot.

Runtime router ID/weight values are device-produced inputs, **not host readback
or code identity**. The producer/domain/ABI, live allocations/generations and
paired tensor directories are bound. Routed IDs must remain in0..255; duplicate
expert picks retain distinct original slots; shared256/slot8 remains explicit.
The live owner checks table/directory coherence, not this host byte predicate.
Weight contents do not become code identity unless they actually alter emitted
code. Tensor/checkpoint identities remain separate.

### SHA256 group dictionary

Every group is required and nonzero. These are raw32-byte digests, not printed
hex or caller-selected short IDs.

|Groups|Ordered identities|
|---|---|
|0..8|Model, config, checkpoint, exact source, compiler, encoder source, semantic spec, independent evaluator executable/source closure, numerical contract|
|9|Router semantic contract|
|10..21|GROUP code/launch, router code/launch, PERM_XO code/launch, SwiGLU code/launch, PERM_ACT code/launch, combine code/launch|
|22..29|GU canonical1152, final code, launch256, existing full manifest, parameter ABI, immutable cbank template, zero-VA QMD template, canonical Quant384|
|30..37|DN identities in the same order|
|38..43|Workload equivalence, approved joint subject, policy, measurement protocol, GU incumbent canonical artifact, GU applicability|
|44|Independently frozen joint qualification-context/bundle identity|
|45,46,47|GU/DN tensor-directory identities, live allocation-owner snapshot identity|
|48..52|Lifecycle protocol, router ABI/domain, permutation tables, barrier dependency DAG, original scale/head8/last-layer combine protocol|
|53,54|DN incumbent canonical artifact, DN applicability (may differ from GU)|
|55..60|Evaluation protocol, paired transformation lineage, original quant/numerical profile, alias/lifetime policy, completion/recovery policy, complete router allowed-domain contract|
|61,62,63|GU operation semantic digest, DN operation semantic digest, evaluator result-custody protocol|

Group44 is an **input context identity**, not a new trusted-record claim. The
parent alone determines whether an existing independent qualification bundle is
accepted. Host fixture hashes never populate production records. No gate masks,
qualification bits or signed receipts are created or modified by these APIs.

### Integration API

All functions are native seeded Omega, nonconcurrent fixed-array calls. Exported
roots return `long`, supported by existing omega-bridge ABI1; no global scratch.

```
mc_pack(scalars view ulong128, hashes view uchar2048,
        windows view ulong1024, wire edit uchar16384) -> long
mc_valid(wire view uchar16384, n ulong) -> long
mc_bind(expected view uchar16384, observed view uchar16384, n ulong,
        instanceSHA edit uchar32) -> long
mc_context(wire view uchar16384, q view ulong64,
           selectedRow view ulong64, role ulong) -> long
mc_bindctx(expected view uchar16384, observed view uchar16384,
           q view ulong64, selectedRow view ulong64,
           role ulong, instanceSHA edit uchar32) -> long
mc_blob(wire view uchar16384, canonical view uchar4096,
        code view uchar16384, codeLength ulong,
        launch view uchar256, role ulong) -> long
mc_custody(wire view uchar16384, retainedInstanceSHA view uchar32,
           frozenExecutorContext view ulong32,
           observedExecutorContext view ulong32) -> long
```

Role0 is GU; role1 DN. Other roles refuse. `mc_context` requires selected family3,
binds q3..6/row17..20 and six q14/18/22/26/30/34 digest groups against
row33/37/41/45/49/53 and this wire. DN uses groups53/54 for its incumbent and
applicability, not GU42/43. It also binds selected row21..24 artifact,
row25..28 launch, row29..32 bundle and row57..60 final code to the correct
paired role. It does **not** run selection, accept row gate masks as authority,
authenticate records, or make an ineligible row eligible.

`mc_bindctx` is the one combined integration root: clear output, check context,
validate both wires, exact-compare bytes, hash actual active11264 with the
unchanged cache SHA256. Failure clears all32 digest bytes. `mc_pack` is the
only field serializer; failure clears the whole output. Status0 means only
binding succeeded;1 malformed/canonicality,2 unsupported semantic/table
contract,3 resource extent/stride/generation/alias,4 unequal valid wires,
5 owner context/custody mismatch,6 canonical payload binding refused.

`mc_blob` uses unchanged Composer validation/launch validation and unchanged
cache SHA256 over actual canonical1152/codeLength/launch256 bytes. It checks
role-specific shapes, exact code lengths, compiler/encoder/spec and operation
semantic digests, canonical embedded code/launch digests, launch15..26 ABI/cbank/
QMD digests and Quant384. The existing full manifest and actual ABI/template
payload validation remain the artifact owner's admission API, never replaced
by matching claimed hashes. Mapped family3 remains refused by the pinned
Composer; this component does not weaken that gate.

`mc_custody` extends #53's result identity boundary without accepting a claimed
PASS or outcome. The32 executor context cells are version1, evidence class1
HOST_ONLY, instanceSHA4, evaluatorSHA4, sourceSHA4, result-protocolSHA4,
approved-joint-subjectSHA4, qualification-contextSHA4, reserved6zero. All SHA
limbs are explicitly LE64. Frozen and observed contexts must match exactly and
match the validated wire and retained instance SHA. Class2/device claims
refuse even when the two supplied arrays agree. This is not a protected executor
or numerical result evaluator; same-process hostile callers remain outside its
custody guarantee, as for `aobserve`/`aattempt`.

The existing integration owner must select once, resolve by immutable ordinal,
independently collect actual metadata/payloads, validate actual manifest/ABI/
templates and paired graph subject, call the byte/context predicates, and retain
that exact selection and snapshot through existing record/submit/completion
leases. No second shape-based lookup or independently selected DN is permitted.
Do not copy the instance hash into q/row identities or overwrite accepted
qualification history. Production record count remains zero until real
independent accepted hardware/full-workload records exist.

### Host controls and remaining gates

`tests/learn53/moe-fixtures.om` is explicitly synthetic HOST_ONLY metadata.
`moe-unit.om` checks every scalar and SHA role, every window cell, reserved
capacity, invalid length, identical malformed wires, GU/DN role mixing, all
selector identity limbs, session context, stale digest clearing, serializer
roundtrip/refusal clearing and executor-custody substitution. Independent literal
one-token and ragged two-window oracles are not a second planner. `moe-export.om`
emits the one-token wire/hash for mode0/1, or the15-token two-window wire/hash
for mode2/3. Read-only external `sha256sum` independently checks both results.

The mounted Wolfram evaluator verified the source-derived layout, without
warnings. These are synthetic layout calculations, not measurements:

```
WolframLanguageEvaluator:
{Expand[32*(9*t)],9*n+8,3072+8*i,128*8+64*32+128*8*8}
-> {288*t,8+9*n,3072+8*i,11264}

WolframLanguageEvaluator:
({#[[1]],#[[2]],9 #[[2]],#[[1]]2048 4,#[[1]]9 512 8,
  #[[1]]9 512 4,#[[1]]9 2048 4,#[[1]]256}& /@ {{0,14},{14,1}})
-> {{0,14,126,0,0,0,0,0},{14,1,9,114688,516096,258048,1032192,3584}}
```

`tests/learn53/moe-host-evidence.json` records the actual native controls,
external SHA oracle, verifier mutants, compiler/source identities and limits.
The three verifier mutants remove exact wire comparison, result-custody
comparison and selected paired-artifact comparison respectively; each must
compile successfully and make the unit executable fail.

To construct either native test: export the exact Git objects listed in
`moe-closure.lock`, verify each source SHA, concatenate in listed order followed
by `native/learn53/moe-closure.om`, `tests/learn53/moe-fixtures.om`, then
`moe-unit.om` or `moe-export.om`. Compile with the lock's independently pinned
compiler, never execute failed/stale output. No production build/runtime,
source registry or new host-language implementation is introduced.

Actual protected evaluator execution, independent finite-FP/numerical/model
quality results, measured full-cost comparison, accepted hardware receipts,
full consuming graph qualification and GPU execution are **NOT_RUN** here.
Matching byte metadata is not a hardware-qualified receipt or graph authority.
