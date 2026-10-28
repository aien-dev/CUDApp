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

## Joint resident MoE operation-instance binding v2

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
|0,1,2|Version2, domain159, exact active length11264|
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
|10..21|Ordered code/launch **series** for GROUP, router, PERM_XO, SwiGLU, PERM_ACT and combine; mandatory even for one entry|
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

Version2 replaces direct entry hashes in groups10..21 with complete ordered
role-series hashes. Version1 is refused; there is no direct-leaf compatibility
path. Nested preimages retain their own version1/domain159 subtype tags.

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

### Canonical nested live-observation preimages

The following pure Omega roots define groups10..21,45..47 in this same owner.
These are **live-instance** identities. They are not Composer launch256
identities, synthesized parameter ABIs, zero-VA templates, qualification
subjects, or new authority. No existing incumbent-glue address normalizer was
found: admitted native templates remain exact admitted bytes under their
existing owner. Live code, bank, QMD, directory and allocation VAs stay live
here. Never zero pointers or hash weight contents to manufacture code identity.
All input arrays and each output32 must be disjoint.

```
mc_code(code view uchar16384, n ulong, out edit uchar32) -> long
mc_launch(meta view ulong32, bank view uchar1024,
          qmdslot view uchar1024, submit view ulong16,
          out edit uchar32) -> long
mc_dir(meta view ulong8, entries view ulong1024, out edit uchar32) -> long
mc_alloc(meta view ulong8, records view ulong1024, out edit uchar32) -> long
```

Every refusal clears output32. `mc_code` requires positive exact encoded
length, multiple16 and at most16384. It hashes exactly that prefix through
the existing `kc_codesha`, excluding the allocation's instruction-prefetch
tail. This bound follows existing `BW_MAX_IR_INSNS=1024`; exceeding it refuses
instead of truncating. Actual code residency/VA-to-byte membership is the
existing allocation owner's responsibility.

`mc_launch` binds the **already emitted command-list entry**, not requested
launch fields or a reimplementation of QMD/default policy:

|Meta cell|Actual retained observation|
|---|---|
|0,1,2|Version1, domain159, live-entry subtype1|
|3|Kernel role0 GROUP,1 router,2 PERM_XO,3 SwiGLU,4 PERM_ACT,5 combine|
|4,5|Active argument-word count1..32; exact encoded code byte length|
|6,7|Entry ordinal; positive command-list record generation|
|8,9|Actual independent flag0/1; actual pending-wait VA0 or eight-byte aligned|
|10|Actual resident code VA, nonzero/16-byte aligned; must equal PROGRAM_ADDRESS decoded from actual QMD1 words32/33|
|11..31|Reserved0|

|Submit cell|Actual consumed submission configuration|
|---|---|
|0,1,2|Version1, domain159, submission subtype4|
|3|Positive record generation, equal launch meta7|
|4,5|Mode0 PCAS/1 INLINE; barrier enum0..4|
|6,7|Resolved timeout in ms, positive; resolved marker spin in us, at most1000000|
|8,9,10|Stamps0/1; skip-launch exactly0; first-barrier-none0/1|
|11|Final recorded launch count1..4096; launch ordinal must be below it|
|12..15|Reserved0|

Canonical preimage order is all32 meta cells as LE64, all16 submit cells as
LE64, actual bank bytes `[0,896+4*nargs)`, actual QMD1 slot bytes `[0,384)`,
then actual QMD0 slot bytes `[512,896)`. This includes the existing224-word
driver bank and actual32-bit argument words; it cannot truncate widened
argument values because it consumes their actual retained bytes. Bank
padding after active arguments and QMD slot gaps are deliberately excluded.
Effective GPR/thread dimensions/shared allocation/grid/code/context pointers
are the actual emitted QMD bytes, not independently recomputed rounded
requested values. These raw snapshots do not independently validate a QMD;
the existing command-list builder/validator remains its owner.

The snapshot must contain the **mode-final representation actually emitted**,
not an earlier retained QMD. The existing emitter patches QMD1 words10/12:
PCAS zeros them, whereas INLINE installs its dependent-QMD fields in a local
copy. In particular, retained storage after PCAS is not the later INLINE
representation. The consuming owner must prepare/capture those same final
bytes before binding and must consume them unchanged. Observe every entry and
every chunk before the **first** queue commit; a refusal after an earlier
chunk was submitted is not an unsubmitted-list refusal.

This preimage binds recorded independence, pending wait, consumed options and
entry QMD/bank bytes. It does not alone bind the complete effective method
stream or chunk schedule: effective barriers also depend on ring capacity,
platform prefix and chunk boundaries. Group51 remains the owning dependency-DAG
contract; callers must not inflate this helper into effective-barrier evidence.

`OmegaGpuLaunch.timeout_ms` and `.spin_us` are ignored by command-list add.
The live digest instead binds the **same resolved submission values actually
consumed**. Observing a global spin default separately, even under the session
lock, is not enough: its original setter did not take that lock. The consuming
owner must capture once and consume unchanged, or invoke binding at the exact
consumption point with the already resolved values. Actual final-list
observation/binding follows recording but precedes any queue commit; refusal
aborts the unsubmitted list without incumbent fallback. This is not terminal
completion evidence, a post-submit receipt, or permission to submit.

`mc_dir` metadata is `{1,159,2,role,count,owner_generation,layer,0}`.
Role0 GU/1 DN, count exactly257, owner generation positive and layer a u32
including zero. Entries0..513 contain all257 ordered `{W_VA,S_VA}` pairs;
each VA is positive/eight-byte aligned. Entries514..1023 must be zero.
SHA256 covers the8 LE64 metadata cells followed by the514 LE64 address cells,
exactly4176 bytes. No weight/scale data or invented contiguous W+S envelope
is included. The existing directory owner supplies and retains this complete
actual snapshot, including mutations through its existing setter.

`mc_alloc` metadata is
`{1,159,3,active_count,graph_owner_gen,tensor_owner_gen,lifecycle_gen,0}`.
Count is1..64; the three owner generations are positive. There are64 records,
each16 LE64 cells:

```
{role,index,handle_generation,kind,allocation_VA,physical_bytes,
 resource_VA,available_bytes,owner_generation,0,0,0,0,0,0,0}
```

Active roles are strictly ascending and below64; role coverage belongs to
the complete operation observer, not a generic invented required-role list.
Index0 is valid; index/generation are u32 and generation is nonzero. Kind2
means data buffer,3 code buffer,4 queue/push **buffer**, never a submission
queue/channel handle. All VAs are positive/eight-byte aligned, extents
positive/nonwrapping; each resource is a real contained allocation subspan.
The same buffer index cannot claim differing generations, kinds, backing
VAs or physical extents. Multiple roles/subranges sharing the same handle
are legitimate and preserved. Unused record cells are zero. SHA256 covers
the8 metadata cells and all1024 record cells as LE64, exactly8256 bytes.

Resource-role generations in the outer wire remain owner-lease generations,
not individual handle generations. This snapshot separately binds the latter.
The observer must cover actual separate W/S allocations, enclosing buffers,
router/glue/leased code and command-list-owned buffers where applicable.
Unused graph buffers are not invented reads. These structural checks do not
attest registry membership, current ownership or lifetime: the existing owner
retains that responsibility under the same observed lease/snapshot.

### Complete ordered code and launch series

```
mc_series(meta view ulong8, leafSHAs view uchar4096,
          ordinals view ulong128, out edit uchar32) -> long
```

This is the only aggregation format for groups10..21, including singleton
roles. Metadata is `{1,159,5,role,kind,count,record_generation,full_count}`.
Roles0..5 have the same order as `mc_launch`; kind0 is encoded-code identity,
kind1 is launch identity; count is1..128. This covers the complete existing
operation bound, not a new64-token limit. Router recording can require multiple
64-row entries, and combine can require a separate entry for every token.
The observer must cover **all** such occurrences; it may not pick first/last,
deduplicate identical code, or truncate. More than128 occurrences is refused.
The full command list can still have4096 entries and actual ordinal4095.

Code metadata requires generation/full_count both0 and all ordinals0. Its
preimage is all8 metadata cells as LE64 followed by all4096 leaf-digest bytes.
Code occurrences retain their actual execution order and count, but do not
acquire runtime addresses, generations or weight data as encoded-code identity.
Launch metadata requires positive actual record generation and full_count
1..4096. Active ordinals must be strictly increasing and below full_count;
inactive ordinals must be0. Its preimage is all8 metadata cells as LE64, all128
ordinals as LE64, then all4096 digest bytes. Each active32-byte digest must be
nonzero and every inactive digest byte must be0 for both kinds. Refusal clears
out; status1 is malformed metadata/reserved data,3 is a zero active digest,
6 is hashing refusal.

The owner gathers both arrays in **one actual-entry traversal**. Code slot
`i` and launch slot `i` must refer to the same actual entry, in ascending actual
ordinal order, with identical series role and count. A launch leaf's role,
ordinal, generation and final-list count must agree with its series metadata
and ordinal. Each code leaf comes from `mc_code` over the actual exact code
bytes; each launch leaf comes from `mc_launch` over that same observed entry.
Code series goes to group`10+2*role`, launch series to`11+2*role`. These
correspondence/provenance obligations cannot be recovered from opaque digests
by `mc_series`, nor does hashing two supplied lists establish them. A valid
leaf from another ordinal/generation must be refused by the observer before
aggregation. No caller qualification flag substitutes for these checks.

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

`tests/learn53/moe-host-evidence.json` remains the immutable historical
Version1 source receipt at commit32af495. The separate
`moe-observe-host-evidence.json` records the Version2 replay, nested preimages,
ordered-series controls, independent byte/SHA oracles and verifier mutants.
The original three mutants removed exact wire, result-custody and selected
paired-artifact comparison. New controls challenge both QMD regions, consumed
options, all257 W/S pairs, buffer aliases/backing, and every series digest/
ordinal cell, including128 occurrences and ordinal4095. These are synthetic
HOST_ONLY tests, not snapshots of actual recorded or submitted GPU work.

To construct a native test, export exact Git objects listed in
`moe-closure.lock`, verify each source SHA and concatenate in listed order,
followed by `native/learn53/moe-closure.om`. For the outer closure, append
`moe-fixtures.om` then `moe-unit.om` or `moe-export.om`. For nested observations,
append `moe-observe-fixtures.om` then `moe-observe-unit.om` or
`moe-observe-export.om`; for ordered series use that same observation fixture
followed by `moe-series-unit.om` or `moe-series-export.om`. Paths are relative to
`tests/learn53/`. Compile with the lock's independently pinned compiler; never
execute failed/stale output. No production build/runtime, source registry or
new host-language implementation is introduced.

Actual protected evaluator execution, independent finite-FP/numerical/model
quality results, measured full-cost comparison, accepted hardware receipts,
full consuming graph qualification and GPU execution are **NOT_RUN** here.
Matching byte metadata is not a hardware-qualified receipt or graph authority.
