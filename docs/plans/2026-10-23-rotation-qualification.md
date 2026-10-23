# CUDApp #49 remaining qualification — L600, personal execution

## Scope and immutable starting points

Native source/evidence base: c652f1b257cbf36d9e0d2fc1a8e2e7ea0f7f9fa4.
External AIEN host reference: aien-sovereign-core f0230a59eec985c3290ac78c0cc04322bacf9f0a.
Native compiler: omega-bridge d2010ab927b608931800c20d73a4c2b25f00a9fa;
SHA256 191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9.
Checkpoint: qwen3.6-35b-a3b-fp8-95a723d0. Freeze config/tokenizer/header and
consumed tensor hashes; no checkpoint mutation, additional GPU model, live restart,
worker delegation or third-party inference engine.

The new Rust adapter is an explicitly test-only external reference instrument,
not new product computation or a native-runtime claim. Every quantization,
Hadamard, rounding and inverse transform remains the seeded Omega executable.
The reference only transports bytes and widens the returned rational values with
unchanged original source scales for its existing f32 matmuls.

## Arms and fixed bytes

Apply to ALL routed and shared expert gate/up/down FP8 matrices in ALL 40 layers;
trunk, router, embedding, head and f32 state are unchanged. This is not a five-matrix
substitution. No activation-based fitting or post-result arm selection.

- fp8: unchanged exact reference/control.
- identity: existing symmetric nearest-even INT4, block128.
- hadamard: existing natural-order unnormalized H128 then INT4.
- zero-aware: identity for a source block containing ANY numerical zero, Hadamard
  otherwise. Thus all original scalar zeros, rows and columns remain zero.

RQU2/RQP2/RQR2 add the zero-aware mode2; modes0/1 remain identical to v1 except
magic. For mode2, bit31 of the existing u32 maximum is the rotated flag; bits0..30
are M. No added byte or external selector. Modes0/1 refuse flagged maxima.
Zero-aware denominator is always 458752; identity blocks multiply their integer
numerators by128. Same 12-byte packed frame and68 bytes/block in all three arms,
plus the SAME original BF16 scale grid. Original v1 accepted/refused behavior stays
unchanged, including mode2 refusal. Both FP8 NaNs still fail closed.

A separate native stream entry reads consecutive complete frames, maximum999999
transactions. It publishes only a fully validated current frame; failure can leave
previous valid frames, never a partial semantic result for the bad frame. Transport
write failures can leave partial bytes and must invalidate the host transaction.
The original one-frame/EOF entry is unchanged. Client checks framing, exact lengths,
child status and native binary fingerprint; no fallback. No on-disk cache is trusted
without source/binary/format identity. Version1 test suite remains a regression gate.

## Model-quality gates frozen before native-arm model runs

Reuse, do not weaken, L355's pre-existing R1/R1b/R2 policy in
`docs/inference/q36/w4-gate.md` at its pinned source. On five512-step public goldens
(p1,p2,p3,p4,chat-max-24) and five512-step held-out goldens (h1..h5), teacher-force
each arm on identical reference tokens with fresh per-prompt state:

- R1 pooled mean-NLL(candidate)/mean-NLL(fp8) <=1.07 in each group.
- R1b AGPL400-token passage mean-NLL increase <=0.069 nats.
- R2 pooled top1 agreement >=0.98 where reference margin>=0.1, and>=0.998 where
  margin>=1.0, in each group. Report denominator counts and all near-tie flips.
- Exact control must match baseline token ids and margins (absolute tolerance1e-3)
  before candidate results are accepted. Preserve the original control failure.
- A compiled nine-level-grid mutant (nonzero even magnitudes rounded down to the
  adjacent odd magnitude; extremes retained) must be rejected by the same gate.
  If it passes, qualification is inconclusive, not approval.

Also publish full-position logit cosine, KL, per-token NLL, expert-route changes,
real activation/output error and zero-support preservation. A numerical failure
can reject an arm but never permits a quality threshold change or default adoption.
Early diagnostic batches do not substitute for the declared complete totals.

## Implementation sequence and verification

1. Native v2/stream: explicit sign-matrix/Fraction oracle tests RED against v1,
   then native implementation GREEN. All254 finite codes, nearest-even ties,
   max frames, mixed zero/nonzero blocks, malformed flags/maxima/counts, chunked
   input, repeated frames, prefix failure and I/O failures; compiled mutants.
2. Test-only reference adapter: isolated `sc-rotation-quality-l600`, feature-off
   behavior unchanged. Write bridge and full-forward tiny-model tests first;
   semantic RED required, then test-only integration. Pin complete consumed closure.
3. Whole-model evaluation: freeze public corpus hashes; baseline control first;
   serial memory-bounded CPU runs through existing host reference. Verify the
   actual number of expert tensors/frames used and record source/output identities.
4. Native GPU integration/full-cost: reconcile CUDApp#16/#17 and existing native
   lowering owner L408. No duplicate compiler implementation or legacy C/Rust
   product workaround. Quality is prerequisite to a deployment candidate; GPU
   inability is NOT a host PASS. Use admitted hardware windows, same model/sequence/
   batch/clock baseline; account for load/setup, transform, resident/temporary
   memory, actual bytes, prefill, decode and energy. No unmeasured speedup.
5. Personally review all changed source, run complete affected suites, replay exact
   committed native source/reference closure, publish source and raw receipts,
   verify GitHub readback. Leave any unmet #49 acceptance explicitly open.

## Reference-cache execution refinement (quality rules unchanged)

The first 32-position all-layer FP8 control passed; native identity-arm execution
showed the per-prompt repeated reconstruction cost is substantial. Reuse exact
native-derived f32 reference matrices across prompts through a test-only cache.
This expands storage to four bytes/weight and is NOT the packed product format,
a GPU kernel, a memory saving or an inference-speed measurement.

Cache keys bind raw tensor/scales, shape, mode and native executable hash. ORC1
stores u32 JSON length, SHA256(JSON), JSON metadata and little-endian f32 values;
metadata separately binds the f32 payload SHA256 and native packed-frame receipt.
Reject corruption, wrong extents, metadata mismatch, symlinks and hardlinks; publish
via no-clobber private file transaction. One precompute writer per (arm,layer),
then read-only model evaluation. This is not hostile concurrent-filesystem security.
Independent tests cover exact bits (including signed zero), native-to-matmul cached
reuse, altered payload/receipt, wrong size, linked entries and victim preservation.

Eight disjoint layer partitions may precompute in parallel on CPU only; no model
is loaded on the GPU. Check free disk with a100GiB remaining floor first. Native
processes, numerical reference processes and an orchestrator are not agents;
all implementation and review remains personal. Full quality totals/thresholds
and all previous failures are retained. Do not treat the short diagnostic run,
metadata sizing, a cache or a launched batch as completion.

## Durable rulings

- User's personal-only instruction overrides delegated implementation/review.
- Existing reference code may be instrumented only as a named external oracle;
  it does not satisfy native inference migration or direct packed GPU execution.
- The fixed zero-aware selector trades potential rotation gains for exact zero
  support at the identical byte budget. Its value is determined by held-out gates.
- Existing unqualified native GPU work is a dependency, not permission to launch
  emitted lab-only artifacts or seize the live device.
