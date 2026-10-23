# Speculation component contract v1

All quantities are unsigned 64-bit integers. Time is microseconds. Probabilities
are unconditional accepted-prefix survival in millionths, monotonically
nonincreasing with depth. There is no floating-point input or nonfinite value;
out-of-range encodings are rejected. Converting an external clock/probability
representation is the caller's responsibility, with its precision disclosed.

## Identity and calibration

`sid[0..31]` contains eight SHA-256 identities, four little-endian words each:
weights; tokenizer; numerical/sampling recipe; CUDA++ backend/library build;
Omega/AIENOS executable build; drafter; candidate generator; experiment and
measurement contract. `sid[32..63]` contains the independently expected values.
Each digest must be nonzero. These are equality checks, not a hash implementation
or authenticated discovery. The existing owner must provide real identities.

`sprofile` contains version (1), nonzero epoch, inclusive context lower bound,
exclusive context ceiling, evidence class (1 synthetic, 2 declared device), and
shape kind (1 chain). The context ceiling is at most 131072. An epoch change
invalidates learned state. Tree verification is explicitly unsupported.

`srows` has seven rows of eleven words. Candidate draft lengths are
0, 1, 2, 4, 8, 16, 31. A row contains:

| Offset | Meaning |
| --- | --- |
| 0 | Supported: 0 or 1. Unsupported rows have no measured cost claim |
| 1 | Draft length |
| 2 | Verification rows, including anchor: draft length + 1 |
| 3..8 | Draft, verify, commit/rollback, host, policy, preparation microseconds |
| 9 | Peak memory bytes required by this shape |
| 10 | Calibration sample count, 5 through 1,000,000 |

Ordinary row zero must be supported with zero draft time. Every enabled row
requires positive verify time and memory; individual phase costs are at most
1,000,000,000 us. Zero in other phases means measured zero, never unknown.
The collector requires an explicit complete phase mask. Total row cost includes
all six phases; cross products are bounded below the unsigned 64-bit limit.

`sprior[31]` is the frozen development estimate. `sbegin(0)` validates and starts
a cold request. `sbegin(1)` retains learned costs/survival only for the same
identity and epoch, resets request counts and probe scheduling, and requires an
already initialized instance. No hidden shared learning is allowed: one instance
per caller-controlled session. The warm flag must be recorded in trial data.

## Selection and observation

Set `scontrol[0]` to require declared device evidence (1), or permit synthetic
diagnostics (0). `scontrol[1]` is sampling mode; zero means greedy and any other
value explicitly selects ordinary decoding.

`sselect(context, remaining, memory, epoch, mode, fixed)` takes six values. Mode
0 is ordinary, 1 is fixed, 2 is adaptive; `fixed` is a draft length. At most
256 output tokens are admitted per request. Candidate rows must fit remaining
output, context and memory. Unsupported fixed lengths are refused rather than
silently changed. Ties retain the smaller length, including ordinary.

The decision's five words are draft length, verification rows, expected output
millionths, predicted whole-round us, and reason. Reasons: 0 predicted gain,
1 explicit mode, 2 unsupported sampling, 3 ordinary fallback, 4 recovery probe.
For a chain of length k, expected output is 1 plus the first k survival values.
This requires the existing verifier's anchor/correction/bonus convention.
Predicted ordinary-normalized gain is derivable from these values and row zero;
it is not an observed speedup.

`sobserve(accepted, emitted, elapsed, eos, complete)` accepts only a completed
round. Accepted is at most k. Normally emitted is accepted+1; EOS may truncate
it. The full elapsed cost must be positive and at most 6,000,000,000 us. This
cost, including adaptation overhead, replaces the last observed cost for that
shape. Prefix survival uses a 7/8 previous + 1/8 observed update, clamped
monotonically. Only completed past rounds teach the policy. Frozen development
costs guide probes; no future fixture result is inspected.

`sstate[6]` counts emitted tokens, `[7]` consumed tokens, `[8]` an emitted but
unconsumed correction/bonus. `sresume()` acknowledges that the actual state owner
consumed that token. Selection is refused until that acknowledgement. Context
and remaining-budget checks follow this boundary. EOS/output exhaustion ends
the request. `scancel()` ends the request without learning or accepting output.
It does not roll back device memory. The caller must use the existing complete
KV, recurrent, convolution and drafter-state owner. These counters cannot prove
that physical state was committed correctly.

## Observation collector

`cstart()` freezes identities, epoch and evidence class and clears collection.
Only supported-shape descriptors are needed; costs may initially be unknown.
Each `cadd()` consumes `cobs[16]`: row index, zero-based ordinal within row,
complete flag, outcome (0 success), six phase durations, full wall duration,
peak memory, verification rows, epoch, evidence class, phase-known mask (63).
Only nonoverlapping phase times may be supplied. Wall duration must cover their
sum and be at most 1,000,000,000 us. Residual wall time is attributed to host
overhead. Duplicate/out-of-order/incomplete/stale observations are refused.

At most 32 observations per shape are retained verbatim in `craw`, row-major,
with 32 records of 16 words per shape. `ccount` is the count per row; `csum`
keeps six phase totals plus wall total; `crange` keeps min/max wall and peak
memory. `cseal()` needs at least five samples for every enabled row and no
active round. It publishes conservatively rounded-up mean phases and peak
memory, then invalidates prior request state. Keep the raw samples and ranges;
a mean alone is insufficient evidence of a hardware cost step.

## Paired report

Each group is one fixture at one initial context (1024 or 32768 tokens) and one
controller cold/warm mode. `pconfig` holds context, warm flag, fixture ID,
minimum usable output length and require-device flag. `parmids` contains the
32 expected identity words for each arm: base, ordinary, fixed, adaptive.
Weights/tokenizer/numerical recipe/protocol must match across all arms. The base
may have a different backend/executable; ordinary/fixed/adaptive share those.
Fixed/adaptive additionally share drafter and candidate generator. `pstart(n)`
freezes these identities and config, for 5 through 32 paired repetitions.

`pobs` has 56 words:

| Offset | Meaning |
| --- | --- |
| 0..2 | Pair, arm (0..3), fixture ID |
| 3..6 | Prompt-token SHA-256 |
| 7..10 | Committed-output-token SHA-256 |
| 11 | Actual generated count, 1..256 |
| 12..14 | Request start, first committed token, request end us |
| 15 | Peak memory bytes |
| 16 | Stop reason: 1 natural EOS, 2 maximum 256 reached |
| 17..19 | Correctness checked (1), complete (1), contamination (0) |
| 20..23 | Profile epoch, cold/warm flag, trial ordinal, initial context |
| 24..55 | Actual arm identity words |

Order rotates the four arms by pair index; exact ordinals are required. Every
attempt, including invalid ones, is retained in `praw` (at most 256). Invalid
attempts poison the group; an accepted-looking retry cannot erase them. Retain
the original group and explain any new campaign separately. Prompt identity is
fixed across pairs. Within each pair all arms must have identical greedy token
digest, length and stop reason. Divergence blocks performance interpretation.
The runtime must supply first-divergence logits/state separately; digests cannot
explain a numerical error.

`preport(0|1|2)` compares adaptive with base/ordinary/fixed. `summary` contains:
pair count, comparator, median paired speed ratio in millionths, lower/upper
95% interval for the population median ratio, adaptive median/p95 full request
time, adaptive median/p95 TTFT, adaptive peak memory, verdict, evidence class.
Output counts agree within pairs, so the time ratio equals the committed-token
throughput ratio. Ratios are rounded down to millionths. Percentiles use nearest
rank; an even-count median averages its two central values.

The interval uses [exact binomial order statistics](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/mediancl.htm): choose the largest k whose
lower-tail probability is at most 1/40, then use sorted ratios k and n-k+1.
It assumes independent paired repetitions. Five pairs cannot give a finite
distribution-free 95% interval, so they produce an inconclusive result with
unbounded upper endpoint. More repeats improve resolution; noise is retained.
Verdicts: 0 inconclusive, 1 improvement (entire interval above 1), 2 no detectable
improvement (both bounds exactly 1 at the stated resolution), 3 regression
(entire interval below 1). A too-short EOS output makes the result inconclusive.
Synthetic evidence remains synthetic regardless of its numerical verdict.

## Native diagnostic stream

`replay.om` reads one bounded binary message from stdin and emits a binary
decision, sealed table or three reports. It performs no model execution and no
hardware verification. Maximum input is 65536 bytes; words are little-endian u64.

Common input: magic 827867987 (`SCX1`), version 1, operation, reserved 0; then
64 identity, 6 profile, 77 row and 31 prior words. The first operation-specific
word is index 182.

- Operation 1: two `scontrol` words followed by the six selection arguments.
- Operation 2: sample count followed by that many 16-word cost observations.
- Operation 3: five `pconfig` words, pair count, attempt count, 128 `parmids`
  words, then that many 56-word trial observations.

Output header: magic, version, operation, evidence class, payload word count.
Operation 1 returns five decision words. Operation 2 returns 77 row words,
7 counts and 21 range words. Operation 3 returns three 12-word summaries in
base/ordinary/fixed order. No partial successful response is emitted after an
admission error. Keep the original input alongside the output as the raw record.
Statuses: 0 success, 65 malformed/range, 69 identity/version/correctness,
74 I/O, 75 incomplete/order/state, 77 unsupported/capacity.
