# Native numerical contract v1

Contract ID: `cudapp.numeric.v1`. Trace schema: `cudapp.trace.v1`.
This is an opt-in comparison interface. It changes no kernel, arithmetic,
ordering, public graph ABI, or serving default.

## Claims and supported modes

| Mode | Required result | What it establishes |
| --- | --- | --- |
| bitwise | Every finite f32 word, integer, ID and token equal | Equality on the supplied artifacts |
| repeatable | Bitwise equality plus identical source and executable pins | Same implementation replay on the supplied artifacts; #13 still requires chip receipts |
| experimental-tolerance | Explicit finite nonnegative atol/rtol, with justification; every element satisfies abs(candidate-reference) <= atol + rtol*abs(reference) | A bounded experimental comparison only |

Tolerance never excuses token or routing ID disagreement. No default logit
or whole-model tolerance is authorized. Same implementation does not imply
cross-implementation equality. Signed zeros differ in bitwise modes and tie
in argmax. Exact modes reject nonzero budgets. Empty, NaN and Inf arrays
are refused, even if both arrays contain the same exceptional word.
`omega_numeric_compare` returns 0 match, 1 mismatch, -1 invalid.
The host CLI returns 0 on a match and 1 on a refusal or mismatch.

## Operations and boundaries

The existing native APIs remain authoritative for accepted shapes and limits.
This contract does not widen those limits or promise chip behavior beyond
qualified shapes. The original graph is hidden 2048, attention 4096, QKV
5120, eight routed experts, intermediate 768, 1..64 layers and at most
2048*2048 vocabulary entries (`omega_gpu_decode_graph.h`). The separate
Qwen3.6 router supports 256 routed experts and one shared expert; this does
not make the original graph a Qwen3.6 implementation.

| Stage | Representation/order/reference | Boundary/exception rule |
| --- | --- | --- |
| FP8 weight decode | e4m3fn bytes, bf16 128x128 scale grid, f32 accumulation; `omega_fp8_gemv_oracle` uses f64 reference | 0x7f/0xff NaN codes refused on upload; signed zero and e4m3 subnormals decoded by `omega_fp8_e4m3_ref` |
| FP8 staging | Existing scale exponent fold and x*2^64; `omega_fp8_pack_scales` and `omega_fp8_pack_x_f32` | Fold overflow/nonfinite scales refused by existing packer. New admission helper refuses nonfinite x or x*2^64 overflow; raw x packer itself does not refuse overflow |
| FP8 quantization | `omega_gpu_fp8_e4m3_rne`, ties to even, finite saturation at +/-448 | Signed zero preserved; subnormal units 2^-9; helper saturates NaN/Inf too, so contract admission must reject these rather than call this model quality |
| RMSNorm | Graph f32 FFMA reduction and MUFU RSQ; host glue f64 sum and 1/sqrt | Same implementation uses fixed instruction/reduction order. No claim of f64 equality; eps and all intermediates must stay finite |
| SwiGLU | Graph MUFU EX2/RCP; host glue expf | Finite supported range only; no global expf equality or accepted model-level error |
| Residual and expert combine | Fixed expert order, multiply/add ordering of current generated code | Every selected expert contributes, including shared expert on paths that have one; absent expert is failure |
| Attention/KV and lm_head | Existing bf16 conversion, native attention and bf16 GEMV references/tests | Preserve current head/context/shape refusals and reduction order; unsupported precision or scaling refused |
| Router and argmax | Descending logits, lowest index on ties | IDs/tokens exact; nonfinite logits refused; small margin never waives this contract |
| Integer/address/encoded instruction work | Existing golden words and C3 integer/math references | Bitwise only; no floating-point tolerance applied to integers |

IEEE subnormal handling is operation specific. Host f32 comparison preserves
finite subnormals. e4m3 decode/quantization tests cover its subnormal grid.
MUFU FTZ/approximation behavior is covered only by the existing simulator and
chip operation gates, with no blanket gradual-underflow promise. Inputs that
produce nonfinite intermediates are outside the finite trace contract.
Reordering a reduction, changing rounding, omitting a scale or changing
precision selects a different experimental implementation and needs new gates.

## Reused references and adversarial gates

`tests/gpu_cmdlist_test.c --sim` executes the actual generated RES/SwiGLU/
ARGMAX programs, with ADD/COMBINE/NORM, scales 1 and 2^64, tied maxima,
missing-expert and wrong-offset mutants. Its existing local RMSNorm and
SwiGLU tolerances are retained as stage-specific historical tests, not
adopted as whole-model tolerances. `tests/gpu_router_test.c --sim` retains
its f64 oracle and mutants; its historical near-margin reporting is not a
waiver for trace token comparison. C3 math and integer gates remain in the
pinned Omega slice harness (`tests/omega/run.sh`); no replacement reference
is introduced. #15 owns reproducibility of that harness.

`make test-numerical-contract` calls the existing FP8 conversion and packers,
checks every finite e4m3 round trip, signed zero, midpoint rounding and
subnormal boundaries, refusal of overflow/nonfinite admission, scale-fold
negative control and strict/tolerance comparator refusals. Existing graph
and router simulator gates are required alongside these new checks.

## Trace exchange

Run `node tools/numerical-trace.mjs reference.json candidate.json [policy.json]`.
Node 18+ is a host development tool only; deployed native runtime has no Node
or third-party inference dependency. JSON files contain these fields:

- `schema`, `contract`, `kind` (`fixture` or `model`), `precision`
  (`fp8-e4m3-bf16-f32`), `context_limit`, integer `seed`.
- SHA256 strings `model_sha256`, `prompt_sha256`, `config_sha256`,
  `binary_sha256`, and full 40-digit `source_sha`.
- Nonempty `rows`, each with `turn`, `position`, `context`,
  `input_sha256`, exact `token`, nonempty integer `router_ids`, nonempty f32 `logits`, and a nonempty
  `stages` object of named f32 arrays. Export stages actually observed,
  such as residual, hidden, router weights and expert outputs. Stage sets
  and dimensions must agree; `router_ids` records observed expert IDs in layer/slot order and is
  compared exactly in every mode.

Positions are absolute KV positions, context includes the current token.
Within a turn both advance by one; across turns turn advances by one and
context/position increase. Separate reset conversations use separate files.
Include all tested turns and endpoints, especially configured context limit.
The tool validates supplied coverage; it cannot prove a producer exported
all prompts, layers or supported contexts. Fixture tests include continuation
and limit cases. Real whole-token, multi-turn and context acceptance is
`NOT_RUN` until coordinated model traces exist.

Policy example for an experiment: `{"mode":"experimental-tolerance",
"atol":0,"rtol":0,"justification":"Explicit exact-budget experiment"}`.
No tolerance is copied from a paper. Any nonzero budget must cite an
independent arithmetic/error analysis and recorded reference distribution.
The parent independently rechecks calculations. Policy hashes travel in
benchmark receipts. Trace matching reports `TRACE_MATCH_ONLY` for model
artifacts, never model-quality PASS.

## Model-quality gate before arithmetic transformations

Freeze model/config/tokenizer/prompt/dataset hashes, reference source and
binary, decoding/seeds, supported batch/context matrix and exact token
policy before measurement. Require complete-model greedy tokens and staged
logits across single-turn, continuing multi-turn and context endpoints;
run #13 repeatability separately. A transformation that intentionally changes
tokens also needs a preregistered quality dataset, metric, sample count,
noninferiority margin justified from that model's reference variation, and
independent review. Until those artifacts and final GB10 integration receipts
exist, quality is `NOT_RUN`, the original baseline remains selectable and
no transformation becomes a default. Simulator fixtures cannot satisfy this.
