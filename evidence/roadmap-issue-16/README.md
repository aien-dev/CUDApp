# Numerical contract host receipt

Executable source pin is in `source-sha.txt`; baseline dependency is
349913f17c083996680c72b32f767b225005155d (canonical candidate carries preserved).
Public main was read-only checked at 232332422bfdc31c8918085b50f13c1e18ab702e.
This evidence-only commit does not identify itself as the executable source.
`input-sha256.txt` pins tracked source/test/build/tool/docs inputs;
`binary-sha256.txt` pins the built library, tests and mutant binaries.
`host-tools.txt` records host/compiler/tool identity and executable hashes.

Commands and raw results:

- RED: `quiet-build timeout 180 make OUT_DIR=build-roadmap-contracts-bench test-numerical-contract`, before the API existed: exit 2 (`red.log`).
- Source target progression is preserved in `target*.log`. Final target:
  `quiet-build timeout 90 make OUT_DIR=build-roadmap-contracts-bench test-numerical-contract`, exit 0 (`target-final.log`).
- Full diagnostic suite: `quiet-build timeout 900 make -j2 OUT_DIR=build-roadmap-contracts-bench test`, exit 2 (`host-suite.log`). The portability table lacked the two new API rows; both were added. This pre-final diagnostic tree is not source-qualified PASS evidence.
- An overlapping full-suite retry was terminated only in this lane's process tree, exit 143 (`host-suite-final.log`). No other owner's processes were killed.
- Final full suite used the source pin's unchanged runtime/test C inputs, same full command, exit 0 (`host-suite-green.log`). It began after the other common builds drained; another worker subsequently began a common build. All output is correctness/simulator evidence, with no timing/performance claim. No timing experiment occurred.

New checks: 285 API cases, 236 production-generated tied-router simulator
checks (128 routed and 256-plus-shared, both fragment orders, scaling and
mask/directory controls), 19 trace fixture tests. The full suite preserves all
33 existing host targets, stage references, thresholds, refusals and mutants.
The isolated wrong-rounding and reversed-router-tie source copies exit nonzero
with the intended assertion in `wrong-rounding.log` and `wrong-tie.log`;
make requires those assertion lines and rejects a surviving mutant. Existing
RES/SwiGLU/ARGMAX and expert-omission checks execute the actual generated code.

JSON trace values and seed/config IDs are fixture inputs in comparator tests.
Fixture matching is not model validation. Root Makefile, graph public ABI,
existing arithmetic and serving defaults were unchanged. #15 owns its separate
host-portability foundation; this AArch64 run uses the carried canonical code
and does not overwrite that worker's changes. Parent should integrate #15
alongside these branches and requalify the resulting artifact.

See `matrix.json` for separate host/simulator/hardware/model/integration rows.
Real model, prompt/config hashes, model binary, loaded driver/chip identity,
whole-model/multiturn/context quality, #13 chip repeatability and #5/#6 final
integration are NOT_RUN. No live model request, GPU operation, server start,
restart, driver change, or timing experiment was performed. Printed host-test
or simulator durations are not benchmark claims. No tolerance was imported
from literature or promoted to a global model acceptance policy.

Provenance clarification: the full numerical make process loaded rules before
the additional reversed-tie mutation recipe was added. Runtime and test C
inputs are identical between a0ef918c3b2da52aa7c68387a42ebd7700bc1edb and
3565287250e4ebd40e79a23d978edd3c936c94f2; only documentation and the extra
mutation recipe changed. The final target run exercises that recipe at the
final numerical source. The child benchmark full-suite run at executable
source 4fdc3dabc5a0243ad93b6106e76497d0ce428d8a also passes all numerical
and benchmark gates together. Its unchanged-runtime proof and raw output
are under evidence/roadmap-issue-17/. This clarification changes no inputs.
