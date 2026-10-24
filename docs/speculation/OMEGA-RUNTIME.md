# Strict Omega runtime prerequisites

The user confirmed strict Omega and authorized extending the existing Omega
compiler/runtime to make this experiment possible. A Rust bridge is not part of
the accepted implementation. This is a component dependency record, not a new
whole-system execution plan.

[Omega PR676](https://github.com/aien-dev/omega-mv/pull/676) extends the existing
native-host compiler with checked device calls and managed file/memory mappings.
Its source d80b428c reconstructs through the original seed chain. The existing
L605 device request rule is reused. L408's original compiler tree, L605's GPU
launcher ownership, L457's canonical backend and L789's compiler correction are
preserved. The new branch is proposed work, not current merged runtime behavior.

| Required connection | Current evidence |
| --- | --- |
| Policy process, pipes, clocks and lifecycle | Existing service component passes337 native checks. |
| Managed native memory and checked device calls | PR676 host tests pass, including stale handles, bounds, rights, exhaustion and compiled controls. |
| Reading actual model files | Native read-only probe passes2,442 mapped/file comparisons over66 shards. This proves access only. |
| Native GPU connection | The canonical NVRM lifecycle is continued in `native/nvrm/context.om`. Private RM/UVM contexts, low-address buffers, empty channels and2,052 FIFO completion writes pass physical probes. See [native device evidence](NATIVE-DEVICE.md). SM kernels, model execution and full runtime qualification remain open. |
| Complete 27B model and speculative state | Existing historical Rust integration remains a behavioral reference. Its numeric transforms, kernel selection, MTP, cache/recurrent state, rollback and correction-token consumption must be attached through the native path. |
| Final compiler qualification | The existing branch-range correction is integrated in Omega078455c9. Its native gate and full host qualification pass; all seven experiment executables reproduce byte-identically. Final independent source review remains required. |
| Four-arm experiment and issue closure | Requires the complete path, full-state correctness, calibration and held-out physical trials. Still NOT_RUN. |

The file probe does not substitute for model execution. Historical fixed-chain
whole-model runs do not qualify this new integration. Do not close #63-66 or
claim adaptive speedup from these prerequisites. Source and evidence for this
step are in `evidence/speculation/model-mapping` and the Omega PR.
