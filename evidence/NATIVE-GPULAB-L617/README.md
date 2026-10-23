# Native AMD/AGX lab qualification

Functional source: `ba3c4e9e2fd64ff88c926b6cb075eb438092c9d1`.
The subsequent publication commit adds receipts only.

## Final clean-source replay

A fresh `git archive` of the functional source was extracted under
`~/handoffs/native-gpulab-L617/replay-ba3c4e9e2fd64ff88c926b6cb075eb438092c9d1` on the Spark.
From that export:

```sh
quiet-build make -f native/gpulab/Makefile test \
  OMEGA_NATIVE_COMPILER=$HOME/handoffs/quantum-development-L600/toolchain/build/bootstrap/omc
sha256sum -c evidence/NATIVE-GPULAB-L617/source-files.sha256
sha256sum -c evidence/NATIVE-GPULAB-L617/native-binaries.sha256
```

All three commands exited 0. Exact logs: `final-replay.log`,
`final-sources.log`, `final-binaries.log`. Those three files are the newly
produced replay records; the export's inherited historical evidence is not
relabeled as fresh execution.

| Gate | Actual result |
|---|---|
| AMD gfx1100 native code generator and virtual execution | 8,224 checks, zero failures |
| Apple G13/G14 native code generator and virtual execution | 8,218 checks, zero failures |
| Cross-target arithmetic, malformed code and transactional outputs | 262,930 checks, zero failures |
| Request parser | PASS |
| Standalone emitter to virtual executor | PASS for both targets |
| Complete AMD depth-64 shader vs independent LLVM assembly | Every byte equal |
| Frozen source hashes | PASS |
| Rebuilt native binary hashes | PASS |

`transport-controls.log` records real process tests: valid targets exit0 with
148/218 bytes; malformed selectors 3..8 exit65 with zero bytes; empty request
exit65; same-length mutated LLVM bytes cause comparator exit3; mutated AMD and
Apple wire programs each cause exit65. The mutator is itself Omega.

Late-lane controls explicitly allow lanes0/1 to complete, fault lane2 and require
all public output slots to remain unchanged. Their independently calculated
instruction offsets are in `WOLFRAM.md`.

## Historical failures retained

`amd-red.log`, `agx-red.log`, `cross-red.log` are outputs of successfully compiled
refusal baselines, each executed with exit1. `request-red-test.log` is empty
because that baseline exits1 without printing. The final implementations exit0.
Early compiler setup failures and two failed full-build attempts are not runtime
mutation kills. `full-first.log` and `full-second.log` retain those build failures;
`full-third.log` is the earlier passing suite. `cross-first.log` used the earlier
262,928-check harness; the final late-lane extension adds two checks.

## Scope and remaining qualification

Every first-party algorithm, generator, virtual interpreter, fixture, scalar
reference, byte comparator and test program is handwritten Omega. Make/Bash and
host utilities remain build/process orchestration. LLVM is an external encoding
oracle and Wolfram an external arithmetic oracle, not production dependencies.

This is a bounded integer GEMV lab, not a complete inference engine. No real AMD
or Apple GPU execution, hardware scheduling/coherency qualification, inference
throughput, or speedup is claimed. No drivers or running services were changed.
Apple's complete generated shader has no real-chip receipt yet. Full AMD byte
agreement with LLVM is stronger independent encoding coverage, not hardware
execution. See `native/gpulab/README.md` for exact ABI and limits.

Implementation was personal, without agents. Review is author review only;
independent acceptance and integration remain with the desk. Publication of this
lab must not close the broader real-hardware or full-inference work.
