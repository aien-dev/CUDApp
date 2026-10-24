# Native Omega compute launch

`native/blackwell/vector.om` emits the existing seeded SM121 integer vector
program in handwritten Omega. Its 512 bytes match Omega `57f5f425`'s
`seed/sm121/vecadd.bin` exactly. It adds two uint32 arrays, with wraparound,
and exits out-of-range threads before memory access.

`native/blackwell/launch.om` continues CUDApp `349913f1`'s VECTOR_1D launch
contract over the native NVRM connection. It builds the driver constants,
two launch descriptors, argument upload, completion markers and final cache
flush. The host waits for completion before reading every result. No C or
Rust runtime, foreign assembler, CUDA library or lab-only capsule is used.

The current interface is deliberately one bounded vector operation, with
64 threads per block and at most 1,048,576 elements. It is not a generic
unvalidated program loader. There is one context and one outstanding packet.
The caller must close the context after a failed open or launch.

The physical test passes sizes 1, 63, 64, 65, 257, 4097, 65537 and 1048576,
compares every uint32 result, and checks up to 64 words beyond the output.
A fresh program allocation that omits one addend must produce a mismatch.
A subsequent fresh correct program must pass again. Invalid zero and
oversized requests are refused, and final context cleanup succeeds.

Host checks compare the emitted program byte for byte with the seed artifact.
For five address/size fixtures, all 2,115 driver/descriptor/argument words
match an independently compiled copy of the pinned C QMD builder. That C
program is a test-only oracle and is never part of the native execution path.

The first physical attempt exited 104 when the test's dynamic CPU loop
exceeded the seeded one-million-iteration guard. A CPU-only reproduction
passed at 1,000,000 and failed at 1,000,001. The shared checker now processes
at most 65,536 values per loop. Its host regression checks 999,999, 1,000,000,
1,000,001 and 1,048,576 elements plus a one-element case. The guard remains
unchanged. The initial failed attempt is retained privately and is not a pass.

Build the GPU test by concatenating `context.om`, `vector.om`, `launch.om`,
the declarations and `ck` function before `fn trial` in
`native-kernel-test.om`, `vector-checks.om`, and the remainder of the kernel
test. Build the CPU regression from the declarations and `ck` before
`fn supply` in `vector-checks-test.om`, `vector-checks.om`, then its remainder.
Compile with the pinned seeded compiler; require successful compilation
before execution. Hardware tests require a confirmed admitted window.

The evidence is a shared-window correctness result, with no performance
claim. FP8 arithmetic, the complete 27B model, speculative-state recovery,
AIENOS attachment and the four-arm comparison remain unfinished. Issues
63 through 66 remain open. Independent final-source review is still pending.
