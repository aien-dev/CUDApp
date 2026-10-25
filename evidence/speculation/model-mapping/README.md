# Real weight-file access from native Omega

Test source: `743416c8a78a148ab4c9f5e06238725a683c5d52`.
Compiler source: Omega `d80b428c6ed9e3864fd7fc02cd7392bd4ad67f56`,
proposed in [Omega PR676](https://github.com/aien-dev/omega-mv/pull/676).

The native Omega executable opened all66 shards referenced by the existing local
27B index. Each shard passed37 comparisons between mapped bytes and independent
file reads: the length prefix, header edges, first/last blocks and32 distributed
blocks. All2,442 comparisons passed. File identity, size and modification metadata
were checked before and after each shard. Each read-only mapping and descriptor
was released. Private checkpoint paths and per-machine records remain local.

This tests native memory/file attachment with actual weights. It is not a complete
safetensors parser, a checksum of every byte, tensor loading onto the GPU, model
execution or inference correctness. Basic header framing checks are deliberate
preconditions for this test, not general format validation. Model numerical
identity, tensor interpretation and complete state attachment remain required.

The test program is handwritten Omega. The existing development Node coordinator
read the already-present index, invoked the executable once per shard, and wrote
the local result record; it performs no tensor interpretation or model execution.
The executable requires only the native compiler's file, mapping and argv words.
All runs used quiet-build. No GPU operation, model service change or benchmark
timing was performed. Hardware experiment status remains NOT_RUN.

To repeat after building the compiler, compile `tests/speculation/model-map-test.om`
to a native executable and pass one existing local shard path as its sole argument.
Success prints `MODEL_MAP_SAMPLES_PASS 37` and exits0. The complete local index
enumeration and artifact identities remain in the private experiment handoff.

The corrected compiler at Omega078455c9 incorporates the existing branch-range
fix. Its full host and native branch gates pass. All seven experiment executables,
including this probe, rebuild byte-identically under that compiler; see
`corrected-compiler-reproduction.json`. The prior real-file run therefore remains
bound to the same executable bytes. No additional model/GPU run is implied.
