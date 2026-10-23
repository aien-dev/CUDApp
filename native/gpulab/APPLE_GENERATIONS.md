# Apple generation coverage: not yet M1–M5 complete

All first-party code, numerical references, request writers, byte comparators and
virtual execution remain handwritten Omega. These are offline lab targets, not
hardware drivers or a full inference runtime.

| Requested family | Request ID | Implemented result | Independent encoding evidence | Hardware |
|---|---:|---|---|---|
| M1 | 101 | G13/G14 bounded GEMV subset | Existing pinned AGX2 tables and byte fixtures | NOT_RUN |
| M2 | 102 | G13/G14 bounded GEMV subset | Same shared subset; not a new per-SKU hardware result | NOT_RUN |
| M3 | 103 | Separate G15 bounded GEMV emitter and decoder | Full K64 shader equals pinned M3 assembler; 140,293 native checks, zero failures | NOT_RUN |
| M4 | 104 | Separate G16 bounded emitter and decoder, short waits | Apple G16 scalar/vector/volatile operand captures; full shared-field assembler K64 match; 140137 native checks, zero failures | NOT_RUN |
| M5 | 105 | REFUSED; implementation pending | Split-memory captures investigated; raw/implicit operand semantics remain unresolved | NOT_RUN |

Do not call a recognized ID support. No claim that every Pro/Max/Ultra variant
shares the tested virtual profile. L641 stays open until all requested generation
rows have their own implemented and qualified path.

## M3 implementation

`m3.om` emits a short move initializing r6, then for each term two 14-byte i32
loads, the eight-byte base-M3 long-wait workaround, and a twelve-byte imadd.
The epilogue stores r6, waits and stops. Program size is `28 + 48*K` bytes,
including 3100 bytes at K64. The separate decoder reads actual bytes; it does
not call the emitter or scalar oracle to infer the answer.

The ABI initializes r0=row*K, r1=0 and r2=row; uniform pairs u0_u1, u2_u3 and
u4_u5 denote the virtual weight, vector and output buffers. Each load's byte
offset is 4*k. The accumulator and arithmetic registers are plain i32 GPRs;
unsupported cache modifiers, groups, operand types and instruction forms refuse.
This is a lab entry state, not a GPU launch ABI. Final public output is committed
only after every lane finishes successfully.

`test-m3.om` exercises all dimensions, exhaustive signed-byte products, every
truncated prefix of K4, whole-dispatch atomicity with a final-lane-only address
fault, and every single-bit mutation of a K1 stream. The mutation run reports
559 decoder refusals, 37 accepted programs whose changed answers are detected by
the scalar oracle, and 12 equivalent-on-these-probes outcomes. The latter are
not claims of universal equivalence. The final two numbers are 140293 checks
and 0 failures. Compile failures are not behavioral RED results.

`test-families.om` exercises the enabled IDs at every depth with three rows and
checks that M5 and unknown IDs cannot execute even when handed valid older-family code.
The Makefile also drives real request -> emitter -> wire-consumer processes for
M1/M2/M3/M4, and checks M5 exit65 with zero output.

## M4 implementation and evidence boundary

`m4.om` is a separate G16 profile using the shared G15/G16 memory and integer
operand fields, with an explicit two-byte `wait 0`. It does not call `m3_run`.
Its program size is `22 + 42*K`: K1=64, K4=190 and K64=2710 bytes.
The lab entry state and transactional output contract are the same as M3.
The native gate passes 140137 checks with zero failures. K1 bit mutations yield
463 refusals, 37 changed answers detected, and 12 equivalent-on-probe outcomes.
Final-lane-only memory faults and all truncated K4 prefixes remain transactional.
G15 and G16 profiles reject each other's different wait sequences.

The evidence has two distinct layers. Apple's independent G16 compiler produces
scalar/vector/volatile kernels whose memory and integer operand fields decode
under the pinned G15 reference. The complete Omega-generated K64 profile equals
the independent shared-field assembler. The latter is not an Apple compiler
byte-for-byte match of the complete kernel: Apple optimizes registers, memory
scheduling and constant programs differently. No physical M4 execution or
all-SKU support is asserted.

## External M3 oracle

Reference: [TellowKrinkle/applegpu M3](https://github.com/TellowKrinkle/applegpu/tree/bde26552eaca66a0ba2d5a6362e08b90693fa4cc), BSD-3-Clause.
The relevant classes are DeviceLoadStoreInstructionDesc, IMAddSubInstructionDesc,
MovImm7InstructionDesc, WaitLongInstructionDesc and their register descriptors.
No upstream interpreter is used as this lab's executor.

The Omega program `m3-oracle-input.om` writes mnemonic assembly independently of
our encoder. The unmodified upstream `assemble.py` was invoked as an external
validation tool on that assembly. Its bytes are frozen in `fixtures/m3-k64.bin`.
The normal build and test path needs no Python; the frozen binary has SHA256
`7988361c60d7154a3e295e80b7cc1a15a373073eae7efea330f115b8ef1fc958`.
The suite verifies fixture identity, regenerated assembly text, all output bytes,
and rejection of a same-length corrupted fixture.

External regeneration (optional, separate from the Omega implementation):

```sh
build/native-gpulab/m3-oracle-input > m3-k64.s
# In a verified checkout of the exact upstream pin above:
python3 assemble.py -i /absolute/path/m3-k64.s -o /absolute/path/m3-k64.bin -quiet
# In this repository:
build/native-gpulab/test-m3-oracle < /absolute/path/m3-k64.bin
```

Upstream source closure SHA256:

```
050a9be1b56e5649c327603673c3ca1ccbac70765a3fb3e3ddd98aa327724a37 applegpu.py
14926001701b8b5dfdb1a9eef1aae71b95c3858cfa9b53b213a8727614482389 assemble.py
98ab1d55aae88fdfbfd94f74eb27dc5027039b8f78c914bdff1cea552aed75ef disassemble.py
06a270fe306d74f5d0c6c0a0d74161db3e8139de867b49bc4605e0d2e4ee8aa7 fma.py
21aafd33b0134634ec5a7bd2ec38370b876658855e83d348fe0f3b4957068205 srgb.py
```

## M5 remains refused; earlier M4 uncertainty was resolved

Investigated [ADevWithAnIdea/agx-re](https://github.com/ADevWithAnIdea/agx-re/tree/c677ff970ca9d5ab98770100e790f8a62c60a2a0)
at the pinned commit, specifically `docs/isa/encoding-tables.md`,
`docs/isa/encoding-tables-m5.md`, and
`experiments/EXP-M5-07-isa-semantics/{report.md,hex_extractions.txt}` plus
`experiments/EXP-M5-11-isa-integration/report.md`.
These are external reports, not measurements reproduced by this task. The GPL
implementation was not copied, built, imported or executed.

The earlier inference that M4 also required the M5 arithmetic format was wrong.
Independent Apple `applegpu_g16g` compilation and the pinned G15 disassembler
show shared G15/G16 operand fields in the scalar, vector and volatile probes.
The `applegpu_g17g` compiler output really differs. The M5
report says its load/store data registers are implicit/positional, calls general
ALU packing raw/unmapped, and reports legacy 0x67/0xe7 occurrence without proving
that arbitrary M3 operand fields preserve semantics. Treating those facts as a
complete GEMV emitter specification would invent support.

The next discriminating experiment is a per-generation compiler/splice corpus:
compile independent scalar signed-i32 dot kernels while varying one source,
destination, pointer slot, index and offset at a time; retain exact GPU/compiler
identity, bytes and numerical read-back; then implement the demonstrated fields
in Omega and compare whole generated shaders against that independent corpus.
Capture the M5 positional-register and address-generation state explicitly,
including load/store dependencies. Do not infer it from mnemonic names.

The Mac is an M2, but the missing compiler was acquired from Apple's public
MetalToolchain 17B54 asset and mounted read-only in the lab workspace.
Apple Metal 32023.830 and the native translator were actually run for G16 and
G17. No system installation or device dispatch was required. M5 evidence work
now uses real compiler captures rather than waiting for a locally installed SDK.
See `../../evidence/NATIVE-APPLE-L641/m4/README.md` for provenance and limitations.
