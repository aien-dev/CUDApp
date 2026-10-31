# sm121dis: Omega's own GB10 disassembler

A disassembler turns the GPU's machine code back into readable instructions. Omega used to borrow NVIDIA's
`nvdisasm` for that, in every check that reads a kernel back. `sm121dis` is our own (row C6 of `docs/OMEGA.md`):
the decoder is `src/omega_sm121_disasm.c`, this folder holds the command line and the comparison with NVIDIA's tool.

## What it proves

- Every kernel the library builds without the chip (each GPU test in its host-only and simulator modes, the
  numeric kernels, and every kernel the hazard checker's sweep builds) is collected and compared word by word with
  `nvdisasm`, text and spacing included. All 2,980 kernels (1,218,992 instruction words) agree (2026-10-30, host).
  The receipts in `evidence/C6-DISASM/` predate the sweep kernels: 331 kernels (59,152 words) on 2026-10-24, 402 kernels
  (74,096 words) on 2026-10-25.
- A copy of the decoder with one planted fault (`SM121DIS_MUTANT`, first source register off by one bit) must fail
  that comparison with a verdict, and does on 2,979 of the 2,980 kernels. The one it cannot touch is a sweep kernel of
  32 words that is only EXIT, a branch and padding: it has no source register for the fault to change.
- A kernel that could not be saved whole (a full disk) fails the collection; it is never compared cut short.
- A word the decoder does not know is printed as `?? <hex>` and fails the check. It is never guessed.

## How to run it

    make test-sm121dis          # collect, compare with nvdisasm, then the red check

`nvdisasm` is used only here, as the reference to compare against. Nothing it prints goes into a kernel.

## Who uses it

The checks that used to call `nvdisasm` now call ours:

- the listing checks inside `tests/gpu_elementwise_test.c`, `gpu_fp8_api_test.c`, `gpu_reconv_test.c`,
  `gpu_attention_test.c` and `bench/prime_race/tests/gb10_sieve_host_test.c` (linked in, `omega_sm121_listing`);
- `tools/divsqrt_nvdisasm_check.sh`, `tools/ldst_nvdisasm_check.sh` and `make test-numeric-reduce-nvdisasm`
  (the `build/sm121dis` command);
- the hazard checker's register cross-check, `tests/bw_hazard_test.c --disasm` (`make test-bw-hazard-nvdisasm`,
  linked in, `omega_sm121_disasm`), over every kernel its sweep builds. Those kernels are in the comparison corpus.

The script and target names are kept so older receipts still line up. `make test-zero-nvdisasm` proves that none
of these checks runs `nvdisasm`.

## Files

| File | Job |
|---|---|
| `sm121dis.c` | command line: a listing, `-b SM121` (called like nvdisasm), `--pairs` (for the comparison), `--version` |
| `corpus.sh` | runs every GPU test with `OMEGA_KERNEL_DUMP_DIR` set, so each kernel the library encodes is saved |
| `check.sh` | compares our text with nvdisasm's for every saved kernel, `SM121DIS_JOBS` at a time (default: every CPU the process may use); PASS only when every word agrees |
| `oracle_pairs.sh` | turns nvdisasm's listing into the same one-line-per-word form |

## Exact MMIO semantic contract (issue #171)

`BW_IR_LDG_MMIO` promises `LDG.E.MMIO.GPU`, not the historical encoder's EF
cache policy. `make test-sm121-mmio OMEGA_MV=/path/to/pinned/omega` concatenates
the production encoder with `om/encoder/mmio_test.om`. The Omega fixture emits
one instruction and checks NVIDIA nvdisasm 13.0.85's exact mnemonic and register,
descriptor and address operands. It then emits an otherwise identical EF
mutant; the same Omega expectation must refuse it with exit status 4.
The seeded compiler SHA-256 remains pinned; the oracle's version and binary
SHA-256 are recorded. Neither decoder output nor mutant bytes enter production.

This tests an IR semantic promise, unlike a decoder-only comparison that can
agree about the wrong instruction. Existing golden receipts are not rewritten.
Before/after corpus identities record any emitted-kernel byte impact separately.
GB10 cache behavior and device execution remain **NOT_RUN**; no live model is
rebuilt, replaced, or deployed by this check.

The recorded pre/post comparison at base `e84326dca9f1ef9afbeaf249fb3e46ae0a2517f7`
has **773 identical existing SASS kernels and four identical captured cubins**
(including a copied bundle used for manifest-tampering controls).
The complete before/after SHA-256 maps and empty changed-kernel/artifact lists
are in `evidence/CUDAPP-SM121-MMIO/corpus-impact.json`, bound to exact source,
generated binding and NVIDIA oracle identities. The existing producer corpus
does not exercise op21; the new single-op fixture does. The generated encoder
and host library bytes change as expected, but no deployed artifact is rebuilt
or loaded. Corpus equality is not cache-behavior or device qualification.

The final canonical replay uses baseline
`c6977b4906229fe83f41103b7eff65b64a0986d6` and corrected implementation
`3c163287d27b6e5a5f5b90c15b61ac83d8f26e2d`. Its complete inventories,
commands, logs, source identities and review are under
`evidence/CUDAPP-SM121-MMIO/c697-final-host/`.
All 773 existing SASS kernels and all four captured cubins remain byte-identical.
Both artifact captures pass all 47 checks. Exact-op and wrong-EF controls,
canonical-baseline refusal, binding regeneration, golden encoder parity and
specification checks pass with the canonical bridge pin
`0c4cb9ff980af278890b231b994eaede4dc7621c`.
The original `corpus-impact.json` remains immutable historical evidence, not
the canonical replay. The complete host suite and device execution are not
claimed by this targeted replay. The review is a source review, not independent
hardware qualification.
