# sm121dis: Omega's own GB10 disassembler

A disassembler turns the GPU's machine code back into readable instructions. Omega used to borrow NVIDIA's
`nvdisasm` for that, in every check that reads a kernel back. `sm121dis` is our own (row C6 of `docs/OMEGA.md`):
the decoder is `src/omega_sm121_disasm.c`, this folder holds the command line and the comparison with NVIDIA's tool.

## What it proves

- Every kernel the library builds without the chip (each GPU test in its host-only and simulator modes, plus the
  numeric kernels) is collected and compared word by word with `nvdisasm`, text and spacing included. All 331
  kernels (59,152 instruction words) agree.
- A copy of the decoder with one planted fault (`SM121DIS_MUTANT`, first source register off by one bit) must fail
  that comparison, and does, on every kernel.
- A word the decoder does not know is printed as `?? <hex>` and fails the check. It is never guessed.

## How to run it

    make test-sm121dis          # collect, compare with nvdisasm, then the red check

`nvdisasm` is used only here, as the reference to compare against. Nothing it prints goes into a kernel.

## Who uses it

The checks that used to call `nvdisasm` now call ours:

- the listing checks inside `tests/gpu_elementwise_test.c`, `gpu_fp8_api_test.c`, `gpu_reconv_test.c`,
  `gpu_attention_test.c` and `bench/prime_race/tests/gb10_sieve_host_test.c` (linked in, `omega_sm121_listing`);
- `tools/divsqrt_nvdisasm_check.sh`, `tools/ldst_nvdisasm_check.sh` and `make test-numeric-reduce-nvdisasm`
  (the `build/sm121dis` command). The script and target names are kept so older receipts still line up.

## Files

| File | Job |
|---|---|
| `sm121dis.c` | command line: a listing, `-b SM121` (called like nvdisasm), `--pairs` (for the comparison), `--version` |
| `corpus.sh` | runs every GPU test with `OMEGA_KERNEL_DUMP_DIR` set, so each kernel the library encodes is saved |
| `check.sh` | compares our text with nvdisasm's for every saved kernel; PASS only when every word agrees |
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
