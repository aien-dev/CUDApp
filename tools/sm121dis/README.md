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
