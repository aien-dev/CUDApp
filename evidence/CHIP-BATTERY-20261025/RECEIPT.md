# Chip battery, 2026-10-25: CUDA++ on the DGX Spark, with the live 35B left running

What this is: a full pass over CUDA++ on the GB10 — host suite, every chip gate, the zero-CUDA
gate, the artifact export, and a real inference check against the resident 35B — plus one defect
found on the way and fixed. Correctness and value checks only: no exclusive window was taken, no
speed is claimed from these runs (the machine was shared with the live model the whole time).

- Machine: DGX Spark, NVIDIA GB10 (sm_121, PCI 000f:01:00.0), Linux 7.0.0-1019-nvidia aarch64,
  open kernel module 580.173.02, no CUDA user-space library used by any program here.
- Source: CUDApp `60d6c57` (branch `feat/device-trace-emitter`, 38 commits ahead of local `main`),
  then `fix/qmd-shared-cap-32k` for the fix below.
- Live model: `aien-serve-http` pid 3906455, 42.5 GiB, port 18091 (`qwen3.6-35b-a3b-fp8`). It was
  serving before, during and after every run in this folder; nothing here stopped, restarted or
  reconfigured it.

## Results

| Check | Command | Result |
| --- | --- | --- |
| Build | `make -j8` | clean, `-Wall -Wextra -Werror` |
| Machine readiness | `make doctor` | READY: module 580.173.02, GB10 qualified, 58.9 GiB available |
| Host suite | `make test` | `CUDA++ HOST TESTS PASS (33 targets)`, rc 0, 417 s (before the fix); see `hosttest-after.log` for the re-run with the new target |
| Zero-CUDA gate | `make zero-cuda-gate` | PASS + self-test (a stub `libcudart` probe is rejected) |
| Artifact export | `make test-native-artifact` | 47 checks, 0 failed |
| First kernel | `make hello-gpu` | PASS, 1048576/1048576 elements, 0 wrong, 0 never written, machine-code digest printed (see `hello-gpu.log`) |
| Chip battery (12 gates) | `run-chip-battery.sh` | 12/12 PASS, 46 s wall (see `battery.txt` and the per-gate `*.json`) |
| Prefill / quant gates | `gpu_pf_mem_test`, `gpu_pfgemm_test --chip`, `gpu_qgemm_test --chip`, `gpu_pfhost_test --chip` | PASS |
| L2 staleness | `gpu_l2_stale_test --mode rewrite|realloc --cached 0|1` | 4/4 FRESH (0 stale or undecoded lines) |
| Inference (live model) | `curl /v1/chat/completions` | answers tokens, greedy; `live-infer.json` has the reply and usage |

Chip battery gates and their verdict lines (`battery.txt`): elementwise 35, matmul_api 97,
attention 219, decode_attn 29, router 17, bf16_gemv 12, fp8gemm 36, moe_window 143, cmdlist 31,
decode_graph 13, tcgemm PASS, prefill_attn 164 checks — 0 failed in every gate.

## Inference on the hardware, and what it is running

The 35B on 18091 is the strongest available proof that inference runs on the chip through CUDA++:
its process has **no CUDA library at all** mapped (only `libc.so.6`, `libm.so.6`, `libgcc_s.so.1`
and the loader), and its build directory carries the archive it was linked against:

```
036b7491d85a2d0781360c194402f345ad2f63720dab1b8f6d0cbc5c09f87d3e  libomega_gpu.a
cc45f45e6abb482d8bb1b4d1c9bf5a30ccb4e854063f8c783a7826ca8deab27e  aien-serve-http
```

`BUILD.txt` in that directory names omega `a647b6a9`, physics `048e5f5e`, aienos `b84c0a67`, and
`native-artifacts.lock` records that archive digest against omega `a647b6a9`. The CUDApp commit
those sources came from is **not** recorded anywhere on this box: `048e5f5e` is a commit in the
`physics` repository (driver client only, no `omega_*` sources), and the digest of the library
built from this repository's `60d6c57` is `e464fa05…`, not `036b7491…`. So the live server runs a
CUDA++ snapshot that no file identifies. That is a provenance gap, not a defect in this tree;
closing it needs a revision string inside `libomega_gpu.a` and a line in the build recipe.

## The defect this pass found

`./build/gpu_pfgemm_test --chip` failed on **all four** production shapes at
`qmd build (shared_bytes)` (`gpu_pfgemm_test.log`). The tensor-core FP8 grouped GEMM asks
`omega_tc2_shared_bytes = 4096 * rt` bytes of shared memory — 32768 at the shipped `rt` 8 — and
`OMEGA_BW_QMD_MAX_SHARED_BYTES` was 16384, so `omega_gpu_pf_gemm()` was refused before any device
work: the whole device-resident prefill GEMM path could not launch.

The cap was not the field. `NVCEC0_QMDV05_00_SHARED_MEMORY_SIZE_SHIFTED7` is `MW(1162:1152)`
(clcec0qmd.h, NVIDIA's open-gpu-doc tree): 11 bits in 128-byte units, up to 262016 bytes. Word 36
already declares `TARGET_SM_CONFIG_SHARED_MEM_SIZE` 9 = 32 KiB and `MAX` 26 = 100 KiB under Mesa
NVK's `size_kb / 4 + 1` rule (`gv100_smem_size_to_hw`, `nak/qmd.rs`). The cap now equals that
declared target (32768); going beyond it needs `MIN_`/`TARGET_` raised the way NVK's
`gv100_pick_smem_size_kb` does, which the header now says.

Fix (`fix/qmd-shared-cap-32k`): cap 16384 → 32768; new host test `test-gpu-qmd-shared` (encoding of
0/1024/4096/16384/32768 bytes, the SM-config fields unchanged, every other QMD word unchanged, and
a negative control one byte over the cap); `gpu_prefill_attn_test` and `gpu_decode_attn_test` check
the macro instead of a literal; stale "QMD cap 16384" comments in `codegen_pattn.h`,
`codegen_moe2.c` and `codegen_qg.c` corrected; `docs/runtime/API.md` states the cap.

After the fix, same chip, same live model running (`pfgemm-fixed.log`): all four shapes PASS,
bit-identity on 48 sampled rows, mutants 1–4 caught, and the 12-gate battery still 12/12.

## What is not claimed

- No speed number. These runs share the machine with a 42.5 GiB model; `hello_gpu` printed 939 us
  against a 64.5 us floor and the pfgemm medians moved between runs. A timing receipt needs a
  `spark-window` and is not in this folder.
- Not a qualification of the fix beyond the gates listed: no window, no signed bundle.
- The pinned live archive (`036b7491…`) was not rebuilt or re-tested here; it is a different source
  snapshot from `60d6c57` and its CUDApp commit is unidentified.
