# D7 same bits on the chip: receipt, 2026-10-27

Issue: [docs/issues/D7-same-bits-chip.md](../../../docs/issues/D7-same-bits-chip.md).
Question: do separate runs of one job on the GB10 print the same output hash, with no setting turned on?
Answer for the four jobs below: yes. Every job ran in 10 separate processes, and each output hash was identical in all 10.

## What ran

| | |
|---|---|
| Chip | NVIDIA GB10 (sm_121), driver 580.173.02, kernel 7.0.0-1019-nvidia |
| Code | aien-dev/CUDApp branch `d7/same-bits-chip` at `0490d98b6bc4bd3a6b09f80b4332cdfe1b9c090f`, no tracked or untracked changes outside `evidence/` (each `*/env.txt`) |
| Binaries | Built at `5be95d9` with `nice -n 19 taskset -c 10-14 make build/hello_gpu build/gpu_bf16_gemv_test build/gpu_elementwise_test build/gpu_decode_attn_test`. `0490d98` changes only `tools/run_d7_same_bits_chip.sh`, so none of these binaries' sources changed. |
| Runner | `flock -w 900 /tmp/aien-gb10.lock tools/run_d7_same_bits_chip.sh JOB 10 evidence/d7-same-bits-chip/2026-10-27` gives one GPU hold per job. The live 35B (pid 79557, 127.0.0.1:18091) kept running beside it, and the guard was checked before and after each hold (`*/gpu-guard.txt`). |
| Setting | None. The runner refuses to start if any `CUBLAS_*`, `CUDA_*`, `CUDNN_*`, `NVIDIA_TF32*`, `TORCH_*`, `PYTORCH_*` or `OMEGA_*DETERMIN*` variable is set. Each `env.txt` records `none`. |
| Processes | Each run is a new process. It opens the device, builds its kernels, uploads its inputs, launches, and prints `<label> output sha256 <hex>` over the output buffer of its first chip call (`tests/out_sha.h`; hello_gpu has its own print). |
| Check | `tools/d7_same_bits_check.sh JOB/run-*.log` writes `JOB/check.txt` |

## Results

| Job | Command | Binary sha256 | What it reduces | Label | Hash (all 10 runs) |
|---|---|---|---|---|---|
| hello | `build/hello_gpu` | `610aa46b…68775` | nothing: c = a + b, 2^20 floats (the D7 baseline) | output | `72d0c1b2928779e83518f0b4c7d101d933cf89f210cd4627e138c33c9ce9b6a6` |
| gemv | `build/gpu_bf16_gemv_test --reps 1 --no-evict` | `1a674f88…a491` | bf16 matrix-vector 151936 x 2048 (the lm_head): a per-lane FFMA chain over K, then a SHFL.DOWN tree across the warp | gemv_f32_151936x2048 | `ed3805976ef5eb04a893ed9d3b9ad1cc8afa17e60a36b7d4dd9dd98e698f9e4b` |
| rmsnorm | `build/gpu_elementwise_test` | `a0a083de…6591` | sum of squares per row across the CTA (SHFL.DOWN, BAR.SYNC, shared memory) | rmsnorm_4x2048 | `fec7ab6c2a6f67496ed95afce7ff8d0083cc0d28ca594087ff944058e4822667` |
| | | | | rmsnorm_1x256 | `2a194f5e2fb95fbe2f584cbc7655a4683d00132738ffe3da3c43e234882d2470` |
| | | | | rmsnorm_2x8192 | `257fcd904344990ba68a1e26d8e4ed227121d50a369f2c62f0aca32664435d88` |
| dattn | `build/gpu_decode_attn_test --reps 1 --no-evict` | `52efc192…ea73` | decode attention, 32 q / 4 kv heads x 128: softmax max and sum reductions, plus the combine across KV splits (grid 1x4 up to 128x4) | dattn_pos0 | `e81b7427ea91d932b787c8b3aab785ebba891780039af46f87e57b557d31ecd8` |
| | | | | dattn_pos6 | `9f1969292c8226ec86e1d91ef5f2706bbc8b2034b831d44351c0b72e651d06fd` |
| | | | | dattn_pos255 | `2a76c594bff6c41fbe33b90f5d4e5fa91c08c3b7945a757417170f301490c889` |
| | | | | dattn_pos300 (sharp q, 2 splits) | `7901e5e5c6499e531e16533afd9aa2e68cf0da8d0bc763b9feb7d62843ff60e8` |
| | | | | dattn_pos1023 (4 splits) | `c538767cf530fdc8495bfae1151c700e8a887cf5d1d4c6444a2ff38dffd41880` |
| | | | | dattn_pos4095 (16 splits) | `01db162e6a0fb8ae267b8dc1b2e97ffb48dc10b3094d015c86e6a6b5765c963c` |
| | | | | dattn_pos8191 (32 splits) | `58fde3abe0308ba8aa23a51c9506595e29929eae3b7dc19462553af2f5acaff6` |
| | | | | dattn_pos32767 (128 splits) | `3741c568cb32da1310250b01feeb882cdcaaa8bf38c28a2247ad462aaca3b798` |

The full binary hashes are in each `env.txt`. All 40 runs exited 0, and every test also passed its own parity checks; the logs are `JOB/run-01.log` to `run-10.log`. The UTC windows were: hello 07:03:37.7 to 07:03:38.7, gemv 07:03:42.4 to 07:03:53.7, rmsnorm 07:03:57.5 to 07:03:58.6, and dattn 07:04:02.2 to 07:04:08.1.

The hello chip hash `72d0c1b2…b6a6` is the same hash the simulator printed in `evidence/dx/d3-faults/hello-sim-spark.txt`.

## Negative control (`negative-control/`)

The checker has to go red when the outputs differ. `negative-control/checker-controls.txt` records five cases:

- **A, a real chip run with different output.** `build/hello_gpu_control --bug=short` launches one element short, so c[1048575] is never written. This was one more GPU hold (`negative-control/hello-bug-short.log`, exit 1, `output sha256 ec24bd3e…df75`). The 10 good hello logs plus that log give `MISMATCH output`, exit 1.
- **B, a one-bit change.** gemv runs 01 to 09, plus a copy of run 10 with the last hex digit of its hash changed from b to a (one bit), give `MISMATCH gemv_f32_151936x2048`, exit 1.
- **C, a missing hash.** dattn runs 01 to 09, plus a copy of run 10 without its context-32768 hash line, give `MISMATCH dattn_pos32767 … missing`, exit 1.
- **D, positive control.** The unmodified 10 gemv runs give PASS, exit 0.
- **E.** A single log is a usage error, exit 2.

## What this does not show

- **Other kernels.** Only these four jobs were checked; this is not a claim about all CUDA++ kernels.
  - scan and sum-reduce (`tests/gpu_scan_test.c`) do not build on main: `mk/c3.mk` and `src/omega_numeric*` arrive with draft PR #112. So they are **NOT_RUN**.
  - softmax as its own kernel, prefill attention, the fp8 grouped GEMM, the GDN and MoE kernels, and the tensor-core GEMMs were not hashed.
- **Other GPUs.** One GB10 and one driver version only.
- **Different inputs.** Each job ran on one fixed input set; run-to-run identity on these inputs is not proof for every input.
- **Concurrent load.** All runs shared the chip with the resident 35B, as it served normally. No run was made with other kernels contending on purpose.
