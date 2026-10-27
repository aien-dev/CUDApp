# ATTN-GENERAL-20261027-4d51f91f

GB10 (sm_121) chip run of feat/attn-general at 4d51f91, made on 2026-10-27 beside the live 35B. It used one flock hold of `/tmp/aien-gb10.lock` for the run itself (`chip.sh`) and a second hold for the A/B v2 (`ab2.sh`). Both holds were health-checked before and after: compute apps showed only 79557, and 18091 reported `"status":"ok"` (`health_pre.txt`, `health_post.txt`, `ab2/health_*.txt`).

Label: chip-EXERCISED. This is not a qualification.

| run | command | result |
|---|---|---|
| general sweep | `gpu_attention_test --general` | 318 of 318 configs pass, PASS 348 checks, 0 failed (`general.json`, `general.log`) |
| hd 64 battery | `gpu_attention_test` | PASS 219 checks, 0 failed (`hd64.json`) |
| hd 128 battery | `gpu_attention_test --hd 128` | PASS 225 checks, 0 failed (`hd128.json`) |
| head sweep | `gpu_attention_test --sweep` | 72 of 72 points; the 8 points with a non-power-of-two GQA ratio (24q / 1, 2, 4, 8 kv) were refused before 0.6.0 and now run on the general kernel. PASS 73 checks, 0 failed (`sweep.json`) |

## General sweep coverage

The sweep draws its configs from seed 0xa77e0601 and compares every one against the f64 host oracle.

- **Tolerance:** `|got - want| <= rel*|want| + 2e-5`, where rel is 2e-4 for kv <= 2048 and 4e-4 above that. The bound grows with kv because the f32 sum rounding grows with n:
  - WolframLanguageEvaluator: `N[4096*2^-24]` -> 2.44e-4
  - WolframLanguageEvaluator: `N[5000*2^-24]` -> 2.98e-4

  The worst scaled error was 0.0062 of the tolerance.
- **Shapes covered:**
  - head_dim: {1, 2, 3, 5, 7, 31, 33, 63, 64, 65, 80, 96, 97, 112, 127, 128, 129, 160, 191, 192, 255, 256}
  - GQA ratios: 1-16 and 1024
  - kv: up to 5000, including kv = 0
  - 115 configs are non-causal and 67 use a sliding window.
  - Entries:
    - general 139
    - gqa_f32 67
    - rows_f32 67
    - paged_batch 22
    - paged_resident 23
- **Mutants:** each one was caught on the chip.
  - GQA_MAP: caught at config 4
  - TAIL_CHUNK: caught at config 4
  - TAIL_DIMS: caught at config 2
  - CAUSAL: caught at config 0
  - SCALE: caught at config 2
- **Routing proof:** the GQA_MAP mutant shows up when the general kernel is forced. It does not show up on the default tuned route (`"routing"` in `general.json`).

## Served-shape A/B: main c30c0be (A) vs 4d51f91 (B)

- **Static check:** all 252 tuned kernels are byte-identical between main and the branch. The set is `omega_gpu_attention_codegen_hd` for hd 64/128 × f32/bf16 × log2bs 0..8 × log2gqa 0..6, and the digest lists are `tuned_kernel_digests_*.txt`, with sha256 bb54fd5c…fcd for both. The lists were produced by `attn_digests.c.txt`. The fused served kernels (dattn, attn256, pattn, pattn256) are not touched by this branch.
- **v1** (`AB-PLAN.md`, sha256 1c1c8836…6cde, posted on BOARD before timing): 10 + 10 clean runs.
  - WolframLanguageEvaluator: Median A = 0.66 ms, Median B = 0.70045 ms, ratio B/A = 1.061; MannWhitneyTest p = 0.853.
  - The ratio is outside the predeclared 1.05 band, so v1 is reported as measured.
- **v2** (`AB-PLAN-v2.md`, posted on BOARD before timing): 40 pairs. 2 A runs were dropped because 18091 was serving (in_flight > 0), leaving 38 A and 40 B clean runs.
  - WolframLanguageEvaluator: Median A = 0.60305 ms, Median B = 0.60245 ms, ratio B/A = 0.9990; MannWhitneyTest p = 0.960.
  - This is within the band, so the result is "no slower".
- Per-run tags are in `ab/clean.tsv` and `ab2/clean.tsv`. The workload is `--timing`: gqa_f32 32q/4kv, hd 64, ctx 2048, 100 calls per run.
