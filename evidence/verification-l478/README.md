# L478 targeted security repair

Supplemental to Omega PR614 (c6/disasm, 19eebe0e04811002f641c0f7b58c1b87ea9b5c42) and PR617 (c5/isolation, f792cd1d35b0e9c191647518c3be651a8d86dc21). Canonical GPU source remains CUDApp. Parent is L457 canonical 349913f17c083996680c72b32f767b225005155d. No original branch or worker modified.

Exact GitHub CodeQL annotations identify insecure capture permissions, environment-derived capture paths, a snprintf offset derived from its return value, five named listing files (one in Omega's sieve), and the isolation dump. Repair uses canonicalized, confined private directories, descriptor-relative exclusive mode-0600 capture creation, fixed-size hex filenames, private listing streams, and exclusive mode-0600 isolation dumps. Capture directories must resolve beneath the current checkout or /tmp, be user-owned and mode 0700. The corpus creator preserves that requirement. Existing instruction, branch, oracle and mutant checks remain intact.

## Executed

- `timeout 240 quiet-build make -j2 OUT_DIR=build-l478-green test-kernel-dump test-gpu-elementwise test-gpu-reconv test-gpu-attention test-gpu-fp8 test-gpu-isolation`: exit 0. Capture 13 checks. Existing binaries report 24,107,119,4997,214,5,219,73,107,227,229 passing checks. Interleaved output is retained verbatim; no full suite run claimed.
- `timeout 60 quiet-build make OUT_DIR=build-l478-green test-isolation-dump`: exit 0, 6 checks. Runs only host codegen/simulator with `--dump-hang`; never launches a kernel.
- Sequential capture controls: original source builds successfully and fails 4/13 checks (exit 1). Four valid behavioral mutants each build successfully and fail: permissive file mode (2 checks), shared directory acceptance (1), existing-file overwrite (1), unconfined directory (1). Repaired source passes 13/13, exit 0.
- Sequential isolation controls: original source builds successfully and fails 5/6 checks (exit 1). Mode-0666 mutant fails 1/6; overwrite mutant fails 3/6. Repaired source passes 6/6, exit 0. JSON records exact commands, source/binary SHA256 and separate build/run exits.
- `timeout 60 quiet-build cc -std=gnu11 -Wall -Wextra -Werror -O2 -Isrc -DOMEGA_GPU_NO_POSIX -ffreestanding -c src/omega_blackwell_codegen.c -o build-l478-free/codegen.o`: exit 0. Undefined symbols remain only memset and sha256_hash. No hosted file API enters the freestanding build.

## Limits

CodeQL official rescan NOT_RUN. GPU/model/hang/recovery/benchmark tests NOT_RUN. No live-server, driver/module, coordinator, cron, judge or PR changes. This repairs demonstrated diagnostic I/O failures; it does not qualify CUDA replacement or admit the original PRs. Omega's sibling source-binding integration and exact-head official gates remain required. R11 living-run ownership failure belongs to L345; prior skip-accepting repairs are held. R13 and hardware failures remain recorded in the companion Omega evidence packet.
