# Naming plan

**Project name in text:** CUDA++. **Repository slug:** `aien-dev/CUDAplusplus` (GitHub slugs cannot hold `+`).
**Prefix where a code name is needed:** `cudapp` (library `libcudapp.a`, lock file `cudapp.lock` in Omega).

## First cut (2026-10-21): no C renames

Every file keeps the path and the C names it had in Omega (`omega_gpu_*`, `omega_blackwell_*`,
`OMEGA_*` macros, `libomega_gpu.a`). Reason: Omega branches that are being worked on right now
(the speed work on the 35B model, the Omegese GPU backend) edit these files. With identical paths
and names, `tools/extract/carry.sh` applies their commits here unchanged. A rename now would turn
each of those branches into a hand merge.

`libcudapp.a` is built as a copy of `libomega_gpu.a`, so new users can link the new name today.

## Step 2: the shared base shrinks

Nine files are kept byte-identical in both repositories (`tools/extract/shared.txt`): `sha256`,
`omega_types.h`, `omega_vector`, `omega_core`, `omega_canonical`. CUDA++ needs them because two
launch headers (`omega_blackwell_submit.h`, `omega_blackwell_realize.h`) take Omega's vector and
matmul spec types, and the kernel digests use SHA-256. Omega owns them; Omega's build checks that its
copies equal the ones in the pinned CUDA++ (`make check-cudapp-shared`), and only one copy is ever
linked into a program (Omega's own build compiles its copies, never CUDA++'s).

Step 2 replaces this with a small `cudapp_base` (SHA-256 plus the two spec structs) owned by CUDA++,
moves the vector execute path that needs the Omega program graph (`omega_vector.c`,
`omega_vector_build_spec_graph`) to Omega, and removes `omega_core`, `omega_canonical` and the Omega
type header from CUDA++. Done when: CUDA++ builds with no file from `shared.txt`, and Omega's golden
tokens are unchanged.

## Step 3: the `cudapp` names

After the open branches listed in Omega's `opencuda/consume` handoff have landed (so nothing is
carried any more), one mechanical rename per prefix, each its own commit with the build and every
host test green and the 35B golden tokens unchanged:

| Now | Then |
| --- | --- |
| `src/omega_gpu_*.{c,h}`, `omega_gpu_*` functions | `src/cudapp_*`, `cudapp_*` |
| `src/omega_blackwell_*`, `omega_blackwell_*` / `bw_*` | `src/cudapp_sm121_*`, `cudapp_sm121_*` |
| `OMEGA_GPU_*`, `OMEGA_BW_*` macros | `CUDAPP_*`, `CUDAPP_SM121_*` |
| `libomega_gpu.a` | `libcudapp.a` only |
| `nvrm_*`, `m16_*` (driver client) | unchanged until L332's own driver replaces the client |

Omega keeps compatibility headers with the old names for one release, then drops them.
