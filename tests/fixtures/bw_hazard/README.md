# Hazard checker fixtures (omega-mv#554)

Encoded sm_121 bytes of the head_dim 256 prefill attention kernels from branch `feat/pattn256` (omega-mv#545), used by
`tests/bw_hazard_test.c` as the red receipt and its control.

| File | Commit | Encoder |
|---|---|---|
| `pattn256_attend_ad5d650b.bin` | ad5d650b | before the fix: IADD3, ISETP_GE and EXIT got a fixed word 3 |
| `pattn256_attend_e3a26848.bin` | e3a26848 | after the fix (PR #553): the scheduled control word is kept |
| `pattn256_prep_ad5d650b.bin` | ad5d650b | before the fix |
| `pattn256_prep_e3a26848.bin` | e3a26848 | after the fix |

`pattn256_attend_ad5d650b.bin` is byte-identical (sha256 2e3f9d41...) to `attend.bin` in the investigation folder
`2026-10-18-pattn256-hang`, the kernel that faulted on the GB10 (Xid 13). How the files were made: build
`libomega_gpu.a` at each commit (PHYSICS_DIR at the pinned physics commit), then write the bytes returned by
`omega_gpu_prefill_attn256_encode(kind, ...)` (kind 1 attend, kind 0 prep; built, encoded and patched, no device).
Checksums in `SHA256SUMS`.

## The rule, checked against ptxas (offline oracle only)

`nvcc 13.0.88 -arch=sm_121 -cubin -O3` on a loop that reloads through a pointer and then advances it (source and
listing sealed in the investigation folder `2026-10-19-omega-554-hazard`, war.cu sha256 50386e2f...):

```
/*0050*/ LDCU.64 UR12, c[0x0][0x358] ;               control 0x000ea200: write barrier 2
/*0080*/ BRA.U !UP0, 0x1e0 ;                         control 0x004fea00: waits on barrier 2 before UR12 is used
/*0140*/ LDG.E.STRONG.GPU R0, desc[UR12][R2.64] ;    control 0x0000a200: write barrier 2, READ barrier 0
/*0160*/ LDG.E.STRONG.GPU R4, desc[UR12][R4.64] ;    control 0x000ea200: write barrier 2, no read barrier
/*01b0*/ IADD.64 R2, R2, UR4 ;                       control 0x001fe400: waits on barrier 0 before rewriting R2
```

ptxas gives the first load a read barrier because its address pair R2:R3 is rewritten while the load may be in flight,
and the rewriting instruction waits on it; the second load, whose address is only replaced by its own result, gets
none. That is the WAR rule the checker applies (and the RAW rule on the descriptor). The chip taught the same rule
twice: omega-mv#545 (this fixture) and the fp8 GEMV hang recorded in `src/omega_bw_em.h`.
