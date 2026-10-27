# C4: streams and events on the GB10 (docs/OMEGA.md row C4)

Row C4: two independent queues of the same work finish in under 0.75 of the sum of their solo times; an event wait
orders two dependent jobs, checked by data, on the chip.

What this branch adds:
- L136's second GPFIFO queue (physics feat/gpu-channels 7476de71, rebased onto physics main as fb2ac4c = physics PR #79, the commit physics.lock now pins; omega 449be575 + 0cf570dd): `omega_gpu_session_extra_queue`
  and `omega_gpu_cmdlist_submit_pair` (list A on queue 0, list B on queue 1, both submitted before either is waited).
- The event: `omega_gpu_cmdlist_wait_for(cl, other, i)`. The next launch added to `cl` starts only after launch `i` of
  `other` has released its semaphore: a host semaphore ACQUIRE (the chip-proven omega-mv#580 ORDER_SEM words) before the
  launch's barrier, so its cache invalidate comes after the wait.
- `first_barrier_none`: no WAIT_FOR_IDLE before the first launch of a submission (a WAIT_FOR_IDLE may also wait for the
  other queue's grids; the test measures both).
- `tests/gpu_streams_test.c`: half-chip spin kernel (24 CTAs x 128 threads, FFMA chain, exact value) and copy kernel,
  both from Omega's IR and our own sm_121 encoder.

| log | what | result |
| --- | --- | --- |
| 01-host-only.log | codegen + encode of both kernels and their mutants; wait_for refusals | PASS (5 checks), 2026-10-24T23:29Z on the train tip |
| 02-sim.log | warp simulator: exact values in both scheduling orders, mutants caught | PASS (9 checks), 2026-10-24T23:29Z on the train tip |
| 03-chip-values-20261022-f792cd1d.log | chip, values only, 2026-10-22 07:06Z on `c5/isolation` f792cd1d (older head; run by the dead worker, log recovered unpushed, durable copy `~/handoffs/evidence-L386-chip-20261022/`): second queue opens; event refusals right; with the wait every Y right (0 wrong of 3072) in both barrier modes; but the NEGATIVE CONTROL IS NOT RED: without the wait Y was also right (0 wrong, both barrier modes), so the run reports `C4 streams values: FAIL (1 of 6)`. Until the control goes red the event proof is not a proof: the two channels may not have run concurrently (Cortex 27f2ae96: two channels of one group ran two decode streams in 0.9990x serial time; whether GB10 time-slices channels of one group is UNKNOWN). Overlap NOT_RUN (values-only). | FAIL (control not red) |
| 04-chip-values-20261025-048e5f5.log | the same values run after the driver cause was found and fixed: each extra channel now allocates its own asynchronous subcontext (physics 048e5f5, physics PR #80), which is what row 03's not-red control predicted. With the wait every Y right (0 wrong of 3072) in both barrier modes AND without the wait 3072 of 3072 wrong, so the control is red at last and the event proof holds | PASS (8 checks), 2026-10-25T00:35Z on `c5/isolation` 02db88d33 with the pin enforced |
| (chip, pending slot) | per-queue timestamps across 3 runs without the wait, then overlap ratio solo vs pair (both barrier modes), serial red check, event + no-wait control, on the train tip | NOT RUN (needs an exclusive window; the C4 overlap ratio is still unmeasured) |

Chip command (GB10 lock; timings need an exclusive window):
`build/gpu_streams_test` (sets AIEN_OMEGA_GPU_CHANNELS=2 itself). PASS needs: pair / (solo A + solo B) < 0.75, the
same two spins in one list on one queue >= 0.75 (red check), every Y right with the wait, and at least one Y wrong
without it (negative control).

Update 2026-10-22 (after an independent review): the C4 code continues on branch c5/isolation, which contains this branch.
There a launch that waits always gets the full WAIT_FOR_IDLE + invalidate after the wait, a list with waits is refused by
plain submit and runs only in submit_pair with the list it waits on, and the chip test checks both refusals. The chip
receipt for C4 will be taken on c5/isolation; merge that branch, not this one alone.
