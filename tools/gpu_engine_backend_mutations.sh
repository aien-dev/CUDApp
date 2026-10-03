#!/bin/sh
# Omega GPU Engine BACKEND mutation sweep (cut A3b2), a thin adapter over
# tools/mutation_runner.sh, table reader. One mutant per protection in
# src/omega_blackwell_engine.c (the Blackwell backend) and in the vector wrapper
# omega_blackwell_execute_vector (src/omega_blackwell_submit.c). For each row the
# runner copies the needed files to a scratch directory, breaks that one
# protection, rebuilds tests/test_omega_blackwell_engine.c there (against the
# simulated driver tests/fake_m16_native.c) and requires it to FAIL. A mutant that
# does not apply, does not build or is not caught fails the sweep (exit nonzero),
# unless its row carries class "redundant" with a rationale.
# Host only, no device, no GPU, no physics C sources (the driver is the fake).
# The link set is the Makefile RX_COMPOSE_GPU_SRCS list plus the three core files the vector builder needs
# (omega_core.c, omega_canonical.c, sha256.c), with the fake driver in place of physics m16_native.c and nvrm.c.
# Shell only (no Python). The real sources carry no marker comments: the rows
# find their text with exact-once matches (R:) or line ranges (sed), and the
# runner reports "edit did not apply" if a row stops matching.
# Each mutant, what it guards and why a redundant one is redundant are listed in
# docs/numeric/OMEGA_GPU_ENGINE.md section 14.1. Set GPU_ENGINE_BMUT_OUT=path.json
# to also keep the JSON receipt. NOT_RUN in Phase A.
#
# Row format: id~files~edit~guards[~class~rationale]. A row cannot contain a tilde,
# so the one line that contains ~ (the poison complement) is matched with "." in a
# sed row. Edit kinds: "R:from@>@to" (from must match exactly once) or a sed
# expression (a "& " in a replacement is written \&; ranges like
# /^static int be_close/,/^}/ keep an edit inside one function).
set -u
cd "$(dirname "$0")/.." || exit 2
unset ONLY

# The scratch copy lives in /tmp, where ../physics does not resolve, so the physics
# path is made absolute here (PHYSICS_DIR, default ../physics).
PHYS=$(cd "${PHYSICS_DIR:-../physics}" 2>/dev/null && pwd -P) || {
    echo "gpu_engine_backend_mutations: physics directory not found (set PHYSICS_DIR)" >&2
    exit 2
}
TP=$PHYS/third_party/nvidia-open-580.173.02
INC="-I$PHYS/m16 -I$PHYS/nvrm -I$TP/src/common/sdk/nvidia/inc -I$TP/kernel-open/common/inc -I$TP/kernel-open/nvidia-uvm -I$TP/src/nvidia/arch/nvalloc/unix/include"
SRCS="src/omega_blackwell_submit.c src/omega_blackwell_engine.c src/omega_gpu_engine.c src/omega_blackwell_matmul.c src/omega_blackwell_codegen.c src/omega_blackwell_encoder.c src/omega_blackwell_qmd.c src/omega_blackwell_realize.c src/omega_vector.c src/omega_validate.c src/omega_core.c src/omega_canonical.c src/sha256.c"
BUILD="gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread -ffunction-sections -fdata-sections -Isrc -Itests $INC -o .gpu_engine_bt tests/test_omega_blackwell_engine.c tests/fake_m16_native.c $SRCS -Wl,--gc-sections -lm"

OUT=${GPU_ENGINE_BMUT_OUT:-}
set -- -k table -m - -t ./.gpu_engine_bt -b "$BUILD" \
    -c "src tests/test_omega_blackwell_engine.c tests/fake_m16_native.c tests/fake_m16_native.h" \
    -f '\[FAIL\]' -B
[ -z "$OUT" ] || set -- "$@" -o "$OUT"

tools/mutation_runner.sh "$@" <<'ROWS'
# ---- allocation checks and frees ------------------------------------------------
BE_ALLOC_CHECK_CPU~src/omega_blackwell_engine.c~R:if (!m->cpu || m->size < bytes ||@>@if (0 || m->size < bytes ||~a buffer without a CPU mapping is refused (ALLOC)
BE_ALLOC_CHECK_SIZE~src/omega_blackwell_engine.c~R:|| m->size < bytes ||@>@|| 0 ||~a buffer smaller than asked is refused (ALLOC)
BE_ALLOC_CHECK_ALIGN~src/omega_blackwell_engine.c~R:|| (m->va & 255u) != 0) {@>@|| 0) {~a GPU address that is not 256-byte aligned is refused (ALLOC)
BE_ALLOC_OWN_CHECK_NOTE~src/omega_blackwell_engine.c~R:note(c, "allocation failed its own checks (CPU mapping, size or 256-byte alignment)");@>@(void)0;~the backend's own refusal text reaches the result
BE_ALLOC_OWN_CHECK_FREE~src/omega_blackwell_engine.c~R:(void)nvrm_free(&c->m.rm, m);@>@(void)0;~a buffer that fails its own checks is freed, not leaked
BE_ROLE_FREE_SKIPPED~src/omega_blackwell_engine.c~R:rc = nvrm_free(&c->m.rm, m);@>@rc = 0;~free_buf really frees a program, input or output buffer
BE_SCRATCH_FREE_SKIPPED~src/omega_blackwell_engine.c~R:int rc = nvrm_free(&c->m.rm, parts[i]);@>@int rc = (parts[i] == NULL) ? -1 : 0;~release_scratch really frees all four SCRATCH parts
BE_SCRATCH_FREE_ERROR_DROPPED~src/omega_blackwell_engine.c~R:if (rc != 0 && first == 0) first = rc;@>@if (0 && rc != 0 && first == 0) first = rc;~a failed SCRATCH part free is reported
BE_SCRATCH_FREE_STOPS_AT_ERROR~src/omega_blackwell_engine.c~R:if (rc != 0 && first == 0) first = rc;@>@if (rc != 0) { first = rc; break; }~one failed SCRATCH part free does not leave the other parts allocated
BE_SCRATCH_ALLOC_FAIL_NO_RELEASE~src/omega_blackwell_engine.c~R:(void)release_scratch(c);@>@(void)0;~a failed SCRATCH allocation frees the parts that did succeed (the core does not)
BE_ERRNO_NOT_RESTORED~src/omega_blackwell_engine.c~R:errno = saved_errno;@>@(void)saved_errno;~the frees after a failed SCRATCH allocation do not clobber the errno the core reads
BE_ALLOC_FAIL_TEXT_RESTORE~src/omega_blackwell_engine.c~R:copy_text(c->text, sizeof c->text, keep);@>@(void)keep;~the failure text survives the cleanup frees~redundant~release_scratch writes neither c->text nor the driver text on a successful free, in the simulated driver and in nvrm_free (physics nvrm.h: free only zeroes the NvrmMem); the restore is a safety net for a driver that writes text on success, which no model here does
BE_SCRATCH_PB_NOT_INSTALLED~src/omega_blackwell_engine.c~R:c->m.pb_mem = c->pb;@>@(void)0;~the large pushbuffer ring replaces the channel's 4 KiB buffer
BE_PB_REPLACED_RESET~src/omega_blackwell_engine.c~R:c->pb_replaced = 0;@>@(void)0;~the flag that records the replaced ring is cleared after SCRATCH is freed~redundant~the zeroing of m.pb_mem stays (it is a separate line); a stale flag only repeats that harmless zeroing, and open memsets the whole context each run. physics m16_native_close is only nvrm_close(&ctx->rm) and never reads pb_mem
# ---- busy flag and open ----------------------------------------------------------
BE_BUSY_CHECK_DROPPED~src/omega_blackwell_engine.c~R:if (atomic_flag_test_and_set(&g_busy)) return -1;@>@(void)atomic_flag_test_and_set(&g_busy);~a second open while a run is retained is refused
BE_OPEN_FAIL_KEEPS_BUSY~src/omega_blackwell_engine.c~/^static int be_open_device/,/^}/ s/atomic_flag_clear(&g_busy);/(void)0;/~a failed open releases the busy flag
BE_CLOSE_BUSY_CLEAR_DROPPED~src/omega_blackwell_engine.c~/^static int be_close/,/^}/ s/atomic_flag_clear(&g_busy);/(void)0;/~close releases the busy flag so the next run can open
BE_OPEN_MEMSET_DROPPED~src/omega_blackwell_engine.c~R:memset(c, 0, sizeof *c);@>@(void)0;~open resets the whole context, including the previous run's snapshot
BE_OPEN_NATIVE_FLAG_DROPPED~src/omega_blackwell_engine.c~R:c->native_open = 1;@>@(void)0;~a failed open still reports the driver's own text
BE_OPEN_ERROR_IGNORED~src/omega_blackwell_engine.c~/^static int be_open_device/,/^}/ s/return rc;/return 0;/~a failed driver open is reported (DEVICE_OPEN)
BE_OPEN_LAST_RC_DROPPED~src/omega_blackwell_engine.c~/^static int be_open_device/,/^}/ s/c->last_rc = rc;/(void)0;/~the failed open's driver code reaches the result
BE_CHANNEL_ERROR_IGNORED~src/omega_blackwell_engine.c~/^static int be_create_channel/,/^}/ s/return rc;/return 0;/~a failed channel create is reported (CHANNEL_CREATE)
BE_CHANNEL_LAST_RC_DROPPED~src/omega_blackwell_engine.c~/^static int be_create_channel/,/^}/ s/c->last_rc = rc;/(void)0;/~the failed channel create's driver code reaches the result
BE_ALLOC_LAST_RC_DROPPED~src/omega_blackwell_engine.c~/^static int alloc_checked/,/^}/ s/c->last_rc = rc;/(void)0;/~the failed allocation's driver code reaches the result
# ---- roles, rounding, copies -------------------------------------------------------
BE_ALLOC_UNKNOWN_ROLE_ACCEPTED~src/omega_blackwell_engine.c~/^static int be_alloc/,/^}/ s/if (!m) {/if (0) {/~alloc refuses an unknown role
BE_ROLE_MEM_UPPER_BOUND~src/omega_blackwell_engine.c~R:|| role >= (int)OMEGA_GPU_BUF_ROLE_COUNT) return NULL;@>@|| 0) return NULL;~a role above the last one has no buffer slot
BE_ROLE_MEM_LOWER_BOUND~src/omega_blackwell_engine.c~R:if (role < (int)OMEGA_GPU_BUF_PROGRAM ||@>@if (0 ||~a negative role has no buffer slot
BE_ROUND_PAGES_ZERO_ACCEPTED~src/omega_blackwell_engine.c~R:if (bytes == 0 ||@>@if (0 ||~a zero-byte allocation is refused
BE_ROUND_PAGES_NO_ROUNDING~src/omega_blackwell_engine.c~s/v = ((uint64_t)bytes + (BE_PAGE - 1u)) & .(uint64_t)(BE_PAGE - 1u);/v = (uint64_t)bytes;/~allocation sizes round up to whole pages
BE_ROUND_PAGES_MINIMUM~src/omega_blackwell_engine.c~R:if (v < BE_PAGE) v = BE_PAGE;@>@(void)0;~one-page minimum~redundant~after the zero check, rounding any nonzero byte count up to a page multiple already gives at least one page, so the minimum can never apply
BE_COPY_IN_NO_MEMCPY~src/omega_blackwell_engine.c~R:memcpy(m->cpu, src, bytes);@>@(void)0;~copy_in really copies the program and the inputs
BE_COPY_IN_BOUNDS~src/omega_blackwell_engine.c~R:!src || (uint64_t)bytes > m->size@>@!src~copy_in refuses a length beyond the buffer
BE_COPY_IN_NULL_SOURCE~src/omega_blackwell_engine.c~R:!m->cpu || !src ||@>@!m->cpu ||~copy_in refuses a NULL source
BE_FILL_POISON_DROPPED~src/omega_blackwell_engine.c~R:memcpy(out->cpu, words, count * 4u);@>@(void)0;~the device output is pre-filled with the poison words
BE_FILL_POISON_SHORT~src/omega_blackwell_engine.c~R:memcpy(out->cpu, words, count * 4u);@>@memcpy(out->cpu, words, (count - 1u) * 4u);~every output word is pre-filled, including the last
BE_FILL_POISON_BOUNDS~src/omega_blackwell_engine.c~R:!words || (uint64_t)count * 4u > out->size@>@!words~fill_poison refuses a length beyond the output buffer
BE_COPY_OUT_NO_MEMCPY~src/omega_blackwell_engine.c~R:memcpy(dst, out->cpu, bytes);@>@(void)0;~copy_out really reads the device output back
BE_COPY_OUT_BOUNDS~src/omega_blackwell_engine.c~R:!dst || (uint64_t)bytes > out->size@>@!dst~copy_out refuses a length beyond the output buffer
# ---- build: arguments, descriptors, sync words, pushbuffer ---------------------------
BE_BUILD_LAYOUT_CHECK~src/omega_blackwell_engine.c~R:if (!job || job->layout != (int)OMEGA_GPU_LAYOUT_VECTOR_1D) {@>@if (0) {~build refuses a missing job and an unsupported layout
BE_BUILD_COUNT_OVERFLOW~src/omega_blackwell_engine.c~R:n == 0 || n > UINT32_MAX - 63u@>@n == 0~grid arithmetic overflow guard~redundant~an element count above UINT32_MAX - 63 needs at least 16 GiB of input, output and input buffers, so the buffer-size check in the same condition chain refuses it first; only the grid rounding (n + 63) could overflow, and it is never reached
BE_BUILD_BUFFERS_SIZE~src/omega_blackwell_engine.c~R:|| a->size < need || b->size < need || out->size < need) {@>@|| need == 1) {~build refuses buffers smaller than the element count needs
BE_C3_FLAG_IGNORED~src/omega_blackwell_engine.c~R:c->c3 = (job->flags & OMEGA_GPU_ENGINE_FLAG_NO_C3) == 0;@>@c->c3 = 1;~the NO_C3 flag removes the L2 flush and the second marker
BE_C3_NEVER~src/omega_blackwell_engine.c~R:c->c3 = (job->flags & OMEGA_GPU_ENGINE_FLAG_NO_C3) == 0;@>@c->c3 = 0;~with flags 0 the L2 flush and the second marker are in the pushbuffer
BE_THREADS_WRONG~src/omega_blackwell_engine.c~R:cfg.threads_per_block = 64;@>@cfg.threads_per_block = 32;~the launch covers every element
BE_GRID_ROUNDS_DOWN~src/omega_blackwell_engine.c~R:cfg.grid_width = (n + 63u) / 64u;@>@cfg.grid_width = n / 64u;~the grid rounds up so a partial last block still runs
BE_ARGS_C_SWAPPED~src/omega_blackwell_engine.c~R:a->va, b->va, out->va, n)@>@a->va, out->va, b->va, n)~the kernel writes the OUTPUT buffer
BE_ARGS_A_B_SWAPPED~src/omega_blackwell_engine.c~R:a->va, b->va, out->va, n)@>@b->va, a->va, out->va, n)~the kernel gets input A and input B in the right argument slots
BE_PROGRAM_VA~src/omega_blackwell_engine.c~R:cfg.code_va = code->va;@>@cfg.code_va = c->qmd.va;~the QMD points at the program buffer
BE_CBANK_VA~src/omega_blackwell_engine.c~R:cfg.cbank_va = c->cbank.va;@>@cfg.cbank_va = c->marker.va;~the QMD points at the constant bank buffer
BE_SEM_VA~src/omega_blackwell_engine.c~R:sem_va = c->qmd.va + BE_SEM_OFFSET;@>@sem_va = c->qmd.va + BE_SEM_OFFSET + 0x100u;~the QMD release semaphore address is the word the host waits on
BE_MARKER_VA~src/omega_blackwell_engine.c~R:marker_va = c->marker.va;@>@marker_va = c->marker.va + 0x20u;~the first release lands on the word the host waits on
BE_SEM_ZERO_DROPPED~src/omega_blackwell_engine.c~R:*c->hsem = 0;@>@(void)0;~the semaphore word is zeroed before the doorbell (no stale DONE)
BE_MARKER_ZERO_DROPPED~src/omega_blackwell_engine.c~R:*c->hmarker = 0;@>@(void)0;~the first marker word is zeroed before the doorbell (no stale payload)
BE_MARKER2_ZERO_DROPPED~src/omega_blackwell_engine.c~R:*c->hmarker2 = 0;@>@(void)0;~the second marker word is zeroed before the doorbell
BE_NOC3_MARKER2_REPORTED~src/omega_blackwell_engine.c~R:if (!c->c3) c->hmarker2 = NULL;@>@(void)0;~with NO_C3 the second marker is neither waited for nor reported
BE_MARKER2_OFFSET_CHANGED~src/omega_blackwell_engine.c~R:#define BE_MARKER2_OFFSET 0x10u@>@#define BE_MARKER2_OFFSET 0x20u~the second marker is 0x10 bytes after the first
BE_MARKER2_PAYLOAD_CHANGED~src/omega_blackwell_engine.c~R:#define BE_MARKER2_PAYLOAD 0x46464646u@>@#define BE_MARKER2_PAYLOAD 0x46464647u~the second marker payload is 0x46464646
BE_RELEASE1_PAYLOAD_ZERO~src/omega_blackwell_engine.c~R:pb_put(c, OMEGA_BW_MARKER_COMPLETION_PAYLOAD);@>@pb_put(c, 0);~the first release writes the completion payload
BE_RELEASE2_PAYLOAD_ZERO~src/omega_blackwell_engine.c~R:pb_put(c, BE_MARKER2_PAYLOAD);@>@pb_put(c, 0);~the second release writes its payload
BE_FLUSH_OP_DROPPED~src/omega_blackwell_engine.c~R:pb_put(c, 0x10u << 27);@>@pb_put(c, 0);~the L2 flush operation word is L2_FLUSH_DIRTY
BE_FLUSH_METHOD_CHANGED~src/omega_blackwell_engine.c~R:pb_put(c, nvrm_mthd(0, 0x0028, 4));@>@pb_put(c, nvrm_mthd(0, 0x002c, 4));~the flush uses the memory-operation method
BE_WFI_FLAG_RELEASE1~src/omega_blackwell_engine.c~/pb_put(c, OMEGA_BW_MARKER_COMPLETION_PAYLOAD);/,+2 s/0x1u | (1u << 20)/0x1u/~the first release waits for idle (WFI)
BE_WFI_FLAG_RELEASE2~src/omega_blackwell_engine.c~/pb_put(c, BE_MARKER2_PAYLOAD);/,+2 s/0x1u | (1u << 20)/0x1u/~the second release waits for idle (WFI)
BE_BARRIER_IN_BUILD~src/omega_blackwell_engine.c~/^static int be_build/,/^}/ s/__asm__ volatile("dsb sy" ::: "memory");/(void)0;/~dsb sy after zeroing the sync words~redundant~a CPU barrier instruction has no effect a host test can observe (the fake driver runs on the same core); it is verified on the chip only (Phase B)
BE_QMD_RELEASE_CHECK~src/omega_blackwell_engine.c~R:if (qmd1_words[9] != 3u ||@>@if (0 ||~extra QMD1 release-word checks~redundant~the real descriptor builder always produces these words (src/omega_blackwell_qmd.c), so no input makes the check fire; it guards a future change of the builder, which tests/test_omega_blackwell_qmd.c style checks cover
BE_PB_OVERFLOW~src/omega_blackwell_engine.c~R:if (c->pb_overflow) {@>@if (0) {~pushbuffer word-capacity guard~redundant~the vector pushbuffer is a fixed 492 words against a 1024-word capacity, so it cannot overflow for any job
# ---- submit, waits, diagnostics ------------------------------------------------------
BE_SUBMIT_NOTHING_BUILT~src/omega_blackwell_engine.c~R:if (c->pb_len == 0) {@>@if (0) {~submit before build is refused~redundant~the driver refuses an empty submission too (physics m16_native.c m16_native_enqueue_methods: count == 0 returns -1), so the result is the same refusal
BE_LAUNCH_NS_DROPPED~src/omega_blackwell_engine.c~R:c->info.launch_ns = now_ns();@>@(void)0;~the launch time is recorded before the doorbell
BE_SUBMIT_ERROR_IGNORED~src/omega_blackwell_engine.c~/^static int be_submit/,/^}/ s/return rc;/return 0;/~a failed submit is reported (SUBMIT), not treated as launched
BE_WAIT_UNMAPPED_ACCEPTED~src/omega_blackwell_engine.c~R:if (!w) {@>@if (0) {~a wait before build or after free is refused without a driver call
BE_WAIT_DRIVER_RC_DROPPED~src/omega_blackwell_engine.c~/^static int wait_word/,/^}/ s/if (rc != 0) {/if (0) {/~the driver's wait code reaches the result
BE_WAIT_RC_MAPPING~src/omega_blackwell_engine.c~R:return rc == -1 ? OMEGA_GPU_BACKEND_TIMEOUT : -2;@>@return rc;~timeout versus driver fault classification~redundant~the engine core treats every nonzero wait result the same (uncertain completion), and the real m16_native_wait_marker returns only 0 or -1 (m16_native.c)
BE_WAIT_EQUALITY_RECHECK~src/omega_blackwell_engine.c~R:if (*snap != want) {@>@if (0) {~an overshoot of the ">=" compare is refused (UNKNOWN_STATE)
BE_WAIT_SNAPSHOT_DROPPED~src/omega_blackwell_engine.c~R:*snap = *w;@>@(void)0;~the observed word is recorded after every wait
BE_MARKER_DONE_NS_ALWAYS~src/omega_blackwell_engine.c~R:if (rc == 0) c->info.marker_done_ns = now_ns();@>@c->info.marker_done_ns = now_ns();~the completion time is recorded only when the first marker wait succeeded
BE_WAIT_MARKER_WRONG_WORD~src/omega_blackwell_engine.c~R:wait_word(c, c->hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD@>@wait_word(c, c->hmarker2, OMEGA_BW_MARKER_COMPLETION_PAYLOAD~wait_marker waits on the first marker word
BE_WAIT_SEMAPHORE_WRONG_VALUE~src/omega_blackwell_engine.c~R:c->hsem, OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE,@>@c->hsem, OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT,~the release semaphore wait is for DONE (6), not the initial value (5)
BE_DIAG_DRV_RC_DROPPED~src/omega_blackwell_engine.c~R:d->drv_rc = c->last_rc;@>@d->drv_rc = 0;~diagnostics report the raw driver code
BE_DIAG_TEXT_OWN_DROPPED~src/omega_blackwell_engine.c~R:if (c->text[0] != '\0') copy_text(@>@if (0) copy_text(~diagnostics report the backend's own text
BE_DIAG_TEXT_DRIVER_DROPPED~src/omega_blackwell_engine.c~R:else if (c->native_open) copy_text(@>@else if (0) copy_text(~diagnostics fall back to the driver's text
BE_DIAG_MARKER_DROPPED~src/omega_blackwell_engine.c~R:d->marker = *c->hmarker;@>@d->marker = 0;~diagnostics report the first marker word
BE_DIAG_MARKER2_DROPPED~src/omega_blackwell_engine.c~R:d->marker2 = *c->hmarker2;@>@d->marker2 = 0;~diagnostics report the second marker word
BE_DIAG_SEMAPHORE_DROPPED~src/omega_blackwell_engine.c~R:d->semaphore[0] = *c->hsem;@>@d->semaphore[0] = 0;~diagnostics report the semaphore word
BE_DIAG_VALID_MARKER_DROPPED~src/omega_blackwell_engine.c~R:d->sync_valid |= OMEGA_GPU_SYNC_MARKER;@>@(void)0;~the first marker is flagged valid only when mapped
BE_DIAG_VALID_MARKER2_DROPPED~src/omega_blackwell_engine.c~R:d->sync_valid |= OMEGA_GPU_SYNC_MARKER2;@>@(void)0;~the second marker is flagged valid only when mapped
BE_DIAG_VALID_SEMAPHORE_DROPPED~src/omega_blackwell_engine.c~R:d->sync_valid |= OMEGA_GPU_SYNC_SEMAPHORE;@>@(void)0;~the semaphore is flagged valid only when mapped
# ---- free_buf and close ---------------------------------------------------------------
BE_FREE_SCRATCH_ROLE_DROPPED~src/omega_blackwell_engine.c~/^static int be_free_buf/,/^}/ s/if (role == (int)OMEGA_GPU_BUF_SCRATCH) {/if (0) {/~free_buf of SCRATCH frees the four SCRATCH parts
BE_FREE_UNKNOWN_ROLE_ACCEPTED~src/omega_blackwell_engine.c~/^static int be_free_buf/,/^}/ s/if (!m) {/if (0) {/~free_buf refuses an unknown role
BE_FREE_ERROR_DROPPED~src/omega_blackwell_engine.c~/^static int be_free_buf/,/^}/ s/return rc;/return 0;/~a failed free is reported (CLEANUP)
BE_RELEASE_KEEPS_MARKER_POINTER~src/omega_blackwell_engine.c~/^static int release_scratch/,/^}/ s/c->hmarker = NULL;/(void)0;/~the first marker word is unmapped before SCRATCH is freed
BE_RELEASE_KEEPS_MARKER2_POINTER~src/omega_blackwell_engine.c~/^static int release_scratch/,/^}/ s/c->hmarker2 = NULL;/(void)0;/~the second marker word is unmapped before SCRATCH is freed
BE_RELEASE_KEEPS_SEM_POINTER~src/omega_blackwell_engine.c~/^static int release_scratch/,/^}/ s/c->hsem = NULL;/(void)0;/~the semaphore word is unmapped before SCRATCH is freed
BE_CLOSE_KEEPS_MARKER_POINTER~src/omega_blackwell_engine.c~/^static int be_close/,/^}/ s/c->hmarker = NULL;/(void)0;/~close unmaps the first marker word
BE_CLOSE_KEEPS_MARKER2_POINTER~src/omega_blackwell_engine.c~/^static int be_close/,/^}/ s/c->hmarker2 = NULL;/(void)0;/~close unmaps the second marker word
BE_CLOSE_KEEPS_SEM_POINTER~src/omega_blackwell_engine.c~/^static int be_close/,/^}/ s/c->hsem = NULL;/(void)0;/~close unmaps the semaphore word
BE_CLOSE_NOT_CALLED~src/omega_blackwell_engine.c~R:rc = m16_native_close(&c->m);@>@rc = 0;~close really closes the driver
BE_CLOSE_ERROR_IGNORED~src/omega_blackwell_engine.c~/^static int be_close/,/^}/ s/return rc;/return 0;/~a failed close is reported (CLEANUP)
# ---- the vector wrapper (src/omega_blackwell_submit.c) ----------------------------------
WRAP_POISON_NOT_COMPLEMENT~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/poison\[i\] = .(uint32_t)(h_a\[i\] + h_b\[i\]);/poison[i] = (uint32_t)(h_a[i] + h_b[i]);/~the poison word is the complement of the expected sum
WRAP_POISON_CONSTANT~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/poison\[i\] = .(uint32_t)(h_a\[i\] + h_b\[i\]);/poison[i] = 0xdeadbeefu;/~the poison word never collides with a legitimate sum (0xdeadbeef is one)
WRAP_ENGINE_FAILURE_IGNORED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/if (erc != OMEGA_GPU_ENGINE_OK) {/if ((void)erc, 0) {/~an engine failure (here a cleanup fault) fails the wrapper
WRAP_ORACLE_IGNORED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/return parity_res;/return 0;/~the host oracle decides success, not the engine
WRAP_PARITY_FLAG_FORCED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->parity_verified = (parity_res == 0);/exec_info->parity_verified = 1;/~parity_verified follows the oracle
WRAP_MARKER_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->completion_marker = info.marker;/exec_info->completion_marker = 0;/~the completion marker is reported
WRAP_SEMAPHORE_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->intermediate_semaphore = info.semaphore;/exec_info->intermediate_semaphore = 0;/~the semaphore value is reported
WRAP_LAUNCH_TIME_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->launch_timestamp_ns = info.launch_ns;/exec_info->launch_timestamp_ns = 0;/~the launch time is reported
WRAP_DONE_TIME_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->completion_timestamp_ns = info.marker_done_ns;/exec_info->completion_timestamp_ns = 0;/~the completion time is reported
WRAP_ELAPSED_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->elapsed_ns = .*$/exec_info->elapsed_ns = 0;/~the elapsed time is reported
WRAP_SM_ARCH_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->sm_architecture = real->sm_architecture;/exec_info->sm_architecture = 0;/~the architecture is reported
WRAP_ELEMENT_COUNT_NOT_REPORTED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/exec_info->element_count = n;/exec_info->element_count = 0;/~the element count is reported
WRAP_NULL_GUARD_DROPPED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/if (!spec || !real || !h_a || !h_b || !h_c_out) return -1;/(void)0;/~a NULL argument is refused before any use
WRAP_ZERO_COUNT_GUARD~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/if (n == 0) return -1;/(void)0;/~a zero element count is refused by the wrapper~redundant~the engine refuses element_count 0 as INVALID_ARGS (src/omega_gpu_engine.c job_valid) and the wrapper then returns -1, so the result is the same refusal
WRAP_MARKER_TIMEOUT~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/job.timeouts.marker_ms = 5000;/job.timeouts.marker_ms = 6000;/~the first marker wait limit is 5000 ms
WRAP_SEMAPHORE_TIMEOUT~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/job.timeouts.release_semaphore_ms = 5000;/job.timeouts.release_semaphore_ms = 6000;/~the semaphore wait limit is 5000 ms
WRAP_MARKER2_TIMEOUT~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/job.timeouts.marker2_ms = 5000;/job.timeouts.marker2_ms = 6000;/~the second marker wait limit is 5000 ms
WRAP_VISIBILITY_TIMEOUT~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/job.timeouts.visibility_ms = 5000;/job.timeouts.visibility_ms = 6000;/~the visibility limit is 5000 ms~redundant~the Blackwell barrier callback ignores its timeout argument (it is a CPU dsb sy that cannot wait), so no host observation depends on the value
WRAP_FLAGS_ALL_PROTECTIONS~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/job.flags = 0;/job.flags = 1;/~the wrapper runs with every protection on (flags 0)
WRAP_BACKEND_NOT_INSTALLED~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/omega_gpu_engine_set_backend(omega_blackwell_engine_backend());/(void)0;/~the wrapper installs the Blackwell backend itself
WRAP_PROGRAM_LENGTH~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/job.program_len = encoded_len;/job.program_len = 16;/~the whole encoded program is handed to the engine
WRAP_MARKER_CHECK~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/if (info.marker != OMEGA_BW_MARKER_COMPLETION_PAYLOAD) return -1;/(void)0;/~second check of the marker payload~redundant~the backend wait_word already refuses any first-marker value other than 0x44444444 (UNKNOWN_STATE), so the engine fails before this line is reached
WRAP_SEMAPHORE_CHECK~src/omega_blackwell_submit.c~/^int omega_blackwell_execute_vector/,/^}/ s/if (info.semaphore != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) return -1;/(void)0;/~second check of the semaphore value~redundant~the backend wait_word already refuses any semaphore value other than DONE (6), so the engine fails before this line is reached
ROWS
