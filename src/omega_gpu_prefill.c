/* omega_gpu_prefill.c: device-resident prefill buffers, kernel cache and launch helper (OWN-30B T3b). See the header. */
#include "omega_gpu_prefill.h"
#include "omega_blackwell_codegen_pf.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_qmma.h"
#include "omega_gpu_platform.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_cmdlist.h"
#include <string.h>

#define PF_SLOTS 64
#define PF_STAGE_BYTES (64ull << 20)
#define PF_TIMEOUT_MS 20000ull
#define PF_SPIN_US 200000u /* spin on the marker: prefill launches are back to back, a sleep-poll costs ~100 us each */

typedef struct { bool valid; OmegaPfKey key; OmegaPfKernel k; OmegaGpuMem code; } Slot;
static struct {
    bool hooked;
    Slot slot[PF_SLOTS];
    OmegaGpuScratch stage;
    uint64_t chip_ns, launches;
} g;

const char *omega_gpu_pf_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_PF_OK: return "OK";
    case OMEGA_GPU_PF_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_PF_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_PF_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_PF_CHIP_FAIL: return "CHIP_FAIL";
    default: return "UNKNOWN";
    }
}

static void on_close(void) {
    for (unsigned i = 0; i < PF_SLOTS; i++) if (g.slot[i].valid) { omega_gpu_session_free(&g.slot[i].code); memset(&g.slot[i], 0, sizeof g.slot[i]); }
    omega_gpu_session_scratch_free(&g.stage);
}
int omega_gpu_pf_open_locked(void) {
    if (!g.hooked) { if (omega_gpu_session_on_close(on_close) != 0) { omega_gpu_session_set_error("prefill: no session hook slot left"); return OMEGA_GPU_PF_CHIP_FAIL; } g.hooked = true; }
    return omega_gpu_session_open() ? OMEGA_GPU_PF_OK : OMEGA_GPU_PF_CHIP_FAIL;
}

void omega_gpu_pf_stats(uint64_t *chip_ns, uint64_t *launches) { if (chip_ns) *chip_ns = g.chip_ns; if (launches) *launches = g.launches; }
void omega_gpu_pf_stats_reset(void) { g.chip_ns = 0; g.launches = 0; }

int omega_gpu_pf_alloc_dev(size_t bytes, OmegaGpuMem *out) {
    if (!out || bytes == 0) return OMEGA_GPU_PF_BAD_ARGS;
    memset(out, 0, sizeof *out);
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    if (rc == OMEGA_GPU_PF_OK) {
        OmegaGpuSession *s = omega_gpu_session_open();
        const OmegaGpuPlatform *p = omega_gpu_session_platform();
        const size_t want = (bytes + 0xfffu) & ~(size_t)0xfffu;
        OmegaGpuBufferDesc d = { .kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_GPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
                                 .gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED, .align = 0x1000u, .bytes = want };
        OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus;
        int prc = (s && p) ? p->dma_alloc(p->ctx, s->dev, &d, &out->handle, &cpu, &va, &bus) : OMEGA_GPU_PLAT_REFUSED_ARGS;
        if (prc != OMEGA_GPU_PLAT_OK || !cpu.ptr || cpu.len < want) {
            if (prc == OMEGA_GPU_PLAT_OK) (void)p->dma_free(p->ctx, out->handle);
            char e[96]; omega_gpu_fmt(e, sizeof e, "prefill: cached alloc of %zu bytes refused (%d)", want, prc);
            omega_gpu_session_set_error(e); memset(out, 0, sizeof *out); rc = OMEGA_GPU_PF_CHIP_FAIL;
        } else { out->kind = OMEGA_GPU_HANDLE_BUF_DATA; out->cpu = cpu.ptr; out->va = va.va; out->size = cpu.len; }
    }
    omega_gpu_session_unlock();
    return rc;
}
int omega_gpu_pf_alloc_host(size_t bytes, OmegaGpuMem *out) {
    if (!out || bytes == 0) return OMEGA_GPU_PF_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    if (rc == OMEGA_GPU_PF_OK && omega_gpu_session_alloc(bytes, out) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc;
}
void omega_gpu_pf_free(OmegaGpuMem *m) { omega_gpu_session_lock(); omega_gpu_session_free(m); omega_gpu_session_unlock(); }

int omega_gpu_pf_kernel_locked(const OmegaPfKey *key, OmegaPfGen gen, OmegaPfKernel *out) {
    for (unsigned i = 0; i < PF_SLOTS; i++) if (g.slot[i].valid && !memcmp(&g.slot[i].key, key, sizeof *key)) { *out = g.slot[i].k; return OMEGA_GPU_PF_OK; }
    Slot *s = NULL;
    for (unsigned i = 0; i < PF_SLOTS && !s; i++) if (!g.slot[i].valid) s = &g.slot[i];
    if (!s) { omega_gpu_session_set_error("prefill kernel cache full"); return OMEGA_GPU_PF_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("prefill codegen: host memory"); return OMEGA_GPU_PF_CODEGEN_FAIL; }
    int cg = gen(key, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("prefill codegen refused the shape"); return OMEGA_GPU_PF_CODEGEN_FAIL; }
    const uint32_t peak = prog->regalloc.peak_gpr_usage, decl = omega_bw_gpr_declared(peak);
    if (decl > 128u) { omega_gpu_host_free(prog); omega_gpu_session_set_error("prefill kernel: more than 128 registers"); return OMEGA_GPU_PF_TOO_LARGE; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_PF_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    if (cg == 0 && omega_tc_patch_code(prog, code.cpu, need) < 0) cg = -1;
    if (cg == 0 && omega_qmma_patch_code(prog, code.cpu, need) < 0) cg = -1;
    omega_gpu_host_free(prog);
    if (cg != 0 || kernel.code_size != need) { omega_gpu_session_free(&code); omega_gpu_session_set_error("prefill encode failed"); return OMEGA_GPU_PF_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_PF_CHIP_FAIL; }
    memset(s, 0, sizeof *s);
    s->valid = true; s->key = *key; s->code = code;
    s->k.code_va = code.va; s->k.gpr_peak = kernel.gpr_count; s->k.gpr_declared = omega_bw_gpr_declared(kernel.gpr_count);
    *out = s->k;
    return OMEGA_GPU_PF_OK;
}

int omega_gpu_pf_launch_locked(const OmegaPfKernel *k, uint32_t threads, uint32_t grid_x, uint32_t grid_y, uint32_t shared_bytes,
                               const uint32_t *args, uint32_t n_args, uint64_t *ns) {
    if (!k || !threads || !grid_x || !grid_y) return OMEGA_GPU_PF_BAD_ARGS;
    if ((uint64_t)threads * k->gpr_declared > 65536u) { omega_gpu_session_set_error("prefill launch: registers x threads exceed the file"); return OMEGA_GPU_PF_TOO_LARGE; }
    if (!omega_bw_gpr_declared_ok(k->gpr_declared, k->gpr_peak)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_PF_CHIP_FAIL; }
    OmegaGpuLaunch L = { .code_va = k->code_va, .gpr_count = k->gpr_declared, .threads_x = threads, .threads_y = 1, .grid_x = grid_x, .grid_y = grid_y,
                         .num_elements = (size_t)threads * grid_x * grid_y, .args = args, .n_args = n_args, .timeout_ms = PF_TIMEOUT_MS,
                         .shared_bytes = shared_bytes, .spin_us = PF_SPIN_US };
    uint64_t t = 0;
    if (omega_gpu_session_launch(&L, &t, NULL) != 0) return OMEGA_GPU_PF_CHIP_FAIL;
    g.chip_ns += t; g.launches++;
    if (ns) *ns += t;
    return OMEGA_GPU_PF_OK;
}

int omega_gpu_pf_record_locked(OmegaGpuCmdList *cl, const OmegaPfKernel *k, uint32_t threads, uint32_t grid_x, uint32_t grid_y,
                               uint32_t shared_bytes, const uint32_t *args, uint32_t n_args) {
    if (!cl || !k || !threads || !grid_x || !grid_y) return OMEGA_GPU_PF_BAD_ARGS;
    if ((uint64_t)threads * k->gpr_declared > 65536u) { omega_gpu_session_set_error("prefill record: registers x threads exceed the file"); return OMEGA_GPU_PF_TOO_LARGE; }
    if (!omega_bw_gpr_declared_ok(k->gpr_declared, k->gpr_peak)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to record"); return OMEGA_GPU_PF_CHIP_FAIL; }
    OmegaGpuLaunch L = { .code_va = k->code_va, .gpr_count = k->gpr_declared, .threads_x = threads, .threads_y = 1, .grid_x = grid_x, .grid_y = grid_y,
                         .num_elements = (size_t)threads * grid_x * grid_y, .args = args, .n_args = n_args, .timeout_ms = PF_TIMEOUT_MS,
                         .shared_bytes = shared_bytes, .spin_us = PF_SPIN_US };
    return omega_gpu_cmdlist_add(cl, &L) < 0 ? OMEGA_GPU_PF_CHIP_FAIL : OMEGA_GPU_PF_OK;
}

static int gen_copy(const OmegaPfKey *key, void *prog) { (void)key; return omega_blackwell_codegen_pf_copy_prog(prog); }
#define KIND_COPY 1u
static int copy_locked(uint64_t dst, uint64_t src, size_t bytes) {
    if ((bytes & 15u) || (dst & 15u) || (src & 15u)) return OMEGA_GPU_PF_BAD_ARGS;
    if (bytes == 0) return OMEGA_GPU_PF_OK;
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = KIND_COPY;
    OmegaPfKernel k; int rc = omega_gpu_pf_kernel_locked(&key, gen_copy, &k);
    for (size_t done = 0; rc == OMEGA_GPU_PF_OK && done < bytes;) {
        size_t part = bytes - done; if (part > (1ull << 30)) part = 1ull << 30;
        const uint32_t units = (uint32_t)(part / 16u), grid = (units + OMEGA_PF_COPY_THREADS - 1u) / OMEGA_PF_COPY_THREADS;
        const uint64_t d = dst + done, s = src + done;
        uint32_t args[5] = { (uint32_t)d, (uint32_t)(d >> 32), (uint32_t)s, (uint32_t)(s >> 32), units };
        rc = omega_gpu_pf_launch_locked(&k, OMEGA_PF_COPY_THREADS, grid, 1, 0, args, 5, NULL);
        done += part;
    }
    return rc;
}
int omega_gpu_pf_copy_va(uint64_t dst_va, uint64_t src_va, size_t bytes) {
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    if (rc == OMEGA_GPU_PF_OK) rc = copy_locked(dst_va, src_va, bytes);
    omega_gpu_session_unlock();
    return rc;
}

static int xfer(const OmegaGpuMem *dev, size_t off, void *host, size_t bytes, int up) {
    if (!dev || !host || (off & 15u) || (bytes & 15u) || off + bytes > dev->size) return OMEGA_GPU_PF_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    if (rc == OMEGA_GPU_PF_OK && omega_gpu_session_scratch(&g.stage, PF_STAGE_BYTES) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
    for (size_t done = 0; rc == OMEGA_GPU_PF_OK && done < bytes;) {
        size_t part = bytes - done; if (part > PF_STAGE_BYTES) part = PF_STAGE_BYTES;
        if (up) {
            memcpy(g.stage.mem.cpu, (const uint8_t *)host + done, part);
            if (omega_gpu_session_sync_to_gpu(&g.stage.mem) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
            else rc = copy_locked(dev->va + off + done, g.stage.mem.va, part);
        } else {
            rc = copy_locked(g.stage.mem.va, dev->va + off + done, part);
            if (rc == OMEGA_GPU_PF_OK && omega_gpu_session_sync_to_cpu(&g.stage.mem) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
            if (rc == OMEGA_GPU_PF_OK) memcpy((uint8_t *)host + done, g.stage.mem.cpu, part);
        }
        done += part;
    }
    omega_gpu_session_unlock();
    return rc;
}
int omega_gpu_pf_upload(const OmegaGpuMem *dst, size_t off, const void *src, size_t bytes) { return xfer(dst, off, (void *)src, bytes, 1); }
int omega_gpu_pf_download(const OmegaGpuMem *src, size_t off, void *dst, size_t bytes) { return xfer(src, off, dst, bytes, 0); }
