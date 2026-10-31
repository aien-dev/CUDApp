/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Verification oracle only: compile the actual incumbent pf_group unchanged.
 * Function sections + linker GC exclude the rest of the graph; no GPU object is linked.
 * pf_group's QMMA branch calls the real host tiler omega_pfq_tiles; run.sh links src/omega_gpu_pfq_api.c for it. */
#include "../../src/omega_gpu_q36_graph.c"
int omega_gpu_fp8_tensor_entry(const OmegaGpuFp8Tensor *t, uint32_t expert, uint64_t *w, uint64_t *s) {
    (void)t; *w = (uint64_t)expert * 256 + 4096; *s = *w + 128; return 0;
}
uint32_t moe_incumbent(uint32_t n, const uint32_t *ids, uint32_t *x, uint32_t *y, uint32_t *seg) {
    OmegaGpuQ36Graph g; memset(&g, 0, sizeof g);
    _Alignas(16) unsigned char mem[32768]; memset(mem, 0, sizeof mem);
    g.pf.hb.cpu = mem; g.lo.lb_stride = 32;
    g.pf.rlb = 0; g.pf.sgu = 4096; g.pf.sdn = 12288;
    g.pf.sgu2 = 20480; g.pf.sdn2 = 20512;
    g.pf.xmap = 20544; g.pf.ymap = 24672;
    g.pf.xmap2 = 28800; g.pf.ymap2 = 29344;
    memcpy(mem, ids, n * 8 * sizeof *ids);
    uint32_t ns = pf_group(&g, 0, n, 0);
    memcpy(x, mem + g.pf.xmap, n * 8 * sizeof *x);
    memcpy(y, mem + g.pf.ymap, n * 8 * sizeof *y);
    memcpy(x + n * 8, mem + g.pf.xmap2, n * sizeof *x);
    memcpy(y + n * 8, mem + g.pf.ymap2, n * sizeof *y);
    const Seg32 *s = (const Seg32 *)(mem + g.pf.sgu);
    for (uint32_t j = 0; j < ns; j++) {
        seg[j * 4] = (uint32_t)((s[j].w_va - 4096) / 256);
        seg[j * 4 + 1] = s[j].row0; seg[j * 4 + 2] = s[j].cnt; seg[j * 4 + 3] = 0;
    }
    if (n) { seg[ns * 4] = 256; seg[ns * 4 + 1] = n * 8; seg[ns * 4 + 2] = n; seg[ns * 4 + 3] = 1; ns++; }
    return ns;
}
