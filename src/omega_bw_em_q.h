/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* omega_bw_em_q.h: private emitters for the native FP8 ops (W-L1G, 2026-10-14). Include after omega_bw_em.h.
 * Both are placeholders patched by omega_qmma_patch_code (omega_blackwell_codegen_qmma.h). */
#ifndef OMEGA_BW_EM_Q_H
#define OMEGA_BW_EM_Q_H
#include "omega_blackwell_codegen_qmma.h"
/* QMMA.16832.F32.E4M3.E4M3 d = a * b + c (c NOREG = RZ); operands may start at a subregister (b pair inside a quad) */
static void qmma_s(Em *e, int d, int a, int b, unsigned bsub, int c, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_HMMA_BF16); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; i.imm = OMEGA_QMMA_MARK;
    i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}
static void qmma(Em *e, int d, int a, int b, int c, uint32_t stall) { qmma_s(e, d, a, b, 0, c, stall); }
/* F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C Rd, Ra, Rb, RZ: Rd = e4m3(Ra) << 8 | e4m3(Rb) (byte order INFERRED, chip probe checks) */
static void f2e4(Em *e, int d, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.imm = OMEGA_QMMA_MARK_F2E4;
    emit(e, i, K_FIXED);
}
#endif
