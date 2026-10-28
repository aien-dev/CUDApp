/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Verification-only deserializer into the existing BlackwellIRProgram.
 * No instructions, allocator, numerical algorithm, or interpreter live here. */
#ifndef COMPONENT_NATIVE_FRAME_H
#define COMPONENT_NATIVE_FRAME_H
static BlackwellIRProgram native_program;
static int frame_word(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (fread(b,1,8,f)!=8) return 0;
    *v=0; for(unsigned j=0;j<8;j++) *v|=(uint64_t)b[j]<<(j*8);
    return 1;
}
static int frame_reg(uint64_t n) { return n==UINT64_MAX ? -1 : (int)n; }
static int frame_load(const char *path,int staged) {
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    uint64_t h[6],q[8],v; int ok=0;
    memset(&native_program,0,sizeof native_program);
    if(staged && (!frame_word(f,&v)||v)) goto done;
    for(unsigned j=0;j<6;j++) if(!frame_word(f,&h[j])) goto done;
    if(!h[0]||h[0]>BW_MAX_IR_INSNS||h[1]>BW_MAX_VREGS||h[2]>BW_MAX_UVREGS||h[4]>256||h[5]>64) goto done;
    native_program.count=(size_t)h[0];
    OmegaRegAlloc *r=&native_program.regalloc;
    r->num_vregs=(int)h[1];r->num_uvregs=(int)h[2];r->peak_gpr_usage=(uint32_t)h[4];r->peak_ugpr_usage=(uint32_t)h[5];
    for(unsigned bank=0;bank<2;bank++) for(int j=0;j<(bank?r->num_uvregs:r->num_vregs);j++) {
        uint64_t width,phys;
        if(!frame_word(f,&width)||!frame_word(f,&phys)||!width||width>128||
           (phys!=UINT64_MAX&&(phys>=(bank?64u:256u)||width>(bank?64u:256u)-phys))) goto done;
        OmegaLiveInterval *l=bank?&r->uintervals[j]:&r->intervals[j];
        l->bundle_size=(uint32_t)width;l->is_pair=width==2;l->is_quad=width==4;l->vreg=j;l->phys_reg=frame_reg(phys);
        if(bank) r->uvreg_to_phys[j]=frame_reg(phys);else r->vreg_to_phys[j]=frame_reg(phys);
    }
    for(size_t j=0;j<native_program.count;j++) {
        for(unsigned k=0;k<8;k++) if(!frame_word(f,&q[k])) goto done;
        BlackwellIRInsn *i=&native_program.insns[j];
        i->op=(BlackwellIROpcode)q[0];i->dst_vreg=frame_reg(q[1]);i->src1_vreg=frame_reg(q[2]);i->src2_vreg=frame_reg(q[3]);i->src3_vreg=frame_reg(q[4]);i->ureg=frame_reg(q[5]);
        i->imm=(uint32_t)q[6];i->control=(uint32_t)(q[6]>>32);
        i->is_uniform=q[7]&1;i->predicate_p0=(q[7]>>1)&1;i->predicate_not=(q[7]>>2)&1;
        i->bar_reg=(uint8_t)(q[7]>>8);i->dst_subreg=(uint8_t)(q[7]>>16);i->src1_subreg=(uint8_t)(q[7]>>24);i->src2_subreg=(uint8_t)(q[7]>>32);i->src3_subreg=(uint8_t)(q[7]>>40);
    }
    /* Code is independently compared/disassembled by run.py, never interpreted here. */
    { size_t bytes=0;while(fgetc(f)!=EOF) bytes++;
      if(ferror(f)||bytes<native_program.count*16||bytes>16384) goto done; }
    ok=1;
done:
    fclose(f);return ok;
}
static int frame_wrong_result(void) {
    for(size_t j=0;j<native_program.count;j++) if(native_program.insns[j].op==BW_IR_FFMA) {
        native_program.insns[j].op=BW_IR_FADD;return 1;
    }
    return 0;
}
#endif
