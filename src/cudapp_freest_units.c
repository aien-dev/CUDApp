/* Generated pure integer-scalar freestanding native ABI v1 binding. */
#include <stdint.h>
#define OMEGA_OK 0u
#define OMEGA_BAD_ARG 32u
#define OMEGA_BAD_RT 34u
typedef struct ombr_rt { const void *entry[32]; } ombr_rt;
uint32_t ombr_enter(const void *code, const uint64_t *args, const ombr_rt *rt, uint64_t *out, void *top)
    __attribute__((visibility("hidden")));
extern const unsigned char ombr_trap[] __attribute__((visibility("hidden")));
extern const unsigned char ombr_no_entry[] __attribute__((visibility("hidden")));
/* ombr_enter saves the callee-saved registers on the caller's stack, keeps that stack pointer in x19 and
 * runs the pure scalar code on the caller's stack. A normal return stores x0 through out and
 * returns 0. A runtime entry restores the stack pointer from x19 and returns a status; out is not
 * written. The seeded compiler never writes x19..x28. Runtime entries are reached by blr/br, so they
 * start with bti jc. */
__asm__(".text\n.balign 4\n.local ombr_enter\n.hidden ombr_enter\n.type ombr_enter,%function\nombr_enter:\n"
        "hint #34\nstp x29,x30,[sp,#-176]!\nmov x29,sp\nstp x19,x20,[sp,#16]\nstp x21,x22,[sp,#32]\n"
        "stp x23,x24,[sp,#48]\nstp x25,x26,[sp,#64]\nstp x27,x28,[sp,#80]\nstp d8,d9,[sp,#96]\n"
        "stp d10,d11,[sp,#112]\nstp d12,d13,[sp,#128]\nstp d14,d15,[sp,#144]\nstr x3,[sp,#160]\n"
        "mov x19,sp\nmov x16,x0\nmov x17,x1\nmov x7,x2\n"
        "ldp x0,x1,[x17]\nldp x2,x3,[x17,#16]\nldp x4,x5,[x17,#32]\nblr x16\n"
        "mov sp,x19\nldr x9,[sp,#160]\nstr x0,[x9]\nmov w0,#0\n"
        "ombr_leave:\nldp d14,d15,[sp,#144]\nldp d12,d13,[sp,#128]\nldp d10,d11,[sp,#112]\n"
        "ldp d8,d9,[sp,#96]\nldp x27,x28,[sp,#80]\nldp x25,x26,[sp,#64]\nldp x23,x24,[sp,#48]\n"
        "ldp x21,x22,[sp,#32]\nldp x19,x20,[sp,#16]\nldp x29,x30,[sp],#176\nret\n.size ombr_enter,.-ombr_enter\n"
        ".local ombr_trap\n.hidden ombr_trap\n.type ombr_trap,%function\nombr_trap:\nhint #38\n"
        "mov sp,x19\nsub x2,x1,#1\nmov w3,#34\ncmp x2,#13\ncsel w0,w1,w3,ls\nb ombr_leave\n"
        ".size ombr_trap,.-ombr_trap\n"
        ".local ombr_no_entry\n.hidden ombr_no_entry\n.type ombr_no_entry,%function\nombr_no_entry:\nhint #38\n"
        "mov sp,x19\nmov w0,#34\nb ombr_leave\n.size ombr_no_entry,.-ombr_no_entry\n");
static const ombr_rt ombr_rt_table = {{ ombr_no_entry, ombr_no_entry, ombr_trap, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry }};
static uint32_t ombr_run(const void *code, const uint64_t *args, uint64_t *result, int has_value)
{
    uint64_t v = 0;
    uint32_t status = ombr_enter(code, args, &ombr_rt_table, &v, 0);
    if (status == OMEGA_OK)
        *result = has_value ? v : 0u;
    return status;
}
__asm__(".section .text.omega_bridge,\"ax\",@progbits\n.balign 4\n.local ombr_code_native_revision\n.hidden ombr_code_native_revision\nombr_code_native_revision:\n.byte 0x01,0x00,0x80,0x92,0x1f,0x00,0x01,0xeb,0x61,0x00,0x00,0x54,0x00,0x00,0x80,0xd2,0xc0,0x03,0x5f,0xd6,0x00,0x04,0x00,0x91,0xc0,0x03,0x5f,0xd6\n.local ombr_entry_native_revision\n.hidden ombr_entry_native_revision\nombr_entry_native_revision:\nhint #34\nb ombr_code_native_revision\n.previous\n");
extern const unsigned char ombr_entry_native_revision[] __attribute__((visibility("hidden")));
__attribute__((visibility("default"))) uint32_t omega_native_revision(uint64_t *result, uint64_t o_previous)
{
    uint64_t a[6] = {0};
    if (!result || (uintptr_t)result % 8u != 0u || (uintptr_t)result > UINTPTR_MAX - 8u)
        return OMEGA_BAD_ARG;
    a[0] = o_previous;
    return ombr_run(ombr_entry_native_revision, a, result, 1);
}
__asm__(".section .text.omega_bridge,\"ax\",@progbits\n.balign 4\n.local ombr_code_native_epoch\n.hidden ombr_code_native_epoch\nombr_code_native_epoch:\n.byte 0x01,0x00,0x80,0x92,0x1f,0x00,0x01,0xeb,0x61,0x00,0x00,0x54,0x00,0x00,0x80,0x92,0xc0,0x03,0x5f,0xd6,0x00,0x04,0x00,0x91,0xc0,0x03,0x5f,0xd6\n.local ombr_entry_native_epoch\n.hidden ombr_entry_native_epoch\nombr_entry_native_epoch:\nhint #34\nb ombr_code_native_epoch\n.previous\n");
extern const unsigned char ombr_entry_native_epoch[] __attribute__((visibility("hidden")));
__attribute__((visibility("default"))) uint32_t omega_native_epoch(uint64_t *result, uint64_t o_previous)
{
    uint64_t a[6] = {0};
    if (!result || (uintptr_t)result % 8u != 0u || (uintptr_t)result > UINTPTR_MAX - 8u)
        return OMEGA_BAD_ARG;
    a[0] = o_previous;
    return ombr_run(ombr_entry_native_epoch, a, result, 1);
}
