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
/* SysV AMD64 adapter to the pinned private Omega frame ABI on the caller's stack. ombr_enter saves
 * the callee-saved registers there, keeps that stack pointer in R12, points R14 at the runtime table
 * and sets the owned-stack floor R15 to 0, so, as in the AArch64 binding, the pure scalar code runs on
 * the caller's stack with no bound check. A normal return stores RAX through out and returns 0. A
 * runtime entry restores the stack pointer from R12 and returns a status; out is not written.
 * CET shadow-stack enforcement is not supported by the nonlocal trap return. */
__asm__(".text\n.balign 16\n.local ombr_enter\n.hidden ombr_enter\n.type ombr_enter,@function\nombr_enter:\n"
        "endbr64\npush %rbx\npush %rbp\npush %r12\npush %r13\npush %r14\npush %r15\npush %rcx\n"
        "mov %rsp,%r12\nmov %rdx,%r14\nxor %r15d,%r15d\nmov %rdi,%r10\n"
        "mov %rsp,%rbp\nsub $256,%rsp\n"
        "mov 0(%rsi),%rax\nmov %rax,-8(%rbp)\nmov 8(%rsi),%rax\nmov %rax,-16(%rbp)\n"
        "mov 16(%rsi),%rax\nmov %rax,-24(%rbp)\nmov 24(%rsi),%rax\nmov %rax,-32(%rbp)\n"
        "mov 32(%rsi),%rax\nmov %rax,-40(%rbp)\nmov 40(%rsi),%rax\nmov %rax,-48(%rbp)\n"
        "call *%r10\nmov %r12,%rsp\nmov (%rsp),%rcx\nmov %rax,(%rcx)\nxor %eax,%eax\n"
        "ombr_leave:\nadd $8,%rsp\npop %r15\npop %r14\npop %r13\npop %r12\npop %rbp\npop %rbx\nret\n"
        ".size ombr_enter,.-ombr_enter\n.local ombr_trap\n.hidden ombr_trap\n.type ombr_trap,@function\nombr_trap:\n"
        "endbr64\nmov %r12,%rsp\nlea -1(%rsi),%eax\ncmp $13,%eax\nmov $34,%eax\ncmovbe %esi,%eax\njmp ombr_leave\n"
        ".size ombr_trap,.-ombr_trap\n.local ombr_no_entry\n.hidden ombr_no_entry\n.type ombr_no_entry,@function\nombr_no_entry:\n"
        "endbr64\nmov %r12,%rsp\nmov $34,%eax\njmp ombr_leave\n.size ombr_no_entry,.-ombr_no_entry\n");
static const ombr_rt ombr_rt_table = {{ ombr_no_entry, ombr_no_entry, ombr_trap, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry, ombr_no_entry }};
static uint32_t ombr_run(const void *code, const uint64_t *args, uint64_t *result, int has_value)
{
    uint64_t v = 0;
    uint32_t status = ombr_enter(code, args, &ombr_rt_table, &v, 0);
    if (status == OMEGA_OK)
        *result = has_value ? v : 0u;
    return status;
}
__asm__(".section .text.omega_bridge,\"ax\",@progbits\n.balign 4\n.local ombr_code_native_revision\n.hidden ombr_code_native_revision\nombr_code_native_revision:\n.quad 0x89499000000001e9,0xec8148e5894855eb,0xf8838b4900000100,0xfff8858948ffffff,0xffffffffb848ffff,0xf0858948ffffffff,0xfff8858b48ffffff,0xfffff08d8b48ffff,0x001a850fc83948ff,0x00000000b8480000,0xf885894800000000,0xfff8858b48ffffff,0xf8858b48c3c9ffff,0x000001b948ffffff,0xc801480000000000,0x48fffffff8858948,0x8948fffffff8858b,0x858b48fffffff885,0xf8858948fffffff8,0xfff8858b48ffffff\n.byte 0xff,0xff,0xc9,0xc3\n.local ombr_entry_native_revision\n.hidden ombr_entry_native_revision\nombr_entry_native_revision:\nendbr64\njmp ombr_code_native_revision\n.previous\n");
extern const unsigned char ombr_entry_native_revision[] __attribute__((visibility("hidden")));
__attribute__((visibility("default"))) uint32_t omega_native_revision(uint64_t *result, uint64_t o_previous)
{
    uint64_t a[6] = {0};
    if (!result || (uintptr_t)result % 8u != 0u || (uintptr_t)result > UINTPTR_MAX - 8u)
        return OMEGA_BAD_ARG;
    a[0] = o_previous;
    return ombr_run(ombr_entry_native_revision, a, result, 1);
}
__asm__(".section .text.omega_bridge,\"ax\",@progbits\n.balign 4\n.local ombr_code_native_epoch\n.hidden ombr_code_native_epoch\nombr_code_native_epoch:\n.quad 0x89499000000001e9,0xec8148e5894855eb,0xf8838b4900000100,0xfff8858948ffffff,0xffffffffb848ffff,0xf0858948ffffffff,0xfff8858b48ffffff,0xfffff08d8b48ffff,0x001a850fc83948ff,0xffffffffb8480000,0xf8858948ffffffff,0xfff8858b48ffffff,0xf8858b48c3c9ffff,0x000001b948ffffff,0xc801480000000000,0x48fffffff8858948,0x8948fffffff8858b,0x858b48fffffff885,0xf8858948fffffff8,0xfff8858b48ffffff\n.byte 0xff,0xff,0xc9,0xc3\n.local ombr_entry_native_epoch\n.hidden ombr_entry_native_epoch\nombr_entry_native_epoch:\nendbr64\njmp ombr_code_native_epoch\n.previous\n");
extern const unsigned char ombr_entry_native_epoch[] __attribute__((visibility("hidden")));
__attribute__((visibility("default"))) uint32_t omega_native_epoch(uint64_t *result, uint64_t o_previous)
{
    uint64_t a[6] = {0};
    if (!result || (uintptr_t)result % 8u != 0u || (uintptr_t)result > UINTPTR_MAX - 8u)
        return OMEGA_BAD_ARG;
    a[0] = o_previous;
    return ombr_run(ombr_entry_native_epoch, a, result, 1);
}
