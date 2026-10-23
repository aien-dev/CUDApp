/* SPDX-License-Identifier: AGPL-3.0-only */
/* cudapp-doctor: is this machine ready for CUDA++, and if not, what exactly to do.
 *
 * Developer bar D2 and D6 (docs/PAIN_POINTS.md). CUDA's install needs a driver, a toolkit, a
 * framework build and an extension build whose versions all agree, and a mismatch shows up later
 * as an undefined symbol or "no kernel image". CUDA++ has one thing to match: the NVIDIA kernel
 * module its driver client speaks to. This program checks that and everything else a first run
 * needs, and prints one line per check. Every failing line says what is wrong, why it matters and
 * the command to run next.
 *
 * It reads files only (/proc, /sys, /dev): no device is opened and no GPU window is needed.
 * CUDAPP_DOCTOR_ROOT=<dir> makes it read <dir>/proc, <dir>/sys and <dir>/dev instead (tests).
 * Exit 0: READY. Exit 1: at least one FAIL line. */
#include "omega_gpu_chip.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifndef CUDAPP_NVIDIA_OPEN_ABI
#error "build with make doctor: CUDAPP_NVIDIA_OPEN_ABI comes from the third_party/nvidia-open-<version> folder"
#endif

static const char *g_root = "";
static int g_fail, g_warn;

static void path(char *out, size_t n, const char *p) { snprintf(out, n, "%s%s", g_root, p); }

static void line(const char *tag, const char *what, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
#include <stdarg.h>
static void line(const char *tag, const char *what, const char *fmt, ...) {
    printf("  %-5s %s: ", tag, what);
    va_list ap; va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    printf("\n");
    if (!strcmp(tag, "FAIL")) g_fail++;
    if (!strcmp(tag, "warn")) g_warn++;
}
static void more(const char *text) { /* indented follow-on lines, one per \n */
    const char *s = text;
    while (*s) {
        const char *e = strchr(s, '\n');
        size_t len = e ? (size_t)(e - s) : strlen(s);
        printf("        %.*s\n", (int)len, s);
        if (!e) break;
        s = e + 1;
    }
}

/* ---- the kernel module: present, open or closed, and the one version CUDA++ must match ---- */
static void check_module(void) {
    char p[4200], buf[1024] = { 0 };
    path(p, sizeof p, "/proc/driver/nvidia/version");
    FILE *f = fopen(p, "r");
    if (!f || !fgets(buf, sizeof buf, f)) {
        if (f) fclose(f);
        line("FAIL", "NVIDIA kernel module", "not loaded (%s/proc/driver/nvidia/version does not exist)", g_root);
        more("CUDA++ uses no NVIDIA user-space software, but it talks to NVIDIA's kernel module.\n"
             "Do this: sudo modprobe nvidia && sudo modprobe nvidia-uvm\n"
             "If modprobe finds no module, install NVIDIA's open kernel module " CUDAPP_NVIDIA_OPEN_ABI
             " (the open-gpu-kernel-modules release of that number), then run make doctor again.");
        return;
    }
    fclose(f);
    int open_module = strstr(buf, "Open Kernel Module") != NULL;
    /* the version is the first word that looks like N.N or N.N.N after "Module" */
    char ver[64] = { 0 };
    const char *m = strstr(buf, "Module");
    for (const char *s = m ? m : buf; *s; s++) {
        if (*s >= '0' && *s <= '9') {
            size_t k = 0;
            while ((s[k] >= '0' && s[k] <= '9') || s[k] == '.') k++;
            if (k > 2 && memchr(s, '.', k) && k < sizeof ver) { memcpy(ver, s, k); break; }
            s += k ? k - 1 : 0;
        }
    }
    if (!ver[0]) {
        line("FAIL", "NVIDIA kernel module", "loaded, but its version could not be read from: %.120s", buf);
        more("Do this: report it with the output of `cat /proc/driver/nvidia/version` attached.");
        return;
    }
    if (strcmp(ver, CUDAPP_NVIDIA_OPEN_ABI) != 0) {
        line("FAIL", "NVIDIA kernel module", "version %s is loaded; CUDA++ speaks the " CUDAPP_NVIDIA_OPEN_ABI " interface", ver);
        more("Why: CUDA++ builds its driver calls from the " CUDAPP_NVIDIA_OPEN_ABI " open-module headers\n"
             "(third_party/nvidia-open-" CUDAPP_NVIDIA_OPEN_ABI "). The layout of those calls can change between module versions,\n"
             "and a wrong layout fails in strange ways, so CUDA++ refuses instead of guessing.\n"
             "This is the only version CUDA++ asks you to match: no toolkit, no framework build, no extension build.\n"
             "Do this: install the " CUDAPP_NVIDIA_OPEN_ABI " open kernel module, or port the client to your version\n"
             "(CONTRIBUTING.md, \"Port a driver version\": copy the new headers, rebuild, run make test and make hello-gpu).");
        return;
    }
    if (!open_module) {
        line("warn", "NVIDIA kernel module", "the closed module %s is loaded; CUDA++ was qualified on the open module of that version", ver);
        more("It uses the same calls and should work. If anything fails, switch to the open module and say so in your report.");
        return;
    }
    line("ok", "NVIDIA kernel module", "open module %s (CUDA++ speaks " CUDAPP_NVIDIA_OPEN_ABI ")", ver);
}

/* ---- device files ---- */
static void check_nodes(void) {
    static const char *const nodes[] = { "/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm" };
    int bad = 0;
    for (size_t i = 0; i < sizeof nodes / sizeof nodes[0]; i++) {
        char p[4200];
        path(p, sizeof p, nodes[i]);
        if (access(p, F_OK) != 0) {
            line("FAIL", "device files", "%s is missing", nodes[i]);
            if (!strcmp(nodes[i], "/dev/nvidia-uvm"))
                more("Do this: sudo modprobe nvidia-uvm   (creates /dev/nvidia-uvm; CUDA++ maps GPU memory through it)");
            else
                more("Do this: sudo modprobe nvidia   (or run nvidia-modprobe once, which creates the device files)");
            bad++;
        } else if (access(p, R_OK | W_OK) != 0) {
            line("FAIL", "device files", "%s exists but this user cannot open it for reading and writing", nodes[i]);
            more("Do this: ask the machine's owner to add you to the group that owns it (ls -l on the file shows it),\n"
                 "or set the module option NVreg_DeviceFileMode=0666.");
            bad++;
        }
    }
    if (!bad) line("ok", "device files", "/dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm can be opened by this user");
}

/* ---- the GPU itself: qualified, or refused with what is missing (D6) ---- */
static void check_gpu(int *unified) {
    OmegaGpuPciDev devs[16];
    int n = omega_gpu_chip_scan(g_root, devs, 16);
    if (n < 0) { line("FAIL", "GPU", "cannot read %s/sys/bus/pci/devices", g_root); more("Do this: run make doctor on the machine itself, not in a container without /sys."); return; }
    if (n == 0) {
        line("FAIL", "GPU", "no NVIDIA GPU found in /sys/bus/pci/devices");
        more("Do this: check the GPU is seated and powered (lspci | grep -i nvidia).");
        return;
    }
    if (n > 16) n = 16;
    for (int i = 0; i < n; i++) {
        const OmegaGpuChipRow *r = omega_gpu_chip_find(devs[i].vendor, devs[i].device);
        char what[96];
        snprintf(what, sizeof what, "GPU %.63s", devs[i].bdf);
        if (r) {
            line("ok", what, "%s (%s), qualified (receipt: %s)", r->name, r->sm, r->receipt);
            if (r->device == 0x2e12u) *unified = 1; /* GB10: one memory for CPU and GPU */
        } else {
            char text[2048];
            omega_gpu_chip_refusal(&devs[i], text, sizeof text);
            const char *nl = strchr(text, '\n');
            const char *first = text + 4 + strlen(devs[i].bdf) + 2; /* after "GPU <bdf>: " (a PCI address has colons in it) */
            line("FAIL", what, "%.*s", (int)(nl - first), first);
            more(nl + 1);
        }
    }
}

/* ---- memory: on a unified-memory chip the GPU's memory is the machine's memory ---- */
static void check_memory(int unified) {
    char p[4200], l[256];
    path(p, sizeof p, "/proc/meminfo");
    FILE *f = fopen(p, "r");
    if (!f) { line("info", "memory", "cannot read /proc/meminfo"); return; }
    unsigned long long total = 0, avail = 0;
    while (fgets(l, sizeof l, f)) {
        sscanf(l, "MemTotal: %llu kB", &total);
        sscanf(l, "MemAvailable: %llu kB", &avail);
    }
    fclose(f);
    double gt = (double)total / 1048576.0, ga = (double)avail / 1048576.0;
    if (unified) line("ok", "memory", "%.1f GiB shared by the CPU and the GPU, %.1f GiB available now", gt, ga);
    else line("info", "memory", "%.1f GiB system memory, %.1f GiB available now", gt, ga);
}

/* ---- the CUDA toolkit: not needed. Say so, so nobody installs it for us. ---- */
static void note_toolkit(void) {
    const char *paths[] = { "/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc" };
    for (size_t i = 0; i < 2; i++) {
        char p[4200];
        path(p, sizeof p, paths[i]);
        if (access(p, X_OK) == 0) { line("info", "CUDA toolkit", "not needed and not used by CUDA++ (one is installed at %s; that is fine)", paths[i]); return; }
    }
    line("info", "CUDA toolkit", "not needed and not used by CUDA++ (none installed)");
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const char *r = getenv("CUDAPP_DOCTOR_ROOT");
    if (r) g_root = r;
    struct utsname u;
    printf("cudapp doctor: is this machine ready for CUDA++?\n");
    if (uname(&u) == 0) line("info", "system", "%s %s %s", u.sysname, u.release, u.machine);
    check_module();
    check_nodes();
    int unified = 0;
    check_gpu(&unified);
    check_memory(unified);
    note_toolkit();
    line("info", "compiler", "this doctor was built by " __VERSION__ "; CUDA++ needs only make and a C compiler");
    if (g_fail) {
        printf("NOT READY: %d problem%s above. Fix the first FAIL line and run make doctor again.\n", g_fail, g_fail == 1 ? "" : "s");
        return 1;
    }
    printf("READY. Next: make hello-gpu\n");
    return 0;
}
