/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* spirv-host: INTERIM C launcher that runs one SPIR-V module from backend/spirv/emit.om on a CPU
 * implementation: PoCL (OpenCL, format SPIRV_CL) or Mesa lavapipe (Vulkan, format SPIRV_VK).
 * It is a host-only conformance harness; it never generates or edits SPIR-V and never chooses results.
 *
 * Why C: seeded Omega has no checked shared-library call yet, so a .om program cannot call
 * libOpenCL or libvulkan (Omega extension proposal E2, ADR 0043, aien-dev/omega-mv#888). Migration-pending: this file only
 * builds with -DCUDAPP_MIGRATION_PENDING_SPIRV_HOST=1, which defaults OFF (mk/multitarget.mk).
 * Successor: backend/vulkan/host.om and backend/opencl/host.om over E2's checked calls.
 *
 * GPU isolation: the OpenCL loader is pinned to one absolute CPU library (MT_POCL_LIBRARY) and only
 * CL_DEVICE_TYPE_CPU is enumerated; the Vulkan loader is pinned to the lavapipe manifest
 * (MT_LVP_ICD) with layers disabled, and the device must report VK_PHYSICAL_DEVICE_TYPE_CPU.
 * Before exit the process scans /proc/self/maps and /proc/self/fd and fails (exit 9) on any mapping
 * or descriptor whose path contains MT_FORBID ("nvidia" by default; the gate builds a control copy
 * with a pattern that must trip).
 *
 * usage: spirv-host cl|vk MODULE.spv KERNEL COUNT IN.bin [IN.bin] OUT.bin
 *   COUNT is the output element count (n); every input file holds COUNT*IN_RATIO floats, where
 *   IN_RATIO = (input bytes) / (4*COUNT) must be a positive integer. Buffers are passed in order
 *   (inputs, then output); cl also gets uint n as the last argument. Output starts as the NaN
 *   pattern 0xFFC0FFEE so an unwritten element can never compare equal to a real result.
 * Exit: 0 ran; 2 usage; 3 environment; 4 input; 5 device; 6 program; 7 kernel; 8 run; 9 isolation.
 */
#if !defined(CUDAPP_MIGRATION_PENDING_SPIRV_HOST) || CUDAPP_MIGRATION_PENDING_SPIRV_HOST != 1
#error "spirv-host is migration-pending C: build with -DCUDAPP_MIGRATION_PENDING_SPIRV_HOST=1 (default OFF)"
#endif
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <vulkan/vulkan.h>

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef MT_POCL_LIBRARY
#define MT_POCL_LIBRARY ""
#endif
#ifndef MT_LVP_ICD
#define MT_LVP_ICD "/usr/share/vulkan/icd.d/lvp_icd.json"
#endif
#ifndef MT_FORBID
#define MT_FORBID "nvidia"
#endif
#define MAX_BUFS 3
#define POISON 0xFFC0FFEEu
#define LOCAL 64u

static int die(int code, const char *what) { fprintf(stderr, "spirv-host: %s\n", what); return code; }

static void *slurp(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    void *p = malloc((size_t)n);
    if (p && fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); p = NULL; }
    fclose(f);
    *size = (size_t)n;
    return p;
}

/* Fails when any mapping or open descriptor of this process names MT_FORBID. */
static int isolation(void)
{
    char line[4096];
    FILE *m = fopen("/proc/self/maps", "r");
    if (!m) return die(9, "isolation: cannot read /proc/self/maps");
    int bad = 0;
    while (fgets(line, sizeof line, m))
        if (strstr(line, MT_FORBID)) { bad = 1; fprintf(stderr, "spirv-host: isolation: mapped %s", line); break; }
    fclose(m);
    DIR *d = opendir("/proc/self/fd");
    if (!d) return die(9, "isolation: cannot read /proc/self/fd");
    struct dirent *e;
    char link[300], target[4096];
    while (!bad && (e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        snprintf(link, sizeof link, "/proc/self/fd/%s", e->d_name);
        ssize_t k = readlink(link, target, sizeof target - 1);
        if (k <= 0) continue;
        target[k] = 0;
        if (strstr(target, MT_FORBID)) { bad = 1; fprintf(stderr, "spirv-host: isolation: open fd %s\n", target); }
    }
    closedir(d);
    if (bad) return die(9, "isolation FAILED: forbidden path \"" MT_FORBID "\" present");
    fprintf(stderr, "spirv-host: isolation: no mapping or descriptor contains \"%s\"\n", MT_FORBID);
    return 0;
}

typedef struct { const char *api, *module, *kernel; uint32_t count; int nin; const char *in[2], *out;
                 void *code; size_t code_size; float *buf[MAX_BUFS]; size_t bytes[MAX_BUFS]; } Job;

static int run_cl(Job *j)
{
    const char *lib = MT_POCL_LIBRARY;
    if (lib[0] != '/') return die(3, "build needs an absolute MT_POCL_LIBRARY");
    if (setenv("OCL_ICD_VENDORS", lib, 1) || unsetenv("OCL_ICD_FILENAMES") || unsetenv("OPENCL_LAYERS") ||
        setenv("OPENCL_LAYER_PATH", "/dev/null", 1) || setenv("POCL_DEVICES", "cpu", 1))
        return die(3, "cannot pin OpenCL loader environment");
    cl_platform_id plat; cl_uint np = 0; cl_int err;
    if (clGetPlatformIDs(1, &plat, &np) != CL_SUCCESS || np != 1) return die(5, "expected exactly one OpenCL platform (pinned PoCL)");
    char text[512];
    if (clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof text, text, NULL) != CL_SUCCESS ||
        strcmp(text, "Portable Computing Language") != 0) return die(5, "platform is not PoCL");
    clGetPlatformInfo(plat, CL_PLATFORM_VERSION, sizeof text, text, NULL);
    fprintf(stderr, "spirv-host: cl library=%s platform=\"%s\"\n", lib, text);
    cl_device_id dev;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &dev, NULL) != CL_SUCCESS) return die(5, "no CPU device");
    cl_device_type ty = 0;
    clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof ty, &ty, NULL);
    if (ty != CL_DEVICE_TYPE_CPU) return die(5, "device is not CL_DEVICE_TYPE_CPU");
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof text, text, NULL);
    fprintf(stderr, "spirv-host: cl device=\"%s\" type=CPU\n", text);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err != CL_SUCCESS) return die(5, "clCreateContext");
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    if (err != CL_SUCCESS) return die(5, "clCreateCommandQueue");
    cl_program prog = clCreateProgramWithIL(ctx, j->code, j->code_size, &err);
    if (err != CL_SUCCESS) return die(6, "clCreateProgramWithIL refused the module");
    if (clBuildProgram(prog, 1, &dev, "", NULL, NULL) != CL_SUCCESS) {
        char log[4096] = { 0 };
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, NULL);
        fprintf(stderr, "%s\n", log);
        return die(6, "clBuildProgram failed");
    }
    cl_kernel k = clCreateKernel(prog, j->kernel, &err);
    if (err != CL_SUCCESS) return die(7, "clCreateKernel: entry point not found");
    cl_mem mem[MAX_BUFS];
    int nb = j->nin + 1;
    for (int i = 0; i < nb; i++) {
        mem[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, j->bytes[i], j->buf[i], &err);
        if (err != CL_SUCCESS || clSetKernelArg(k, (cl_uint)i, sizeof mem[i], &mem[i]) != CL_SUCCESS)
            return die(7, "buffer or argument");
    }
    cl_uint n = j->count;
    if (clSetKernelArg(k, (cl_uint)nb, sizeof n, &n) != CL_SUCCESS) return die(7, "argument n");
    size_t global = ((size_t)j->count + LOCAL - 1) / LOCAL * LOCAL, local = LOCAL;
    if (clEnqueueNDRangeKernel(q, k, 1, NULL, &global, &local, 0, NULL, NULL) != CL_SUCCESS ||
        clEnqueueReadBuffer(q, mem[nb - 1], CL_TRUE, 0, j->bytes[nb - 1], j->buf[nb - 1], 0, NULL, NULL) != CL_SUCCESS ||
        clFinish(q) != CL_SUCCESS)
        return die(8, "enqueue, read back or finish failed");
    fprintf(stderr, "spirv-host: cl dispatched global=%zu local=%zu\n", global, local);
    for (int i = 0; i < nb; i++) clReleaseMemObject(mem[i]);
    clReleaseKernel(k); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}

#define VKOK(x, code, what) do { if ((x) != VK_SUCCESS) return die(code, what); } while (0)

static int memtype(VkPhysicalDevice pd, uint32_t bits)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}

static int run_vk(Job *j)
{
    if (setenv("VK_DRIVER_FILES", MT_LVP_ICD, 1) || setenv("VK_ICD_FILENAMES", MT_LVP_ICD, 1) ||
        unsetenv("VK_ADD_DRIVER_FILES") || unsetenv("VK_INSTANCE_LAYERS") || unsetenv("VK_LAYER_PATH") ||
        unsetenv("VK_ADD_LAYER_PATH") || setenv("VK_LOADER_LAYERS_DISABLE", "~all~", 1))
        return die(3, "cannot pin Vulkan loader environment");
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "cudapp-spirv-host",
                              .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance inst;
    VKOK(vkCreateInstance(&ici, NULL, &inst), 5, "vkCreateInstance");
    uint32_t npd = 0;
    VKOK(vkEnumeratePhysicalDevices(inst, &npd, NULL), 5, "vkEnumeratePhysicalDevices");
    if (npd != 1) return die(5, "expected exactly one Vulkan device (pinned lavapipe)");
    VkPhysicalDevice pd;
    VKOK(vkEnumeratePhysicalDevices(inst, &npd, &pd), 5, "vkEnumeratePhysicalDevices");
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(pd, &pp);
    if (pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) return die(5, "device is not VK_PHYSICAL_DEVICE_TYPE_CPU");
    fprintf(stderr, "spirv-host: vk icd=%s device=\"%s\" type=CPU api=%u.%u.%u driver=0x%x\n", MT_LVP_ICD, pp.deviceName,
            VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion), VK_API_VERSION_PATCH(pp.apiVersion),
            pp.driverVersion);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qf[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t fam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = i; break; }
    if (fam == UINT32_MAX) return die(5, "no compute queue");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = fam,
                                    .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    VkDevice dev;
    VKOK(vkCreateDevice(pd, &dci, NULL, &dev), 5, "vkCreateDevice");
    VkQueue q;
    vkGetDeviceQueue(dev, fam, 0, &q);
    int nb = j->nin + 1;
    VkBuffer b[MAX_BUFS]; VkDeviceMemory m[MAX_BUFS]; void *map[MAX_BUFS];
    for (int i = 0; i < nb; i++) {
        VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = j->bytes[i],
                                   .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
        VKOK(vkCreateBuffer(dev, &bci, NULL, &b[i]), 5, "vkCreateBuffer");
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b[i], &mr);
        int t = memtype(pd, mr.memoryTypeBits);
        if (t < 0) return die(5, "no host-visible coherent memory type");
        VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = (uint32_t)t };
        VKOK(vkAllocateMemory(dev, &mai, NULL, &m[i]), 5, "vkAllocateMemory");
        VKOK(vkBindBufferMemory(dev, b[i], m[i], 0), 5, "vkBindBufferMemory");
        VKOK(vkMapMemory(dev, m[i], 0, j->bytes[i], 0, &map[i]), 5, "vkMapMemory");
        memcpy(map[i], j->buf[i], j->bytes[i]);
    }
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = j->code_size, .pCode = j->code };
    VkShaderModule sm;
    VKOK(vkCreateShaderModule(dev, &smci, NULL, &sm), 6, "vkCreateShaderModule refused the module");
    VkDescriptorSetLayoutBinding lb[MAX_BUFS];
    for (int i = 0; i < nb; i++)
        lb[i] = (VkDescriptorSetLayoutBinding){ .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = (uint32_t)nb, .pBindings = lb };
    VkDescriptorSetLayout dsl;
    VKOK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl), 6, "vkCreateDescriptorSetLayout");
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl };
    VkPipelineLayout pl;
    VKOK(vkCreatePipelineLayout(dev, &plci, NULL, &pl), 6, "vkCreatePipelineLayout");
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pl,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module = sm, .pName = j->kernel } };
    VkPipeline pipe;
    VKOK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe), 7, "vkCreateComputePipelines (entry point?)");
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)nb };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    VkDescriptorPool dp;
    VKOK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp), 7, "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
    VkDescriptorSet ds;
    VKOK(vkAllocateDescriptorSets(dev, &dsai, &ds), 7, "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo dbi[MAX_BUFS];
    VkWriteDescriptorSet wds[MAX_BUFS];
    for (int i = 0; i < nb; i++) {
        dbi[i] = (VkDescriptorBufferInfo){ b[i], 0, j->bytes[i] };
        wds[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i] };
    }
    vkUpdateDescriptorSets(dev, (uint32_t)nb, wds, 0, NULL);
    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = fam };
    VkCommandPool cp;
    VKOK(vkCreateCommandPool(dev, &cpi, NULL, &cp), 8, "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cb;
    VKOK(vkAllocateCommandBuffers(dev, &cbai, &cb), 8, "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VKOK(vkBeginCommandBuffer(cb, &cbbi), 8, "vkBeginCommandBuffer");
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    uint32_t groups = (j->count + LOCAL - 1) / LOCAL;
    vkCmdDispatch(cb, groups, 1, 1);
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VKOK(vkEndCommandBuffer(cb), 8, "vkEndCommandBuffer");
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    VKOK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), 8, "vkQueueSubmit");
    VKOK(vkQueueWaitIdle(q), 8, "vkQueueWaitIdle");
    fprintf(stderr, "spirv-host: vk dispatched groups=%u local=%u\n", groups, LOCAL);
    memcpy(j->buf[nb - 1], map[nb - 1], j->bytes[nb - 1]);
    vkDestroyCommandPool(dev, cp, NULL); vkDestroyDescriptorPool(dev, dp, NULL); vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL); vkDestroyDescriptorSetLayout(dev, dsl, NULL); vkDestroyShaderModule(dev, sm, NULL);
    for (int i = 0; i < nb; i++) { vkUnmapMemory(dev, m[i]); vkDestroyBuffer(dev, b[i], NULL); vkFreeMemory(dev, m[i], NULL); }
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8) return die(2, "usage: spirv-host cl|vk MODULE.spv KERNEL COUNT IN.bin [IN.bin] OUT.bin");
    Job j = { .api = argv[1], .module = argv[2], .kernel = argv[3], .nin = argc - 6, .out = argv[argc - 1] };
    if (strcmp(j.api, "cl") != 0 && strcmp(j.api, "vk") != 0) return die(2, "api must be cl or vk");
    char *end;
    unsigned long c = strtoul(argv[4], &end, 10);
    if (*end || c == 0 || c > (1ul << 26)) return die(2, "COUNT must be 1..67108864");
    j.count = (uint32_t)c;
    for (int i = 0; i < j.nin; i++) j.in[i] = argv[5 + i];
    j.code = slurp(j.module, &j.code_size);
    if (!j.code || j.code_size % 4) return die(4, "module unreadable or not whole 32-bit words");
    for (int i = 0; i < j.nin; i++) {
        j.buf[i] = slurp(j.in[i], &j.bytes[i]);
        if (!j.buf[i] || j.bytes[i] % ((size_t)4 * j.count)) return die(4, "input unreadable or not a multiple of COUNT floats");
    }
    j.bytes[j.nin] = (size_t)4 * j.count;
    uint32_t *o = malloc(j.bytes[j.nin]);
    if (!o) return die(4, "out of memory");
    for (uint32_t i = 0; i < j.count; i++) o[i] = POISON;
    j.buf[j.nin] = (float *)o;
    int rc = strcmp(j.api, "cl") == 0 ? run_cl(&j) : run_vk(&j);
    if (rc) return rc;
    rc = isolation();
    if (rc) return rc;
    /* 0600, not fopen's 0666 & ~umask (CodeQL cpp/world-writable-file-creation) */
    int fd = open(j.out, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (!f && fd >= 0) close(fd);
    if (!f || fwrite(o, 1, j.bytes[j.nin], f) != j.bytes[j.nin] || fclose(f) != 0) return die(4, "cannot write output");
    return 0;
}
