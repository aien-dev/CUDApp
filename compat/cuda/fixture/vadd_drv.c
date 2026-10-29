/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* CUDApp#184 C0 reference application: an independently authored CUDA Driver API program.
 * It stands in for an unchanged third-party CUDA executable; it is a qualification input, not
 * CUDA++ or Omega implementation code. Written from the public CUDA Driver API reference
 * (PROVENANCE class A). Built against the toolkit's cuda.h so the imported symbol variants are
 * exactly what that header selects (compat/cuda/profile-c0-*.tsv records them).
 *
 * usage: vadd_drv <kernel.ptx>
 * exit 0  VADD_U32 PASS, every element and every tail sentinel correct
 * exit 1  wrong output (element or tail sentinel)
 * exit 2  a Driver API call returned an error (call name and CUresult printed)
 * exit 3  usage or PTX file error
 * Contract fixed before implementation: N=4099 elements, 256 threads per block, 17 blocks,
 * a[i] = i*2654435761 mod 2^32, b[i] = (2^32-1) - (7*i mod 2^32), c[i] = a[i]+b[i] mod 2^32,
 * 64 tail sentinels 0xA5A5A5A5 after c[N-1] must be unchanged, checksum = sum c[i] as u64. */
#include <cuda.h>
#include <stdio.h>

#define VADD_N 4099u
#define VADD_BLOCK 256u
#define VADD_TAIL 64u
#define VADD_SENTINEL 0xA5A5A5A5u

static unsigned int ha[VADD_N];
static unsigned int hb[VADD_N];
static unsigned int hc[VADD_N + VADD_TAIL];
static char ptx[16384];

static int check(CUresult rc, const char *call)
{
	if (rc != CUDA_SUCCESS) {
		printf("VADD_U32 CUDA_ERROR call=%s rc=%d\n", call, (int)rc);
		return 1;
	}
	return 0;
}

#define CALL(expr) do { if (check((expr), #expr)) return 2; } while (0)

int main(int argc, char **argv)
{
	CUdevice dev;
	CUcontext ctx;
	CUmodule mod;
	CUfunction fn;
	CUdeviceptr da, db, dc;
	unsigned int n = VADD_N;
	unsigned int grid = (VADD_N + VADD_BLOCK - 1u) / VADD_BLOCK;
	unsigned long long sum = 0;
	unsigned int bad = 0;
	size_t len;
	FILE *f;
	void *params[4];

	if (argc != 2) {
		printf("VADD_U32 USAGE vadd_drv <kernel.ptx>\n");
		return 3;
	}
	f = fopen(argv[1], "rb");
	if (f == NULL) {
		printf("VADD_U32 PTX_OPEN_FAILED\n");
		return 3;
	}
	len = fread(ptx, 1, sizeof(ptx) - 1u, f);
	if (fclose(f) != 0 || len == 0 || len == sizeof(ptx) - 1u) {
		printf("VADD_U32 PTX_READ_FAILED bytes=%zu\n", len);
		return 3;
	}
	ptx[len] = '\0';
	for (unsigned int i = 0; i < VADD_N; i++) {
		ha[i] = i * 2654435761u;
		hb[i] = 0xFFFFFFFFu - 7u * i;
	}
	for (unsigned int i = 0; i < VADD_N + VADD_TAIL; i++)
		hc[i] = VADD_SENTINEL;

	CALL(cuInit(0));
	CALL(cuDeviceGet(&dev, 0));
	CALL(cuCtxCreate(&ctx, NULL, 0, dev));
	CALL(cuModuleLoadData(&mod, ptx));
	CALL(cuModuleGetFunction(&fn, mod, "vadd_u32"));
	CALL(cuMemAlloc(&da, sizeof(ha)));
	CALL(cuMemAlloc(&db, sizeof(hb)));
	CALL(cuMemAlloc(&dc, sizeof(hc)));
	CALL(cuMemcpyHtoD(da, ha, sizeof(ha)));
	CALL(cuMemcpyHtoD(db, hb, sizeof(hb)));
	CALL(cuMemcpyHtoD(dc, hc, sizeof(hc)));
	params[0] = &da;
	params[1] = &db;
	params[2] = &dc;
	params[3] = &n;
	CALL(cuLaunchKernel(fn, grid, 1, 1, VADD_BLOCK, 1, 1, 0, NULL, params, NULL));
	CALL(cuCtxSynchronize());
	CALL(cuMemcpyDtoH(hc, dc, sizeof(hc)));
	for (unsigned int i = 0; i < VADD_N; i++) {
		if (hc[i] != ha[i] + hb[i])
			bad++;
		sum += hc[i];
	}
	for (unsigned int i = VADD_N; i < VADD_N + VADD_TAIL; i++)
		if (hc[i] != VADD_SENTINEL)
			bad++;
	CALL(cuMemFree(da));
	CALL(cuMemFree(db));
	CALL(cuMemFree(dc));
	CALL(cuModuleUnload(mod));
	CALL(cuCtxDestroy(ctx));
	if (bad != 0) {
		printf("VADD_U32 FAIL n=%u grid=%u block=%u bad=%u checksum=%llu\n", n, grid, VADD_BLOCK, bad, sum);
		return 1;
	}
	printf("VADD_U32 PASS n=%u grid=%u block=%u checksum=%llu\n", n, grid, VADD_BLOCK, sum);
	return 0;
}
