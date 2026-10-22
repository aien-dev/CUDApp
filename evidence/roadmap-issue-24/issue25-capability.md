# Issue25 capability packet (inspection only)

Issue24 executable source pin: 264c775 (full pin in pins.txt). Host fixture tests
validate its explicit adapter; shared weights are refused. #16/#17 integrated
hardware gates and #14 disjoint process overhead remain prerequisites, NOT_RUN.
No issue25 integration or GPU experiment is implemented here.

Current CUDApp base 349913f17c083996680c72b32f767b225005155d has
`nvrm_alloc`, `nvrm_alloc_gpu_uncached`, `nvrm_free`, session allocate/free/sync,
and the platform DMA allocate/free table. There is no public file fd/page import,
registration of CPU mmap pages, dma_export, or dma_import API in this artifact.
`alloc_with_cacheability` allocates a new RM NV01_MEMORY_SYSTEM object, rounds to
4096 bytes, maps it to a chosen CPU VA, then maps that same driver object via UVM.
CPU pointer equals GPU virtual address for these allocations. This does not
establish GPU access to pages of an independently mmapped file.

Existing candidate inspected read-only, without modifying another checkout:

- physics `feat/shared-mem` 82467f1925bb426c8a74c99f3daf1eb90fb9c93c:
  `NvrmShare { hclient, hmemory, size }`,
  `nvrm_export(Nvrm*, const NvrmMem*, NvrmShare*)`,
  `nvrm_import(Nvrm*, const NvrmShare*, int read_only, NvrmMem*)`.
  Export grants same-user duplication of an existing RM allocation. Import
  duplicates that RM object with NV_ESC_RM_DUP_OBJECT, reserves a 4096-aligned
  local VA, maps the duplicate CPU-side with PROT_READ/NVOS33 read-only flags
  when requested, and passes read_only through its UVM mapping. It adds a live
  allocation record; teardown uses nvrm_free. The API imports an RM object,
  not arbitrary file pages. Its source contract says duplicates retain pages
  after owner free/exit; that claim is not requalified by this worker.
- Omega `feat/shared-weights-ship`
  97b63ab208915e15d97dcb33625d39a03f872811: optional platform dma_export /
  dma_import and `omega_gpu_session_export` / `_import`. Share tokens carry RM
  client/object, bytes and GPU-cache flag. Imported rights are GPU_READ, with
  CPU_READ only for uncached imports. Source claims read-only CPU/GPU mappings.
  Its ABI minor 1.3 means shared-token fields; current CUDApp's 1.3 means the
  widened handle table. Equal version numbers do not establish binary layout
  compatibility. Do not copy old platform structs over canonical carries.

Next parent decision: reconcile the canonical ABI and existing shared-object
lifetime evidence before considering an import admission category. Charge
physical shared pages once under a durable object identity plus importer leases;
charge private buffers and measured driver/process overhead separately. Budget
release must follow final proven importer teardown, not publisher PID death.
Current issue24 refuses shared weight reservations so it cannot falsely grant
that accounting contract.

For true file-backed page import, first locate and prove a native API taking the
actual file pages, their pin/residency lifetime and GPU permissions. The inspected
APIs do not provide it. Unsupported import must refuse; a copied resident buffer
is a copy baseline. Any future proof needs bounded allocation, page identity,
file replacement/truncation policy, early unmap/process exit and stale token
controls, read-only enforcement, uncached markers, cold/warm separation,
RSS/PSS and complete-model logits/tokens on the integrated final artifact.

GB10 mapping, pinning, read-only enforcement, no-copy page identity, process
teardown, model parity, load/decode timings and serving readiness: NOT_RUN.
No kernel/driver/module/server changes were made. Existing sharing handoff:
/home/drakestapleton/handoffs/2026-10-21-shared-weights.md. It is historical
context, not a fresh hardware receipt for this artifact.
