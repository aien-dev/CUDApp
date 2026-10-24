# Native GPU connection

`native/nvrm/context.om` continues the canonical NVRM lifecycle from
`349913f17c083996680c72b32f767b225005155d` in seeded Omega. It uses the
NVIDIA580.173.02 public ABI definitions already retained with their notices in
this repository. No C or Rust runtime helper is linked or executed.

The compiler extension is in Omega PR676. It adds scoped RM parameter binding,
an exact UVM request allowlist, owned memory addresses, ordered mapped words,
unused low-address reservations and binding limited to those reservations.
The source-born compiler has no foreign compiler, assembler or linker in its
construction chain. Independent GCC header-layout and GNU assembler encoding
oracles were used during development; shell and Node coordinated checks.

The physical probes cover private RM/UVM context creation, a GPU address space,
coherent system-memory objects, their CPU and GPU mappings, an uncached queue,
the compute channel, a mapped notification page, submission and bounded waiting.
`nvstart` queries the device, `nvfull` attaches UVM, `nvalloc` returns an owned
mapping, `nvchan` creates an empty channel, `nvsend` submits one packet and
`nvwait` observes its completion. `nvdrop` releases an allocation and `nvstop`
tears down the entire context. Failed operations retain stage/error information.

There is one outstanding packet at a time. Pending work prevents buffer release
and another submission. A missed completion latches a fault; further submission
and context reuse are refused. Partial allocation state remains recorded until
cleanup. The table has96 live allocation slots; the compiler has128 mapping
slots. These are bounded runtime resources, not model-sized allocation arenas.

GPU buffers first reserve unused page-aligned CPU addresses below1TiB. File
binding can replace only that owned reservation. The same address is serialized
into the GPU mappings. This matters because a command entry's address field is
40bits; ordinary Linux-selected CPU addresses can exceed it. The existing
canonical GP entry word, including its level bit, is retained. No other runtime
or shared compiler tree was modified.

The driver reports the GB10 SM identifier as2564, matching the repository's
historical M16 receipt. That field uses a different numbering contract from the
CUDA target name. An initial test incorrectly expected3073 and was corrected.
Its initial negative-value diagnostic also required correction. Both failures
are retained locally. A UVM attempt launched before scheduling confirmation is
excluded from qualified evidence; subsequent probes required confirmed admission.

| Physical check | Result |
| --- | --- |
| Device and private address-space lifecycle | Three complete cycles per probe passed. |
| Buffer lifecycle | Twelve buffers over three contexts; full written-byte comparison, explicit release and context-owned cleanup passed. |
| Empty channel lifecycle | Three cycles passed, including refusal to free its active queue. |
| FIFO submission | 2,052 distinct completion payloads passed across two contexts, including ring wrap. A deliberately wrong expected payload timed out and latched the fault. |

These are correctness probes under shared hardware admission. The FIFO test
executes semaphore commands. The subsequent [native kernel qualification](NATIVE-KERNEL.md)
separately covers integer vector arithmetic. There is no model inference,
AIENOS endpoint integration or timing comparison in this device evidence.
Full-state rollback, all four experiment arms and issues63-66
remain open. Cleanup refusal injection and independent final-source review remain
required alongside the next kernel/model stages.

To build a probe, concatenate `native/nvrm/context.om` with the selected
`tests/speculation/native-{device,memory,channel,submit}-test.om`, compile it with
the pinned native compiler and run only in an admitted hardware window. Current
source/artifact identities and bounded results are in `evidence/speculation/native-device`.
