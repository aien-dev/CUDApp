# Teardown refusal and retry fixture

Refs CUDApp #65 and #67. This is a bounded cleanup repair and a synthetic host
fixture. It does not qualify model inference, GPU execution, or deployment.

## Source contract

`nvstate` records the teardown phase:

| Value | Meaning | Permitted cleanup |
| --- | --- | --- |
| 0 | Closed | Return success without another effect |
| 1 | Driver resources live, including partial construction | Existing RM/UVM teardown and one close attempt per descriptor |
| 2 | Descriptors closed; CPU mappings remain | Retry owned BAR, channel, and buffer unmaps only |

The compiler's `mapdrop` returns a nonzero syscall result without retiring its
owned mapping slot. The runtime must therefore retain the mapping handle until
unmap succeeds. After descriptor closure, driver-owned memory flags can retire,
but a failed CPU mapping retains its handle, address and length. A failed
cleanup latches `nvfault`; successful cleanup does not make the context reusable.
Descriptor numbers are retired after their first close attempt, including an
error, so a subsequent call cannot close a newly reused descriptor number.

The same ownership rule applies before `nvcpumap` publishes a mapping to its
caller. If descriptor close fails and its immediate unmap is refused,
`nvunpub` retains that mapping for `nvmapfinish`. A faulted or inactive
context refuses another `nvcpumap`, so the retained handle cannot be overwritten.
Failed `mapbind` is different: the pinned compiler contract retires that handle.

## Fixture and isolation

The handwritten `tests/speculation/teardown-host-test.om` runs the actual
`nvstop`/`nvmapfinish`/`nvmforce` control flow in a private source-derived unit.
The disclosed shell composer replaces only these OS effects:

- Device operations become a bounded fake that refuses any call after the
  descriptor phase has ended.
- Device opens become an immediate test assertion failure.
- Unmap and close become stateful fakes that count calls, reject double release,
  and can refuse an otherwise valid synthetic mapping.
- The unpublished-mapping case allows one synthetic descriptor open and replaces
  `mapnew` with a fake that returns a fourth owned mapping; no real mapping or
  device open is performed.

The composer refuses an unbound device/open/close/unmap verb. It never changes
the tracked runtime file. The fixture does not open `/dev/nvidia*`, allocate GPU
memory, or submit work. It follows the existing development practice of building
private native translation units and compiling behavioral negative controls;
the shell is test orchestration, not a new product runtime.

Cases cover normal and repeated shutdown, a BAR unmap refusal, a channel-map
refusal, a buffer unmap refusal, a driver teardown refusal followed by mapping
refusal, successful retry, and a descriptor close error. Each refused mapping
must survive two shutdown attempts, then release when refusal is removed.
Retries must make no extra descriptor closes or driver calls. The context fault
must remain set, and `nvstart` must still refuse reuse. A direct `nvmforce` call
while driver resources are live must refuse without changing ownership flags,
handles, addresses or lengths and without making an OS effect.
The additional close-error/unmap-refusal case checks that the unpublished
mapping survives two shutdown attempts, is released on a later successful
retry, and is never double-unmapped. It also checks refusal of another mapping
attempt while that context is faulted.

## Reproduction

Use Linux AArch64 and the compiler from the existing native-device receipt:

- Omega source: `d7390d88d07dbf48cfbb0583070599c350f814d3`.
- Compiler SHA-256:
  `9749f38841a55917d41924a68bce31463ec2339910bddd5e6cbf6ea4420551db`.

```sh
sh tests/speculation/run-teardown-host.sh /absolute/path/to/pinned-omc
```

The script retains a private temporary directory with exact source units, ELF
files, compiler logs, stdout/stderr, exit statuses, and hashes. It compiles the
original `f9cd649c1dca92ad81009bcd05a959f34c25c76e` context with the same fixture
first: it must fail by a test assertion (exit 1), not compilation error, trap or
timeout. The repaired source must then compile and exit 0 with the PASS marker.
An unsupported host returns 77 and `NOT_RUN`; it is not a test pass.

## Current qualification limit

The source-compatibility review uses the exact pinned `seed/omc/omc.om`, not
a different compiler profile. Its `topname` and `s_fn` readers limit globals,
texts and parameter names to eight bytes; function names may be 63 bytes.
The initial source review missed overlong names in both this repair and the
inherited fixture. They are now shortened without changing cleanup behavior.
The fixture's custom calls use scalar arguments only; the compiler's prohibition
on passing globals to ordinary `view`/`edit` parameters does not apply to them.
The `tmapnew` fake preserves the native word's five scalar types and long result.
This is source inspection, not evidence that either unit compiles or executes.

Native compilation and execution are **NOT_RUN** in the merge-review workspace,
which is x86_64 and has no runnable AArch64 compiler or emulator. The fixture and
repair remain candidates for source review and supported-host qualification.
Existing hardware receipts describe older source and must not be reused to
approve this change. Real kernel/model behavior and production activation remain
outside this host fixture.

## Supported-host qualification

On 2026-10-24 a candidate based on `main` (22e9d9b plus this repair, code commit
2e24216) was compiled with the pinned compiler rebuilt from seed and run on
Linux AArch64: the original cleanup failed by assertion and the repair passed.
Mutants, the GPU vector rerun and the remaining fixture gap are recorded in
`evidence/speculation/release-candidate-teardown/`.
