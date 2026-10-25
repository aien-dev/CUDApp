# Device trace wire contract v1, host reference only

## Status and ownership

PROPOSED wire contract for CUDApp #82 and Omega #778. The handwritten
`native/blackwell/trace.om` is a bounded host reference and codec, not a GPU
producer. It does not emit SASS, allocate a device buffer or read a device
clock. The semantic fixtures and compiler source have run through an external
reference interpreter. Native ELF execution and all device behavior are
NOT_RUN. See REPRODUCE.md for pins, commands and limitations.
The existing `vector.om`, `launch.om` and `context.om` are unchanged.

This first slice uses one fixed producer, lane 0 of one selected warp in one
selected CTA. The existing VECTOR_1D launch has 64 threads per CTA, at most
1,048,576 logical threads. CTA is 0..16383 and warp is 0..1. It is not an
all-thread view. The source contains no atomics, spin loops or barriers.

## Bytes and identities

All integers are unsigned little-endian. Header and record are each 64 bytes.
Capacity is 0..64. The exact transport extent is 64  + 64*capacity, at most4160
bytes. This is a chosen bound, not a hardware limit. No implicit alignment,
padding, pointers, strings, tensor data or prompt text are serialized.

Header offsets:

| Byte | Width | Field |
| --- | --- | --- |
| 0 | 4 | Magic bytes `OMGT` (1413958991) |
| 4 | 4 | Version 1 |
| 8 | 4 | Exact transport extent |
| 12 | 4 | Capacity |
| 16 | 4 | Written prefix length |
| 20 | 4 | Dropped selected valid events, saturating at 4294967295 |
| 24 | 4 | Status: 0 recording, 1 complete, 2 partial, 3 failed |
| 28 | 4 | Clock domain 1: raw device globaltimer low32, units UNKNOWN |
| 32 | 8 | Nonzero generation ID |
| 40 | 8 | Nonzero kernel ID |
| 48 | 4 | Selected CTA |
| 52 | 4 | Selected warp |
| 56 | 4 | Nonzero manifest ID |
| 60 | 4 | Loss saturation flag: 0 exact, 1 further loss was clipped |

Each retained record, at64  + 64*sequence:

| Byte | Width | Field |
| --- | --- | --- |
| 0 | 4 | Nonzero numeric site ID |
| 4 | 4 | Kind: 1 mark, 2 begin, 3 end |
| 8 | 4 | Retained producer sequence, starting 0 |
| 12 | 4 | Raw low32 stamp |
| 16 | 8 | Generation ID |
| 24 | 8 | Kernel ID |
| 32 | 4 | CTA |
| 36 | 4 | Warp |
| 40 | 4 | Lane, always 0 in v1 |
| 44 | 4 | Explicit numeric payload, default 0 |
| 48 | 16 | Reserved zero |

All unused slots are zero. An exported capture carries the exact configured
extent, not only the written prefix. The final partial record is never valid.

IDs are lookup keys, not hashes or proof. The offline manifest must bind them
without collision within the run to full source, compiler, generated-code,
launch-descriptor and configuration digests, plus site location/name/kind.
A consumer must possess that exact manifest and expected generation/kernel
before calling `gtvalidate`. After structural validation, the #778 consumer
must resolve every nonzero site ID and reject unknown sites or site/kind
mismatches. The codec does not perform that lookup. It checks matching numeric
generation/kernel/manifest keys, not the external manifest's truth.

## Native host reference API

Concatenate the module and a unit with `fn main`. Its globals are private to
one host process and one capture, not reentrant or thread-safe. This is an
explicit reference for a future GPU producer, not host timestamps disguised
as GPU execution.

- `gtinit(gen ulong, kernel ulong, manifest uint, cap uint, cta uint, warp uint)`
  returns bool. It validates all bounds before calculating extent, zeros all
  storage and initializes the capture. Any refusal deactivates the producer;
  previously serialized bytes are not a newly accepted capture.
- `gtappend(gen ulong, site uint, kind uint, who uint, stamp uint, payload uint)`
  takes at most six parameters, as required by the pinned seeded compiler.
  `who` is the existing logical thread index, CTA*64 + thread. It returns 0
  recorded, 1 filtered, 2 capacity exhausted, -1 inactive, -2 stale generation,
  or -3 invalid event/identity. Invalid or filtered events are not losses.
- `gtextent()` returns the configured byte extent after a successful init.
- `gtfinish(status uint)` accepts 1..3 once and deactivates the producer.
  Status 1 must only be selected by a future launcher after its existing
  completion and visibility protocol succeeds. The reference cannot prove
  completion itself. Timeout/fault does not authorize recycling live storage.
- `gtvalidate(bytes ulong, gen ulong, kernel ulong, manifest uint)` validates
  `gtbytes` against the actual supplied byte count and expected provenance.
  It refuses malformed headers, overflow-sized extents, unknown versions,
  noncanonical unused bytes, invalid selection, inconsistent loss, and records
  whose sequence, kind, identity or reserved bytes disagree.
- `gtput32/64`, `gtget32/64`, `gtrowvalid` and `gtstate` are internal helpers.
  Only `gtvalidate` accepts untrusted lengths. A caller must not mutate the
  producer state. Tests mutate it only to reach the saturation boundary.

The producer retains a prefix. When full, it changes only loss accounting.
It never wraps onto existing records, waits for space or vetoes application
work. A zero-capacity capture is valid and loses every selected valid event.

## Clock and ordering, deliberately not interpreted

Clock 1 names the intended source, not a qualified opcode. Test stamps are
synthetic. The historical S2R special-register value 0x52 is only migration
reference. The new timer encoding, scheduling and selected-lane stores still
need independent word/control checks and an admitted sm_121 hardware receipt.

There is no conversion to nanoseconds, cycles, wall time or CPU time here.
There is no 64-bit timer, cross-SM comparison, cross-kernel ordering or claim
that issuing asynchronous work means it completed. A begin near 2^32 followed
by a small end is preserved exactly, not silently converted to a duration.
A low32 delta alone cannot distinguish multiple wraps. A later consumer must
have independently established capture-duration bounds in the same clock
before interpreting any interval, otherwise timing is unavailable.

Sequence gives order only for this single producer. Nested ranges are out of
scope. A structurally valid unmatched end, missing end, partial capture or
loss remains incomplete; `gtvalidate` is not a range-completeness validator.
Omega #778 owns symbol resolution, pairing and presentation.

## Next device slice, still required

Keep tracing absent from default unit composition. Default generated kernel
bytes, allocations, arguments and completion work must remain identical.
Before adding an enabled branch, prove the timer/store/control encodings,
register/predicate liveness and resource counts against the existing emitter.
Then add separately owned trace storage, initialization and readback only
following the already qualified completion/cache path. Do not insert a new
barrier merely to make a plot appear orderly. GB10 correctness, overflow,
canaries, mutants, repeated runs and paired off/on perturbation remain open.
