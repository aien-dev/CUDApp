# Bounded trace contract first slice

Issue: https://github.com/aien-dev/CUDApp/issues/82
Consumer: https://github.com/aien-dev/omega-mv/issues/778
Base: a520cd30e54c3b0ddd553ee3ccb3f5e673345cd6

1. Freeze a small little-endian wire layout and one-producer selection policy.
   The builder's choice is fixed slots instead of a shared atomic ring. This
   avoids contention, cross-thread ordering claims and any new GPU barrier.
2. Write native Omega tests before the codec. Cover capacity, loss, selection,
   malformed inputs, stale provenance, truncated buffers and disabled bytes.
3. Implement the bounded host reference and codec in native Omega. Preserve
   the existing vector emitter and launcher unchanged. Do not emit an
   unqualified timer instruction or claim this is already a device producer.
4. Run available independent host checks and record their exact limits. Run
   the native tests only through a verified pinned AArch64 compiler. This
   x86_64 environment cannot execute that path, so it is NOT_RUN here.
5. Obtain independent review. The next implementation step is the existing
   emitter's timer, selected-lane stores and buffer ownership, with golden
   opcode/control tests followed by a separately authorized GB10 receipt.

The low32 historical timer is not a coherent64 clock. No duration, units,
CPU correlation or cross-SM order is inferred from this slice. Native tests
are supplied for later execution; an external audit is not their substitute.
