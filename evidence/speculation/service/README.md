# Persistent policy process qualification

Functional and test source: `b5eaf26f2bc590fe6b0d54a793b43827079ed624`.
The source was exported from that immutable revision before compilation.
[receipt.json](receipt.json) records source ordering, compiler and executable
digests, exits, exact test outputs, and each compiled control's literal edit.

- Native parent/child pipe integration: **337 checks, exit 0**.
- Six deliberately broken service implementations compiled and were each
  rejected by the unchanged native integration test with exit 1.
- Existing policy 106, cost collector 125, paired reporter 299 and independent
  oracle 12,801 checks also passed under this native-host compiler profile.
- Existing upstream host qualification passed nine positive groups, 41 refusal
  cases and 70 original fixture comparisons after source-only reconstruction.

The six controls bypass request sequencing, bypass protocol version, admit
synthetic data when device evidence is required, silently accept EOF during a
request, retain an old response payload after refusal, and report zero elapsed
policy time. Compile errors, crashes and timeouts do not count as rejection.

The test starts eight native worker processes and checks initialization,
cold/warm requests, adaptive selection, incomplete observations, pending-token
acknowledgement, context mismatch, cancellation, EOS, shutdown, truncated headers
and bodies, wrong sequences, wrong versions, bad lengths and unsupported commands.
It validates that the service's elapsed time is positive and no larger than
the parent's measured complete exchange. These timing assertions are process
correctness checks; they are not a performance benchmark.

A traced replay also passed 337 checks. The syscall summary shows eight child
creations and reaps, nine native executable launches including the test client,
pipe reads/writes, polling, descriptor operations and monotonic clock calls.
`readelf` finds two load segments and no interpreter/dynamic dependency in the
worker. Raw development traces remain local; no model or GPU was involved.

The initial partial compiler export compared zero original fixtures because the
upstream gate obtains its fixture list from the Git index. That attempt is not
counted as a complete qualification. Restoring the pinned fixture paths and their
index produced the required 70 comparisons. Both local logs are retained.

See [SERVICE.md](../../../docs/speculation/SERVICE.md) for reconstruction and the
wire contract. Existing shell and Node.js development orchestration are disclosed;
the delivered service and integration test are handwritten seeded Omega. No new
compiler implementation, shell runtime, C/Rust connection or Python was added.

The real 27B engine connection, its state correctness, hardware cost curves,
full comparison, and independent final-source review remain pending. This receipt
does not close #63, #64, #65 or #66 and does not imply a speedup.
