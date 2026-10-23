# Runtime-API negative-fixture portability: L639 / L640

Canonical repository: aien-dev/CUDApp. Functional repair source:
`a91726cd24dbff5a6f67a1246fea0ddf1422f93a`.

Only `tools/runtime_api_negative.sh` changes functionally: POSIX sed writes a
scratch temporary file, rejects an edit error or unchanged fixture, then replaces
only that fixture. All nine original mutations, diagnostic checks and aggregation
are retained. This does not change the public API, ABI, docs or driver.

## Results

- macOS focused runtime-API target: 91 functions, 28 structs, 236 fields,
  159 constants and all nine negative controls passed.
- External edit-error and no-op probes failed closed across all nine controls.
- Independent L640 review found no lost/weakened control; see `REVIEW-L640.md`.
- Complete Linux make-declared `HOST_TESTS`: **33 declared, 33 executed, all
  exit 0**, in original order, each separately bounded. No retries or skips.
- The original 180-second aggregate run **timed out at exit 124**. That result
  and raw output remain in `linux-aggregate-timeout.*`; the replacement changes
  orchestration, not a test or acceptance rule.
- The full macOS suite **failed** in unchanged Linux NVRM driver compilation
  (`_IOC_READ`, `_IOC_WRITE`, `MAP_FIXED_NOREPLACE`); no complete macOS pass is claimed.

## Complete Linux raw proof

`linux-all-declared-host-targets.tar.gz` holds FINAL.json, the exact list obtained
from make, results.json, the stale final progress marker and all 33 raw logs.
`EXPORT-MANIFEST.json` inside it hashes every original member. All 37 original
member hashes were verified after transport. The terminal FINAL.json, not the
last RUNNING progress marker, is authoritative.

Archive SHA-256:
`2e011f27c82d9bc1a8461e0904d60398199ce065f3393a085ec5e53b256e405a`.

The original per-target orchestration is preserved separately in the evidence
and Spark `~/handoffs/qualify-L639-targets.py`. No GPU/model/performance or deployed
qualification is implied by host correctness tests. No required original
omega#594 API-prose/Opus review was substituted or waived; provenance for that
separate review remains unresolved. Source qualification is not PR/merge/issue
closure. Renda remains the sole publisher.
