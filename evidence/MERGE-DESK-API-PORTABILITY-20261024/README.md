# Merge desk: portable runtime API fixtures

The existing L639 patch a91726c was applied unchanged to current main c196256. The new executable source is 7678607. Only the existing shell negative-fixture builder changes; no runtime, device, compiler, or model source changes.

The focused runtime API gate passes, and the existing independent refusal oracle rejects both a failing sed command and a no-op sed command across all nine controls. The required full host run exits 2 because the existing Blackwell engine emits AArch64 `dsb sy` on this x86_64 host. Its already-started mutation target finishes with all 49 controls killed. The failure is retained and not waived. This candidate is suitable for a draft update and still needs the required complete host qualification on a supported build environment before merge.

See receipt.json for exact source identities, commands, outcomes and hashes. No GPU, Spark service, or remote branch was changed.
