<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Scanner error refusal regression

Base: b348277e4383ab1f37f45bb702ea7d4fb4363782. Linux host-only.
No repository-wide clean-audit conclusion is established by this fixture.
Issue #89 remains the owner of complete audit execution and independent review.

`bash tools/extract/audit-error-selftest.sh` against original audit-scan.sh:
exit 1, `grep phase -naP returned 0, expected 2`. This is a reproduced false
success from an injected search-tool error, not evidence about real secrets.

`bash tools/extract/audit-selftest.sh` against the fixed scanner: exit 0.
The original 17 planted detections still pass. Added controls verify:
- injected errors in raw-secret, raw-private, message-secret and message-private scans return 2;
- invalid secret/private patterns return 2;
- corrupted compressed content returns 2 and remains listed as unscanned;
- clean synthetic history returns 0;
- a synthetic secret in a filename containing the former formatting delimiter returns 1 and is retained.

The fix distinguishes grep status 1 (no matches) from errors, writes matches
before formatting them, and uses literal output prefixes. Formatting pipeline
errors remain subject to pipefail. Binary-count pipeline failures are also
checked before classification. A dedicated CI job runs both scanner test sets.

Tested source identities:
3dfe636aa78689dcfe443e68205b49c1230f94edd2d1566f1eb5526bb4f86e2c  tools/extract/audit-scan.sh
4596ffdcb2f6171bc235f6855ccb0064d6827a030297e6a4983337802b61820a  tools/extract/audit-selftest.sh
30f8866924b793e1925d40b90157ee0fa1c6b1985de9c0ebcec5922c89d81e59  tools/extract/audit-error-selftest.sh
