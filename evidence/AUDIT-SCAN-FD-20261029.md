<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Match-output open failure distinction

Follow-up to source 69f51413. The scanner now opens its match-output descriptor
before invoking grep, so a shell output-redirection failure cannot be confused
with grep's no-match exit status. No detection pattern changes.

`bash tools/extract/audit-selftest.sh` passes after this change, retaining the
original 17 detections and every error-refusal control described in
AUDIT-SCAN-ERRORS-20261029.md. The earlier receipt remains bound to its earlier
source, not relabeled. This is host scanner verification, not a clean-history
claim. Tested scanner SHA-256:
bc726f90851ba5e9a33939a7bf88b10c9424a0d0e3281562c9c925e089bd7bbf  tools/extract/audit-scan.sh
