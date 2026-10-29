<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Output-open failure control

Follow-up to c67fc277. The synthetic mktemp wrapper creates a directory where
match output must be opened. The scanner now reports incomplete and exits 2.
The complete audit-selftest exits 0 with original planted detections and all
error controls passing. This adds an executed output-open control to the
previously recorded search, regex, compression and filename controls.
No real repository-wide clean result is claimed.

Tested source SHA-256:
5538f3ddab502e44e16b6dac5101c1ef27fe28fc72057416f6395444127b8a98  tools/extract/audit-scan.sh
289c1cb9ba451f861411cd9a7d8ff1c78e5993f2ba40008e9698882f0db9ec24  tools/extract/audit-error-selftest.sh
