#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Report host checks that this machine does not run. Exit 0 means the report
# was printed. It is not a pass of those checks. Invoking one of them directly
# exits 2.
set -eu
host=$(uname -m)
if [ "$host" = aarch64 ] || [ "$host" = arm64 ]; then
    echo "target-only report: NOT_RUN (this host is $host; make test runs the seeded binding and the pinned-compiler rows)"
    exit 0
fi
cat <<EOF
test-om-spec-encoder: NOT_RUN (requires the pinned AArch64 omc and nvdisasm; not part of the portable host suite)
test-portable-receipts: NOT_RUN (requires the pinned AArch64 omc; not part of the portable host suite)
test-model-capacity: NOT_RUN (requires the pinned AArch64 omc; not part of the portable host suite)
target-only report: the lines above are refusals. make test-om-spec-encoder and the other named targets exit 2 unless OMEGA_OMC is given and executes on this host. This report exiting 0 is not a pass of them.
EOF
