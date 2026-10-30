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
test-om-encoder: NOT_RUN (requires AArch64 Linux; src/cudapp_om_units.c is AArch64 machine code and was not executed)
test-om-cubin: NOT_RUN (requires AArch64 Linux; the seeded cubin packer was not executed)
test-codegen-freestanding hosted comparison: NOT_RUN (requires AArch64 Linux; the .om versus C differential was not run)
test-gpu-native-runtime: NOT_RUN (requires AArch64 Linux; the seeded native units were not executed)
test-om-spec-encoder: NOT_RUN (requires the pinned omc and nvdisasm; not part of the portable host suite)
test-portable-receipts: NOT_RUN (requires the pinned omc; not part of the portable host suite)
test-model-capacity: NOT_RUN (requires the pinned omc; not part of the portable host suite)
target-only report: the lines above are refusals. make test-om-encoder and the other named targets exit 2. This report exiting 0 is not a pass of them.
EOF
