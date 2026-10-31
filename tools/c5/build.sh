#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# C5 sharing (docs/OMEGA.md row C5): build what tools/c5/chip.sh runs.
#   build/gpu_isolation_test         the imported test, unchanged (make target)
#   build/gpu_isolation_test_mutant  the same source with one change made here, never committed: the chip work kernel
#                                    is codegen mutant 1 (out = 2x + tag + 1), so the chip values check must go red
#   build/c5_matrix                  tools/c5/c5_matrix.om compiled by the pinned seeded omc (tools/c5/omc.pin)
# OMC=path skips the fetch of the pinned omc; its sha256 is checked either way.
set -eu
cd "$(dirname "$0")/../.."
make build/gpu_isolation_test
src=tests/gpu_isolation_test.c
mut=build/gpu_isolation_test_mutant.c
sed 's/if (codegen(hang, 0, &k->prog) != 0) return -1;/if (codegen(hang, 1, \&k->prog) != 0) return -1;/' "$src" > "$mut"
[ "$(diff "$src" "$mut" | grep -c '^>')" -eq 1 ] || { echo "FAIL: the mutant edit did not apply exactly once" >&2; exit 1; }
# the mutant compiles with the Makefile's own CC and CFLAGS, through a one-rule makefile on stdin
printf '%s\n\t%s\n' 'build/gpu_isolation_test_mutant: build/gpu_isolation_test_mutant.c build/libomega_gpu.a' \
    '$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ $< build/libomega_gpu.a -lpthread -lm' |
    make -f Makefile -f - build/gpu_isolation_test_mutant
read -r repo sha path want < tools/c5/omc.pin
if [ -z "${OMC:-}" ]; then
    OMC=build/omc-pinned
    curl -fsSL "https://raw.githubusercontent.com/$repo/$sha/$path" -o "$OMC"
fi
got=$(sha256sum "$OMC" | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "FAIL: omc sha256 $got, pin $want" >&2; exit 1; }
chmod +x "$OMC"
"$OMC" < tools/c5/c5_matrix.om > build/c5_matrix 2> build/c5_matrix.err || { cat build/c5_matrix.err >&2; exit 1; }
[ ! -s build/c5_matrix.err ] || { cat build/c5_matrix.err >&2; exit 1; }
chmod +x build/c5_matrix
sha256sum build/gpu_isolation_test build/gpu_isolation_test_mutant build/c5_matrix "$OMC"
