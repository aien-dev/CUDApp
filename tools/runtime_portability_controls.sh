#!/bin/sh
# Negative and positive controls for tools/runtime_portability_check.sh (omega-mv#596, slice R3).
#   usage: runtime_portability_controls.sh <scratch-dir> <table.tsv> <engine.mk>
# Each control builds a scratch copy of the GPU source files, breaks one thing, and requires the
# checker to FAIL with the expected message (or, for the positive control, to still PASS).
# A check that cannot fail proves nothing. Exit 0 only when every control behaves as required.
scratch=$1; table=$2; mk=$3
here=$(cd "$(dirname "$0")" && pwd)
chk="$here/runtime_portability_check.sh"
[ -n "$scratch" ] && [ -f "$table" ] && [ -f "$mk" ] || { echo "usage: $0 <scratch-dir> <table.tsv> <engine.mk>"; exit 2; }
ok=0

fresh() { # fresh <name>: scratch root with a copy of the GPU files and the table
	r="$scratch/$1"; rm -rf "$r"; mkdir -p "$r/src"
	cp src/omega_gpu_* src/omega_blackwell_* src/omega_bw_* "$r/src/"
	cp "$table" "$r/table.tsv"
}
expect_fail() { # expect_fail <name> <pattern> <what was broken>
	out=$("$chk" "$scratch/$1" "$scratch/$1/table.tsv" "$mk" 2>&1); rc=$?
	if [ $rc -eq 1 ] && printf '%s\n' "$out" | grep -q "$2"; then
		echo "CONTROL $1: FAIL as required ($3)"; printf '%s\n' "$out" | grep "$2" | head -2 | sed 's/^/    /'
	else
		echo "CONTROL $1: WRONG, the checker exit was $rc and did not report '$2' ($3)"; printf '%s\n' "$out" | head -5 | sed 's/^/    /'; ok=1
	fi
}
expect_pass() {
	out=$("$chk" "$scratch/$1" "$scratch/$1/table.tsv" "$mk" 2>&1); rc=$?
	if [ $rc -eq 0 ]; then echo "CONTROL $1: PASS as required ($2)"
	else echo "CONTROL $1: WRONG, the checker failed ($2)"; printf '%s\n' "$out" | head -5 | sed 's/^/    /'; ok=1; fi
}
setclass() { # setclass <name> <src/file> <class>
	awk -F'\t' -v f="$2" -v c="$3" 'BEGIN{OFS="\t"} $1==f{$2=c} {print}' "$scratch/$1/table.tsv" > "$scratch/$1/t.new" && mv "$scratch/$1/t.new" "$scratch/$1/table.tsv"
}

fresh unchanged;      expect_pass unchanged "scratch copy of the real tree and table"

fresh new-file-no-row
printf 'int omega_gpu_zz_dummy(void) { return 0; }\n' > "$scratch/new-file-no-row/src/omega_gpu_zz_dummy.c"
expect_fail new-file-no-row "no row in the table for src/omega_gpu_zz_dummy.c" "a new omega_gpu_ file with no row"

fresh row-missing-file
printf 'src/omega_gpu_zz_ghost.c\tPORTABLE\tghost\tNO_HOSTED_INCLUDE\n' >> "$scratch/row-missing-file/table.tsv"
expect_fail row-missing-file "row names a missing file: src/omega_gpu_zz_ghost.c" "a row for a file that does not exist"

fresh portable-includes-nvrm
printf '#include "nvrm.h"\n' | cat - "$scratch/portable-includes-nvrm/src/omega_gpu_wait.c" > "$scratch/p.c" && mv "$scratch/p.c" "$scratch/portable-includes-nvrm/src/omega_gpu_wait.c"
expect_fail portable-includes-nvrm "omega_gpu_wait.c: PORTABLE but includes a Linux/NVRM header: nvrm.h" "a PORTABLE file that includes nvrm.h"

fresh portable-row-on-linux-file
setclass portable-row-on-linux-file src/omega_gpu_platform_linux.c PORTABLE
expect_fail portable-row-on-linux-file "omega_gpu_platform_linux.c: PORTABLE but includes a Linux/NVRM header" "the real Linux backend relabelled PORTABLE"

fresh portable-row-on-chip-file
setclass portable-row-on-chip-file src/omega_blackwell_qmd.h PORTABLE
setclass portable-row-on-chip-file src/omega_gpu_attention_api.c PORTABLE
expect_fail portable-row-on-chip-file "omega_gpu_attention_api.c: PORTABLE but includes a Blackwell chip header" "a file that includes Blackwell headers relabelled PORTABLE"

fresh guarded-include
{ printf '#ifndef OMEGA_GPU_NO_POSIX\n#include "nvrm.h"\n#include <pthread.h>\n#endif\n'; cat "$scratch/guarded-include/src/omega_gpu_engine.c"; } > "$scratch/g.c" && mv "$scratch/g.c" "$scratch/guarded-include/src/omega_gpu_engine.c"
expect_pass guarded-include "an nvrm.h include inside #ifndef OMEGA_GPU_NO_POSIX is the hosted-only build and must not count"

fresh freestanding-claim-broken
printf '#include <stdlib.h>\n' | cat - "$scratch/freestanding-claim-broken/src/omega_gpu_fmt.c" > "$scratch/g.c" && mv "$scratch/g.c" "$scratch/freestanding-claim-broken/src/omega_gpu_fmt.c"
expect_fail freestanding-claim-broken "omega_gpu_fmt.c: is in GPU_FS_SRCS but the include scan finds stdlib.h" "a freestanding-closure file that gained stdlib.h"

[ $ok -eq 0 ] && echo "ALL CONTROLS BEHAVED AS REQUIRED" || echo "A CONTROL DID NOT BEHAVE: the check is not trustworthy"
exit $ok
