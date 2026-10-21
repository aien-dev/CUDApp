#!/bin/sh
# Omega runtime R3 (omega-mv#596): the portability table must match the source tree.
#   usage: runtime_portability_check.sh <root> <table.tsv> <engine.mk>
# Host only, no compiler, no device. POSIX sh + awk.
#
# Fails (exit 1, one line per problem) when:
#   1. a src/omega_gpu_*, src/omega_blackwell_* or src/omega_bw_* file has no row in the table
#   2. a row names a file that does not exist, or is duplicated, or has a bad class / binding / empty reason
#   3. a PORTABLE row's file includes a Linux or NVRM header (nvrm.h, m16_native.h, omega_gpu_platform_linux.h,
#      pthread.h, unistd.h, fcntl.h, sys/*, linux/*) or a Blackwell chip header (omega_blackwell_*, omega_bw_*)
#   4. the os_binding column disagrees with what the include scan finds, or a .c file in GPU_FS_SRCS
#      (engine.mk, the make gpu-engine-freestanding list) is not FREESTANDING_PROVEN
# It proves every file has a row and that the rows agree with the includes. It does not prove that
# a row's reason is right.
#
# Include scan: direct #include lines only. Lines inside "#ifdef OMEGA_ATTN_DEBUG" (a debug trace, off by default) and "#ifndef OMEGA_GPU_NO_POSIX" (or the #else
# of "#ifdef OMEGA_GPU_NO_POSIX") are the hosted-only build and are not counted, because the
# freestanding build never sees them.
root=$1; table=$2; mk=$3
[ -d "$root/src" ] && [ -f "$table" ] && [ -f "$mk" ] || { echo "usage: $0 <root> <table.tsv> <engine.mk>"; exit 2; }

fail=0
bad() { echo "FAIL: $*"; fail=1; }

# unguarded includes of one file, one per line, as written between <> or ""
includes() {
	awk '
	function hosted(  i) { for (i = 1; i <= d; i++) if (h[i]) return 1; return 0 }
	/^[ \t]*#[ \t]*ifdef[ \t]+OMEGA_ATTN_DEBUG/ { d++; h[d] = 1; isdef[d] = 0; next }
	/^[ \t]*#[ \t]*if/ && !/^[ \t]*#[ \t]*(ifdef|ifndef)[ \t]+OMEGA_GPU_NO_POSIX/ {
		d++; h[d] = 0; isdef[d] = 0
		if ($0 ~ /^[ \t]*#[ \t]*if[ \t]+!defined[ \t]*\(?[ \t]*OMEGA_GPU_NO_POSIX/) h[d] = 1
		next }
	/^[ \t]*#[ \t]*ifndef[ \t]+OMEGA_GPU_NO_POSIX/ { d++; h[d] = 1; isdef[d] = 0; next }
	/^[ \t]*#[ \t]*ifdef[ \t]+OMEGA_GPU_NO_POSIX/  { d++; h[d] = 0; isdef[d] = 1; next }
	/^[ \t]*#[ \t]*(else|elif)/ { if (d > 0 && isdef[d]) h[d] = 1; next }
	/^[ \t]*#[ \t]*endif/ { if (d > 0) d--; next }
	/^[ \t]*#[ \t]*include/ {
		if (hosted()) next
		s = $0; sub(/^[ \t]*#[ \t]*include[ \t]*[<"]/, "", s); sub(/[>"].*$/, "", s); print s }
	' "$1"
}

# --- row syntax and file existence -----------------------------------------------------------
rows=$(grep -v '^#' "$table" | grep -v '^file	' || true)
tmp_rows=$(mktemp); printf '%s\n' "$rows" | grep -v '^$' > "$tmp_rows"
fs_srcs=$(awk '/^GPU_FS_SRCS[ \t]*=/ {on=1} on {print} on && !/\\$/ {exit}' "$mk" | tr -d '\\' | tr ' \t' '\n\n' | grep '^src/.*\.c$' || true)
[ -n "$fs_srcs" ] || bad "no GPU_FS_SRCS list found in $mk"

dups=$(cut -f1 "$tmp_rows" | sort | uniq -d)
[ -z "$dups" ] || bad "duplicate rows: $(echo $dups)"

while IFS='	' read -r f class reason bind extra; do
	[ -n "$f" ] || continue
	[ -f "$root/$f" ] || { bad "row names a missing file: $f"; continue; }
	case "$class" in CHIP_SPECIFIC|PORTABLE) ;; *) bad "$f: class '$class' is not CHIP_SPECIFIC or PORTABLE" ;; esac
	case "$bind" in FREESTANDING_PROVEN|NO_HOSTED_INCLUDE|HOSTED_LIBC|LINUX_DRIVER) ;; *) bad "$f: os_binding '$bind' is not one of FREESTANDING_PROVEN NO_HOSTED_INCLUDE HOSTED_LIBC LINUX_DRIVER" ;; esac
	[ -n "$reason" ] || bad "$f: empty reason"
	[ -z "$extra" ] || bad "$f: more than four columns"

	inc=$(includes "$root/$f")
	linux=$(printf '%s\n' "$inc" | grep -E '^(nvrm\.h|m16_native\.h|omega_gpu_platform_linux\.h|pthread\.h|unistd\.h|fcntl\.h|sys/.*|linux/.*)$' || true)
	driver=$(printf '%s\n' "$linux" | grep -E '^(nvrm\.h|m16_native\.h|omega_gpu_platform_linux\.h|sys/.*|linux/.*)$' || true)
	hosted=$(printf '%s\n' "$inc" | grep -E '^(stdio\.h|stdlib\.h|time\.h|errno\.h|math\.h|pthread\.h|unistd\.h|fcntl\.h|assert\.h|inttypes\.h|signal\.h|ctype\.h|sys/.*|linux/.*)$' || true)
	chip=$(printf '%s\n' "$inc" | grep -E '^omega_(blackwell_|bw_)' || true)

	# derived binding
	if [ -n "$driver" ]; then want=LINUX_DRIVER
	elif [ -n "$hosted" ]; then want=HOSTED_LIBC
	else want=NO_HOSTED_INCLUDE; fi
	if printf '%s\n' "$fs_srcs" | grep -qx "$f"; then
		[ "$want" = NO_HOSTED_INCLUDE ] || bad "$f: is in GPU_FS_SRCS but the include scan finds $(echo $driver $hosted)"
		want=FREESTANDING_PROVEN
	fi
	[ "$bind" = "$want" ] || bad "$f: os_binding is $bind, the include scan and GPU_FS_SRCS say $want"

	if [ "$class" = PORTABLE ]; then
		[ -z "$linux" ] || bad "$f: PORTABLE but includes a Linux/NVRM header: $(echo $linux)"
		[ -z "$chip" ] || bad "$f: PORTABLE but includes a Blackwell chip header: $(echo $chip)"
	fi
done < "$tmp_rows"

# --- every file has a row ----------------------------------------------------------------------
for p in "$root"/src/omega_gpu_* "$root"/src/omega_blackwell_* "$root"/src/omega_bw_*; do
	[ -f "$p" ] || continue
	f=src/$(basename "$p")
	cut -f1 "$tmp_rows" | grep -qx "$f" || bad "no row in the table for $f"
done

n=$(wc -l < "$tmp_rows" | tr -d ' ')
rm -f "$tmp_rows"
if [ "$fail" = 0 ]; then echo "PASS: $n rows, every omega_gpu_* / omega_blackwell_* / omega_bw_* file in src/ has one and the rows agree with the includes"; exit 0; fi
echo "FAIL: portability table does not match the tree"; exit 1
