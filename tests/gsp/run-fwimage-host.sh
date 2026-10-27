#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Disclosed verification orchestration. The parser under test is handwritten seeded Omega
# (native/gsp/fwimage.om + omega-mv om/tls/sha2.om). This script only coordinates the build and
# compares it with independent oracles: readelf -SW (binutils), dd + sha256sum (coreutils), and
# L332's C tool ck_gsp_fw_inspect.c, built into a temporary directory as a verification oracle only.
# Read-only: no device access; the firmware file is only read; mutated copies live in a temp dir.
#
# usage: run-fwimage-host.sh /abs/path/to/pinned-omc /abs/path/to/gsp_ga10x.bin
# The pinned omc must sit in an omega-mv checkout (seed/out/omc), whose om/tls/sha2.om is used.
# CK_GSP_ROOT overrides the oracle tree (default: L332's aienos-own-gpu-driver worktree).
# Exit: 0 FWIMAGE PASS; 1 FAIL; 64 usage; 69 pin refused; 74 I/O; 77 NOT_RUN.
set -u
PATH=/usr/bin:/bin
LC_ALL=C
export PATH LC_ALL

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != aarch64 ]; then
  echo 'NOT_RUN: fwimage host test needs Linux AArch64 and the pinned seeded omc' >&2
  exit 77
fi
if [ "$#" -ne 2 ]; then
  echo 'usage: run-fwimage-host.sh /absolute/path/to/pinned-omc /absolute/path/to/gsp_ga10x.bin' >&2
  exit 64
fi
omc=$1
fw=$2
case $omc in /*) ;; *) echo 'usage: omc path must be absolute' >&2; exit 64 ;; esac
case $fw in /*) ;; *) echo 'usage: firmware path must be absolute' >&2; exit 64 ;; esac

# Pins. omc: omega-mv 369dbc96 seed/out/omc, equal to its committed seed/out/omc.sha256.
# sha2.om: omega-mv 369dbc96 om/tls/sha2.om. Firmware: 580.173.02 gsp_ga10x.bin as installed.
omc_pin=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
sha2_pin=5cf0f98116def6917d3a373f2bd4c3932d46e3ff84d9567bbae9e720366fdb33
fw_pin=32cb7cf3e77f97e92f6e92135939ccb3ba382748144569c0dfccf3338d43d963

digest() { d=$(sha256sum < "$1") || return 1; printf '%s\n' "${d%% *}"; }

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd) || exit 74
sha2=$(CDPATH= cd -- "$(dirname -- "$omc")/../.." && pwd)/om/tls/sha2.om || exit 74
expected=$root/tests/gsp/expected-580.173.02.txt
src=$root/native/gsp/fwimage.om
ck_root=${CK_GSP_ROOT:-/home/drakestapleton/workspace/aienos-own-gpu-driver/native}

[ "$(digest "$omc")" = "$omc_pin" ] || { echo "REFUSED: $omc digest differs from pin $omc_pin" >&2; exit 69; }
[ "$(digest "$sha2")" = "$sha2_pin" ] || { echo "REFUSED: $sha2 digest differs from pin $sha2_pin" >&2; exit 69; }
[ "$(digest "$fw")" = "$fw_pin" ] || { echo "REFUSED: $fw digest differs from pin $fw_pin" >&2; exit 69; }
echo "PIN omc $omc_pin"
echo "PIN sha2.om $sha2_pin"
echo "PIN firmware $fw_pin"
echo "SRC fwimage.om $(digest "$src")"

out=$(mktemp -d "${TMPDIR:-/tmp}/cudapp-fwimage.XXXXXX") || exit 74
echo "Outputs: $out"
fail=0
bad() { echo "FAIL: $*" >&2; fail=1; }

# 1. Compile: fwimage.om then sha2.om, one OM-2 unit.
cat "$src" "$sha2" > "$out/unit.om" || exit 74
timeout 60 "$omc" < "$out/unit.om" > "$out/fwimage" 2> "$out/compile.err"
rc=$?
if [ "$rc" -ne 0 ] || [ -s "$out/compile.err" ] || [ ! -s "$out/fwimage" ]; then
  cat "$out/compile.err" >&2
  echo "FAIL: compile exit $rc" >&2
  exit 1
fi
chmod 700 "$out/fwimage" || exit 74
echo "COMPILE PASS $(digest "$out/fwimage")"

# 2. Run on the real file: no family, gb20y, gb10y.
for arm in none gb20y gb10y; do
  if [ "$arm" = none ]; then set -- "$fw"; else set -- "$fw" "$arm"; fi
  timeout 120 "$out/fwimage" "$@" > "$out/run-$arm.out" 2> "$out/run-$arm.err"
  rc=$?
  if [ "$rc" -ne 0 ] || [ -s "$out/run-$arm.err" ]; then
    cat "$out/run-$arm.err" >&2
    bad "fwimage $arm exit $rc"
  fi
done

# 3. Reference report from readelf -SW, dd and sha256sum only (never from fwimage output).
sect() { # FILE -> "index name off size" rows (hex without 0x)
  readelf -SW "$1" | sed -n 's/^ *\[ *\([0-9][0-9]*\)\] /\1 /p' | awk '$1 > 0 { print $1, $2, $5, $6 }'
}
span_sha() { # FILE OFFHEX SIZEHEX
  dd if="$1" iflag=skip_bytes,count_bytes skip=$((0x$2)) count=$((0x$3)) bs=1048576 status=none \
    | sha256sum | cut -d' ' -f1
}
gen_ref() { # FILE -> reference report, including both SELECT lines
  sect "$1" > "$out/ref.rows" || return 1
  [ -s "$out/ref.rows" ] || return 1
  while read -r i name off size; do
    printf 'SECTION %s off=0x%x size=0x%x\n' "$name" "0x$off" "0x$size"
  done < "$out/ref.rows"
  set -- "$1" $(awk '$2 == ".fwversion" { print $3, $4 }' "$out/ref.rows")
  printf 'FWVERSION %s\n' "$(dd if="$1" iflag=skip_bytes,count_bytes skip=$((0x$2)) count=$((0x$3)) status=none | tr -d '\000')"
  while read -r i name off size; do
    case $name in .fwimage|.fwsignature_*) printf 'SHA256 %s %s\n' "$name" "$(span_sha "$1" "$off" "$size")" ;; esac
  done < "$out/ref.rows"
  set -- "$1" $(awk '$2 == ".fwimage" { print $3, $4 }' "$out/ref.rows")
  for f in gb20y gb10y; do
    s=$(awk -v n=".fwsignature_$f" '$2 == n { print $3, $4 }' "$out/ref.rows")
    printf 'SELECT family=%s image_off=0x%x image_size=0x%x sig_section=.fwsignature_%s sig_sha256=%s\n' \
      "$f" "0x$2" "0x$3" "$f" "$(span_sha "$1" ${s})"
  done
}
gen_ref "$fw" > "$out/reference.txt" || { echo 'FAIL: readelf/dd reference generation' >&2; exit 1; }

# compare EXPECTED_FILE: the committed expectation must equal the fresh tool reference, and each
# fwimage run must equal the expectation (base lines, plus the one SELECT line for its family).
compare() {
  grep -v '^#' "$1" > "$out/exp.body" || return 1
  cmp -s "$out/exp.body" "$out/reference.txt" || { echo "  differs from readelf/dd reference: $1" >&2; diff "$out/exp.body" "$out/reference.txt" >&2; return 1; }
  grep -v '^SELECT ' "$out/exp.body" > "$out/exp.none"
  cmp -s "$out/exp.none" "$out/run-none.out" || { diff "$out/exp.none" "$out/run-none.out" >&2; return 1; }
  for f in gb20y gb10y; do
    { cat "$out/exp.none"; grep "^SELECT family=$f " "$out/exp.body"; } > "$out/exp.$f"
    cmp -s "$out/exp.$f" "$out/run-$f.out" || { diff "$out/exp.$f" "$out/run-$f.out" >&2; return 1; }
  done
  return 0
}
if compare "$expected"; then
  echo "EXPECTED PASS $(grep -vc '^#' "$expected") lines = readelf/dd/sha256sum reference = fwimage output (3 runs)"
else
  bad 'fwimage output, expected file and readelf/dd reference disagree'
fi

# 4. Checker negative control: one altered digit in the expectation must make compare fail.
awk '/^SHA256 \.fwimage / { p = length("SHA256 .fwimage ") + 11; c = substr($0, p, 1); $0 = substr($0, 1, p - 1) (c == "0" ? "1" : "0") substr($0, p + 1) } { print }' \
  "$expected" > "$out/expected-mutant.txt"
if [ "$(cmp -l "$expected" "$out/expected-mutant.txt" | wc -l)" -ne 1 ]; then
  bad 'checker mutant did not change exactly one byte'
elif compare "$out/expected-mutant.txt" 2> "$out/checker-mutant.diff"; then
  bad 'checker accepted an altered expected line'
else
  echo 'CHECKER-NEGATIVE PASS altered expected digest line is rejected'
fi

# 5. C oracle (verification only): compare the lines it shares with fwimage.
if [ ! -r "$ck_root/kernel/tools/ck_gsp_fw_inspect.c" ] || [ ! -r "$ck_root/argus/sha256.c" ]; then
  echo "NOT_RUN: C oracle source absent under $ck_root" >&2
  exit 77
fi
echo "ORACLE source $(digest "$ck_root/kernel/tools/ck_gsp_fw_inspect.c") ck_gsp_fw_inspect.c ($(git -C "$ck_root" rev-parse --short HEAD 2>/dev/null || echo unknown))"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -I"$ck_root/argus" -o "$out/ck_gsp_fw_inspect" \
  "$ck_root/kernel/tools/ck_gsp_fw_inspect.c" "$ck_root/argus/sha256.c" 2> "$out/oracle-build.err" \
  || { cat "$out/oracle-build.err" >&2; bad 'C oracle build'; }
for f in gb20y gb10y; do
  "$out/ck_gsp_fw_inspect" inspect "$fw" --family "$f" > "$out/oracle-$f.txt" 2>&1 || bad "C oracle refused $f"
  grep -q '^RESULT OK$' "$out/oracle-$f.txt" || bad "C oracle RESULT not OK for $f"
  awk -v fam="$f" '
    $1 == "section" && $2 > 0 { split($5, o, "="); split($6, s, "="); printf "SECTION %s off=0x%x size=0x%x\n", $3, o[2], s[2]; if ($3 == ".fwimage") { io = o[2]; is = s[2] } }
    $1 == "fwversion" { print "FWVERSION " $2 }
    $1 == "sha256_fwimage" { print "SHA256 .fwimage " $2 }
    $1 == "signature_section" { sig = $2 }
    $1 == "sha256_signature" { sh = $2 }
    END { printf "SELECT family=%s image_off=0x%x image_size=0x%x sig_section=%s sig_sha256=%s\n", fam, io, is, sig, sh }
  ' "$out/oracle-$f.txt" > "$out/oracle-$f.lines"
  grep -E '^(SECTION |FWVERSION |SHA256 \.fwimage |SELECT )' "$out/run-$f.out" > "$out/fwimage-$f.lines"
  if cmp -s "$out/oracle-$f.lines" "$out/fwimage-$f.lines"; then
    echo "ORACLE PASS $f $(wc -l < "$out/oracle-$f.lines") shared lines equal (sections, version, image digest, selection)"
  else
    diff "$out/oracle-$f.lines" "$out/fwimage-$f.lines" >&2
    bad "C oracle disagrees for $f"
  fi
done

# 6. Negative controls on mutated copies. Each must refuse with its documented code, print nothing
# on stdout, and the copy must differ from the original only where intended.
neg=$out/neg
mkdir "$neg" || exit 74
le8() { # VALUE OFFSET FILE: write VALUE as 8 little-endian bytes at OFFSET
  v=$1 k=0
  while [ "$k" -lt 8 ]; do
    printf "\\$(printf '%03o' $((v & 255)))"
    v=$((v >> 8)) k=$((k + 1))
  done | dd of="$3" bs=1 seek="$2" conv=notrunc status=none
}
poke() { # OFFSET CHAR FILE
  printf '%s' "$2" | dd of="$3" bs=1 seek="$1" conv=notrunc status=none
}
expect_refusal() { # NAME CODE CHANGED_BYTES FILE [FAMILY]
  name=$1 code=$2 nbytes=$3 file=$4
  shift 4
  if [ "$nbytes" != - ]; then
    n=$(cmp -l "$fw" "$file" 2>/dev/null | wc -l)
    [ "$n" -eq "$nbytes" ] || { bad "control $name changed $n bytes, intended $nbytes"; return; }
  fi
  timeout 60 "$out/fwimage" "$file" "$@" > "$neg/$name.out" 2> "$neg/$name.err"
  rc=$?
  if [ "$rc" -eq "$code" ] && [ ! -s "$neg/$name.out" ] && grep -q '^FWIMAGE REFUSE ' "$neg/$name.err"; then
    echo "NEGATIVE PASS $name exit $rc: $(cat "$neg/$name.err")"
  else
    bad "control $name exit $rc (want $code), stdout $(wc -c < "$neg/$name.out") bytes"
  fi
}
shoff=$(readelf -hW "$fw" | awk -F: '/Start of section headers/ { split($2, a, " "); print a[1] }')
fsize=$(wc -c < "$fw")
row() { awk -v n="$1" '$2 == n { print $1, $3, $4 }' "$out/ref.rows"; }
set -- $(row .shstrtab); strtab=$((0x$2))
nameoff() { od -An -tu4 -j $((shoff + $1 * 64)) -N4 "$fw" | tr -d ' '; }

head -c 1000 "$fw" > "$neg/truncated.bin"
expect_refusal truncated 12 - "$neg/truncated.bin"

cp "$fw" "$neg/magic.bin" && poke 1 X "$neg/magic.bin"
expect_refusal magic 10 1 "$neg/magic.bin"
rm -f "$neg/magic.bin"

set -- $(row .fwversion); vidx=$1
cp "$fw" "$neg/offpast.bin" && le8 $((fsize + 4096)) $((shoff + vidx * 64 + 24)) "$neg/offpast.bin"
expect_refusal offpast 14 - "$neg/offpast.bin"
rm -f "$neg/offpast.bin"

cp "$fw" "$neg/rename.bin" && poke $((strtab + $(nameoff "$vidx") + 3)) x "$neg/rename.bin"
expect_refusal rename-fwversion 16 1 "$neg/rename.bin"
rm -f "$neg/rename.bin"

set -- $(row .fwsignature_gb10x); didx=$1
cp "$fw" "$neg/dup.bin" && poke $((strtab + $(nameoff "$didx") + 17)) y "$neg/dup.bin"
expect_refusal duplicate-name 18 1 "$neg/dup.bin"
rm -f "$neg/dup.bin"

expect_refusal unknown-family 17 0 "$fw" gb99z

if [ "$fail" -eq 0 ]; then
  echo 'FWIMAGE PASS'
  exit 0
fi
echo 'FWIMAGE FAIL' >&2
exit 1
