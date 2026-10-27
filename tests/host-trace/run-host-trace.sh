#!/bin/sh
# Host checks for the unified trace (docs/trace/UNIFIED_TRACE.md). No device is opened.
#
# Every check is a seeded Omega program compiled by the pinned seed compiler; this script only
# composes units, runs them and compares exit codes and output lines. It also applies four
# deliberate source mutations and requires each to make a check fail (negative controls).
#
# Usage: OMEGA_COMPILER=/path/to/seed/out/omc sh tests/host-trace/run-host-trace.sh [OUTDIR]
#        OUTDIR keeps the sample trace files (default: a temporary directory, removed on exit).
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
compiler=${OMEGA_COMPILER:-}
[ -n "$compiler" ] && [ -x "$compiler" ] || {
  echo 'OMEGA_COMPILER must name the pinned seed compiler (seed/out/omc)' >&2
  exit 69
}
work=${TMPDIR:-/tmp}/cudapp-host-trace.$$
mkdir -p "$work"
trap 'rm -rf -- "$work"' EXIT HUP INT TERM
out=${1:-$work/out}
mkdir -p "$out"
t=$root/tests/host-trace
n=$root/native
echo "compiler $(sha256sum "$compiler" | cut -d' ' -f1)"
fail() { echo "FAIL: $*" >&2; exit 1; }
build() { # build NAME FILE...
  name=$1; shift
  cat "$@" > "$work/$name.om"
  "$compiler" < "$work/$name.om" > "$work/$name" || return 1
  chmod +x "$work/$name"
}
launch_units="$t/fakenv.om $n/blackwell/vector.om $n/blackwell/launch.om $t/fakedev.om $n/trace/htrec.om $n/trace/htlaunch.om $t/launch-test.om"

# 1. recorder
build recorder "$n/trace/htrec.om" "$t/recorder-test.om" || fail 'recorder-test does not compile'
"$work/recorder" > "$work/recorder.out" || fail "recorder-test exit $?: $(cat "$work/recorder.out")"
grep -qx 'HOSTTRACE_RECORDER_PASS [0-9]*' "$work/recorder.out" || fail 'recorder-test'
echo "recorder: $(cat "$work/recorder.out")"

# 2. omtrace itself, then the launch test traced end to end (compile phases + runtime fragment)
build omtrace "$n/trace/htrec.om" "$root/tools/omtrace/omtrace.om" || fail 'omtrace does not compile'
cd "$root"
# shellcheck disable=SC2086
"$work/omtrace" f "$compiler" "$out/launch" "$out/launch.otr" $launch_units > "$work/launch.out" 2>&1 ||
  fail "omtrace f exit $?: $(cat "$work/launch.out")"
grep -qx 'HOSTTRACE_LAUNCH_PASS [0-9]*' "$work/launch.out" || fail "launch-test: $(cat "$work/launch.out")"
grep -qx 'OMTRACE_OK [0-9]*' "$work/launch.out" || fail 'omtrace f did not report OK'
echo "launch: $(paste -sd' ' "$work/launch.out")"
for k in '"cat":"kernel.launch"' '"stage_name":"host.wait"' '"cat":"mem.copy"' '"name":"omc.compile"' \
         '"name":"device.timing.NOT_RUN"' '"name":"kernel:VECTOR_1D/vector.om:gbvec"'; do
  grep -qF "$k" "$out/launch.trace.json" || fail "trace lacks $k"
done
grep -qF '"parentSpanId":"' "$out/launch.otlp.json" || fail 'otlp has no parent links'
"$work/omtrace" v "$out/launch.otr" > /dev/null 2>&1 || fail 'the recorded fragment does not validate'
grep -qF '"dropped":0,"refused":0,"open_spans":0}' "$out/launch.trace.json" || fail 'clean trace reports losses'
# a full recording that lost events must say so in both exports (and must export at all)
"$work/omtrace" f "$compiler" "$work/lossy" "$work/lossy.otr" "$n/trace/htrec.om" "$t/lossy.om" \
  > "$work/lossy.out" 2>&1 || fail "omtrace f on the lossy fixture exit $?: $(cat "$work/lossy.out")"
grep -qF '"dropped":2,"refused":1,"open_spans":1}' "$work/lossy.trace.json" || fail 'chrome export hides losses'
for k in '"aien.dropped","value":{"intValue":"2"}' '"aien.refused","value":{"intValue":"1"}' \
         '"aien.open_spans","value":{"intValue":"1"}'; do
  grep -qF "$k" "$work/lossy.otlp.json" || fail "otlp export lacks $k"
done
echo "lossy: $(cat "$work/lossy.out") with dropped 2, refused 1, open_spans 1 in both exports"

# 3. compile diagnostics and traps are located in the unit that caused them
printf 'fn f a ulong to ulong\nif gt a 3\ntrap 3\nend\nret a\nend\n' > "$work/t1.om"
printf 'fn main to int\nr = f 9\nret 0\nend\n' > "$work/t2.om"
printf 'om2 diag\nfn main to int\nret int add zz 1\nend\n' > "$work/d2.om"
s=0; "$work/omtrace" r "$compiler" "$work/trap" - "$work/t1.om" "$work/t2.om" > /dev/null 2>&1 || s=$?
[ "$s" = 0 ] || fail "omtrace r on a trapping program exit $s"
grep -qF '"trap_name":"range","candidates":"'"$work"'/t1.om:3"' "$work/trap.trace.json" || fail 'trap not located'
s=0; "$work/omtrace" c "$compiler" "$work/diag" - "$work/d2.om" > /dev/null 2>&1 || s=$?
[ "$s" = 3 ] || fail "omtrace c on a failing unit exit $s, expected 3"
grep -qF '"status":3,"line":3,"col":13,"file":"'"$work"'/d2.om"' "$work/diag.trace.json" || fail 'diagnostic not located'
echo 'locate: trap and diagnostic located'

# 4. malformed fragments: each mutant must be rejected with its own reason
build badfrag "$t/badfrag.om" || fail 'badfrag does not compile'
"$work/badfrag" "$out/launch.otr" "$work/mut" || fail 'badfrag could not write mutants'
expect='20 21 22 23 24 25 26 27 28 29 30 31 27'
m=0
for e in $expect; do
  s=0; "$work/omtrace" v "$work/mut.$m.otr" > /dev/null 2>&1 || s=$?
  [ "$s" = "$e" ] || fail "mutant $m exit $s, expected $e"
  m=$((m + 1))
done
s=0; "$work/omtrace" v "$work/missing.otr" > /dev/null 2>&1 || s=$?
[ "$s" = 6 ] || fail "missing fragment exit $s, expected 6"
echo "reject: $m malformed fragments rejected with distinct reasons, missing file exit 6"

# 5. source mutations: each must turn a passing check into a failing one
mutate() { # mutate FILE OLDLINE NEWLINE OUT
  awk -v old="$2" -v new="$3" '$0 == old { print new; hit++; next } { print } END { exit hit != 1 }' "$1" > "$4" ||
    fail "mutation anchor not found once in $1: $2"
}
# A mutant that no longer compiles proves nothing, so each must compile and then fail its check.
killed() { # killed NAME MUTATED-SOURCE FILE...: build, run; true when the run fails
  name=$1; shift
  build "$name" "$@" || fail "mutant $name does not compile"
  ! "$work/$name" > /dev/null 2>&1
}
mutate "$n/trace/htrec.om" 'if ne get htfl i 0' 'if false' "$work/m1.src"
killed m1 "$work/m1.src" "$t/recorder-test.om" || fail 'mutant: second end accepted, test still passes'
mutate "$n/trace/htrec.om" 'ret or or eq b 58 eq b 47 eq b 45' 'ret true' "$work/m2.src"
killed m2 "$work/m2.src" "$t/recorder-test.om" || fail 'mutant: any name byte accepted, test still passes'
mutate "$n/trace/htrec.om" 'ret eq get htst 0 1' 'ret true' "$work/m3.src"
killed m3 "$work/m3.src" "$t/recorder-test.om" || fail 'mutant: recorder on by default, test still passes'
mutate "$root/tools/omtrace/omtrace.om" 'if lt t1 t0' 'if false' "$work/m4.src"
build m4 "$n/trace/htrec.om" "$work/m4.src" || fail 'mutant m4 does not compile'
s=0; "$work/m4" v "$work/mut.8.otr" > /dev/null 2>&1 || s=$?
[ "$s" != 28 ] || fail 'mutant: end-before-start check removed, still rejected as time'
echo 'mutations: 4 of 4 compile and are detected'
echo 'HOST TRACE CHECKS PASS'
