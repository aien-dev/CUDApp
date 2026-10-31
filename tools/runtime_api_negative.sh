#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# runtime_api_negative.sh -- negative controls for tools/runtime_api_check.sh (R1, omega-mv#594).
#
#   tools/runtime_api_negative.sh MANIFEST ROOT WORK_DIR API_MD
#
# Copies the headers, API.md, the manifest and its INTERNAL list to WORK_DIR, breaks one copy in one way at a time and
# runs the check on the copies. Each break must make
# the check FAIL and say the expected thing; if the check still passes, this script fails (the check cannot see that
# class of change). Then it breaks copies of the manifest and runs tools/runtime_api_semver.sh against the real one:
# each too-small bump must FAIL, and one correct bump must PASS. The real headers and manifest are never touched.
set -u
[ $# -eq 4 ] || { echo "usage: $0 MANIFEST ROOT WORK_DIR API_MD" >&2; exit 2; }
MANIFEST=$1; ROOT=$(cd "$2" && pwd) || exit 2; WORK=$3; APIMD=$4
HERE=$(dirname "$0")
bad=0
# control NAME EXPECTED_TEXT FILE SED_EXPR   (FILE: a header under src/, API.md, manifest, or internal)
control() {
  name=$1; expect=$2; file=$3; expr=$4
  d=$WORK/neg-$name; rm -rf "$d"; mkdir -p "$d/src"; cp "$ROOT"/src/omega_*.h "$d/src/"; cp "$APIMD" "$d/API.md"
  # the check reads the INTERNAL list next to the manifest (.manifest -> .internal), so both copies keep that pairing
  cp "$MANIFEST" "$d/api.manifest"; cp "${MANIFEST%.manifest}.internal" "$d/api.internal"
  case $file in
    API.md) target=$d/API.md ;;
    manifest) target=$d/api.manifest ;;
    internal) target=$d/api.internal ;;
    *) target=$d/src/$file ;;
  esac
  # POSIX sed has no -i. Generate a private replacement, then verify the mutation
  # before moving it over the scratch fixture; never edit the real headers/docs.
  if ! sed -e "$expr" "$target" > "$d/edit.tmp"; then
    echo "NEGATIVE CONTROL $name: BROKEN (fixture edit failed)"; bad=1; return
  fi
  if cmp -s "$target" "$d/edit.tmp"; then
    echo "NEGATIVE CONTROL $name: BROKEN (the edit changed nothing)"; bad=1; return
  fi
  if ! mv "$d/edit.tmp" "$target"; then
    echo "NEGATIVE CONTROL $name: BROKEN (fixture replacement failed)"; bad=1; return
  fi
  out=$("$HERE/runtime_api_check.sh" "$d/api.manifest" "$d" "$d/work" "$d/API.md" 2>&1); rc=$?
  if [ $rc -ne 0 ] && printf '%s\n' "$out" | grep -qF "$expect"; then echo "NEGATIVE CONTROL $name: FAILED as it must ($expect)"
  else echo "NEGATIVE CONTROL $name: NOT CAUGHT (rc=$rc, wanted \"$expect\")"; printf '%s\n' "$out" | head -5; bad=1; fi
}
control removed-function     "header disagrees with manifest: fn omega_gpu_session_alloc_code" omega_gpu_session.h '/^int omega_gpu_session_alloc_code(/d'
control missing-version      "version macro missing: OMEGA_RUNTIME_API_VERSION_MAJOR"           omega_runtime_api.h '/define OMEGA_RUNTIME_API_VERSION_MAJOR/d'
control changed-signature    "header disagrees with manifest: fn omega_gpu_session_free"       omega_gpu_session.h 's/^void omega_gpu_session_free(OmegaGpuMem \*m);/void omega_gpu_session_free(const OmegaGpuMem *m);/'
control changed-struct       'manifest says `struct OmegaGpuLaunch '           omega_gpu_session.h 's/uint64_t timeout_ms;         \/\*/uint32_t timeout_ms;         \/*/'
control changed-constant     'manifest says `const OMEGA_GPU_SESSION_MAX_ARGS ' omega_gpu_session.h 's/OMEGA_GPU_SESSION_MAX_ARGS 32u/OMEGA_GPU_SESSION_MAX_ARGS 64u/'
control version-doc-mismatch "must contain the line \"Omega Runtime API version:"             API.md 's/^Omega Runtime API version: .*/Omega Runtime API version: 9.9/'
control stable-without-doc   "STABLE entry omega_gpu_session_launch has no"                    API.md 's/^#### `omega_gpu_session_launch`/#### launch/'
control wrong-quoted-signature "signature quoted in" API.md 's/^uint32_t omega_gpu_session_open_count(void);/uint64_t omega_gpu_session_open_count(void);/'
control signature-inside-another "signature quoted in" API.md 's/^uint32_t omega_gpu_session_open_count(void);/int32_t omega_gpu_session_open_count(void);/'
control missing-patch-version "version macro missing: OMEGA_RUNTIME_API_VERSION_PATCH"           omega_runtime_api.h '/define OMEGA_RUNTIME_API_VERSION_PATCH/d'
control added-undeclared-function "undeclared public entry (in a manifest header, not in the manifest or the INTERNAL list): fn omega_gpu_session_sneaky" \
  omega_gpu_session.h 's/^void omega_gpu_session_set_error(const char \*stage);/&\nint omega_gpu_session_sneaky(void);/'
control added-undeclared-constant "undeclared public entry (in a manifest header, not in the manifest or the INTERNAL list): const OMEGA_GPU_SESSION_SNEAKY" \
  omega_gpu_session.h 's/^#define OMEGA_GPU_SESSION_MAX_ARGS 32u/&\n#define OMEGA_GPU_SESSION_SNEAKY 7u/'
control added-undeclared-enumerator "undeclared public entry (in a manifest header, not in the manifest or the INTERNAL list): const OMEGA_GPU_CMDLIST_BARRIER_SNEAKY" \
  omega_gpu_cmdlist.h 's/OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA = 4 }/OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA = 4, OMEGA_GPU_CMDLIST_BARRIER_SNEAKY = 5 }/'
control stale-internal-entry "stale INTERNAL entry (no manifest header declares it): fn omega_gpu_session_set_error" \
  omega_gpu_session.h '/^void omega_gpu_session_set_error(const char \*stage);/d'
# the controls the omega-mv copy of this gate carried (omega-mv tests/gates/omega-runtime-api.gate, retired there when
# the runtime API moved here) that the list above did not have yet
control removed-recover      "header disagrees with manifest: fn omega_gpu_session_recover"     omega_gpu_session.h '/^int omega_gpu_session_recover(void);/d'
control removed-padding-field "header disagrees with manifest: field OmegaGpuCmdListOpts first_barrier_none" omega_gpu_cmdlist.h '/int first_barrier_none;/,/\*\/$/d'
control new-fnptr-function   "undeclared public entry (in a manifest header, not in the manifest or the INTERNAL list): fn omega_gpu_session_zzz_fp" \
  omega_gpu_session.h '$a\
int (*omega_gpu_session_zzz_fp(void))(int);'
control manifest-fn-dropped  "undeclared public entry (in a manifest header, not in the manifest or the INTERNAL list): fn omega_gpu_serving_seal" \
  manifest '/^fn omega_gpu_serving_seal /d'
control internal-unlisted    "undeclared public entry (in a manifest header, not in the manifest or the INTERNAL list): fn omega_gpu_session_set_error" \
  internal '/^fn omega_gpu_session_set_error /d'
control internal-also-manifest "entry is both in the manifest and INTERNAL: fn omega_gpu_session_launch" \
  internal '$a\
fn omega_gpu_session_launch # control: a manifest fn also named INTERNAL'
control internal-without-reason "INTERNAL entry fn omega_gpu_session_set_error gives no reason" \
  internal 's/^fn omega_gpu_session_set_error #.*/fn omega_gpu_session_set_error/'
# a stale API.md: a section for an entry the manifest no longer has
control stale-doc-section    "stale API.md section (no manifest fn or struct): omega_gpu_session_retired" \
  API.md '$a\
#### `omega_gpu_session_retired`'

# semver NAME WANT(FAIL|PASS) EXPECTED_TEXT SED_EXPR [BASE_EXPR] -- a manifest copy edited by SED_EXPR, checked against the
# real manifest (or against a copy edited by BASE_EXPR, e.g. to stand in for a base already past 1.0)
semver() {
  name=$1; want=$2; expect=$3; expr=$4; bexpr=${5:-}
  d=$WORK/neg-semver-$name; rm -rf "$d"; mkdir -p "$d"
  sed -e "$expr" "$MANIFEST" > "$d/head.manifest" || { echo "NEGATIVE CONTROL semver-$name: BROKEN (fixture edit failed)"; bad=1; return; }
  if [ -n "$bexpr" ]; then sed -e "$bexpr" "$MANIFEST" > "$d/base.manifest"; else cp "$MANIFEST" "$d/base.manifest"; fi
  if cmp -s "$d/base.manifest" "$d/head.manifest"; then echo "NEGATIVE CONTROL semver-$name: BROKEN (the edit changed nothing)"; bad=1; return; fi
  out=$("$HERE/runtime_api_semver.sh" "$d/base.manifest" "$d/head.manifest" 2>&1); rc=$?
  if [ "$want" = FAIL ]; then
    if [ $rc -eq 1 ] && printf '%s\n' "$out" | grep -qF "$expect"; then echo "NEGATIVE CONTROL semver-$name: FAILED as it must ($expect)"
    else echo "NEGATIVE CONTROL semver-$name: NOT CAUGHT (rc=$rc, wanted \"$expect\")"; printf '%s\n' "$out" | head -5; bad=1; fi
  else
    if [ $rc -eq 0 ] && printf '%s\n' "$out" | grep -qF "$expect"; then echo "POSITIVE CONTROL semver-$name: PASSED as it must ($expect)"
    else echo "POSITIVE CONTROL semver-$name: WRONGLY FAILED (rc=$rc)"; printf '%s\n' "$out" | head -5; bad=1; fi
  fi
}
minor_up='s/^\(const OMEGA_RUNTIME_API_VERSION_MINOR \)\([0-9]*\)$/\1999/'
major_one='s/^const OMEGA_RUNTIME_API_VERSION_MAJOR [0-9]*$/const OMEGA_RUNTIME_API_VERSION_MAJOR 1/'
# major version zero (API.md section 2, semver 2.0.0 item 4): a STABLE removal passes with a MINOR bump while the base is 0.x,
# fails with no bump, and fails with a MINOR-only bump once the base is 1.x
semver removed-stable-minor-under-zero PASS "the entry changes need MINOR (MAJOR-class under major version zero)" "/^fn omega_gpu_session_alloc_code /d; $minor_up"
semver removed-stable-unbumped-under-zero FAIL "needs at least a MINOR bump" "/^fn omega_gpu_session_alloc_code /d"
semver removed-stable-minor-only-at-one FAIL "needs a MAJOR bump" "/^fn omega_gpu_session_alloc_code /d; $major_one; $minor_up" "$major_one"
semver changed-stable-unbumped FAIL "needs at least a MINOR bump" 's/^\(fn omega_gpu_session_free \)fn(\*mut OmegaGpuMem)/\1fn(*const OmegaGpuMem)/'
semver added-unbumped FAIL "needs at least a MINOR bump" '$a\
fn omega_gpu_session_sneaky fn() -> i32 # EXPERIMENTAL'
semver version-down FAIL "version goes down" 's/^const OMEGA_RUNTIME_API_VERSION_MINOR [0-9]*$/const OMEGA_RUNTIME_API_VERSION_MINOR 0/; s/^const OMEGA_RUNTIME_API_VERSION_MAJOR [0-9]*$/const OMEGA_RUNTIME_API_VERSION_MAJOR 0/'
semver added-with-minor PASS "the entry changes need MINOR" "$minor_up"'
$a\
fn omega_gpu_session_sneaky fn() -> i32 # EXPERIMENTAL'
[ $bad = 0 ] && { echo "RESULT: PASS (every negative control was caught)"; exit 0; }
echo "RESULT: FAIL (a negative control was not caught)"; exit 1
