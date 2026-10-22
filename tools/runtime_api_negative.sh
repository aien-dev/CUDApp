#!/bin/sh
# runtime_api_negative.sh -- negative controls for tools/runtime_api_check.sh (R1, omega-mv#594).
#
#   tools/runtime_api_negative.sh MANIFEST ROOT WORK_DIR API_MD
#
# Copies the headers to WORK_DIR, breaks the copy in one way at a time and runs the check on it. Each break must make
# the check FAIL and say the expected thing; if the check still passes, this script fails (the check cannot see that
# class of change). The real headers are never touched.
set -u
[ $# -eq 4 ] || { echo "usage: $0 MANIFEST ROOT WORK_DIR API_MD" >&2; exit 2; }
MANIFEST=$1; ROOT=$(cd "$2" && pwd) || exit 2; WORK=$3; APIMD=$4
HERE=$(dirname "$0")
bad=0
# control NAME EXPECTED_TEXT SED_FILE-or-"" SED_EXPR
control() {
  name=$1; expect=$2; file=$3; expr=$4
  d=$WORK/neg-$name; rm -rf "$d"; mkdir -p "$d/src"; cp "$ROOT"/src/omega_*.h "$d/src/"; cp "$APIMD" "$d/API.md"
  case $file in
    API.md) sed -i "$expr" "$d/API.md" ;;
    *) sed -i "$expr" "$d/src/$file" ;;
  esac
  if cmp -s "$ROOT/src/$file" "$d/src/$file" 2>/dev/null && [ "$file" != API.md ]; then echo "NEGATIVE CONTROL $name: BROKEN (the edit changed nothing)"; bad=1; return; fi
  out=$("$HERE/runtime_api_check.sh" "$MANIFEST" "$d" "$d/work" "$d/API.md" 2>&1); rc=$?
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
[ $bad = 0 ] && { echo "RESULT: PASS (every negative control was caught)"; exit 0; }
echo "RESULT: FAIL (a negative control was not caught)"; exit 1
