#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# runtime_api_semver.sh -- does a manifest change carry the version bump docs/runtime/API.md section 2 asks for?
#
#   tools/runtime_api_semver.sh BASE_MANIFEST HEAD_MANIFEST
#
# BASE_MANIFEST is the manifest on the branch being merged into (CI: `git show BASE:docs/runtime/omega_runtime_api.manifest`),
# HEAD_MANIFEST the one being proposed. tools/runtime_api_check.sh already proves HEAD matches the headers; this script
# compares the two lists entry by entry and works out the least bump the change needs:
#   MAJOR  a STABLE fn/struct removed; a STABLE fn signature, struct size/alignment or field changed; a STABLE entry moved
#          down to EXPERIMENTAL; a constant removed or its value changed (constants count as STABLE unless the line carries
#          `# EXPERIMENTAL`); OMEGA_GPU_PLATFORM_ABI_MAJOR changed.
#   MINOR  anything else that adds, changes or removes an entry (EXPERIMENTAL changes, additions, promotions,
#          OMEGA_GPU_PLATFORM_ABI_MINOR).
#   none   no entry changed; the version may stay or go up (PATCH for a fix, MINOR for a documented behaviour addition).
# Major version zero (semver 2.0.0 item 4, API.md section 2): while the BASE major is 0, a MAJOR-class change needs only
# a MINOR bump (0.5 -> 0.6); from 1.0 on it needs MAJOR. A MAJOR-class change with no bump fails either way.
# The version (OMEGA_RUNTIME_API_VERSION_MAJOR/_MINOR/_PATCH consts; PATCH 0 when the base has none) must never go down and
# must carry at least the needed bump. Exit 0 = PASS, 1 = FAIL (an undeclared or too-small change), 2 = usage.
set -u
[ $# -eq 2 ] || { echo "usage: $0 BASE_MANIFEST HEAD_MANIFEST" >&2; exit 2; }
[ -r "$1" ] && [ -r "$2" ] || { echo "FAIL: cannot read $1 or $2"; exit 2; }
awk '
function tier(raw) { return (raw ~ /#[ \t]*EXPERIMENTAL([ \t]|$)/) ? "EXPERIMENTAL" : (raw ~ /#[ \t]*STABLE([ \t]|$)/) ? "STABLE" : "" }
function rd(side, raw,   line, w, n, key, val, t) {
  line = raw; sub(/#.*/, "", line); gsub(/^[ \t]+|[ \t]+$/, "", line); if (line == "") return
  n = split(line, w, /[ \t]+/)
  if (w[1] == "const" && w[2] ~ /^OMEGA_RUNTIME_API_VERSION(_MAJOR|_MINOR|_PATCH)?$/) { ver[side, w[2]] = w[3]; return }
  if (w[1] == "fn") { key = "fn " w[2]; val = line; sub(/^fn[ \t]+[^ \t]+[ \t]+/, "", val); t = tier(raw) }
  else if (w[1] == "struct") { key = "struct " w[2]; val = w[3] " " w[4]; t = tier(raw); stier[side, w[2]] = t }
  else if (w[1] == "field") { key = "field " w[2] " " w[3]; val = w[4] " " w[5] " " w[6] " " w[7]; t = "@" w[2] }
  else if (w[1] == "const") { key = "const " w[2]; val = w[3]; t = (tier(raw) == "EXPERIMENTAL") ? "EXPERIMENTAL" : "STABLE" }
  else return
  v[side, key] = val; tr[side, key] = t; keys[key] = 1
}
FNR == 1 { side++ }
{ rd(side, $0) }
function realtier(side, key) { t = tr[side, key]; if (substr(t, 1, 1) == "@") return stier[side, substr(t, 2)]; return t }
function need(level, why) { print "  " level ": " why; if (level == "MAJOR") major = 1; else minor = 1 }
END {
  if (side != 2) { print "FAIL: expected two manifests"; exit 2 }
  for (key in keys) {
    inb = ((1, key) in v); inh = ((2, key) in v); tb = realtier(1, key); th = realtier(2, key)
    if (inb && !inh) { if (tb == "STABLE") need("MAJOR", key " removed (STABLE)"); else need("MINOR", key " removed (" tb ")"); continue }
    if (!inb && inh) { need("MINOR", key " added (" th ")"); continue }
    if (v[1, key] != v[2, key]) {
      if (key == "const OMEGA_GPU_PLATFORM_ABI_MAJOR") need("MAJOR", key " " v[1, key] " -> " v[2, key])
      else if (key == "const OMEGA_GPU_PLATFORM_ABI_MINOR") need("MINOR", key " " v[1, key] " -> " v[2, key])
      else if (tb == "STABLE") need("MAJOR", key " changed (STABLE): `" v[1, key] "` -> `" v[2, key] "`")
      else need("MINOR", key " changed (" tb "): `" v[1, key] "` -> `" v[2, key] "`")
      continue }
    if (key !~ /^field / && tb != th) {
      if (tb == "STABLE") need("MAJOR", key " moved down " tb " -> " th); else need("MINOR", key " tier " tb " -> " th) }
  }
  for (s = 1; s <= 2; s++) {
    if (!((s, "OMEGA_RUNTIME_API_VERSION_MAJOR") in ver) || !((s, "OMEGA_RUNTIME_API_VERSION_MINOR") in ver)) {
      print "FAIL: " (s == 1 ? "base" : "head") " manifest has no OMEGA_RUNTIME_API_VERSION_MAJOR/_MINOR const"; exit 1 }
    M[s] = ver[s, "OMEGA_RUNTIME_API_VERSION_MAJOR"] + 0; m[s] = ver[s, "OMEGA_RUNTIME_API_VERSION_MINOR"] + 0
    p[s] = ((s, "OMEGA_RUNTIME_API_VERSION_PATCH") in ver) ? ver[s, "OMEGA_RUNTIME_API_VERSION_PATCH"] + 0 : 0
  }
  from = M[1] "." m[1] "." p[1]; to = M[2] "." m[2] "." p[2]
  cmp = (M[2] != M[1]) ? M[2] - M[1] : (m[2] != m[1]) ? m[2] - m[1] : p[2] - p[1]
  zero = (major && M[1] == 0); if (zero) { major = 0; minor = 1 }
  want = major ? "MAJOR" : zero ? "MINOR (MAJOR-class under major version zero)" : minor ? "MINOR" : "none"
  if (cmp < 0) { print "FAIL: version goes down: " from " -> " to; exit 1 }
  if (major && M[2] <= M[1]) { print "FAIL: undeclared change: the manifest change needs a MAJOR bump, version " from " -> " to; exit 1 }
  if (minor && !major && M[2] == M[1] && m[2] <= m[1]) { print "FAIL: undeclared change: the manifest change needs at least a MINOR bump, version " from " -> " to; exit 1 }
  print "RESULT: PASS (version " from " -> " to "; the entry changes need " want ")"
}' "$1" "$2" > "${TMPDIR:-/tmp}/runtime_api_semver.$$" ; rc=$?
# changes first, then the verdict, in one block
out=$(cat "${TMPDIR:-/tmp}/runtime_api_semver.$$"); rm -f "${TMPDIR:-/tmp}/runtime_api_semver.$$"
printf '%s\n' "$out" | grep '^  ' | LC_ALL=C sort
printf '%s\n' "$out" | grep -v '^  '
exit $rc
