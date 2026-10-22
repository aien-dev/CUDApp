#!/bin/sh
# runtime_api_check.sh -- host check of the Omega Runtime API manifest against the C headers (no GPU, no physics).
#
#   tools/runtime_api_check.sh [--bless] MANIFEST ROOT WORK_DIR [API_MD]
#
# MANIFEST  docs/runtime/omega_runtime_api.manifest (line format: docs/runtime/API.md, "The manifest")
# ROOT      directory the manifest's `header` paths are relative to (the omega checkout; a copy in the negative controls)
# WORK_DIR  scratch directory for the generated probe
# API_MD    optional: also check that API.md states the version and has a section for every STABLE entry
#
# What it does: writes a C probe from the manifest and compiles it against the headers named there.
#   - every `fn` line becomes a compile-time type-equality check on the function itself, so a removed function, a changed argument
#     type or a changed return type is a compile error (-Werror=incompatible-pointer-types);
#   - every `struct`/`field`/`const` line becomes a printf of sizeof/alignof/offsetof/value, compared with the manifest;
#   - the version macros OMEGA_RUNTIME_API_VERSION_MAJOR/MINOR must exist (#error otherwise) and equal the manifest.
# It does not link the library and does not touch the GPU. A field added inside struct padding is not seen
# (the struct size would not change); sovereign-core's Rust checker, which reads field names, covers that later (R12).
# --bless prints the manifest to stdout with struct/field/const numbers taken from the headers (review the diff).
# Exit 0 = PASS, 1 = FAIL (each mismatch printed as "FAIL: ..."), 2 = usage.
set -u
BLESS=0
if [ "${1:-}" = "--bless" ]; then BLESS=1; shift; fi
[ $# -ge 3 ] || { echo "usage: $0 [--bless] MANIFEST ROOT WORK_DIR [API_MD]" >&2; exit 2; }
MANIFEST=$1; ROOT=$(cd "$2" && pwd) || exit 2; WORK=$3; APIMD=${4:-}
CC=${CC:-cc}
mkdir -p "$WORK" || exit 2
PROBE=$WORK/probe.c; ITEMS=$WORK/items.txt; GOT=$WORK/got.txt; WANT=$WORK/want.txt
fail=0
say_fail() { echo "FAIL: $*"; fail=1; }

# ---- 1. manifest -> probe source -------------------------------------------------------------------------
awk -v root="$ROOT" -v items="$ITEMS" '
function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
function base(t) {
  if (t == "u8") return "uint8_t"; if (t == "u16") return "uint16_t"; if (t == "u32") return "uint32_t"
  if (t == "u64") return "uint64_t"; if (t == "i8") return "int8_t"; if (t == "i16") return "int16_t"
  if (t == "i32") return "int32_t"; if (t == "i64") return "int64_t"; if (t == "f32") return "float"
  if (t == "f64") return "double"; if (t == "usize") return "size_t"; if (t == "c_char") return "char"
  if (t == "c_void" || t == "()" || t == "") return "void"
  return t
}
function ctype(t, r) {
  t = trim(t)
  if (t ~ /^\*const /) { r = ctype(substr(t, 8)); return (r ~ /\*$/) ? r " const *" : "const " r " *" }
  if (t ~ /^\*mut /) { return ctype(substr(t, 6)) " *" }
  if (t ~ /^fn\(/) { nfp++; fptd = fptd "typedef " fnsig(t, "(*FP" nfp ")") ";\n"; return "FP" nfp }
  return base(t)
}
# "fn(a, b) -> r" as a C declarator named nm
function fnsig(s, nm,   depth, i, c, cl, params, ret, n, parts, k, out) {
  s = trim(s); sub(/^fn/, "", s); s = trim(s)
  depth = 0; cl = 0
  for (i = 1; i <= length(s); i++) { c = substr(s, i, 1)
    if (c == "(") depth++
    if (c == ")") { depth--; if (depth == 0) { cl = i; break } } }
  params = substr(s, 2, cl - 2); ret = trim(substr(s, cl + 1)); sub(/^->/, "", ret)
  n = splitparams(params, parts); out = ""
  for (k = 1; k <= n; k++) out = out (k > 1 ? ", " : "") ctype(parts[k])
  if (n == 0) out = "void"
  return ctype(ret) " " nm "(" out ")"
}
function splitparams(s, parts,   i, c, depth, cur, n) {
  n = 0; cur = ""; depth = 0
  for (i = 1; i <= length(s); i++) { c = substr(s, i, 1)
    if (c == "(") depth++
    if (c == ")") depth--
    if (c == "," && depth == 0) { parts[++n] = trim(cur); cur = ""; continue }
    cur = cur c }
  if (trim(cur) != "") parts[++n] = trim(cur)
  return n
}
function item(line) { k++; print k "\t" line > items; return sprintf("#if !defined(ONLY) || ONLY == %d\n", k) }
BEGIN { k = 0; nfp = 0; fptd = "" }
{ line = $0; sub(/#.*/, "", line); line = trim(line); if (line == "") next
  nw = split(line, w, /[ \t]+/)
  if (w[1] == "header") { inc = inc "#include \"" root "/" w[2] "\"\n"; next }
  if (w[1] == "abi") next
  if (w[1] == "struct") { body = body item(line) "  printf(\"struct " w[2] " %zu %zu\\n\", sizeof(" w[2] "), _Alignof(" w[2] "));\n#endif\n"; next }
  if (w[1] == "field") { body = body item(line) "  printf(\"field " w[2] " " w[3] " %zu %zu\\n\", offsetof(" w[2] ", " w[3] "), sizeof(((" w[2] " *)0)->" w[3] "));\n#endif\n"; next }
  if (w[1] == "const") { body = body item(line) "  printf(\"const " w[2] " %lld\\n\", (long long)(" w[2] "));\n#endif\n"; next }
  if (w[1] == "fn") { rest = line; sub(/^fn[ \t]+[^ \t]+[ \t]+/, "", rest)
    decl = decl item(line) "_Static_assert(__builtin_types_compatible_p(__typeof__(" w[2] "), " fnsig(rest, "") "), \"signature of " w[2] "\");\n#endif\n"; next }
  print "manifest: unknown or malformed line: " $0 > "/dev/stderr"; bad = 1 }
END {
  print "#include <stdio.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <stdbool.h>"
  print "#include \"" root "/src/omega_runtime_api.h\""
  print "#ifndef OMEGA_RUNTIME_API_VERSION_MAJOR\n#error \"version macro missing: OMEGA_RUNTIME_API_VERSION_MAJOR\"\n#endif"
  print "#ifndef OMEGA_RUNTIME_API_VERSION_MINOR\n#error \"version macro missing: OMEGA_RUNTIME_API_VERSION_MINOR\"\n#endif"
  printf "%s", inc
  printf "%s", fptd
  printf "%s", decl
  print "int main(void) {\n  printf(\"const OMEGA_RUNTIME_API_VERSION_MAJOR %lld\\n\", (long long)(OMEGA_RUNTIME_API_VERSION_MAJOR));"
  print "  printf(\"const OMEGA_RUNTIME_API_VERSION_MINOR %lld\\n\", (long long)(OMEGA_RUNTIME_API_VERSION_MINOR));"
  printf "%s", body
  print "  return 0;\n}"
  exit bad
}' "$MANIFEST" > "$PROBE" || { echo "FAIL: manifest could not be turned into a probe"; exit 1; }

# ---- 2. compile (all at once; on failure name each failing manifest line) --------------------------------
CFL="-std=gnu11 -Wall -Wextra -Werror -Werror=implicit-function-declaration -D_GNU_SOURCE -I$ROOT/src"
if ! $CC $CFL -o "$WORK/probe" "$PROBE" 2> "$WORK/cc.err"; then
  named=0
  if ! $CC $CFL -DONLY=0 -fsyntax-only "$PROBE" 2> "$WORK/cc0.err"; then
    say_fail "version macros or headers do not compile:"; grep -m3 error "$WORK/cc0.err" | sed "s/^/      /"
  else
    while IFS="	" read -r n text; do
      if ! $CC $CFL -DONLY="$n" -fsyntax-only "$PROBE" 2> /dev/null; then say_fail "header disagrees with manifest: $text"; named=1; fi
    done < "$ITEMS"
    if [ $named = 0 ]; then say_fail "probe does not compile:"; grep -m5 'error' "$WORK/cc.err" | sed 's/^/      /'; fi
  fi
  echo "RESULT: FAIL (runtime API check, compile stage)"; exit 1
fi
"$WORK/probe" > "$GOT" || { echo "FAIL: probe crashed"; exit 1; }

# ---- 3. bless or compare ---------------------------------------------------------------------------------
if [ $BLESS = 1 ]; then
  awk 'NR == FNR { g[$1 " " $2 (($1 == "field") ? " " $3 : "")] = $0; next }
    { raw = $0; line = raw; sub(/#.*/, "", line); n = split(line, w, /[ \t]+/); cm = ""
      if (match(raw, /#.*/)) cm = " " substr(raw, RSTART)
      if (w[1] == "struct") { split(g["struct " w[2]], v, " "); print "struct " w[2] " " v[3] " " v[4] cm; next }
      if (w[1] == "field") { split(g["field " w[2] " " w[3]], v, " "); print "field " w[2] " " w[3] " " v[4] " " v[5] " " w[6] " " w[7] cm; next }
      if (w[1] == "const") { split(g["const " w[2]], v, " "); print "const " w[2] " " v[3] cm; next }
      print raw }' "$GOT" "$MANIFEST"
  exit 0
fi

# manifest facts without kind/count and without comments, in file order
awk '{ sub(/#.*/, ""); n = split($0, w, /[ \t]+/); if (w[1] == "struct") print "struct " w[2] " " w[3] " " w[4]
  else if (w[1] == "field") print "field " w[2] " " w[3] " " w[4] " " w[5]; else if (w[1] == "const") print "const " w[2] " " w[3] }' "$MANIFEST" > "$WANT"
while IFS= read -r l; do
  grep -qxF "$l" "$GOT" || { key=$(echo "$l" | awk '{ print $1 " " $2 (($1 == "field") ? " " $3 : "") }')
    have=$(grep -F "$key " "$GOT" | head -1); say_fail "manifest says \`$l\` but the header gives \`${have:-(absent)}\`"; }
done < "$WANT"

# manifest sanity: tier tag on every struct/fn line, field size matches its kind, version consts present
awk '{ raw = $0; line = raw; sub(/#.*/, "", line); n = split(line, w, /[ \t]+/)
  if (w[1] == "struct" || w[1] == "fn") { if (raw !~ /#[ \t]*(STABLE|EXPERIMENTAL)([ \t]|$)/) print "FAIL: " w[1] " " w[2] " has no STABLE/EXPERIMENTAL tag" }
  if (w[1] == "field") { ks["u8"]=1; ks["i8"]=1; ks["bool"]=1; ks["c_char"]=1; ks["u16"]=2; ks["i16"]=2; ks["u32"]=4; ks["i32"]=4; ks["f32"]=4
    ks["enum"]=4; ks["u64"]=8; ks["i64"]=8; ks["f64"]=8; ks["ptr"]=8
    if (w[6] in ks && w[5] != ks[w[6]] * w[7]) print "FAIL: field " w[2] "." w[3] " size " w[5] " does not match kind " w[6] " x " w[7] } }' "$MANIFEST" > "$WORK/sanity.txt"
if [ -s "$WORK/sanity.txt" ]; then cat "$WORK/sanity.txt"; fail=1; fi
vmaj=$(awk '$1 == "const" && $2 == "OMEGA_RUNTIME_API_VERSION_MAJOR" { print $3 }' "$MANIFEST")
vmin=$(awk '$1 == "const" && $2 == "OMEGA_RUNTIME_API_VERSION_MINOR" { print $3 }' "$MANIFEST")
[ -n "$vmaj" ] && [ -n "$vmin" ] || say_fail "manifest does not list OMEGA_RUNTIME_API_VERSION_MAJOR and _MINOR"

# API.md: states the version, has a section for every STABLE entry and mentions every EXPERIMENTAL one
if [ -n "$APIMD" ]; then
  grep -q "^Omega Runtime API version: $vmaj\.$vmin\$" "$APIMD" || say_fail "$APIMD must contain the line \"Omega Runtime API version: $vmaj.$vmin\""
  awk '{ raw = $0; line = raw; sub(/#.*/, "", line); split(line, w, /[ \t]+/)
    if ((w[1] == "fn" || w[1] == "struct") && match(raw, /#[ \t]*(STABLE|EXPERIMENTAL)/)) { t = substr(raw, RSTART); print w[2], (t ~ /STABLE/ && t !~ /EXPERIMENTAL/) ? "STABLE" : "EXPERIMENTAL" } }' "$MANIFEST" |
  while read -r name tier; do
    if [ "$tier" = STABLE ]; then grep -q "^#### \`$name\`" "$APIMD" || echo "FAIL: STABLE entry $name has no \"#### \`$name\`\" section in $APIMD"
    else grep -qF "\`$name\`" "$APIMD" || echo "FAIL: EXPERIMENTAL entry $name is not mentioned in $APIMD"; fi
  done > "$WORK/apimd.txt"
  if [ -s "$WORK/apimd.txt" ]; then cat "$WORK/apimd.txt"; fail=1; fi
  # every C signature quoted in API.md (lines inside a plain ```c fence that end in `);`) must be in a header, comments removed
  for h in $(awk '$1 == "header" { print $2 }' "$MANIFEST"); do perl -0777 -pe 's{/\*.*?\*/}{}gs' "$ROOT/$h"; done | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g' | sed 's/^/ /' > "$WORK/headers.norm"
  awk '/^```c$/ { f = 1; next } /^```/ { f = 0 } f && /\);[ \t]*$/ { print }' "$APIMD" | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//' |
  while IFS= read -r sig; do grep -qF -- " $sig" "$WORK/headers.norm" || echo "FAIL: signature quoted in $APIMD but in no header: $sig"; done > "$WORK/sigs.txt"
  if [ -s "$WORK/sigs.txt" ]; then cat "$WORK/sigs.txt"; fail=1; fi
fi

nfn=$(grep -c '^fn ' "$MANIFEST"); nst=$(grep -c '^struct ' "$MANIFEST"); nfl=$(grep -c '^field ' "$MANIFEST"); nco=$(grep -c '^const ' "$MANIFEST")
if [ $fail = 0 ]; then echo "RESULT: PASS (runtime API $vmaj.$vmin: $nfn functions, $nst structs, $nfl fields, $nco constants match the headers)"; exit 0; fi
echo "RESULT: FAIL (runtime API check)"; exit 1
