#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Disclosed test orchestration for the release-candidate receipt; not product runtime.
# Mutation check for CUDApp RC: each mutant edits one behavior of the repaired
# native/nvrm/context.om (blob 5280fea0) and must be killed by the fixture's own
# assertion (after arm exit 1 + NATIVE_TEARDOWN_HOST_FAIL), never by compile error.
# usage: mutants.sh <cudapp-worktree> <pinned-omc> <outdir>
set -u
src=$1; omc=$2; out=$3; mkdir -p "$out"
[ "$(git -C "$src" rev-parse HEAD:native/nvrm/context.om | cut -c1-8)" = 5280fea0 ] || { echo REFUSED: unexpected context blob; exit 69; }
run() { # name awk-program
  name=$1; prog=$2; w="$out/$name"; rm -rf "$w"; mkdir -p "$w"
  git -C "$src" worktree add -q --detach "$w/tree" HEAD || exit 74
  awk "$prog" "$src/native/nvrm/context.om" > "$w/context.om" || exit 74
  cmp -s "$w/context.om" "$src/native/nvrm/context.om" && { echo "$name INVALID (no change)"; return; }
  cp "$w/context.om" "$w/tree/native/nvrm/context.om"
  TMPDIR="$w" sh "$w/tree/tests/speculation/run-teardown-host.sh" "$omc" > "$w/run.log" 2>&1; rc=$?
  d=$(ls -d "$w"/cudapp-teardown-host.* | head -1)
  ae=$(cat "$d/after.exit" 2>/dev/null || echo none)
  if [ "$rc" -eq 1 ] && [ "$ae" = 1 ] && grep -q '^NATIVE_TEARDOWN_HOST_FAIL$' "$d/after.stderr"; then v=KILLED
  elif [ "$rc" -eq 0 ]; then v=SURVIVED
  else v="INVALID(rc=$rc after.exit=$ae)"; fi
  echo "$name $v diff=$(diff "$src/native/nvrm/context.om" "$w/context.om" | grep -c '^[<>]') sha=$(sha256sum "$w/context.om" | cut -c1-16)"
  git -C "$src" worktree remove --force "$w/tree"
}
run M1-unpub-not-retained  'NR==508{next}{print}'
run M2-unpub-dropped-on-refusal 'NR==258{print;print "put nvunpub 0 0";next}{print}'
run M3-bar-dropped-on-refusal 'NR==267{print;print "put nvbar 0 0";next}{print}'
run M4-retry-skipped       'NR==156{print "ret true";next}{print}'
run M5-retry-redoes-close-and-driver 'NR==285{print;print "put nvstate 0 1";next}{print}'
run M6-fault-cleared       'NR==300{print "put nvfault 0 0";print;next}{print}'
run M7-faulted-map-allowed 'NR>=448&&NR<=451{next}{print}'
run M8-descriptor-not-retired 'NR==233{next}{print}'
run M7b-guard-state-only   'NR==448{print "if ne get nvstate 0 1";next}{print}'
run M7c-guard-fault-only   'NR==448{print "if ne get nvfault 0 0";next}{print}'
