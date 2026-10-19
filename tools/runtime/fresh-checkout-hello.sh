#!/usr/bin/env bash
# fresh-checkout-hello.sh <branch> [log-file]
# Proves the README's "Run your first kernel" works from nothing: clones <branch> from GitHub into an
# empty temporary folder, takes the commands from that section of the README (the first bash block), runs
# only those, and records the output with the commit it ran. Exit code is that of the last README command
# (the hello program: 0 PASS, 1 FAIL, 2 GPU not usable). This is a correctness check, not a timing run.
# It uses the GPU, so run it under the shared lock:  quiet-measure flock /tmp/aien-gb10.lock <this script> <branch> <log>
set -uo pipefail
branch=${1:?usage: fresh-checkout-hello.sh <branch> [log-file]}
log=${2:-/dev/stdout}
url=${OMEGA_URL:-https://github.com/aien-dev/omega-mv}
work=$(mktemp -d "${TMPDIR:-/tmp}/omega-fresh.XXXXXX")
cleanup() { rm -rf "$work"; }
trap cleanup EXIT
{
	echo "# fresh-checkout-hello: branch=$branch url=$url"
	echo "# date: $(date -u +%FT%TZ)  host: $(uname -srm)  work: $work"
	# the README commands come from the pushed branch itself
	git clone -q --branch "$branch" "$url" "$work/src.readme" 2>&1 || { echo "FAIL: cannot clone $branch"; exit 3; }
	cmds=$(awk '/^## Run your first kernel/{s=1;next} s&&/^```bash/{b=1;next} s&&b&&/^```/{exit} s&&b{print}' "$work/src.readme/README.md")
	[ -n "$cmds" ] || { echo "FAIL: no bash block under 'Run your first kernel' in README.md"; exit 3; }
	echo "# README commands:"; echo "$cmds" | sed 's/^/#   /'
	rm -rf "$work/src.readme"
	# run them in a clean folder; the clone line is pointed at the branch under test
	mkdir "$work/run" && cd "$work/run" || exit 3
	cmds=$(echo "$cmds" | sed -E "s#^git clone #git clone --branch $branch #; s#https://github.com/aien-dev/omega-mv#$url#")
	rc=0
	while IFS= read -r line; do
		[ -n "$line" ] || continue
		echo "\$ $line"
		bash -c "$line" 2>&1 || { rc=$?; echo "# command exited $rc"; break; }
		case $line in "cd "*) cd "${line#cd }" || exit 3; echo "# commit: $(git rev-parse HEAD)";; esac
	done <<<"$cmds"
	echo "# fresh-checkout-hello exit code: $rc"
	exit $rc
} > >(tee "$log") 2>&1
