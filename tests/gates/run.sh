#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# tests/gates/run.sh: the one entry point for the local verification gates of CUDApp. Every step runs the SAME command
# as the hosted job it mirrors, so a gate manifest under tests/gates/ names this script plus one step word and needs
# no shell expansion (ADR 0028 P6, the layout physics and omega-mv use).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
step="${1:?usage: tests/gates/run.sh <step>}"
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }

case "$step" in
  runtime-api)
    # Mirrors .github/workflows/runtime-api.yml step "Manifest, surface and negative controls" (and the
    # test-runtime-api entry of `make test`): docs/runtime/omega_runtime_api.manifest against API.md and the 22
    # manifest headers (functions, structs, fields, constants, version, public surface, INTERNAL list), then every
    # negative control of tools/runtime_api_negative.sh. The final line must be the negative script's RESULT: PASS,
    # and the real tree's own RESULT: PASS line must be there too, so an empty or truncated run cannot pass.
    log=$(mktemp)
    trap 'rm -f "$log"' EXIT
    rc=0
    make test-runtime-api > "$log" 2>&1 || rc=$?
    cat "$log"
    [ "$rc" -eq 0 ] || exit "$rc"
    real=$(grep -c '^RESULT: PASS (runtime API ' "$log" || true)
    last=$(tail -n 1 "$log")
    case $last in
      "RESULT: PASS (every negative control was caught)") ;;
      *) echo "run.sh runtime-api: make exited 0 without the negative controls' final RESULT: PASS line: $last" >&2; exit 1 ;;
    esac
    [ "$real" -eq 1 ] || { echo "run.sh runtime-api: expected one real-tree RESULT: PASS line, saw $real" >&2; exit 1; }
    obs runtime_api_negative_controls "$(grep -c '^NEGATIVE CONTROL .*: FAILED as it must' "$log")"
    obs runtime_api_not_caught "$(grep -c 'NOT CAUGHT\|BROKEN' "$log" || true)"
    ;;
  *)
    echo "run.sh: unknown step '$step'" >&2
    exit 2
    ;;
esac
