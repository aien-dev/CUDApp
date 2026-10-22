#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
step="${1:?usage: tests/gates/run.sh <step>}"
obs() { printf 'AIEN-OBS %s %s\n' "$1" "$2"; }
case "$step" in

  zero-disk-secrets)
    found=false
    if find . -type f \( -name ".env" -o -name ".env.*" \) \
         ! -name "*.example" ! -name "*.sample" ! -path "./target/*" | head -1 | grep -q .; then
      found=true
      find . -type f \( -name ".env" -o -name ".env.*" \) \
        ! -name "*.example" ! -name "*.sample" ! -path "./target/*" | head -5 >&2
    fi
    obs env_files_found "$found"
    [ "$found" = false ]
    ;;

  host-tests)   make -j"$(nproc)" test ;;
  zero-cuda)    make zero-cuda-gate ;;
  m16-requal)   sh tests/run_m16_requalification.sh ;;
  nvrm-lifecycle) sh tests/run_nvrm_lifecycle_gates.sh ;;
  *) echo "run.sh: unknown step $step" >&2; exit 2 ;;
esac
