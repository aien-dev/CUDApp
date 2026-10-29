#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# `make doctor` part 2: the target table from backend/targets.om (seeded Omega), one row per GPU
# family with its honest status. Only the GB10 (sm_121) is SUPPORTED; every other GPU row is PLANNED
# with hardware NOT_RUN. Compiles with the pinned omc and opens no device.
# usage: targets-doctor.sh OMC OUTDIR   (OMC may be empty: prints how to get it and exits 0)
set -u
LC_ALL=C
export LC_ALL
omc_pin=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd) || exit 74
omc=${1:-}; out=${2:?usage: targets-doctor.sh OMC OUTDIR}
echo
echo "GPU targets (backend/targets.om):"
if [ -z "$omc" ] || [ ! -x "$omc" ]; then
  echo "  NOT_RUN: the target table is seeded Omega and needs the pinned omc."
  echo "  Fix: make doctor OMC=/absolute/path/to/omega-mv/seed/out/omc  (omega-mv 369dbc96, sha256 $omc_pin)"
  exit 0
fi
d=$(sha256sum < "$omc"); d=${d%% *}
if [ "$d" != "$omc_pin" ]; then
  echo "  FAIL: $omc has sha256 $d, not the pin $omc_pin."
  echo "  Fix: use seed/out/omc from aien-dev/omega-mv at 369dbc96."
  exit 1
fi
mkdir -p "$out" || exit 74
if ! "$omc" < "$root/backend/targets.om" > "$out/targets" 2> "$out/targets.compile.err" || [ -s "$out/targets.compile.err" ]; then
  cat "$out/targets.compile.err"; echo "  FAIL: backend/targets.om did not compile"; exit 1
fi
chmod 700 "$out/targets" || exit 74
"$out/targets" doctor
