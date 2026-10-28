#!/bin/sh
# Generate architecture-independent checked C from the seeded Omega units.
# The additive bridge-seeded frontend checks exactly the same source bytes with
# omega-mv@e5b32567 before emitting C. No C oracle replaces an Omega unit.
# Generation runs on AArch64; the committed GNU C11/POSIX output builds and
# runs natively on x86-64 without Omega binaries, emulation or GPU access.
# OMEGA_OMC: pinned seeded compiler; OMEGA_BRIDGE: converter extension checkout.
# --check compares every generated TU, declaration and source-to-C identity.
set -eu
OMC_PIN=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
CONVERTER_PIN=e1d1de0a83e27b34792041cd9e30a5c353c57fa1dc1aff05361bc950278d418b
FRONTEND_PIN=23ff7a2356bb5d980060a4685a27ac46587013069b7d24b37c2873d7ee659b37
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNITS="${OMEGA_UNITS:-om/encoder/bw_enc_sm121.om om/cubin/cubin_elf.om om/encoder/ampere_enc_sm86.om om/encoder/amp_runtime.om}"
DEST="${OMEGA_UNITS_DEST:-$ROOT/src}"
OUTS="cudapp_native.h cudapp_om_units.ident"
[ -n "${OMEGA_OMC:-}" ] && [ -n "${OMEGA_BRIDGE:-}" ] || { echo "NOT_RUN: set OMEGA_OMC and OMEGA_BRIDGE"; exit 3; }
[ "$(sha256sum < "$OMEGA_OMC" | cut -c1-64)" = "$OMC_PIN" ] || { echo "NOT_RUN: omc is not the pinned compiler"; exit 3; }
[ "$(sha256sum < "$OMEGA_BRIDGE/src/toc.om" | cut -c1-64)" = "$CONVERTER_PIN" ] || { echo "NOT_RUN: toc.om is not the pinned converter"; exit 3; }
[ "$(sha256sum < "$OMEGA_BRIDGE/tools/bridge-seeded" | cut -c1-64)" = "$FRONTEND_PIN" ] || { echo "NOT_RUN: bridge-seeded is not the pinned frontend"; exit 3; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cat > "$T/cudapp_native.h" <<'HEADER'
/* Generated checked portable Omega declarations. Regenerate with om/units.sh.
 * Status: 0 done, 1..14 trap, 32 argument refused, 33 reentry.
 * Only a successful call writes result. Views use exact declared lengths.
 * Implementation is Omega lowered by omega-bridge, not the native AArch64 ABI. */
#ifndef CUDAPP_PORTABLE_OMEGA_H
#define CUDAPP_PORTABLE_OMEGA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define OMEGA_OK 0u
#define OMEGA_BAD_ARG 32u
#define OMEGA_REENTRY 33u
HEADER
: > "$T/cudapp_om_units.ident"
for u in $UNITS; do
  case "$u" in /*) src=$u ;; *) src=$ROOT/$u ;; esac
  name=$(basename "$u" .om | tr -c 'A-Za-z0-9_\n' '_')
  cp "$src" "$T/$name.om"
  src="$T/$name.om"
  file=cudapp_om_$name.c
  OMEGA_SEEDED_OMC="$OMEGA_OMC" sh "$OMEGA_BRIDGE/tools/bridge-seeded" c "$src" > "$T/$file"
  OMEGA_SEEDED_OMC="$OMEGA_OMC" sh "$OMEGA_BRIDGE/tools/bridge-seeded" h "$src" >> "$T/cudapp_native.h"
  identity="omega-ident 2 unit=$name abi=portable-seeded-1 src=$(sha256sum < "$src" | cut -c1-64) c=$(sha256sum < "$T/$file" | cut -c1-64) omc=$OMC_PIN converter=$CONVERTER_PIN frontend=$FRONTEND_PIN path=$u"
  printf '%s\n' "$identity" > "$T/cudapp_om_$name.ident"
  printf '%s\n' "$identity" >> "$T/cudapp_om_units.ident"
  OUTS="$OUTS $file cudapp_om_$name.ident"
done
cat >> "$T/cudapp_native.h" <<'HEADER'
#ifdef __cplusplus
}
#endif
#endif
HEADER
if [ "${1:-}" = --check ]; then
  bad=0
  for f in $OUTS; do cmp -s "$T/$f" "$DEST/$f" || { echo "FAIL $DEST/$f differs from pinned tools"; bad=1; }; done
  [ "$bad" = 0 ] || exit 1
  echo "PASS portable Omega units regenerate byte for byte"; exit 0
fi
for f in $OUTS; do cp "$T/$f" "$DEST/$f"; done
echo "wrote checked portable Omega units to $DEST"
