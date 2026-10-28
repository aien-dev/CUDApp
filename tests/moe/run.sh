#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Host only. No GPU library or device node is opened.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
: "${OMEGA_OMC:?Set the om/units.sh pinned compiler}"
: "${OMEGA_BRIDGE:?Set the om/units.sh pinned bridge checkout}"
: "${QUANT_ROOT:?Set canonical QuantSchema checkout}"
: "${CACHE_ROOT:?Set canonical KernelCache checkout}"
: "${COMPOSE_ROOT:?Set canonical KernelCompose checkout}"
: "${PIPELINE_ROOT:?Set canonical PipelineSelect checkout}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT HUP INT TERM
OMEGA_UNITS="om/moe/moe_plan.om om/moe/moe_group.om om/moe/moe_ids32.om om/moe/moe_maps32.om om/moe/moe_segs32.om om/moe/moe_windows.om tests/moe/moe_verify.om" OMEGA_UNITS_DEST="$T" sh "$ROOT/om/units.sh"
${CC:-cc} -std=c11 -O2 -Wall -Wextra -Werror -ffunction-sections -fdata-sections -I"$ROOT/src" -c "$ROOT/tests/moe/incumbent.c" -o "$T/incumbent.o"
${CC:-cc} -std=c11 -O2 -Wall -Wextra -Werror -I"$T" "$ROOT/tests/moe/caller.c" "$T/cudapp_om_units.c" "$T/incumbent.o" -Wl,--gc-sections -o "$T/moe-test"
"$T/moe-test"
cat "$QUANT_ROOT/om/quant/quant.om" "$COMPOSE_ROOT/om/components/realize.om" "$CACHE_ROOT/om/cache/digest.om" "$CACHE_ROOT/om/cache/residency.om" "$PIPELINE_ROOT/om/pipeline/select.om" "$PIPELINE_ROOT/om/pipeline/graph.om" "$ROOT/om/moe/moe_group.om" "$ROOT/om/moe/moe_admit.om" "$PIPELINE_ROOT/tests/pipeline/fixture.om" "$ROOT/tests/moe/admit-caller.om" > "$T/admit.om"
"$OMEGA_OMC" < "$T/admit.om" > "$T/admit"; chmod +x "$T/admit"
"$T/admit"
echo "MOE-ADMIT HOST PASS: actual selector/cache, metadata pin/acquire/retire/unpin, missing/stale/wrong-family refusals"
sha256sum "$QUANT_ROOT/om/quant/quant.om" "$COMPOSE_ROOT/om/components/realize.om" "$CACHE_ROOT/om/cache/digest.om" "$CACHE_ROOT/om/cache/residency.om" "$PIPELINE_ROOT/om/pipeline/select.om" "$PIPELINE_ROOT/om/pipeline/graph.om" "$ROOT/om/moe/moe_admit.om" "$T/admit"
sha256sum "$OMEGA_OMC" "$ROOT"/om/moe/*.om "$ROOT/tests/moe/moe_verify.om" "$T/moe-test"
