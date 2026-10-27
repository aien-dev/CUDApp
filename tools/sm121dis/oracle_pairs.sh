#!/bin/sh
# oracle_pairs.sh <nvdisasm -hex output>: one line per instruction, "<addr> <low 64 bits> <high 64 bits> <text> ;",
# text with spacing collapsed but the space before the semicolon kept (nvdisasm leaves it out on padding words).
# The C6 oracle side (docs/OMEGA.md): nvdisasm decodes, nothing it prints enters a kernel.
awk '
/^[[:space:]]*\/\*[0-9a-f]+\*\// {
    line = $0
    addr = line; sub(/^[[:space:]]*\/\*/, "", addr); sub(/\*\/.*/, "", addr)
    txt = line; sub(/^[[:space:]]*\/\*[0-9a-f]+\*\/[[:space:]]*/, "", txt); sub(/;.*$/, ";", txt)
    lo = line; sub(/.*\/\* 0x/, "", lo); sub(/ \*\/.*/, "", lo)
    gsub(/[[:space:]]+/, " ", txt)
    pend = 1; next
}
pend && /\/\* 0x[0-9a-f]+ \*\// { hi = $0; sub(/.*\/\* 0x/, "", hi); sub(/ \*\/.*/, "", hi); print addr, lo, hi, txt; pend = 0 }
' "$1"
