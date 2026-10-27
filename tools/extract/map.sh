#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# map.sh: file-level map of the GPU stack between Omega (aien-dev/omega-mv) and CUDA++.
#
#   tools/extract/map.sh OMEGA_DIR OMEGA_REF [CUDAPP_REF] [PHYSICS_DIR PHYSICS_REF]
#
# Prints TSV on stdout: omega_path  cudapp_path  status  omega_blob  cudapp_blob
#   status: same | differs | cudapp-only | omega-only
# Paths are 1:1 (filemap.tsv is the identity map). The set compared is the GPU part:
#   every path in filemap.tsv, shared.txt and extra-paths.txt (directories expanded),
#   every src/omega_gpu_* / src/omega_blackwell_* / src/omega_bw_* / src/omega_sm121_* file on either side,
#   every tests/, mk/, examples/ and docs/runtime/ file CUDA++ holds or Omega holds under
#   a CUDA++ name pattern (tests/gpu_*, tests/bw_*, mk/omega-gpu-*, mk/runtime-*, mk/sm121dis.mk,
#   tools/sm121dis/, evidence/C3-* .. evidence/C6-*).
# With PHYSICS_DIR/PHYSICS_REF the driver-client paths (physics-paths.txt) are compared
# against that physics checkout instead of Omega; their omega_path column reads physics:PATH.
# Blobs are git object ids, so `same` means byte-identical. Opens nothing but git.
set -euo pipefail
export LC_ALL=C

omega=${1:?omega dir}; oref=${2:?omega ref}; cref=${3:-HEAD}
physics=${4:-}; pref=${5:-}
here=$(cd "$(dirname "$0")" && pwd)
top=$(git -C "$here" rev-parse --show-toplevel)
otip=$(git -C "$omega" rev-parse --verify "$oref^{commit}")
ctip=$(git -C "$top" rev-parse --verify "$cref^{commit}")

lst() { grep -v '^#' "$1" | grep . || true; }
tree() { git -C "$1" ls-tree -r "$2" | awk -F'\t' '{split($1,a," "); print $2 "\t" a[3]}'; }

scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
tree "$omega" "$otip" | sort > "$scratch/o"
tree "$top" "$ctip" | sort > "$scratch/c"
: > "$scratch/p"
if [[ -n $physics ]]; then
    ptip=$(git -C "$physics" rev-parse --verify "$pref^{commit}")
    tree "$physics" "$ptip" | sort > "$scratch/p"
fi
lst "$here/physics-paths.txt" > "$scratch/phys-roots"

{
    cut -f1 "$here/filemap.tsv"
    lst "$here/shared.txt"
    lst "$here/extra-paths.txt"
    cat "$scratch/phys-roots"
    cut -f1 "$scratch/o" "$scratch/c" | grep -E '^src/omega_(gpu|blackwell|bw|sm121)_|^tests/(gpu|bw)_|^mk/(omega-gpu|runtime)-|^mk/sm121dis\.mk$|^tools/sm121dis/|^evidence/C[3-6]-'
    cut -f1 "$scratch/c" | grep -E '^(src|tests|mk|examples|docs/runtime|nvrm|m16|native)/|^Makefile$'
} | sort -u > "$scratch/roots"

# Expand directory roots against both trees, keep files only.
cut -f1 "$scratch/o" "$scratch/c" "$scratch/p" | sort -u > "$scratch/all"
awk 'NR==FNR{r[$0]=1; next} {p=$0; if (p in r) {print; next} n=split(p,s,"/"); d=""; for(i=1;i<n;i++){d=d (i>1?"/":"") s[i]; if (d in r) {print; next}}}' \
    "$scratch/roots" "$scratch/all" | sort -u > "$scratch/paths"

# A path is a driver-client path when it equals or sits under a physics-paths.txt root.
isphys() { awk -v p="$1" 'BEGIN{r=1} {if (p==$0 || index(p, $0 "/")==1) {r=0; exit}} END{exit r}' "$scratch/phys-roots"; }

printf 'omega_path\tcudapp_path\tstatus\tomega_blob\tcudapp_blob\n'
join -t$'\t' -a1 -e- -o 0,2.2 "$scratch/paths" "$scratch/o" > "$scratch/jo"
join -t$'\t' -a1 -e- -o 0,2.2 "$scratch/paths" "$scratch/c" > "$scratch/jc"
join -t$'\t' -a1 -e- -o 0,2.2 "$scratch/paths" "$scratch/p" > "$scratch/jp"
paste "$scratch/jo" "$scratch/jc" "$scratch/jp" | while IFS=$'\t' read -r path ob _ cb _ pb; do
    src=$path
    if [[ -n $physics ]] && isphys "$path"; then ob=$pb; src="physics:$path"; fi
    if [[ $ob == - && $cb == - ]]; then continue
    elif [[ $ob == - ]]; then st=cudapp-only
    elif [[ $cb == - ]]; then st=omega-only
    elif [[ $ob == "$cb" ]]; then st=same
    else st=differs; fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$src" "$path" "$st" "$ob" "$cb"
done
