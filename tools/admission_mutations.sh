#!/bin/sh
# SPDX-License-Identifier: MIT
# Behavioral mutants of the actual implementation, never production switches.
set -eu
set -f
cc=$1
flags=$2
out=$3
mkdir -p "$out/mutants"
for mutant in budget ownership pid-start boot seal missing-memory generation adapter-bypass; do
    dir="$out/mutants/$mutant"
    mkdir -p "$dir"
    cp src/omega_gpu_admission.c "$dir/omega_gpu_admission.c"
    mode=
    case "$mutant" in
        budget)
            sed -i 's/bytes > a->budget - total || total > avail/0 || total > avail/' "$dir/omega_gpu_admission.c"
            mode=--contenders ;;
        ownership)
            sed -i 's/ || !owned(a, r)//g' "$dir/omega_gpu_admission.c"
            mode=--ownership ;;
        pid-start)
            sed -i 's/if (start != r->start) dead = 1;/(void)start;/' "$dir/omega_gpu_admission.c"
            mode=--ownership ;;
        boot)
            sed -i 's/if (strcmp(r->boot, a->boot)) dead = 1;/if (0) dead = 1;/' "$dir/omega_gpu_admission.c"
            mode=--ownership ;;
        seal)
            sed -i 's/if (a->sealed) return -EPERM;/if (0) return -EPERM;/;s/(to == ACQUIRED \&\& a->sealed)/(0)/' "$dir/omega_gpu_admission.c" ;;
        missing-memory)
            sed -i '/static int available(/,/^}/s/if (fd < 0) return neg();/if (fd < 0) { *out = UINT64_MAX; return 0; }/' "$dir/omega_gpu_admission.c" ;;
        generation)
            sed -i 's/ || r->generation != t.generation//g' "$dir/omega_gpu_admission.c" ;;
        adapter-bypass)
            sed -i '/rc = reserve_locked(a, bytes, usage, \&t);/{n;s/if (rc) return leave(a, rc, 0);/if (rc) { (void)p->dma_alloc(p->ctx, dev, d, out, cpu, gpu, bus); return leave(a, rc, 0); }/;}' "$dir/omega_gpu_admission.c"
            mode=--adapter ;;
    esac
    if cmp -s src/omega_gpu_admission.c "$dir/omega_gpu_admission.c"; then echo "FAIL no mutation: $mutant"; exit 1; fi
    # Intentional splitting of the compiler flags supplied by make; globbing off.
    $cc $flags -Itests -o "$dir/test" tests/test_omega_gpu_admission.c "$dir/omega_gpu_admission.c" \
        src/omega_gpu_platform.c src/omega_gpu_platform_linux.c tests/fake_m16_native.c -Wl,--wrap=pwrite -lpthread > "$dir/build.log" 2>&1
    rc=0
    timeout 45 "$dir/test" ${mode:+"$mode"} > "$dir/run.log" 2>&1 || rc=$?
    printf 'exit_code=%s\n' "$rc" >> "$dir/run.log"
    # A timeout, compile error, or crash is not an assertion kill.
    if [ "$rc" -ne 1 ] || ! rg -q '^FAIL ' "$dir/run.log"; then
        echo "FAIL mutant $mutant did not fail an assertion (exit $rc)"; exit 1
    fi
    echo "PASS mutant $mutant rejected by behavioral assertion"
done
