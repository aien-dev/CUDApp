#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Host-only regressions for drift.sh. Uses local git repositories, no remote or device.
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
export GIT_AUTHOR_NAME=drift-test GIT_AUTHOR_EMAIL=drift@test.invalid
export GIT_COMMITTER_NAME=$GIT_AUTHOR_NAME GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL
fail=0

fixture() {
    local name=$1
    c=$scratch/$name/cudapp; o=$scratch/$name/omega
    mkdir -p "$c/tools/extract" "$c/src" "$o/src"
    git init -q "$c"; git init -q "$o"
    cp "$here/drift.sh" "$here/map.sh" "$c/tools/extract/"
    for file in filemap.tsv shared.txt extra-paths.txt physics-paths.txt; do
        printf '# fixture\n' > "$c/tools/extract/$file"
    done
}
commit_pair() {
    git -C "$c" add .; git -C "$c" commit -qm fixture
    git -C "$o" add .; git -C "$o" commit -qm fixture
    base=$scratch/baseline
    git -C "$o" ls-tree -r HEAD src | awk '{print $4 "\t" $3}' > "$base"
}
run() {
    local name=$1 want_rc=$2 want_summary=$3; shift 3
    local rc=0 out=$scratch/output
    "$@" > "$out" 2>&1 || rc=$?
    echo "=== $name (exit $rc, expected $want_rc) ==="
    cat "$out"
    if [[ $rc -ne $want_rc ]] || ! grep -qxF "$want_summary" "$out"; then
        echo "FAIL: $name"; fail=1
    elif [[ $want_rc -eq 2 ]] && grep -qE '^(DIVERGENCE:|DRIFT-GATE: PASS|SELF-TEST: PASS)' "$out"; then
        echo "FAIL: $name printed a success summary after a setup error"; fail=1
    else echo "PASS: $name"; fi
}

# Each case has one shared path, with literal expected counts. Header-like words
# retain the existing diagnostic heuristic; they do not bypass the blob gate.
for kind in code header mixed identical large binary; do
    fixture "$kind"
    case $kind in
        code)
            printf 'int value = 1;\n' > "$o/src/omega_gpu_case.c"
            printf 'int value = 2;\n' > "$c/src/omega_gpu_case.c" ;;
        header)
            printf '// SPDX-License-Identifier: MIT\nint value = 1;\n' > "$o/src/omega_gpu_case.c"
            printf '// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\nint value = 1;\n' > "$c/src/omega_gpu_case.c" ;;
        mixed)
            printf '// SPDX-License-Identifier: MIT\nint value = 1;\n' > "$o/src/omega_gpu_case.c"
            printf '// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\nint value = 2;\n' > "$c/src/omega_gpu_case.c" ;;
        identical)
            printf 'int value = 1;\n' > "$o/src/omega_gpu_case.c"
            cp "$o/src/omega_gpu_case.c" "$c/src/omega_gpu_case.c" ;;
        large)
            awk 'BEGIN {for(i=0;i<20000;i++) print "int value_" i " = 1;"}' > "$o/src/omega_gpu_case.c"
            awk 'BEGIN {for(i=0;i<20000;i++) print "int value_" i " = 2;"}' > "$c/src/omega_gpu_case.c" ;;
        binary)
            printf 'value\0one\n' > "$o/src/omega_gpu_case.c"
            printf 'value\0two\n' > "$c/src/omega_gpu_case.c" ;;
    esac
    commit_pair
    case $kind in
        identical) summary='DIVERGENCE: 1 shared, 1 same, 0 differ (0 licence-header only, 0 code); 0 cudapp-only, 0 omega-only' ;;
        header) summary='DIVERGENCE: 1 shared, 0 same, 1 differ (1 licence-header only, 0 code); 0 cudapp-only, 0 omega-only' ;;
        *) summary='DIVERGENCE: 1 shared, 0 same, 1 differ (0 licence-header only, 1 code); 0 cudapp-only, 0 omega-only' ;;
    esac
    run "$kind classification at baseline" 0 "$summary" bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"
    # A changed baseline must still reject a header-only blob difference.
    printf 'src/omega_gpu_case.c\t0000000000000000000000000000000000000000\n' > "$base"
    rc=1; [[ $kind != identical ]] || rc=0
    run "$kind blob gate" "$rc" "$summary" bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"
done

fixture controls
printf 'int value = 1;\n' > "$o/src/omega_gpu_case.c"
printf 'int value = 2;\n' > "$c/src/omega_gpu_case.c"
printf 'int same = 1;\n' > "$o/src/omega_gpu_same.c"
cp "$o/src/omega_gpu_same.c" "$c/src/omega_gpu_same.c"
commit_pair
run 'built-in negative controls' 0 'SELF-TEST: PASS (both controls red, real check green)' \
    bash "$c/tools/extract/drift.sh" --self-test "$o" HEAD "$base"
run 'invalid ref in direct mode' 2 'DRIFT-GATE: NOT_RUN (map failed)' \
    bash "$c/tools/extract/drift.sh" "$o" invalid-ref "$base"
run 'invalid ref in self-test mode' 2 'DRIFT-GATE: NOT_RUN (map failed)' \
    bash "$c/tools/extract/drift.sh" --self-test "$o" invalid-ref "$base"

# A directory can satisfy -s. It is not a readable baseline file; awk failure
# must never be interpreted as no drift. Reject unreadable files as well.
mkdir "$scratch/baseline-directory"
printf 'not a baseline\n' > "$scratch/baseline-directory/entry"
run 'baseline is a directory' 2 "DRIFT-GATE: NOT_RUN (baseline $scratch/baseline-directory is not a readable nonempty file)" \
    bash "$c/tools/extract/drift.sh" "$o" HEAD "$scratch/baseline-directory"
chmod 000 "$base"
if cat "$base" > /dev/null 2>&1; then
    # Root/CAP_DAC_OVERRIDE may still read mode 000. That is a usable baseline,
    # not an unreadable-file fixture. Check real readability independently.
    run 'mode-000 baseline remains readable' 0 'DIVERGENCE: 2 shared, 1 same, 1 differ (0 licence-header only, 1 code); 0 cudapp-only, 0 omega-only' \
        bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"
else
    run 'baseline is unreadable' 2 "DRIFT-GATE: NOT_RUN (baseline $base is not a readable nonempty file)" \
        bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"
fi
chmod 600 "$base"

# Also propagate a read failure after the baseline's initial validation.
mkdir "$scratch/awk-bin"
cat > "$scratch/awk-bin/awk" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    if [[ $arg == "$DRIFT_TEST_BASELINE" ]]; then echo 'injected awk read error' >&2; exit 2; fi
done
exec "$DRIFT_TEST_REAL_AWK" "$@"
EOF
chmod +x "$scratch/awk-bin/awk"
run 'baseline comparison read error' 2 'DRIFT-GATE: NOT_RUN (baseline comparison failed)' \
    env PATH="$scratch/awk-bin:$PATH" DRIFT_TEST_BASELINE="$base" DRIFT_TEST_REAL_AWK="$(command -v awk)" \
    bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"

# ls-tree can still map a missing blob. Reading its content must fail closed,
# not masquerade as a licence-header-only difference via process substitution.
for repo in "$o" "$c"; do
    blob=$(git -C "$repo" rev-parse HEAD:src/omega_gpu_case.c)
    object=$repo/.git/objects/${blob:0:2}/${blob:2}
    mv "$object" "$scratch/saved-blob"
    run 'missing blob' 2 'DRIFT-GATE: NOT_RUN (summary failed)' \
        bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"
    mv "$scratch/saved-blob" "$object"
done

# Deterministic diff I/O-error injection at the external command boundary.
# All git mapping and blob extraction remain real.
mkdir "$scratch/bin"
printf '#!/usr/bin/env bash\necho "injected diff error" >&2\nexit 2\n' > "$scratch/bin/diff"
chmod +x "$scratch/bin/diff"
run 'diff error' 2 'DRIFT-GATE: NOT_RUN (summary failed)' \
    env PATH="$scratch/bin:$PATH" bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"
run 'diff error in self-test mode' 2 'DRIFT-GATE: NOT_RUN (summary failed)' \
    env PATH="$scratch/bin:$PATH" bash "$c/tools/extract/drift.sh" --self-test "$o" HEAD "$base"

# A map row labelled differs must not silently accept an empty diff as a header.
# Identical real blobs are covered above: map.sh labels them same and skips diff.
printf '#!/usr/bin/env bash\nexit 0\n' > "$scratch/bin/diff"
run 'diff reports identical for distinct blobs' 2 'DRIFT-GATE: NOT_RUN (summary failed)' \
    env PATH="$scratch/bin:$PATH" bash "$c/tools/extract/drift.sh" "$o" HEAD "$base"

# A setup error in either negative control must retain status 2 even when the
# later real check would succeed. Do not swallow it with `check ... || true`.
cat > "$scratch/bin/diff" <<'EOF'
#!/usr/bin/env bash
set -eu
n=0
if [[ -f $DRIFT_TEST_DIFF_COUNT ]]; then read -r n < "$DRIFT_TEST_DIFF_COUNT"; fi
n=$((n + 1)); echo "$n" > "$DRIFT_TEST_DIFF_COUNT"
if [[ $n -eq $DRIFT_TEST_DIFF_FAIL_AT ]]; then echo 'injected one-time diff error' >&2; exit 2; fi
exec "$DRIFT_TEST_REAL_DIFF" "$@"
EOF
for control in 1 2; do
    run "one-time diff error in control $control" 2 'DRIFT-GATE: NOT_RUN (summary failed)' \
        env PATH="$scratch/bin:$PATH" DRIFT_TEST_DIFF_COUNT="$scratch/count-$control" \
        DRIFT_TEST_DIFF_FAIL_AT="$control" DRIFT_TEST_REAL_DIFF="$(command -v diff)" \
        bash "$c/tools/extract/drift.sh" --self-test "$o" HEAD "$base"
done

if [[ $fail -eq 0 ]]; then echo 'DRIFT REGRESSION TESTS PASS'; else echo 'DRIFT REGRESSION TESTS FAIL'; fi
exit "$fail"
