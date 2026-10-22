#!/bin/sh
# One mutation runner (POSIX sh, no Python). Deep module, small interface.
#
# A MUTATION is one record:
#   { name, file(s), edit, test command, what the check guards, equivalence }
# The runner copies the source tree to a scratch directory (the source tree is
# never edited), applies the edit there, checks that the files really changed
# (otherwise ERROR "edit did not apply"), runs the optional build command
# (failure => ERROR), then runs the test command inside the scratch tree:
#   exit nonzero => KILLED      exit 0 => SURVIVED
# EQUIVALENCE is a data column: class none|redundant|gap plus rationale text.
# A SURVIVED mutant with class redundant or gap is reported SURVIVED_REDUNDANT
# or SURVIVED_GAP and the JSON note is "<class>: <rationale>" (the same note
# tools/c4_requal_mutants.sh writes today). Class none (or no class) leaves it
# SURVIVED, which fails the sweep.
#
# EDIT KINDS
#   sed         files comma-separated, expressions separated by @@ (one per
#               file, in order), as tools/c4_requal_mutants.sh does today.
#   marker-sed  one file, one sed expression applied only to lines containing
#               MUT:<name>, as tensor/autodiff/m20 mutation sweeps do today.
#   replace     edit text "R:<from>@>@<to>[@@<from>@>@<to>...]": <from> must be
#               found exactly once in its file (else ERROR), @NL@ in <to> is a
#               newline, as numeric_oracle_mutations.sh does today.
#
# READERS (two; the older sweeps become adapters over them)
#   table   rows  id~files~edit~guards[~class~rationale]    (kinds sed, replace)
#           Legacy rows are accepted: 4 columns (no equivalence), or 5 columns
#           whose last column is prose starting "redundant:" or "gap:".
#           Lines that are empty or start with # are skipped.
#   marker  rows  name|file|sed-expr   (kind marker-sed; the expression may
#           contain |). Marker rows have no equivalence column: a survivor is
#           always a failure.
#
# USAGE
#   tools/mutation_runner.sh -k table|marker -m ROWS|- -t TESTCMD
#        [-d SRCDIR] [-b BUILDCMD] [-e TEXT] [-c "PATH ..."] [-o out.json] [-f PREFIX] [-B]
#   -k reader kind. -m rows file, or - for stdin. -t test command (sh -c, run
#   in the scratch tree). -d source tree (default .). -b build command run in
#   the scratch tree before the test. -e TEXT optional: when the build step
#   fails and its log is empty (make killed or never started) the row note
#   is "build failed: TEXT" instead of "build failed: <first error line>";
#   without -e nothing changes. -c paths (relative to SRCDIR) to copy
#   (default: whole tree). -o JSON receipt path. -f line prefix that marks a
#   failing case in test output, used only for the KILLED note. -B run build
#   and test on the unmutated tree first; if that fails the sweep FAILS.
#   Env ONLY="id id" restricts the sweep to those ids.
#
# OUTPUT
#   stderr: one line per mutant "<id> <STATUS>  <guards> (<note>)", STATUS is
#           KILLED, SURVIVED, SURVIVED_REDUNDANT, SURVIVED_GAP or ERROR.
#   stdout: one summary line.
#   JSON (-o), exactly the c4 receipt layout, same whitespace, key order and
#   escaping (note has double quotes and backslashes removed with tr -d):
#     {"mutants":[{"id","file","checks","status","note"}, ...]}
#   Exit 0 only if at least one mutant ran and every mutant is KILLED or
#   justified-survived; 2 on bad usage.
set -u

kind=; rows=; test_cmd=; src=.; build_cmd=; copy=.; out=; fprefix=; baseline=0; empty_note=
while getopts k:m:t:d:b:e:c:o:f:B opt; do
    case $opt in
        k) kind=$OPTARG;; m) rows=$OPTARG;; t) test_cmd=$OPTARG;; d) src=$OPTARG;;
        b) build_cmd=$OPTARG;; e) empty_note=$OPTARG;; c) copy=$OPTARG;; o) out=$OPTARG;; f) fprefix=$OPTARG;;
        B) baseline=1;; *) kind=;;
    esac
done
case $kind in table|marker) ;; *) kind=;; esac
if [ -z "$kind" ] || [ -z "$rows" ] || [ -z "$test_cmd" ]; then
    echo "usage: mutation_runner.sh -k table|marker -m ROWS|- -t TESTCMD [-d SRCDIR] [-b BUILDCMD] [-e TEXT] [-c PATHS] [-o out.json] [-f PREFIX] [-B]" >&2
    exit 2
fi
src=$(cd "$src" 2>/dev/null && pwd) || { echo "mutation_runner: no such source dir" >&2; exit 2; }
work=$(mktemp -d "${TMPDIR:-/tmp}/mutrun.XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT INT TERM
if [ "$rows" = - ]; then cat > "$work/rows"; rows=$work/rows; fi
[ -r "$rows" ] || { echo "mutation_runner: cannot read rows" >&2; exit 2; }
[ -n "$out" ] || out=$work/out.json

# Fresh scratch copy of the tree in $1.
copy_tree() {
    mkdir -p "$1"
    for p in $copy; do
        if [ "$p" = . ]; then cp -R "$src/." "$1/"
        else
            mkdir -p "$1/$(dirname "$p")"
            cp -R "$src/$p" "$1/$p" 2>/dev/null
        fi
    done
}

if [ $baseline -eq 1 ]; then
    copy_tree "$work/base"
    if { [ -z "$build_cmd" ] || (cd "$work/base" && sh -c "$build_cmd") </dev/null >"$work/base.log" 2>&1; } \
        && (cd "$work/base" && sh -c "$test_cmd") </dev/null >>"$work/base.log" 2>&1; then :
    else
        head -5 "$work/base.log" >&2
        echo "mutation runner: FAIL (unmutated baseline does not pass)"
        exit 1
    fi
    rm -rf "$work/base"
fi

# Edit step. Sets $note on failure and returns nonzero.
apply_edit() { # dir files ekind edit id
    a_dir=$1; a_files=$(echo "$2" | tr ',' ' '); a_kind=$3; a_edit=$4; a_id=$5
    for f in $a_files; do
        [ -f "$a_dir/$f" ] || { note="edit did not apply (no such file $f)"; return 1; }
    done
    a_before=$(for f in $a_files; do sha256sum "$a_dir/$f"; done | sha256sum)
    i=1
    for f in $a_files; do
        e=$(printf '%s\n' "$a_edit" | awk -F'@@' -v n=$i '{print $n}')
        case $a_kind in
            sed) sed -i "$e" "$a_dir/$f" ;;
            marker-sed) sed -i "/MUT:$a_id/ $e" "$a_dir/$f" ;;
            replace)
                e=${e#R:}
                FROM=${e%%@>@*} TO=${e#*@>@} perl -0pi -e '$n += s/\Q$ENV{FROM}\E/$ENV{TO} =~ s{\@NL\@}{\n}gr/ge; END { $? = $n == 1 ? 0 : 3 }' "$a_dir/$f" \
                    || { note="replace text not found exactly once in $f"; return 1; }
                ;;
        esac
        i=$((i + 1))
    done
    a_after=$(for f in $a_files; do sha256sum "$a_dir/$f"; done | sha256sum)
    [ "$a_before" != "$a_after" ] || { note="edit did not apply"; return 1; }
}

n=0 k=0 r=0 g=0
first=1
printf '{\n  "mutants": [\n' > "$out"

# run_one id files ekind edit guards justification   (justification is the
# prose "redundant: ..." / "gap: ..." or empty)
run_one() {
    id=$1; files=$2; ekind=$3; edit=$4; what=$5; just=$6
    if [ -n "${ONLY:-}" ]; then case " $ONLY " in *" $id "*) ;; *) return 0;; esac; fi
    dir=$work/m.$id
    copy_tree "$dir"
    note=
    if ! apply_edit "$dir" "$files" "$ekind" "$edit" "$id"; then status=ERROR
    else
        log=$work/$id.log
        if [ -n "$build_cmd" ] && ! (cd "$dir" && sh -c "$build_cmd") </dev/null >"$log" 2>&1; then
            status=ERROR; note="build failed: $(grep -m1 -E "error:|Error " "$log" | cut -c1-120)"
            [ -z "$empty_note" ] || [ -s "$log" ] || note="build failed: $empty_note"
        else
            (cd "$dir" && sh -c "$test_cmd") </dev/null >"$log.run" 2>&1
            rc=$?
            nf=0
            [ -z "$fprefix" ] || nf=$(grep -c "^$fprefix" "$log.run")
            if [ $rc -ne 0 ] && [ "$nf" -gt 0 ]; then
                cut_from=$((${#fprefix} + 2))
                status=KILLED; note="$nf failing case(s), first: $(grep -m1 "^$fprefix" "$log.run" | cut -c"$cut_from"-$((cut_from + 64)))"
            elif [ $rc -ne 0 ]; then status=KILLED; note="harness exit $rc (crash or abort)"
            else
                case "$just" in
                    redundant:*) status=SURVIVED_REDUNDANT; note="$just" ;;
                    gap:*) status=SURVIVED_GAP; note="$just" ;;
                    *) status=SURVIVED; note="harness still PASS, no justification recorded" ;;
                esac
            fi
        fi
    fi
    echo "$id $status  $what ($note)" >&2
    [ $first -eq 1 ] || printf ',\n' >> "$out"
    first=0
    printf '    {"id": "%s", "file": "%s", "checks": "%s", "status": "%s", "note": "%s"}' "$id" "$files" "$what" "$status" "$(printf '%s\n' "$note" | tr -d '"\\')" >> "$out"
    n=$((n + 1))
    case $status in KILLED) k=$((k + 1));; SURVIVED_REDUNDANT) r=$((r + 1));; SURVIVED_GAP) g=$((g + 1));; esac
    rm -rf "$dir"
}

# Reader 1: id~files~edit~guards[~class~rationale]
read_table() {
    while IFS='~' read -r t_id t_files t_edit t_guards t_c5 t_c6; do
        case $t_id in ''|'#'*) continue;; esac
        t_kind=sed
        case $t_edit in R:*) t_kind=replace;; esac
        case $t_c5 in
            none|redundant|gap)
                if [ "$t_c5" = none ]; then t_just=; else t_just="$t_c5: $t_c6"; fi ;;
            *) t_just=$t_c5; [ -z "$t_c6" ] || t_just="$t_c5~$t_c6" ;;
        esac
        run_one "$t_id" "$t_files" "$t_kind" "$t_edit" "$t_guards" "$t_just"
    done < "$rows"
}

# Reader 2: name|file|sed-expr (expr may contain |)
read_marker() {
    while IFS='|' read -r m_name m_file m_expr; do
        case $m_name in ''|'#'*) continue;; esac
        run_one "$m_name" "$m_file" marker-sed "$m_expr" "MUT:$m_name" ""
    done < "$rows"
}

read_$kind

printf '\n  ]\n}\n' >> "$out"
echo "mutants: $n total, $k killed, $r survived as redundant, $g survived as test gaps (receipt $out)"
# Fail on any unjustified survivor or build/edit error, and on an empty sweep.
[ "$n" -gt 0 ] && [ $((k + r + g)) -eq "$n" ]
