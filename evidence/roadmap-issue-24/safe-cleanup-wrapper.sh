#!/bin/sh
# L475 host harness compatibility: no recursive rm under workspace. Existing
# test scratch cleanup is restricted to this owner's output and /tmp, then
# performed by find without following symlinks. No repository source deletion.
recursive=0
for arg in "$@"; do
    case "$arg" in
        --) ;;
        -*) case "$arg" in *r*|*R*) recursive=1;; esac ;;
    esac
done
if [ "$recursive" -eq 0 ]; then exec /usr/bin/rm "$@"; fi
for path in "$@"; do
    case "$path" in -*) continue;; esac
    [ -e "$path" ] || [ -L "$path" ] || continue
    resolved=$(realpath -m -- "$path") || exit 2
    case "$resolved" in
        /home/drakestapleton/workspace/cudapp-roadmap-memory24/build-roadmap-reservations24/*|/tmp/mutrun.*/*|/tmp/tmp.*/*|/tmp/mutrun.*|/tmp/tmp.*|/tmp/cudapp-doctor-test.*) ;;
        *) echo "refused cleanup outside L475 output: $path" >&2; exit 2;;
    esac
    if [ -L "$path" ]; then /usr/bin/rm -- "$path"; else find "$path" -depth -delete || exit; fi
done
