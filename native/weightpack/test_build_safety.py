#!/usr/bin/env python3
"""Exercise reused receipts and unsafe build outputs, using disposable owned trees."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    qualify = repo/'native/weightpack/qualify.py'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for case in ('missing-cli-compiler', 'wrong-cli-compiler', 'missing-make-compiler',
                 'symlink-build', 'hardlink-output'):
        with tempfile.TemporaryDirectory(prefix='weightpack-control-', dir=args.out.parent) as directory:
            root = Path(directory)
            evidence = root/'evidence'
            evidence.mkdir()
            receipt = evidence/'qualification.json'
            receipt.write_text('{"old_PASS": true}')
            build = root/'build'
            build.mkdir()
            victim = root/'victim'
            victim.write_bytes(b'PRESERVE ME')
            if case == 'symlink-build':
                build = root/'linked-build'
                build.symlink_to(root/'build', target_is_directory=True)
            elif case == 'hardlink-output':
                (build/'weightpack').hardlink_to(victim)
            compiler = args.compiler
            if case == 'missing-cli-compiler':
                compiler = str(root/'absent')
            if case == 'wrong-cli-compiler':
                compiler = str(victim)
            if case == 'missing-make-compiler':
                # Make's public entry has its own early dependency check.
                outdir = root/'make-output'
                evidence = outdir/'weightpack/evidence'
                evidence.mkdir(parents=True)
                receipt = evidence/'qualification.json'
                receipt.write_text('{"old_PASS": true}')
                cmd = ['make', 'test-native-weightpack', 'OMEGA_NATIVE_COMPILER=', 'OUT_DIR='+str(outdir)]
            else:
                cmd = ['python3', str(qualify), '--compiler', compiler, '--build', str(build), '--evidence', str(evidence)]
            p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=15)
            preserved = victim.read_bytes() == b'PRESERVE ME'
            retired = not receipt.exists()
            archives = list(evidence.glob('qualification.previous-*.json'))
            old_preserved = any(x.read_text() == '{"old_PASS": true}' for x in archives)
            ok = p.returncode != 0 and preserved and retired and old_preserved
            records.append({'case': case, 'pass': ok, 'rc': p.returncode,
                            'victim_preserved': preserved, 'no_stale_current_PASS': retired,
                            'old_receipt_archived': old_preserved})
    result = {'total': len(records), 'passed': sum(x['pass'] for x in records), 'cases': records}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return int(result['passed'] != result['total'])


if __name__ == '__main__':
    raise SystemExit(main())
