#!/usr/bin/env python3
"""Owned disposable filesystem and qualification-freshness controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compiler', required=True)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    qualify = repo/'native/rotquant/qualify_rotation.py'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    names = ('missing-compiler', 'wrong-pin', 'optimized-cli', 'missing-make-compiler',
             'optimized-make', 'symlink-build', 'hardlink-output', 'fifo-output')
    for case in names:
        with tempfile.TemporaryDirectory(prefix='rotation-control-', dir=args.out.parent) as name:
            root = Path(name).resolve()
            build = root/'build'
            evidence = root/'evidence'
            build.mkdir()
            evidence.mkdir()
            victim = root/'victim'
            victim.write_bytes(b'PRESERVE')
            marker = root/'compiler-executed'
            compiler = args.compiler
            if case == 'missing-compiler':
                compiler = str(root/'absent')
            elif case in ('wrong-pin', 'optimized-cli'):
                fake = root/'wrong-compiler'
                fake.write_text('#!/usr/bin/env python3\nfrom pathlib import Path\nPath('+repr(str(marker))+').write_text("EXECUTED")\n')
                fake.chmod(0o700)
                compiler = str(fake)
            elif case == 'symlink-build':
                build = root/'linked'
                build.symlink_to(root/'build', target_is_directory=True)
            elif case == 'hardlink-output':
                (build/'rotquant').hardlink_to(victim)
            elif case == 'fifo-output':
                os.mkfifo(build/'rotquant')
            env = dict(os.environ)
            env.pop('PYTHONOPTIMIZE', None)
            if case.startswith('optimized'):
                env['PYTHONOPTIMIZE'] = '1'
            if case.endswith('make') or case == 'missing-make-compiler':
                out = root/'make-output'
                evidence = out/'rotquant/evidence'
                evidence.mkdir(parents=True)
                if case == 'missing-make-compiler':
                    compiler = ''
                cmd = ['make', 'test-native-rotquant', 'OMEGA_NATIVE_COMPILER='+compiler, 'OUT_DIR='+str(out)]
            else:
                cmd = ['python3', str(qualify), '--compiler', compiler, '--build', str(build), '--evidence', str(evidence)]
            receipt = evidence/'qualification.json'
            receipt.write_text('{"old_PASS":true}')
            p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=15)
            archived = any(x.read_text() == '{"old_PASS":true}' for x in evidence.glob('qualification.previous-*.json'))
            ok = p.returncode != 0 and not receipt.exists() and archived and victim.read_bytes() == b'PRESERVE' and not marker.exists()
            records.append({'case': case, 'pass': ok, 'rc': p.returncode, 'old_receipt_archived': archived,
                            'no_current_PASS': not receipt.exists(), 'victim_preserved': victim.read_bytes() == b'PRESERVE',
                            'wrong_compiler_not_executed': not marker.exists()})
    result = {'total': len(records), 'passed': sum(x['pass'] for x in records), 'cases': records}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return int(result['passed'] != result['total'])


if __name__ == '__main__':
    raise SystemExit(main())
