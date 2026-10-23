#!/usr/bin/env python3
"""External harness oracle, not production runtime or a replacement API checker."""
from pathlib import Path
import os, subprocess, tempfile, json, shutil
root=Path(__file__).resolve().parents[2]
scratch=Path.home()/'.hermes/cache/scratch'
results=[]
expected=['removed-function','missing-version','changed-signature','changed-struct','changed-constant','version-doc-mismatch','stable-without-doc','wrong-quoted-signature','signature-inside-another']
with tempfile.TemporaryDirectory(prefix='L639-sed-controls-',dir=scratch) as tmp:
    tmp=Path(tmp); commands=tmp/'bin'; commands.mkdir()
    for case,body in [('edit-failure','#!/bin/sh\nexit 42\n'),('edit-noop','#!/bin/sh\nfor last; do :; done\ncat "$last"\n')]:
        sed=commands/'sed'; sed.write_text(body); sed.chmod(0o755)
        env=dict(os.environ,PATH=str(commands)+os.pathsep+os.environ['PATH'],TMPDIR=str(scratch))
        p=subprocess.run(['sh','tools/runtime_api_negative.sh','docs/runtime/omega_runtime_api.manifest','.',str(tmp/case),'docs/runtime/API.md'],cwd=root,env=env,capture_output=True,text=True)
        assert p.returncode==1,(case,p.returncode,p.stdout,p.stderr)
        assert all(f'NEGATIVE CONTROL {name}: BROKEN' in p.stdout for name in expected),p.stdout
        assert 'RESULT: PASS' not in p.stdout
        results.append(dict(case=case,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
print(json.dumps(results,indent=2))
