#!/usr/bin/env python3
"""Bounded full declared-host-target qualification; no retry or skipped target."""
from pathlib import Path
import json,subprocess,os,signal,time
h=Path.home(); source=h/'workspace/cudapp'; tree=h/'workspace/hermes-desk/cudapp-api-qualified-L639'; out=h/'handoffs/L639-linux-targets'; sha='a91726cd24dbff5a6f67a1246fea0ddf1422f93a'
assert not tree.exists()
subprocess.run(['git','-C',str(source),'worktree','add','--detach',str(tree),sha],check=True)
assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()==sha
out.mkdir(exist_ok=False)
# Ask make itself for its exact declared suite rather than copying or shrinking a list.
rule='L639-print-host-targets: ; @printf "%s\\n" "$(HOST_TESTS)"'
raw=subprocess.check_output(['make','-s','--eval',rule,'L639-print-host-targets'],cwd=tree,text=True)
targets=raw.split(); assert targets and len(targets)==len(set(targets))
(out/'declared-targets.json').write_text(json.dumps(targets,indent=2)+'\n')
tmp=out/'tmp';tmp.mkdir(); env=dict(os.environ,TMPDIR=str(tmp),PATH=str(h/'.local/bin')+os.pathsep+os.environ['PATH'])
results=[]
for i,target in enumerate(targets):
    assert target.replace('-','').replace('_','').isalnum(),target
    log=out/f'{i:02d}-{target}.log'
    (out/'current.json').write_text(json.dumps(dict(target=target,ordinal=i,state='RUNNING',source=sha))+'\n')
    with log.open('w') as f:
        p=subprocess.Popen(['quiet-build','nice','-n','19','make','-j2',target],cwd=tree,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            rc=124
    row=dict(target=target,exit=rc,log=str(log));results.append(row)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if rc!=0:break
complete=len(results)==len(targets) and all(r['exit']==0 for r in results)
final=dict(source=sha,declared_count=len(targets),executed_count=len(results),all_declared_pass=complete,results=results,original_aggregate_exit=124,scope='All original HOST_TESTS targets, per-target900s including quiet wait; no retries/skips, no GPU/model or performance qualification')
(out/'FINAL.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
raise SystemExit(0 if complete else 1)
