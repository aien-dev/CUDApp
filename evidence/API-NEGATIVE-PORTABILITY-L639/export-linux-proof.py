#!/usr/bin/env python3
"""Export completed L639 proof read-only, only its own verified paths."""
from pathlib import Path
import gzip,hashlib,io,json,subprocess,sys,tarfile
home=Path.home();root=home/'handoffs/L639-linux-targets';sha='a91726cd24dbff5a6f67a1246fea0ddf1422f93a'
final=json.loads((root/'FINAL.json').read_text());targets=json.loads((root/'declared-targets.json').read_text())
assert final['source']==sha and final['all_declared_pass'] is True
assert len(targets)==len(set(targets))==final['declared_count']==final['executed_count']==len(final['results'])
assert [r['target'] for r in final['results']]==targets and all(r['exit']==0 for r in final['results'])
tree=home/'workspace/hermes-desk/cudapp-api-qualified-L639'
assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()==sha
files=[root/n for n in ['FINAL.json','declared-targets.json','results.json','current.json']]
for r in final['results']:
    p=Path(r['log']);assert p.resolve().parent==root.resolve() and p.is_file();files.append(p)
manifest={'functional_source':sha,'declared_count':len(targets),'all_targets_pass':True,'original_aggregate_exit':124,
 'scope':'Complete make-declared HOST_TESTS, separately bounded, no retries/skips, host only; not a GPU/model/performance qualification or original API-prose Opus review.',
 'progress_note':'current.json is the stale last RUNNING marker; FINAL.json is the terminal result authority.',
 'members':[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',mtime=0) as gz:
    with tarfile.open(fileobj=gz,mode='w|') as tar:
        for p in files:tar.add(p,arcname=p.name,recursive=False)
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('EXPORT-MANIFEST.json');info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
