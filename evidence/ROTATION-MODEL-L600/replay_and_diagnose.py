from pathlib import Path
import hashlib,json,os,re,subprocess
h=Path.home(); hand=h/'handoffs/quantum-development-L600'; source=hand/'reference-final-export'
pin=(hand/'activation-final-source.txt').read_text().strip()
env=os.environ.copy();env.update(AIEN_FORCE_CPU_STUB='1',CARGO_TARGET_DIR=str(hand/'reference-activation-final-target'),CARGO_BUILD_JOBS='4',RAYON_NUM_THREADS='4',AIEN_ROTQUANT_BIN=str(h/'workspace/cudapp-rotation-int4/build/rotquant-v2-final/rotquant-stream'),AIEN_ROTQUANT_SHA256='2343c13036b002d65b8c7cbd26fbbb6f4c966de21ad3b8e80c77ff2c0b451ce3',AIEN_ROTQUANT_NOOP=str(hand/'native-noop'),AIEN_ROTQUANT_CACHE_TESTROOT=str(hand/'activation-replay-fixtures'))
status={'source':pin,'steps':[],'complete':False}
status_path=hand/'reference-final-export-status.json'
def save():status_path.write_text(json.dumps(status,indent=2))
def run(cmd,name):
    with (hand/name).open('x') as f:p=subprocess.run(list(map(str,cmd)),cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT)
    status['steps'].append({'command':list(map(str,cmd)),'log':name,'exit_code':p.returncode});save()
    if p.returncode:raise RuntimeError(f'{name}: {p.returncode}')
    print('STEP_COMPLETE',name,flush=True)
base=[h/'.local/bin/quiet-build',h/'.cargo/bin/cargo']
run(base+['test','--offline','--release','-p','aien-inference-abi','--features','rotation-qualification','--','--test-threads=1'],'reference-final-export-suite.log')
run(base+['build','--offline','--release','-p','aien-inference-abi','--features','rotation-qualification','--bin','q36_rotation_quality'],'reference-final-export-build.log')
binary=hand/'reference-activation-final-target/release/q36_rotation_quality'
status['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
results=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', (hand/'reference-final-export-suite.log').read_text())
status['tests']={'passed':sum(int(x[0]) for x in results),'failed':sum(int(x[1]) for x in results),'ignored':sum(int(x[2]) for x in results),'groups':len(results)}
root=hand/'rotation-activation32';root.mkdir()
for arm in ['identity','hadamard','zero-aware']:
    cache=hand/'rotation-full-v1'/arm/'reference-f32-cache'
    if not (cache.parent/'precompute-complete.json').exists():raise RuntimeError('precompute prerequisite absent '+arm)
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0])*1024 < 24*1024**3:raise RuntimeError('diagnostic memory floor')
    env['AIEN_ROTQUANT_CACHE']=str(cache)
    run([h/'.local/bin/quiet-build','nice','-n','10',binary,h/'models/qwen3.6-35b-a3b-fp8-95a723d0',hand/'rotation-corpus/p1.json','32',root/arm,arm],'activation32-'+arm+'.log')
    new=json.loads((root/arm/(arm+'.json')).read_text());old=json.loads((hand/'rotation-smoke32'/(arm+'.json')).read_text())
    if new['logits_sha256']!=old['logits_sha256'] or new['positions']!=old['positions']:raise RuntimeError('diagnostic changed actual logits '+arm)
    if json.loads((root/arm/'fp8.json').read_text())['logits_sha256']!=json.loads((hand/'rotation-smoke32/fp8.json').read_text())['logits_sha256']:raise RuntimeError('diagnostic changed baseline logits')
    tensors=json.loads((root/arm/(arm+'-tensors.json')).read_text())
    probes=tensors['real_activation_probes']
    shared=[p for p in probes if '.layers.0.mlp.shared_expert.' in p['tensor']]
    if len(shared)!=3 or tensors['final_child_frames']!=0:raise RuntimeError('incomplete shared probes or unexpected cache reconstruction')
    metrics=json.loads((root/arm/(arm+'-activation-errors.json')).read_text())
    if sorted(x['layer'] for x in metrics['layers'])!=list(range(40)):raise RuntimeError('incomplete real activation coverage')
    status.setdefault('diagnostics',[]).append({'arm':arm,'scored_positions':32,'layers':40,'shared_linear_probes':len(shared),'logits_identical_to_earlier_smoke':True,'all_cache_hits':True});save()
if hashlib.sha256((h/'workspace/targets/l600-rotation-reference/release/q36_rotation_quality').read_bytes()).hexdigest()!='f66d514c4849a4513e52395a23537b45ff941b9e857d928d6b70b59fbf3f1fe9':raise RuntimeError('primary batch executable changed')
status['complete']=True;save();print(json.dumps(status,indent=2),flush=True)
