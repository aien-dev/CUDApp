#!/usr/bin/env python3
"""Post-run independent aggregation and lossless evidence publication.
No model/quantizer execution, fitting, acceptance changes or GPU access.
"""
from pathlib import Path
import hashlib,json,math,gzip,subprocess,datetime
h=Path.home();hand=h/'handoffs/quantum-development-L600';root=hand/'rotation-full-v1';repo=h/'workspace/cudapp-rotation-int4';out=repo/'evidence/ROTATION-MODEL-L600'
arms=['identity','hadamard','zero-aware','nine-level-mutant'];names=['p1','p2','p3','p4','chat-max-24','h1','h2','h3','h4','h5','agpl']
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():p.read_bytes()
    p.write_bytes(b)
def dump(p,d):write(p,(json.dumps(d,indent=2)+'\n').encode())
def archive(src,dst):
    b=src.read_bytes();c=gzip.compress(b,mtime=0);require(gzip.decompress(c)==b,'gzip roundtrip');write(dst,c)
    return {'source':str(src.relative_to(root)),'published':str(dst.relative_to(out)),'raw_bytes':len(b),'raw_sha256':sha(b),'gzip_sha256':sha(c)}
def close(x,y):require(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),f'aggregate differs: {x} {y}')
require(load(hand/'rotation-full-v1-status.json')['exit_code']==0,'batch nonzero')
q=load(root/'qualification.json');require([v['arm'] for v in q['arms']]==arms,'arm membership')
inputs=load(root/'inputs.json');manifest=load(hand/'rotation-corpus/manifest.json')
require(set(manifest)==set(names),'corpus membership');require(sha((hand/'rotation-corpus/manifest.json').read_bytes())==inputs['corpus_manifest_sha256'],'manifest changed')
require(sha((hand/'rotation-precompute-inventory.json').read_bytes())==inputs['inventory_sha256'],'inventory changed')
for name,m in manifest.items():require(sha(Path(m['path']).read_bytes())==m['frozen_sha256'],'corpus changed '+name)
for key,digest in [('reference',inputs['argv']['reference_sha']),('native','2343c13036b002d65b8c7cbd26fbbb6f4c966de21ad3b8e80c77ff2c0b451ce3'),('mutant','48e5a4db4e8ab779825b588da9d058afaf3ad32c31a79921814a58251da398a2')]:require(sha(Path(inputs['argv'][key]).read_bytes())==digest,'binary changed '+key)
# Read the full original checkpoint closure again, not just header/mtime checks.
closure=load(hand/'whole-model-input-closure.json');files=[]
for f in closure['files']:
    p=Path(closure['root'])/f['file'];before=p.stat();digest=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):digest.update(b)
    after=p.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'source changed during hash')
    require(digest.hexdigest()==f['sha256'] and before.st_size==f['bytes'],'checkpoint changed '+f['file'])
    files.append({'file':f['file'],'sha256':digest.hexdigest(),'bytes':before.st_size})
require(sum(x['file'].startswith('layers-') and x['file'].endswith('.safetensors') for x in files)==40,'checkpoint layer closure')
dump(out/'whole-model-postrun-closure.json',{'all_hashes_match_initial_closure':True,'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files})
print('POSTRUN_CHECKPOINT_IDENTICAL',len(files),flush=True)
archive_index=[];cases=[];reports=[];baselines={};changed_route_count=0
for arm,v in zip(arms,q['arms']):
    label='hadamard' if arm=='nine-level-mutant' else arm;p=root/arm
    require({x.name for x in p.glob('quality-*') if x.is_dir()}=={'quality-'+n for n in names},'case membership '+arm)
    prec={}
    for f in p.glob('layers-*/layer-*.json'):
        for name,t in load(f)['native']['tensors'].items():require(name not in prec,'duplicate precompute');prec[name]=t
    require(len(prec)==30840,'precompute count')
    grouped={'golden':[],'heldout':[],'agpl':[]}
    for name in names:
        case=p/('quality-'+name);n=399 if name=='agpl' else 512;group='agpl' if name=='agpl' else ('heldout' if name.startswith('h') else 'golden')
        status=load(p/('quality-'+name+'.status.json'));done=load(case/'complete.json');base=load(case/'fp8.json');cand=load(case/(label+'.json'))
        require(status['exit_code']==0 and done['completed'] and done['exact_control'] and not done['gpu_tested'] and not done['production_changed'],'case control '+arm+name)
        require(done['scored_positions']==base['scored_positions']==cand['scored_positions']==n,'case count')
        require(base['input_sha256']==manifest[name]['frozen_sha256'],'input hash')
        signature={k:base[k] for k in ['input_sha256','logits_sha256','moe_activation_sha256','positions','tokens']}
        if name in baselines:require(signature==baselines[name],'baseline differs across arms '+name)
        else:baselines[name]=signature
        require(base['exact_control'] and cand['full_model_layers']==40 and not cand['direct_packed_gpu_execution'],'scope')
        require([r['position'] for r in cand['positions']]==list(range(n)),'position membership')
        require(len(base['positions'])==n,'baseline rows')
        for r,b in zip(cand['positions'],base['positions']):
            require(r['position']==b['position'] and r['target']==b['target'] and r['reference_argmax']==b['argmax'],'paired token identity')
            require(r['reference_margin']==b['margin'] and r['reference_nll']==b['nll'],'paired reference values')
            require(all(math.isfinite(r[k]) for k in ['nll','reference_nll','reference_margin','logit_cosine','kl_reference_candidate']),'nonfinite metric')
            require(0<=r['layers_with_changed_ordered_routes']<=40,'route count range')
        a=load(case/(label+'-tensors.json'));f=load(case/'fp8-tensors.json')
        require(a['final_child_frames']==0,'cache miss');require(a['unique_expert_tensors']==len(a['tensors']),'actual tensor count')
        require(f['unique_expert_tensors']==len(f['tensors']),'baseline tensor count')
        for name2,t in a['tensors'].items():
            expected=prec[name2]
            for k,value in expected.items():require(t[k]==value,'candidate/precompute mismatch '+arm+name+name2+k)
        for name2,t in f['tensors'].items():
            for k in ['source_sha256','scale_sha256','shape','raw_bytes','scale_bytes']:require(t[k]==prec[name2][k],'baseline source mismatch')
        grouped[group].extend(cand['positions'])
        cases.append({'arm':arm,'case':name,'positions':n,'control':True,'candidate_tensors':len(a['tensors']),'fp8_tensors':len(f['tensors']),'candidate_logits_sha256':cand['logits_sha256']})
        expected_files={'complete.json','fp8.json',label+'.json','fp8-tensors.json',label+'-tensors.json'}
        require({x.name for x in case.iterdir()}==expected_files,'unexpected/missing raw case file')
        for src in sorted(case.iterdir()):archive_index.append(archive(src,out/'full-quality'/arm/case.name/(src.name+'.gz')))
        for suffix in ['.log','.status.json']:
            src=p/(case.name+suffix);archive_index.append(archive(src,out/'full-quality'/arm/(src.name+'.gz')))
    rr={'arm':arm,'groups':{}}
    for group,rows in grouped.items():
        require(len(rows)==(399 if group=='agpl' else 2560),'group count')
        bn=math.fsum(r['reference_nll'] for r in rows)/len(rows);cn=math.fsum(r['nll'] for r in rows)/len(rows)
        stats={'positions':len(rows),'reference_mean_nll':bn,'candidate_mean_nll':cn,'mean_nll_increase':cn-bn,'nll_ratio':cn/bn,'reference_teacher_forced_perplexity':math.exp(bn),'candidate_teacher_forced_perplexity':math.exp(cn),'perplexity_ratio':math.exp(cn-bn),'all_top1_disagreements':sum(r['argmax']!=r['reference_argmax'] for r in rows),'near_tie_disagreements_margin_below_0_1':sum(r['argmax']!=r['reference_argmax'] and r['reference_margin']<0.1 for r in rows),'positions_with_route_changes':sum(r['layers_with_changed_ordered_routes']>0 for r in rows),'mean_changed_route_layers':math.fsum(r['layers_with_changed_ordered_routes'] for r in rows)/len(rows)}
        if group=='agpl':
            close(cn-bn,v['agpl']['mean_nll_rise']);passed=cn-bn<=0.069;require(passed==v['agpl']['pass'],'AGPL verdict mismatch')
        else:
            old=v['groups'][group];close(bn,old['reference_mean_nll']);close(cn,old['candidate_mean_nll']);close(cn/bn,old['nll_ratio'])
            close(min(r['logit_cosine'] for r in rows),old['minimum_logit_cosine']);close(math.fsum(r['kl_reference_candidate'] for r in rows)/len(rows),old['mean_kl'])
            passed=cn/bn<=1.07;stats['agreement']={}
            for margin,floor in [(0.1,0.98),(1.0,0.998)]:
                chosen=[r for r in rows if r['reference_margin']>=margin];agree=sum(r['argmax']==r['reference_argmax'] for r in chosen);old_ag=old['agreement'][str(margin)];require((len(chosen),agree)==(old_ag['positions'],old_ag['agree']),'agreement counts')
                ok=agree/len(chosen)>=floor;require(ok==old_ag['pass'],'agreement verdict');passed=passed and ok
                stats['agreement'][str(margin)]={'positions':len(chosen),'agree':agree,'fraction':agree/len(chosen),'pass':ok}
            require(passed==old['pass'],'group verdict mismatch')
        stats['pass']=passed;rr['groups'][group]=stats
    rr['candidate_quality_pass']=all(x['pass'] for x in rr['groups'].values());require(rr['candidate_quality_pass']==v['candidate_quality_pass'],'arm verdict mismatch');reports.append(rr)
    write(out/'full-quality'/arm/'verdict.json',(p/'verdict.json').read_bytes());print('ARM_AUDITED',arm,len(cases),flush=True)
require(len(cases)==44 and len({(x['arm'],x['case']) for x in cases})==44,'complete case count')
require(sum(x['positions'] for x in cases)==22076,'complete position count')
require(not reports[-1]['candidate_quality_pass'] and q['negative_control_rejected'] and q['quality_gate_valid'],'negative control')
require(not any(r['candidate_quality_pass'] for r in reports[:3]),'unexpected candidate approval')
write(out/'full-quality/qualification.json',(root/'qualification.json').read_bytes());write(out/'full-quality/batch-status.json',(hand/'rotation-full-v1-status.json').read_bytes())
write(out/'full-quality/batch.log.gz',gzip.compress((hand/'rotation-full-v1.log').read_bytes(),mtime=0))
dump(out/'full-quality/independent-audit.json',{'scope':'independent post-run sums/counts/source validation; no new model execution','completed_cases':len(cases),'candidate_scored_positions':sum(x['positions'] for x in cases),'scored_positions_per_arm':sum(x['positions'] for x in cases if x['arm']=='identity'),'baseline_identical_across_all_arms':True,'native_cache_receipts_match_complete_precompute':True,'negative_control_rejected':True,'candidate_decision':'NO_GO: all three frozen INT4 arms fail accuracy','cases':cases,'recomputed':reports,'raw_archive_index':archive_index})
print('FINAL_COUNTS',len(cases),sum(x['positions'] for x in cases),len(archive_index),flush=True)
for r in reports:
 g=r['groups']['golden'];v=r['groups']['heldout'];a=r['groups']['agpl']
 print('| '+r['arm']+' | '+f"{100*(g['nll_ratio']-1):.2f}% | {100*(v['nll_ratio']-1):.2f}% | {100*v['agreement']['0.1']['fraction']:.4f}% | {100*v['agreement']['1.0']['fraction']:.4f}% | {a['mean_nll_increase']:+.6f} | REJECT |",flush=True)
 print('PERPLEXITY',r['arm'],[(k,x['reference_teacher_forced_perplexity'],x['candidate_teacher_forced_perplexity']) for k,x in r['groups'].items()],flush=True)
print('FINAL_AUDIT_COMPLETE',flush=True)
