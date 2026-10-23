#!/usr/bin/env python3
"""CPU-only qualification orchestration. No model maths, tuning or GPU runtime."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def quality(rows):
    require(bool(rows), 'empty quality rows')
    base = sum(r['reference_nll'] for r in rows)/len(rows)
    candidate = sum(r['nll'] for r in rows)/len(rows)
    require(base > 0, 'nonpositive baseline NLL')
    thresholds = {}
    for margin, floor in [(0.1, 0.98), (1.0, 0.998)]:
        selected = [r for r in rows if r['reference_margin'] >= margin]
        require(bool(selected), 'no qualified margin denominator')
        agree = sum(r['argmax'] == r['reference_argmax'] for r in selected)
        thresholds[str(margin)] = {'positions': len(selected), 'agree': agree,
                                  'fraction': agree/len(selected), 'floor': floor,
                                  'pass': agree/len(selected) >= floor}
    return {'positions': len(rows), 'reference_mean_nll': base, 'candidate_mean_nll': candidate,
            'nll_ratio': candidate/base, 'nll_ratio_limit': 1.07,
            'agreement': thresholds, 'minimum_logit_cosine': min(r['logit_cosine'] for r in rows),
            'mean_kl': sum(r['kl_reference_candidate'] for r in rows)/len(rows),
            'pass': candidate/base <= 1.07 and all(t['pass'] for t in thresholds.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha', required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--mutant', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--precompute-workers', type=int, default=8)
    args = parser.parse_args()
    require(__debug__, 'optimized Python forbidden')
    require(1 <= args.precompute_workers <= 8, 'worker cap')
    require(sha(args.reference) == args.reference_sha, 'reference binary changed')
    require(sha(args.native) == '2343c13036b002d65b8c7cbd26fbbb6f4c966de21ad3b8e80c77ff2c0b451ce3', 'native binary changed')
    require(sha(args.mutant) == '48e5a4db4e8ab779825b588da9d058afaf3ad32c31a79921814a58251da398a2', 'mutant binary changed')
    corpus = json.loads((args.corpus/'manifest.json').read_text())
    inventory = json.loads(args.inventory.read_text())
    require(len(corpus) == 11 and inventory['tensor_count'] == 30840, 'incomplete frozen inputs')
    for name, item in corpus.items():
        require(sha(Path(item['path'])) == item['frozen_sha256'], 'changed corpus '+name)
    for name in ['config', 'tokenizer']:
        require(sha(args.model/(name+'.json')) == inventory[name+'_sha256'], 'changed model '+name)
    required = inventory['four_arm_reference_f32_cache_bytes_excluding_headers']
    require(shutil.disk_usage(args.root.parent).free > required + 100*1024**3, 'cache disk floor')
    require(not args.root.exists(), 'fresh job root required; never reuse stale PASS')
    args.root.mkdir(mode=0o700)
    record(args.root/'inputs.json', {'argv': vars(args) | {k:str(v) for k,v in vars(args).items() if isinstance(v,Path)},
                                     'corpus_manifest_sha256': sha(args.corpus/'manifest.json'),
                                     'inventory_sha256': sha(args.inventory), 'quality_rules_frozen': True})
    quiet = Path.home()/'.local/bin/quiet-build'
    arms = [('identity',0,args.native), ('hadamard',1,args.native),
            ('zero-aware',2,args.native), ('nine-level-mutant',1,args.mutant)]
    roots = {}

    def launch(cmd, env, logfile):
        begin = time.time()
        with logfile.open('x') as log:
            p = subprocess.run([str(quiet), 'nice', '-n', '10'] + list(map(str,cmd)),
                               env=env, stdout=log, stderr=subprocess.STDOUT)
        record(logfile.with_suffix('.status.json'), {'argv': list(map(str,cmd)), 'exit_code': p.returncode,
               'started_unix': begin, 'ended_unix': time.time(), 'not_a_runtime_benchmark': True})
        require(p.returncode == 0, 'command failed: '+str(logfile))

    for arm, mode, native in arms:
        armroot = args.root/arm; armroot.mkdir(); cache = armroot/'reference-f32-cache';cache.mkdir(mode=0o700)
        roots[arm] = armroot
        env = os.environ.copy()
        env.update(AIEN_ROTQUANT_BIN=str(native), AIEN_ROTQUANT_SHA256=sha(native),
                   AIEN_ROTQUANT_MODE=str(mode), AIEN_ROTQUANT_CACHE=str(cache), RAYON_NUM_THREADS='1')
        print('PRECOMPUTE_START', arm, flush=True)
        def part(first):
            output = armroot/('layers-'+str(first))
            launch([args.reference,'--precompute',args.model,first,first+5,output],env,armroot/('precompute-'+str(first)+'.log'))
            return output
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.precompute_workers) as pool:
            outputs = list(pool.map(part, range(0,40,5)))
        layers = []
        for out in outputs:
            complete = json.loads((out/'complete.json').read_text());require(complete['complete'], 'incomplete partition')
            for layer in range(complete['first_layer'],complete['end_layer']):
                path = out/('layer-'+str(layer)+'.json');r=json.loads(path.read_text());expected=inventory['layers'][layer]
                require(r['layer']==layer and r['tensor_count']==expected['tensor_count'] and r['header_sha256']==expected['header_sha256'], 'precompute membership/source mismatch')
                layers.append({'layer':layer,'tensor_count':r['tensor_count'],'receipt_sha256':sha(path)})
        require(sorted(r['layer'] for r in layers)==list(range(40)), 'incomplete/disjoint layer coverage')
        require(sum(r['tensor_count'] for r in layers)==30840, 'precompute tensor total mismatch')
        cachefiles=list(cache.glob('*.orc'));require(len(cachefiles)==30840, 'cache membership mismatch')
        record(armroot/'precompute-complete.json',{'layers':layers,'tensor_count':30840,'expanded_reference_cache_bytes':sum(p.stat().st_size for p in cachefiles), 'not_packed_product_storage':True})
        print('PRECOMPUTE_COMPLETE',arm,30840,flush=True)

    def evaluate(arm_info):
        arm, mode, native = arm_info; armroot=roots[arm]
        env=os.environ.copy();env.update(AIEN_ROTQUANT_BIN=str(native),AIEN_ROTQUANT_SHA256=sha(native),
                                         AIEN_ROTQUANT_CACHE=str(armroot/'reference-f32-cache'),RAYON_NUM_THREADS='4')
        env.pop('AIEN_ROTQUANT_MODE',None)
        label = 'hadamard' if arm=='nine-level-mutant' else arm
        for name in ['p1','p2','p3','p4','chat-max-24','h1','h2','h3','h4','h5','agpl']:
            count=400 if name=='agpl' else 512;out=armroot/('quality-'+name)
            print('QUALITY_START',arm,name,flush=True)
            launch([args.reference,args.model,corpus[name]['path'],count,out,label],env,armroot/('quality-'+name+'.log'))
            done=json.loads((out/'complete.json').read_text());expected=399 if name=='agpl' else 512
            require(done['completed'] and done['exact_control'] and done['scored_positions']==expected, 'model count/control failed')
            receipt=json.loads((out/(label+'-tensors.json')).read_text())
            require(receipt['final_child_frames']==0, 'unprepared cache entry was reconstructed during quality run')
            print('QUALITY_COMPLETE',arm,name,expected,flush=True)
        rows={}
        for group,names in [('golden',['p1','p2','p3','p4','chat-max-24']),('heldout',['h1','h2','h3','h4','h5'])]:
            rows[group]=[r for name in names for r in json.loads((armroot/('quality-'+name)/(label+'.json')).read_text())['positions']]
            require(len(rows[group])==2560,'quality group total mismatch')
        passage=json.loads((armroot/'quality-agpl'/(label+'.json')).read_text())['positions'];require(len(passage)==399,'passage total mismatch')
        rise=sum(r['nll']-r['reference_nll'] for r in passage)/len(passage)
        groups={name:quality(items) for name,items in rows.items()}
        verdict={'arm':arm,'groups':groups,'agpl':{'positions':399,'mean_nll_rise':rise,'limit':0.069,'pass':rise<=0.069},
                 'candidate_quality_pass':all(g['pass'] for g in groups.values()) and rise<=0.069,
                 'gpu_execution_tested':False,'inference_acceleration_tested':False,'production_changed':False}
        record(armroot/'verdict.json',verdict)
        return verdict

    # Two host references, bounded memory. These are numerical jobs, not agents.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verdicts=list(pool.map(evaluate,arms))
    mutant=next(v for v in verdicts if v['arm']=='nine-level-mutant')
    valid=not mutant['candidate_quality_pass']
    record(args.root/'qualification.json',{'arms':verdicts,'quality_gate_valid':valid,'negative_control_rejected':valid,
           'gpu_acceptance':'NOT_RUN: requires qualified native IR/runtime interface and admitted hardware',
           'issue49_complete':False,'no_deployment_authorization':True})
    require(valid,'nine-level mutant passed: gate inconclusive, never adoption')
    print('QUALITY_QUALIFICATION_RECORDED',args.root/'qualification.json',flush=True)


if __name__=='__main__':
    main()
