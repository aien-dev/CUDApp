// SPDX-License-Identifier: MIT
import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fixture as trace} from './numerical_trace_test.mjs';
import {digest} from '../tools/numerical-trace.mjs';
import {SCHEMA,validateRun,report,compareRuns,importCapture,measured,unknown,readable,quantile,encodeJSON} from '../tools/native-benchmark.mjs';
export function fixture() {
  const policy={mode:'bitwise'},t=trace();
  const r={schema:SCHEMA,contract:t.contract,kind:'fixture',source_sha:t.source_sha,model_sha256:t.model_sha256,
    prompt_sha256:t.prompt_sha256,config_sha256:t.config_sha256,binary_sha256:t.binary_sha256,driver_sha256:'1'.repeat(64),
    precision:t.precision,chip:'fixture-no-hardware',backend:'native-graph',policy,policy_sha256:digest(JSON.stringify(policy)),
    supported_context_limit:8,workloads:[{batch:1,context:8,input_tokens:8,output_tokens:1,seed:7}],
    conditions:{thermal_clock_protocol_sha256:'2'.repeat(64),description:'Fixture only; no timing measurements'},
    correctness:{reference:t,candidate:structuredClone(t)},samples:[]};
  for(const session of ['a','b','c']) for(const stamps of [false,true]) for(const phase of ['cold','warm']) {
    const total=phase==='cold'?200:100, metrics={};
    for(const k of ['setup_ns','ttft_ns','prefill_ns','host_submit_ns','barriers','rss_bytes','pss_bytes','driver_process_bytes','weight_bytes','kv_bytes','intermediate_bytes']) metrics[k]=unknown('fixture unavailable');
    for(const [k,v] of Object.entries({decode_ns:total,total_ns:total+(stamps?10:0),host_record_ns:10,native_completion_ns:80,
      launches:2,submits:1,allocations:0,allocation_bytes:0})) metrics[k]=measured(v,'fixture only');
    metrics.gpu_ns=stamps?measured(50,'fixture timestamp span'):unknown('stamps off');
    r.samples.push({session,ordinal:0,phase,stamps,workload:{batch:1,context:8,input_tokens:8,output_tokens:1,seed:7},metrics,
      timestamps:stamps?['100','120','150']:[],kinds:stamps?[0,1]:[],correctness_sha256:digest(JSON.stringify(t))});
  }
  return r;
}
test('quantiles from independent samples; raw sessions and unknowns retained',()=>{
  const r=report(fixture()); assert.equal(r.groups.length,4); assert.equal(r.groups[0].metrics.total_ns.p50,200);
  assert.equal(r.groups[0].metrics.total_ns.p95,200); assert.equal(r.instrumentation.pairs.length,6);
  assert.equal(r.groups[0].metrics.host_submit_ns.category,'unknown'); assert.equal(r.hardware,'NOT_RUN');
  assert.equal(quantile([9,1,4],.5),4); assert.equal(quantile([9,1,4],.95),9);
  assert.match(readable(r),/p50=200 p95=200/);
});
test('unfavorable candidate preserved and fixture never claims speedup',()=>{
  const a=fixture(),b=fixture(); for(const s of b.samples) s.metrics.total_ns.value*=2;
  const c=compareRuns(a,b); assert.equal(c.pairs[0].ratio,.5); assert.equal(c.groups[0].unfavorable_pairs,3);
  assert.equal(c.groups[0].decision,'NONE_FIXTURE');
});
const mutations={
  'contract mismatch':r=>r.contract='other',
  'missing hash':r=>delete r.model_sha256,
  'policy hash':r=>r.policy.mode='repeatable',
  empty:r=>r.samples=[],
  nonfinite:r=>r.samples[0].metrics.total_ns.value=Infinity,
  'missing metric':r=>delete r.samples[0].metrics.barriers,
  'unknown as zero':r=>r.samples[0].metrics.barriers.value=0,
  'estimated counter':r=>r.samples[0].metrics.launches.category='estimated',
  'measured estimate':r=>r.samples[0].metrics.weight_bytes=measured(123,'fiction'),
  'timestamp order':r=>r.samples[2].timestamps=['100','90','150'],
  'timestamp poison':r=>r.samples[2].timestamps[2]='18446744073709551615',
  'unknown kind':r=>r.samples[2].kinds[0]=99,
  'record exceeds total':r=>r.samples[0].metrics.host_record_ns.value=999,
  'completion exceeds total':r=>r.samples[0].metrics.native_completion_ns.value=999,
  'fake gpu span':r=>r.samples[2].metrics.gpu_ns.value=1,
  'stamps off fake gpu':r=>r.samples[0].metrics.gpu_ns=measured(0,'fake'),
  'zero total':r=>r.samples[0].metrics.total_ns.value=0,
  'duplicate session':r=>r.samples.push(structuredClone(r.samples[0])),
  'correlated ordinal':r=>{const x=structuredClone(r.samples[0]);x.ordinal=1;r.samples.push(x);},
  'missing correctness':r=>delete r.correctness,
  'wrong correctness binary':r=>r.correctness.candidate.binary_sha256='0'.repeat(64),
  'wrong correctness binding':r=>r.samples[0].correctness_sha256='0'.repeat(64),
  'changed token':r=>r.correctness.candidate.rows[0].token=2,
  'uncovered generated tokens':r=>{r.workloads[0].input_tokens=7;r.workloads[0].output_tokens=2;r.samples[0].workload={...r.workloads[0]};},
  'unsupported context':r=>r.samples[0].workload.context=9,
  'uncovered context':r=>r.samples[0].workload.context=6,
  'unsupported batch':r=>r.samples[0].workload.batch=2,
  'seed not validated':r=>r.samples[0].workload.seed=999,
};
for(const [name,mutate] of Object.entries(mutations)) test(`reject ${name}`,()=>{const r=fixture();mutate(r);assert.throws(()=>validateRun(r));});
for(const k of ['driver_sha256','model_sha256','prompt_sha256','config_sha256','policy_sha256']) test(`paired rejects ${k}`,()=>{
  const a=fixture(),b=fixture(); b[k]='0'.repeat(64); assert.throws(()=>compareRuns(a,b));
});
test('unequal workload/seed, thermal protocol, sample count refuse',()=>{
  for(const mutate of [r=>r.samples[0].workload.input_tokens++,r=>r.samples[0].workload.seed++,
    r=>r.conditions.description='different conditions',r=>r.samples.pop()]) {
    const a=fixture(),b=fixture();mutate(b);assert.throws(()=>compareRuns(a,b));
  }
});
test('adapter capture import usable with unknown fields',()=>{
  const r=fixture(),s=r.samples[0];
  const e={schema:'cudapp.graph-sample.v1',contract:r.contract,session:s.session,ordinal:0,phase:'warm',stamps:false,
    token:1,workload:s.workload,kinds:[],measured:{total_ns:100,decode_ns:100,launches:2},timestamps:[]};
  const result=importCapture(r,[e],r.correctness); assert.equal(report(result).groups.length,1);
  assert.equal(result.samples[0].metrics.barriers.value,null);
  assert.throws(()=>importCapture(r,[],r.correctness));
  assert.throws(()=>importCapture(r,[{...e,token:2}],r.correctness));
  assert.throws(()=>importCapture(r,[{...e,measured:{...e.measured,barriers:NaN}}],r.correctness));
});
test('CLI prepare never executes and report/compare/import have JSON and text',()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'cudapp-bench-fixture-'));
  try {
    const r=fixture(),file=path.join(dir,'run.json');fs.writeFileSync(file,JSON.stringify(r));
    const cli=(...args)=>spawnSync(process.execPath,['tools/native-benchmark.mjs',...args],{encoding:'utf8'});
    let p=cli('prepare',file,path.join(dir,'plan.json'));assert.equal(p.status,0,p.stderr);
    assert.equal(JSON.parse(fs.readFileSync(path.join(dir,'plan.json'))).status,'NOT_RUN');
    assert.equal(cli('prepare',file,path.join(dir,'plan.json')).status,1); // no clobber
    assert.equal(JSON.parse(cli('report',file).stdout).kind,'fixture');
    assert.match(cli('compare',file,file,'--text').stdout,/NONE_FIXTURE/);
    fs.writeFileSync(file,'{"bad":'); assert.equal(cli('report',file).status,1);
  } finally {fs.rmSync(dir,{recursive:true});}
});

test('C graph capture imports and reports through CLI',()=>{
  const adapter=process.env.CUDAPP_BENCH_ADAPTER;
  assert.ok(adapter,'run make test-native-benchmark to provide compiled adapter fixture');
  const event=spawnSync(adapter,['--fixture'],{encoding:'utf8'});assert.equal(event.status,0,event.stderr);
  const r=fixture(),data=importCapture(r,event.stdout.trim().split('\n').map(JSON.parse),r.correctness);
  assert.equal(report(data).groups[0].metrics.gpu_ns.p50,50);
  assert.equal(report(data).groups[0].metrics.barriers.category,'unknown');
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'cudapp-import-fixture-'));
  try {
    for(const [name,body] of Object.entries({'manifest.json':JSON.stringify(r),'capture.jsonl':event.stdout,'correctness.json':JSON.stringify(r.correctness)}))fs.writeFileSync(path.join(dir,name),body);
    const p=spawnSync(process.execPath,['tools/native-benchmark.mjs','import',...['manifest.json','capture.jsonl','correctness.json','imported.json'].map(x=>path.join(dir,x))],{encoding:'utf8'});
    assert.equal(p.status,0,p.stderr);assert.equal(JSON.parse(fs.readFileSync(path.join(dir,'imported.json'))).samples.length,1);
  }finally{fs.rmSync(dir,{recursive:true});}
});

test('artifact serialization preserves exact signed zero',()=>{
  const r=fixture();r.correctness.reference.rows[0].stages.hidden[0]=-0;r.correctness.candidate.rows[0].stages.hidden[0]=-0;
  for(const s of r.samples)s.correctness_sha256=digest(encodeJSON(r.correctness.candidate));
  const parsed=JSON.parse(encodeJSON(r));assert.ok(Object.is(parsed.correctness.candidate.rows[0].stages.hidden[0],-0));
  assert.equal(report(parsed).correctness.status,'PASS');
});

test('serving request clocks validate TTFT, prefill and decode; clock mutations refuse',()=>{
  const r=fixture(),e={schema:'cudapp.graph-sample.v1',contract:r.contract,session:'request-a',ordinal:0,phase:'warm',stamps:false,
    token:1,workload:r.workloads[0],timestamps:[],kinds:[],request_clock:['100','120','130','200'],
    measured:{total_ns:100,ttft_ns:30,prefill_ns:20,decode_ns:80}};
  const imported=importCapture(r,[e],r.correctness);assert.equal(report(imported).groups[0].metrics.ttft_ns.p50,30);
  for(const bad of [{...e,request_clock:['100','140','130','200']},{...e,measured:{...e.measured,ttft_ns:1}},
    {...e,request_clock:['100','nan','130','200']}])assert.throws(()=>importCapture(r,[bad],r.correctness));
});
test('traffic estimates and #14 memory scopes stay separate from measured counters',()=>{
  const r=fixture(),e={schema:'cudapp.graph-sample.v1',contract:r.contract,session:'memory-a',ordinal:0,phase:'warm',stamps:false,
    token:1,workload:r.workloads[0],timestamps:[],kinds:[],measured:{total_ns:100,rss_bytes:4096,pss_bytes:2048},
    methods:{rss_bytes:'#14 post-warmup process snapshot; shared mappings included',pss_bytes:'#14 post-warmup proportional shared mapping snapshot'},
    estimated:{weight_bytes:{value:128,method:'fixture expert byte sum only'}}};
  const g=report(importCapture(r,[e],r.correctness)).groups[0];assert.equal(g.metrics.weight_bytes.category,'estimated');
  assert.equal(g.metrics.rss_bytes.p50,4096);assert.equal(g.metrics.pss_bytes.p50,2048);
  assert.throws(()=>importCapture(r,[{...e,methods:{}}],r.correctness));
  assert.throws(()=>importCapture(r,[{...e,estimated:{launches:{value:1,method:'fake'}}}],r.correctness));
});
test('fabricated faster fixture cannot become a performance decision',()=>{
  const a=fixture(),b=fixture();for(const s of b.samples) {s.metrics.total_ns.value=10;s.metrics.decode_ns.value=10;s.metrics.native_completion_ns.value=1;s.metrics.host_record_ns.value=1;}
  assert.ok(compareRuns(a,b).groups.every(g=>g.decision==='NONE_FIXTURE'));
});
