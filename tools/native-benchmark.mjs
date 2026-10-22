// SPDX-License-Identifier: MIT
// Native host receipt tooling. Never launches a workload or connects to a server.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {CONTRACT,requireThat as need,sha,digest,integer,finite,equal,validateTrace,compareTraces} from './numerical-trace.mjs';
export const SCHEMA='cudapp.benchmark.v1';
// JSON numbers can carry -0; JSON.stringify would erase its exact f32 sign.
export function encodeJSON(x) {
  if(typeof x==='number') {need(finite(x),'nonfinite JSON number');return Object.is(x,-0)?'-0':JSON.stringify(x);}
  if(x===null || typeof x==='string' || typeof x==='boolean')return JSON.stringify(x);
  if(Array.isArray(x))return '['+x.map(v=>v===undefined?'null':encodeJSON(v)).join(',')+']';
  need(x && typeof x==='object','unsupported JSON value');
  return '{'+Object.keys(x).filter(k=>x[k]!==undefined).map(k=>JSON.stringify(k)+':'+encodeJSON(x[k])).join(',')+'}';
}
const measuredKeys=['setup_ns','ttft_ns','prefill_ns','decode_ns','total_ns','host_record_ns','host_submit_ns',
  'native_completion_ns','gpu_ns','launches','submits','barriers','allocations','allocation_bytes','rss_bytes','pss_bytes','driver_process_bytes'];
const estimateKeys=['weight_bytes','kv_bytes','intermediate_bytes'];
function observation(v,label) {
  need(v && ['measured','estimated','unknown'].includes(v.category),`${label} category`);
  if(v.category==='unknown') need(v.value===null && typeof v.reason==='string' && v.reason.length>0,`${label} unknown reason`);
  else need(finite(v.value) && v.value>=0 && typeof v.method==='string' && v.method.length>0,`${label} value/method`);
}
export const unknown=reason=>({category:'unknown',value:null,reason});
export const measured=(value,method)=>({category:'measured',value,method});
function pins(m) {
  need(m?.schema===SCHEMA && m.contract===CONTRACT,'benchmark schema/contract');
  need(['fixture','model'].includes(m.kind),'benchmark kind');
  need(/^[a-f0-9]{40}$/.test(m.source_sha),'source pin');
  for(const k of ['model_sha256','prompt_sha256','config_sha256','binary_sha256','driver_sha256','policy_sha256']) need(sha(m[k]),`pin ${k}`);
  need(m.precision==='fp8-e4m3-bf16-f32' && typeof m.chip==='string' && m.chip.length>0,'precision/chip');
  need(m.backend==='native-graph' || m.backend==='graph-cuda','backend');
  need(m.policy && digest(JSON.stringify(m.policy))===m.policy_sha256,'policy hash');
  need(integer(m.supported_context_limit,1) && Array.isArray(m.workloads) && m.workloads.length>0,'declared supported workload sweep');
  const shapeKeys=new Set();
  for(const w of m.workloads) {workload(w); need(w.batch===1 && w.context<=m.supported_context_limit,'unsupported sweep shape'); const k=JSON.stringify(w);need(!shapeKeys.has(k),'duplicate sweep shape');shapeKeys.add(k);}
  need(m.conditions && sha(m.conditions.thermal_clock_protocol_sha256) && typeof m.conditions.description==='string' && m.conditions.description.length>0,'thermal/clock protocol');
}
function workload(w) {
  need(w && integer(w.batch,1) && integer(w.context,1) && integer(w.input_tokens,1) && integer(w.output_tokens,1) && integer(w.seed) && w.input_tokens+w.output_tokens-1<=w.context,'workload/seed');
}
export function validateRun(r) {
  pins(r);
  need(r.correctness && r.correctness.reference && r.correctness.candidate,'complete-model trace artifacts required');
  for(const t of [r.correctness.reference,r.correctness.candidate]) {
    validateTrace(t);
    for(const k of ['contract','kind','model_sha256','prompt_sha256','config_sha256','precision']) need(t[k]===r[k],`correctness pin ${k}`);
  }
  for(const k of ['source_sha','binary_sha256']) need(r.correctness.candidate[k]===r[k],`candidate correctness pin ${k}`);
  const correctness=compareTraces(r.correctness.reference,r.correctness.candidate,r.policy);
  need(Array.isArray(r.samples) && r.samples.length>0,'empty samples');
  const keys=new Set(), sessionSeeds=new Map();
  for(const s of r.samples) {
    need(typeof s.session==='string' && /^[A-Za-z0-9_.-]+$/.test(s.session) && integer(s.ordinal),'session/ordinal');
    need(['cold','warm'].includes(s.phase) && typeof s.stamps==='boolean','phase/stamps');
    workload(s.workload);
    need(r.workloads.some(w=>equal(w,s.workload)),'sample outside declared sweep');
    need(s.workload.batch===1,'v1 native graph batch support is one');
    need(s.workload.seed===r.correctness.candidate.seed && s.workload.seed===r.correctness.reference.seed,'trace/workload seed');
    need(r.correctness.candidate.rows.some(row=>row.context===s.workload.context),'workload context absent from correctness trace');
    need(s.workload.output_tokens<=r.correctness.candidate.rows.length,'insufficient correctness tokens');
    for(let c=s.workload.context-s.workload.output_tokens+1;c<=s.workload.context;c++)
      need(r.correctness.candidate.rows.some(row=>row.context===c),'missing output context in correctness trace');
    need(s.workload.context<=r.correctness.candidate.context_limit,'workload context unsupported by trace contract');
    // One imported sample is one independent session/shape/phase/instrumentation observation.
    const key=JSON.stringify([s.session,s.phase,s.stamps,s.workload,s.ordinal]); need(!keys.has(key),'duplicate sample'); keys.add(key);
    const sk=JSON.stringify([s.session,s.phase,s.stamps,s.workload]); need(!sessionSeeds.has(sk),'correlated samples in independent-session group'); sessionSeeds.set(sk,true);
    if(s.request_clock!==undefined && s.request_clock!==null) {
      need(Array.isArray(s.request_clock) && s.request_clock.length===4 && s.request_clock.every(t=>typeof t==='string' && /^\d+$/.test(t) && BigInt(t)<2n**64n-1n),'request clock');
      const [start,prefill,first,end]=s.request_clock.map(BigInt);
      need(start<=prefill && prefill<=first && first<=end && prefill<end,'request timestamp order');
      for(const [k,v] of Object.entries({ttft_ns:first-start,prefill_ns:prefill-start,decode_ns:end-prefill,total_ns:end-start}))
        need(Number.isSafeInteger(Number(v)) && s.metrics[k]?.category==='measured' && s.metrics[k].value===Number(v),'request metrics/clock mismatch');
    }
    need(Array.isArray(s.kinds) && s.kinds.length===(s.stamps ? s.timestamps?.length-1 : 0) && s.kinds.every(k=>integer(k) && k<12),'timeline kinds');
    need(s.metrics && typeof s.metrics==='object','metrics');
    for(const k of measuredKeys) {
      observation(s.metrics[k],k); need(s.metrics[k].category!=='estimated',`counter cannot be estimated ${k}`);
      if(s.metrics[k].value!==null) need(Number.isSafeInteger(s.metrics[k].value),`integer counter ${k}`);
    }
    for(const k of estimateKeys) { observation(s.metrics[k],k); need(s.metrics[k].category!=='measured',`traffic estimate label ${k}`); }
    need(s.metrics.host_record_ns.value===null || s.metrics.host_record_ns.value<=s.metrics.total_ns.value,'record exceeds total');
    need(s.metrics.native_completion_ns.value===null || s.metrics.native_completion_ns.value<=s.metrics.total_ns.value,'completion exceeds total');
    need(s.metrics.submits.value===null || s.metrics.launches.value===null || s.metrics.submits.value<=s.metrics.launches.value,'submits exceed launches');
    need(s.metrics.total_ns.category==='measured' && s.metrics.total_ns.value>0,'measured positive total required');
    need(s.metrics.decode_ns.value===null || s.metrics.decode_ns.value>0,'decode duration positive');
    if(s.metrics.ttft_ns.value!==null) need(s.metrics.ttft_ns.value<=s.metrics.total_ns.value,'TTFT exceeds total');
    if(s.metrics.prefill_ns.value!==null && s.metrics.ttft_ns.value!==null) need(s.metrics.prefill_ns.value<=s.metrics.ttft_ns.value,'prefill exceeds TTFT');
    if(s.metrics.decode_ns.value!==null) need(s.metrics.decode_ns.value<=s.metrics.total_ns.value,'decode exceeds total');
    need(Array.isArray(s.timestamps),'timestamps array');
    need(s.timestamps.length===(s.stamps ? s.metrics.launches.value+1 : 0),'stamp/launch count');
    let previous;
    for(const t of s.timestamps) {
      need(typeof t==='string' && /^\d+$/.test(t) && BigInt(t)<2n**64n-1n,'timestamp uint64/poison');
      need(previous===undefined || BigInt(t)>previous,'timestamp order'); previous=BigInt(t);
    }
    if(s.stamps) {
      const span=Number(BigInt(s.timestamps.at(-1))-BigInt(s.timestamps[0]));
      need(Number.isSafeInteger(span) && s.metrics.gpu_ns.category==='measured' && s.metrics.gpu_ns.value===span,'GPU span disagrees with timestamps');
    } else need(s.metrics.gpu_ns.category==='unknown','GPU time requires device timestamps');
    need(typeof s.correctness_sha256==='string' && s.correctness_sha256===digest(encodeJSON(r.correctness.candidate)),'sample correctness binding');
  }
  return correctness;
}
export function quantile(a,q) {
  need(a.length>0 && a.every(finite),'quantile data'); const v=[...a].sort((x,y)=>x-y);
  // Nearest rank on independent raw session observations. No interpolation.
  return v[Math.max(0,Math.ceil(q*v.length)-1)];
}
function groupKey(s,stamps=true) { return JSON.stringify([s.phase,...(stamps?[s.stamps]:[]),s.workload]); }
function summary(obs) {
  const categories=[...new Set(obs.map(x=>x.category))];
  if(categories.length!==1 || categories[0]==='unknown') return {category:'unknown',value:null,reason:'missing or mixed observations',known_samples:obs.filter(x=>x.value!==null).length};
  const values=obs.map(x=>x.value);
  return {category:categories[0],n:values.length,p50:quantile(values,.5),p95:quantile(values,.95),min:Math.min(...values),max:Math.max(...values)};
}
function identity(r) { return Object.fromEntries(['source_sha','model_sha256','prompt_sha256','config_sha256','binary_sha256','driver_sha256','policy_sha256','contract','precision','chip','backend','conditions','workloads','supported_context_limit'].map(k=>[k,r[k]])); }
export function report(r) {
  const correctness=validateRun(r), groups=new Map();
  for(const s of r.samples) {const key=groupKey(s); if(!groups.has(key))groups.set(key,[]); groups.get(key).push(s);}
  const result=[];
  for(const [key,ss] of groups) {
    const metrics=Object.fromEntries([...measuredKeys,...estimateKeys].map(k=>[k,summary(ss.map(s=>s.metrics[k]))]));
    const stageKinds=[...new Set(ss.flatMap(s=>s.kinds))];
    const stages=Object.fromEntries(stageKinds.map(k=>[k,summary(ss.map(s=>{
      let sum=0n; for(let i=0;i<s.kinds.length;i++) if(s.kinds[i]===k) sum+=BigInt(s.timestamps[i+1])-BigInt(s.timestamps[i]);
      return measured(Number(sum),'instrumented timeline spans including launch barrier/report cost');
    }))]));
    metrics.throughput_tokens_s=summary(ss.map(s=>s.metrics.decode_ns.value===null?unknown('decode duration unavailable'):
      measured(s.workload.output_tokens*1e9/s.metrics.decode_ns.value,'output tokens / decode duration; independent session')));
    result.push({group:JSON.parse(key),sessions:ss.map(s=>s.session),metrics,stage_ns:stages});
  }
  const overhead=[];
  for(const off of r.samples.filter(s=>!s.stamps)) {
    const on=r.samples.find(s=>s.stamps && s.session===off.session && s.ordinal===off.ordinal && groupKey(s,false)===groupKey(off,false));
    if(on) overhead.push({session:off.session,phase:off.phase,workload:off.workload,
      total_delta_ns:on.metrics.total_ns.value-off.metrics.total_ns.value,
      fraction:(on.metrics.total_ns.value-off.metrics.total_ns.value)/off.metrics.total_ns.value});
  }
  return {schema:SCHEMA,contract:CONTRACT,kind:r.kind,identity:identity(r),run_sha256:digest(encodeJSON(r)),raw_samples:r.samples,groups:result,
    instrumentation:overhead.length?{category:'measured',pairs:overhead}:{category:'unknown',value:null,reason:'paired stamps off/on sessions absent'},
    correctness,model_quality:'NOT_RUN',hardware:r.kind==='fixture'?'NOT_RUN':'IMPORTED_UNVERIFIED',
    graph_cuda_comparison:'NOT_RUN',performance_claim:r.kind==='fixture'?'NONE_FIXTURE':'RAW_IMPORTED_ONLY',
    bandwidth_reference:'No physical mixed-traffic ceiling inferred from access-pattern-specific probes.'};
}
export function compareRuns(a,b) {
  validateRun(a); validateRun(b);
  for(const k of ['contract','kind','model_sha256','prompt_sha256','config_sha256','driver_sha256','precision','chip','policy_sha256']) need(a[k]===b[k],`paired pin mismatch ${k}`);
  need(equal(a.conditions,b.conditions),'paired thermal/clock conditions');
  need(a.samples.length===b.samples.length,'unequal sample counts');
  // Baseline and candidate must preserve complete-model outputs as well as their own references.
  compareTraces(a.correctness.candidate,b.correctness.candidate,a.policy);
  const key=s=>JSON.stringify([s.session,s.ordinal,s.phase,s.stamps,s.workload]);
  const candidate=new Map(b.samples.map(s=>[key(s),s]));
  const pairs=a.samples.map(s=>{
    const c=candidate.get(key(s)); need(c,'unpaired workloads/seeds/session');
    const av=s.metrics.total_ns.value,bv=c.metrics.total_ns.value;
    return {session:s.session,phase:s.phase,stamps:s.stamps,workload:s.workload,baseline_ns:av,candidate_ns:bv,delta_ns:bv-av,ratio:av/bv};
  });
  const groups=new Map();
  for(let i=0;i<a.samples.length;i++) {const k=groupKey(a.samples[i]);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(pairs[i]);}
  return {schema:SCHEMA,contract:CONTRACT,kind:a.kind,baseline_source:a.source_sha,candidate_source:b.source_sha,
    baseline_identity:identity(a),candidate_identity:identity(b),baseline_run_sha256:digest(encodeJSON(a)),candidate_run_sha256:digest(encodeJSON(b)),
    pairs,groups:[...groups].map(([group,p])=>({group:JSON.parse(group),n:p.length,ratio_p50:quantile(p.map(x=>x.ratio),.5),
      ratio_p95:quantile(p.map(x=>x.ratio),.95),unfavorable_pairs:p.filter(x=>x.delta_ns>0).length,
      decision:a.kind==='fixture'?'NONE_FIXTURE':p.length<3?'INSUFFICIENT_SESSIONS':'REVIEW_REQUIRED'})),
    correctness:'TRACE_MATCH_ONLY',model_quality:'NOT_RUN',graph_cuda_comparison:a.backend!==b.backend?'IMPORTED_UNVERIFIED':'NOT_RUN'};
}
export function readable(r) {
  return [`${r.schema} contract=${r.contract} kind=${r.kind}`,`model quality: ${r.model_quality}`,
    ...(r.groups??[]).map(g=>g.metrics?`${JSON.stringify(g.group)}: n=${g.sessions.length} total_ns p50=${g.metrics.total_ns.p50} p95=${g.metrics.total_ns.p95}`:
      `${JSON.stringify(g.group)}: n=${g.n} paired ratio p50=${g.ratio_p50} unfavorable=${g.unfavorable_pairs} decision=${g.decision}`),
    r.instrumentation?`instrumentation: ${r.instrumentation.category}`:'',`graph-enabled CUDA comparison: ${r.graph_cuda_comparison}`].filter(Boolean).join('\n');
}
export function importCapture(manifest,events,correctness) {
  pins(manifest); need(events.length>0,'empty capture');
  const samples=events.map(e=>{
    need(e.schema==='cudapp.graph-sample.v1' && e.contract===CONTRACT,'capture schema/contract');
    need(integer(e.token) && correctness.candidate.rows.some(r=>r.context===e.workload?.context && r.token===e.token),'capture token/context mismatch');
    need(e.measured && typeof e.measured==='object','capture measurements');
    const metrics=Object.fromEntries([...measuredKeys,...estimateKeys].map(k=>[k,unknown('not exposed by graph adapter')]));
    for(const [key,value] of Object.entries(e.measured)) {
      need(measuredKeys.includes(key),'unsupported captured measurement');
      const processMemory=['rss_bytes','pss_bytes','driver_process_bytes'].includes(key);
      if(processMemory) need(typeof e.methods?.[key]==='string' && e.methods[key].trim().length>0,'process-memory scope/method required');
      metrics[key]=measured(value,e.methods?.[key]??`native graph adapter ${key}`);
    }
    for(const [key,observationValue] of Object.entries(e.estimated??{})) {
      need(estimateKeys.includes(key),'unsupported estimate');
      metrics[key]={category:'estimated',value:observationValue.value,method:observationValue.method};
      observation(metrics[key],key);need(Number.isSafeInteger(metrics[key].value),'estimated bytes integer');
    }
    return {session:e.session,ordinal:e.ordinal,phase:e.phase,stamps:e.stamps,workload:e.workload,
      metrics,timestamps:e.timestamps,kinds:e.kinds,request_clock:e.request_clock??null,correctness_sha256:digest(encodeJSON(correctness.candidate))};
  });
  const run={...manifest,correctness,samples}; validateRun(run); return run;
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  try {
    const [cmd,...args]=process.argv.slice(2),read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
    let out;
    if(cmd==='report' && (args.length===1 || (args.length===2 && args[1]==='--text'))) out=report(read(args[0]));
    else if(cmd==='compare' && (args.length===2 || (args.length===3 && args[2]==='--text'))) out=compareRuns(read(args[0]),read(args[1]));
    else if(cmd==='import' && args.length===4) {
      const lines=fs.readFileSync(args[1],'utf8').trim(); need(lines,'empty capture file');
      out=importCapture(read(args[0]),lines.split('\n').map(x=>JSON.parse(x)),read(args[2]));
      fs.writeFileSync(args[3],encodeJSON(out)+'\n',{flag:'wx'}); console.log(`imported ${out.samples.length} samples`); process.exit(0);
    } else if((cmd==='prepare' || cmd==='capture') && args.length===2) {
      out=read(args[0]); pins(out);
      out={...out,status:'NOT_RUN',collection:'Link src/omega_native_benchmark.c into the approved graph workload; call omega_benchmark_graph_token; save JSONL; import with reference/candidate traces.',
        command:['node','tools/native-benchmark.mjs','import',path.resolve(args[0]),'capture.jsonl','correctness.json','run.json']};
      fs.writeFileSync(args[1],encodeJSON(out)+'\n',{flag:'wx'}); console.log('prepared NOT_RUN manifest; no workload executed'); process.exit(0);
    } else throw Error('usage: native-benchmark.mjs prepare manifest.json plan.json | import manifest.json capture.jsonl correctness.json run.json | report run.json [--text] | compare baseline.json candidate.json [--text]');
    need(!args.at(-1)?.startsWith('--') || args.at(-1)==='--text','unknown option');
    console.log(args.at(-1)==='--text'?readable(out):encodeJSON(out));
  } catch(e) { console.error(`FAIL: ${e.message}`); process.exitCode=1; }
}
