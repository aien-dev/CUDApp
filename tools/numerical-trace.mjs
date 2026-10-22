// SPDX-License-Identifier: MIT
// Host artifact comparison only. No inference, networking, or device access.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
export const CONTRACT='cudapp.numeric.v1';
export const SCHEMA='cudapp.trace.v1';
export function requireThat(ok, message) { if (!ok) throw Error(message); }
export function sha(value) { return typeof value==='string' && /^[a-f0-9]{64}$/.test(value); }
export function digest(bytes) { return createHash('sha256').update(bytes).digest('hex'); }
export function integer(n, lo=0) { return Number.isSafeInteger(n) && n>=lo; }
export function finite(n) { return typeof n==='number' && Number.isFinite(n); }
export function equal(a,b) { return JSON.stringify(a)===JSON.stringify(b); }
export function f32Array(a) {
  return Array.isArray(a) && a.length>0 && a.every(x=>finite(x) && Object.is(Math.fround(x),x));
}
export function argmax(a) { let k=0; for(let i=1;i<a.length;i++) if(a[i]>a[k]) k=i; return k; }
export function validateTrace(t) {
  requireThat(t?.schema===SCHEMA && t.contract===CONTRACT,'trace schema/contract');
  requireThat(['fixture','model'].includes(t.kind),'trace kind');
  for(const k of ['model_sha256','prompt_sha256','config_sha256','binary_sha256']) requireThat(sha(t[k]),`trace pin ${k}`);
  requireThat(/^[a-f0-9]{40}$/.test(t.source_sha),'trace source_sha');
  requireThat(t.precision==='fp8-e4m3-bf16-f32' && integer(t.context_limit,1) && integer(t.seed),'trace precision/context');
  requireThat(Array.isArray(t.rows) && t.rows.length>0,'trace rows');
  let prev;
  for(const r of t.rows) {
    requireThat(integer(r.turn) && integer(r.position) && integer(r.context,1) && r.context<=t.context_limit && r.position<r.context,'trace coordinates');
    requireThat(!prev || (r.turn===prev.turn && r.position===prev.position+1 && r.context===prev.context+1) ||
      (r.turn===prev.turn+1 && r.position>prev.position && r.context>prev.context),'trace order/context continuation');
    requireThat(sha(r.input_sha256) && f32Array(r.logits),'trace input/logits');
    requireThat(integer(r.token) && r.token===argmax(r.logits),'trace token/lowest-index tie');
    requireThat(Array.isArray(r.router_ids) && r.router_ids.length>0 && r.router_ids.every(x=>integer(x)), 'trace router IDs');
    requireThat(r.stages && typeof r.stages==='object' && !Array.isArray(r.stages) && Object.keys(r.stages).length>0,'trace stages');
    for(const [k,v] of Object.entries(r.stages)) requireThat(/^[a-z][a-z0-9_.-]*$/.test(k) && f32Array(v),`trace stage ${k}`);
    if(prev) requireThat(r.logits.length===prev.logits.length,'trace vocab changed');
    prev=r;
  }
  return t;
}
export function compareTraces(a,b,policy={mode:'bitwise'}) {
  validateTrace(a); validateTrace(b);
  requireThat(['bitwise','repeatable','experimental-tolerance'].includes(policy.mode),'comparison mode');
  const tolerance=policy.mode==='experimental-tolerance';
  if(tolerance) requireThat(finite(policy.atol) && finite(policy.rtol) && policy.atol>=0 && policy.rtol>=0 &&
    typeof policy.justification==='string' && policy.justification.trim().length>=12,'explicit experimental tolerance justification');
  else requireThat((policy.atol??0)===0 && (policy.rtol??0)===0,'exact mode budgets');
  for(const k of ['contract','kind','model_sha256','prompt_sha256','config_sha256','precision','context_limit','seed']) requireThat(a[k]===b[k],`trace mismatch ${k}`);
  if(policy.mode==='repeatable') for(const k of ['source_sha','binary_sha256']) requireThat(a[k]===b[k],`repeatability pin ${k}`);
  requireThat(a.rows.length===b.rows.length,'trace row count');
  let elements=0;
  const arrays=(x,y,label)=>{
    requireThat(x.length===y.length,`${label} length`);
    for(let i=0;i<x.length;i++) {
      const budget=tolerance ? policy.atol+policy.rtol*Math.abs(x[i]) : 0;
      requireThat(Number.isFinite(budget),'tolerance overflow');
      requireThat(tolerance ? Math.abs(x[i]-y[i])<=budget : Object.is(x[i],y[i]),`${label}[${i}] mismatch`); elements++;
    }
  };
  a.rows.forEach((x,i)=>{
    const y=b.rows[i];
    for(const k of ['turn','position','context','input_sha256','token']) requireThat(x[k]===y[k],`row ${i} ${k}`);
    requireThat(equal(x.router_ids,y.router_ids),'router IDs mismatch');
    requireThat(equal(Object.keys(x.stages).sort(),Object.keys(y.stages).sort()),'stage set mismatch');
    arrays(x.logits,y.logits,`row ${i} logits`);
    for(const k of Object.keys(x.stages)) arrays(x.stages[k],y.stages[k],`row ${i} stage ${k}`);
  });
  return {contract:CONTRACT,status:'PASS',kind:a.kind,mode:policy.mode,rows:a.rows.length,elements,
    model_validation:a.kind==='fixture'?'NOT_RUN':'TRACE_MATCH_ONLY',model_quality:'NOT_RUN'};
}
if(process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  try {
    const [a,b,p,...rest]=process.argv.slice(2);
    requireThat(a && b && !rest.length,'usage: node tools/numerical-trace.mjs reference.json candidate.json [policy.json]');
    const read=x=>JSON.parse(fs.readFileSync(x,'utf8'));
    console.log(JSON.stringify(compareTraces(read(a),read(b),p?read(p):undefined),null,2));
  } catch(e) { console.error(`FAIL: ${e.message}`); process.exitCode=1; }
}
