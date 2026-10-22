// SPDX-License-Identifier: MIT
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {validateTrace,compareTraces,CONTRACT,SCHEMA} from '../tools/numerical-trace.mjs';
export function fixture() {
  return {schema:SCHEMA,contract:CONTRACT,kind:'fixture',model_sha256:'a'.repeat(64),prompt_sha256:'b'.repeat(64),
    config_sha256:'c'.repeat(64),binary_sha256:'d'.repeat(64),source_sha:'e'.repeat(40),precision:'fp8-e4m3-bf16-f32',context_limit:8,seed:7,
    rows:[{turn:0,position:0,context:1,input_sha256:'f'.repeat(64),router_ids:[0,1,2,3,4,5,6,7],token:1,logits:[-1,2,2],stages:{residual:[1,2],hidden:[1,2]}},
      {turn:0,position:1,context:2,input_sha256:'f'.repeat(64),router_ids:[0,1,2,3,4,5,6,7],token:1,logits:[-1,2,2],stages:{residual:[1,2],hidden:[1,2]}},
      {turn:1,position:7,context:8,input_sha256:'f'.repeat(64),router_ids:[0,1,2,3,4,5,6,7],token:1,logits:[-1,2,2],stages:{residual:[1,2],hidden:[1,2]}}]};
}
test('whole-token, multiturn, boundary-context fixtures match',()=>{
  const r=compareTraces(fixture(),fixture()); assert.equal(r.rows,3); assert.equal(r.model_validation,'NOT_RUN');
});
for(const [name,change] of Object.entries({
  scale:t=>t.rows[0].stages.hidden[0]=2**64,
  'missing expert':t=>t.rows[0].stages.residual[0]=0,
  rounding:t=>t.rows[0].logits[0]=Math.fround(-1+2**-24),
  'wrong routing tie':t=>t.rows[0].router_ids.reverse(),
  'argmax tie':t=>t.rows[0].token=2,
  'model pin':t=>t.model_sha256='0'.repeat(64),
  'prompt pin':t=>t.prompt_sha256='0'.repeat(64),
  contract:t=>t.contract='other',
  'missing stage':t=>delete t.rows[0].stages.hidden,
  'missing turn':t=>t.rows.splice(2,1),
  'context overflow':t=>t.rows[2].context=9,
  'out of order':t=>t.rows.reverse(),
  'nonfinite':t=>t.rows[0].logits[0]=Infinity,
  'empty':t=>t.rows=[],
  'non-f32':t=>t.rows[0].logits[0]=0.1,
  'missing pin':t=>delete t.binary_sha256
})) test(`refuses ${name}`,()=>{ const t=fixture(); change(t); assert.throws(()=>compareTraces(fixture(),t)); });
test('repeatability requires identical executable pins',()=>{
  const t=fixture(); t.source_sha='0'.repeat(40);
  assert.throws(()=>compareTraces(fixture(),t,{mode:'repeatable'}));
  assert.equal(compareTraces(fixture(),t).status,'PASS');
});
test('signed zeros exact; tolerance explicit and experimental',()=>{
  const a=fixture(),b=fixture(); a.rows[0].stages.hidden=[0]; b.rows[0].stages.hidden=[-0];
  assert.throws(()=>compareTraces(a,b));
  const policy={mode:'experimental-tolerance',atol:0,rtol:0,justification:'Fixture signed zero control only'};
  assert.equal(compareTraces(a,b,policy).model_quality,'NOT_RUN');
  assert.throws(()=>compareTraces(a,b,{mode:'experimental-tolerance',atol:1,rtol:0}));
  assert.throws(()=>compareTraces(a,b,{mode:'bitwise',atol:1}));
  assert.throws(()=>validateTrace({}));
});
