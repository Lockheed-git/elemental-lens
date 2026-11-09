import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {readModel,makeCue,predictVisible,renderPixels,STYLES} from '../site/engine.mjs';

const root=new URL('../site/',import.meta.url);
const w=192,h=128,n=w*h;
const rgba=new Uint8ClampedArray(n*4);
for (let i=0;i<n;i++) {
  const x=i%w,y=Math.floor(i/w);
  const tone=35+195*Math.exp(-((x-90-y*.12)**2/1500+(y-60)**2/2200));
  rgba.set([tone,tone*.94,tone*.83,255],4*i);
}
const a=makeCue(rgba,w,h,65,false);
const b=makeCue(rgba,w,h,65,true);
const rough=makeCue(rgba,w,h,0,false);
assert.ok(a.every((value,i)=>Math.abs(value+b[i])<1e-6));
assert.ok(a.some((value,i)=>Math.abs(value-rough[i])>.05));

function distance(a,b) {
  let sum=0;
  for (let i=0;i<a.length;i++) sum+=Math.abs(a[i]-b[i]);
  return sum/a.length;
}

const outputs=[];
for (const style of STYLES) {
  const file=await readFile(fileURLToPath(new URL(`models/${style}.eln`,root)));
  const model=readModel(file.buffer.slice(file.byteOffset,file.byteOffset+file.byteLength));
  const start=performance.now();
  const prediction=predictVisible(a,w,h,model);
  const ms=Math.round(performance.now()-start);
  assert.ok(prediction.every(v=>Number.isFinite(v) && Math.abs(v)<=1));
  const sensitivity=distance(prediction,predictVisible(b,w,h,model));
  assert.ok(sensitivity>.05,`${style} ignored input`);
  const pixels=renderPixels(prediction,w,h);
  assert.equal(pixels.length,n*4);
  assert.ok(pixels.every((value,i)=>i%4!==3 || value===255));
  outputs.push(prediction);
  console.log(`${style}: ${w}×${h} ${ms} ms, input sensitivity ${sensitivity.toFixed(3)}`);
}
for (let i=0;i<outputs.length;i++) {
  for(let j=0;j<i;j++) assert.ok(distance(outputs[i],outputs[j])>.08,'styles are too similar');
}
