const $ = id => document.getElementById(id);
const names = {fire:'火焰',wave:'海浪',cloud:'云'};
const original=$('original'), result=$('result'), detail=$('detail'), invert=$('invert');
const input=document.createElement('canvas');
const inputCtx=input.getContext('2d',{willReadFrequently:true});
const getStyle=()=>document.querySelector('input[name=style]:checked').value;
const MAX_FILE=20*1024*1024, MAX_SIDE=192;
let worker, imageReady=false, serial=0, sourceVersion=0, filename='手形示例', basename='hand';
let last=null, modelPending=false;

function status(text,error=false) {
  $('status').textContent=text;
  $('status').classList.toggle('error',error);
}

function clearOutput() {
  result.getContext('2d').clearRect(0,0,result.width,result.height);
  $('save').disabled=true;
  last=null;
}

function setWorking(working) {
  modelPending=working;
  $('generate').disabled=working || !imageReady;
  $('result-frame').classList.toggle('pending',working);
  $('result-frame').setAttribute('aria-busy',String(working));
}

function getWorker() {
  if (worker) return worker;
  worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
    if (data.id!==serial) return;
    setWorking(false);
    if (data.error) {
      status(data.error+'。请重试。',true);
      return;
    }
    result.width=data.w; result.height=data.h;
    result.getContext('2d').putImageData(new ImageData(data.pixels,data.w,data.h),0,0);
    const name=names[getStyle()];
    $('output-label').textContent=name+'重构';
    $('save').disabled=false;
    last={basename,style:getStyle()};
    status(`${name}重构完成 · ${data.w} × ${data.h} · 本机推理 ${(data.ms/1000).toFixed(2)} 秒`);
  };
  worker.onerror=()=>{
    worker?.terminate();
    worker=null;
    setWorking(false);
    status('浏览器未能启动生成器。请用现代浏览器通过网站地址打开页面。',true);
  };
  return worker;
}

function generate() {
  if (!imageReady) return;
  const id=++serial, style=getStyle(), w=input.width,h=input.height;
  clearOutput();
  setWorking(true);
  $('output-label').textContent=names[style]+'重构';
  status('正在本机生成'+names[style]+'风格…');
  try {
    const rgba=inputCtx.getImageData(0,0,w,h).data;
    getWorker().postMessage({id,style,rgba:rgba.buffer,w,h,detail:Number(detail.value),invert:invert.checked},[rgba.buffer]);
  } catch {
    setWorking(false);
    status('无法读取这张图片。请试用 JPG、PNG 或 WebP。',true);
  }
}

async function decode(url) {
  const img=new Image();
  img.src=url;
  await img.decode();
  if (!img.naturalWidth || !img.naturalHeight) throw Error('图像尺寸无效');
  return img;
}

function acceptImage(img,name,stem) {
  const scale=MAX_SIDE/Math.max(img.naturalWidth,img.naturalHeight);
  const dim=n=>Math.max(8,Math.round(n*scale/4)*4);
  const w=dim(img.naturalWidth), h=dim(img.naturalHeight);
  input.width=w; input.height=h;
  inputCtx.drawImage(img,0,0,w,h);
  const p=Math.min(1,768/Math.max(img.naturalWidth,img.naturalHeight));
  original.width=Math.max(1,Math.round(img.naturalWidth*p));
  original.height=Math.max(1,Math.round(img.naturalHeight*p));
  original.getContext('2d').drawImage(img,0,0,original.width,original.height);
  filename=name; basename=stem;
  $('filename').textContent=filename;
  $('filename').title=filename;
  imageReady=true;
  generate();
}

async function loadSample() {
  const version=++sourceVersion;
  try {
    const img=await decode(new URL('./assets/hand.svg',import.meta.url).href);
    if (version===sourceVersion) acceptImage(img,'手形示例','hand');
  } catch {
    if (version===sourceVersion) status('无法加载示例。你可以选择自己的图片。',true);
  }
}

async function loadFile(file) {
  if (!file) return;
  if (file.size>MAX_FILE) {
    status('这张图超过 20 MB，请换一张更小的图片。',true);
    return;
  }
  if (file.type && !file.type.startsWith('image/')) {
    status('请选择图片文件。',true);
    return;
  }
  const version=++sourceVersion;
  const url=URL.createObjectURL(file);
  try {
    const img=await decode(url);
    if (version===sourceVersion) acceptImage(img,file.name,file.name.replace(/\.[^.]+$/,'') || 'image');
  } catch {
    if (version===sourceVersion) status('无法打开这张图片。请尝试 JPG、PNG 或 WebP。',true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

const pick=$('pick');
pick.addEventListener('click',()=>$('file').click());
$('file').addEventListener('change',e=>{
  loadFile(e.target.files[0]);
  e.target.value='';
});
for (const event of ['dragenter','dragover']) {
  pick.addEventListener(event,e=>{e.preventDefault();pick.classList.add('drag');});
}
pick.addEventListener('dragleave',()=>pick.classList.remove('drag'));
pick.addEventListener('drop',e=>{
  e.preventDefault();
  pick.classList.remove('drag');
  loadFile(e.dataTransfer.files[0]);
});
$('sample').addEventListener('click',loadSample);
$('generate').addEventListener('click',generate);
document.querySelector('.style-picker').addEventListener('change',generate);
detail.addEventListener('input',()=>{$('detail-value').value=detail.value;});
detail.addEventListener('change',generate);
invert.addEventListener('change',generate);
$('save').addEventListener('click',()=>{
  if (!last || modelPending) return;
  const name=`${last.basename}-${last.style}.png`;
  result.toBlob(blob=>{
    if (!blob) return;
    const a=document.createElement('a');
    const url=URL.createObjectURL(blob);
    a.href=url; a.download=name;
    a.click();
    setTimeout(()=>URL.revokeObjectURL(url),3000);
  },'image/png');
});
loadSample();
