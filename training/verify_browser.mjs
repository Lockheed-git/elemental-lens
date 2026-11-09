import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFile, mkdir} from 'node:fs/promises';
import {extname, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {deflateSync} from 'node:zlib';

const pw=process.env.ELEMENTAL_PLAYWRIGHT
  ? await import(pathToFileURL(process.env.ELEMENTAL_PLAYWRIGHT))
  : await import('playwright');
const folder=fileURLToPath(new URL('../site/',import.meta.url));
const types={'.mjs':'text/javascript','.html':'text/html; charset=utf-8','.css':'text/css','.svg':'image/svg+xml'};
const server=createServer(async (req,res)=>{
  const path=new URL(req.url,'http://localhost').pathname;
  const filename=resolve(folder,decodeURIComponent(path==='/lab/'?'index.html':path.slice(5)));
  if (!filename.startsWith(resolve(folder)+sep)) { res.writeHead(404).end(); return; }
  try {
    const data=await readFile(filename);
    res.writeHead(200,{'Content-Type':types[extname(filename)]||'application/octet-stream'});
    res.end(data);
  } catch {
    res.writeHead(404).end();
  }
});
const pngSignature=Buffer.from([137,80,78,71,13,10,26,10]);
function crc32(buf) {
  let crc=-1;
  for (const val of buf) {
    crc^=val;
    for (let n=0;n<8;n++) crc=(crc>>>1)^((crc&1) ? 0xEDB88320 : 0);
  }
  return (~crc)>>>0;
}
function chunk(name,data) {
  const body=Buffer.concat([Buffer.from(name),data]);
  const header=Buffer.alloc(4), footer=Buffer.alloc(4);
  header.writeUInt32BE(data.length);
  footer.writeUInt32BE(crc32(body));
  return Buffer.concat([header,body,footer]);
}
function fixture() {
  const w=320,h=200,raw=Buffer.alloc(h*(w*3+1)),ihdr=Buffer.alloc(13);
  ihdr.writeUInt32BE(w); ihdr.writeUInt32BE(h,4);
  ihdr[8]=8; ihdr[9]=2;
  for (let y=0;y<h;y++) {
    for (let x=0;x<w;x++) {
      const wave=Math.sin(x*.032)*22+85, i=y*(w*3+1)+1+x*3;
      const n=y>wave ? 220-((y-wave)*.6) : 20+(x/w)*30;
      raw[i]=n; raw[i+1]=n*.92; raw[i+2]=n*.76;
    }
  }
  return Buffer.concat([pngSignature,chunk('IHDR',ihdr),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]);
}
const output=fileURLToPath(new URL('./runs/browser/',import.meta.url));
await mkdir(output,{recursive:true});
await new Promise((done,fail)=>server.listen(0,'127.0.0.1',done).once('error',fail));
const url=`http://127.0.0.1:${server.address().port}/lab/`;
let browser;
try {
  browser=await pw.chromium.launch({
    executablePath:process.env.ELEMENTAL_CHROMIUM || undefined,headless:true,
    args:['--no-first-run'],
  });
  const page=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});
  const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>requests.push({url:r.url(),method:r.method()}));
  await page.goto(process.env.ELEMENTAL_TEST_URL || url);
  const get=(id)=>page.locator('#'+id);
  const complete=async name=> {
    await page.waitForFunction(n=>document.getElementById('status').textContent.startsWith(n+'重构完成'),name);
    assert.equal(await get('save').isEnabled(),true);
  };
  await complete('火焰');
  await page.screenshot({path:resolve(output,'desktop.png'),fullPage:true});
  const canvas=()=>get('result').evaluate(el=>el.toDataURL());
  const modes=[];
  modes.push(await canvas());
  for (const [style,name] of [['wave','海浪'],['cloud','云']]) {
    await page.locator(`input[value="${style}"]`).check();
    await complete(name);
    modes.push(await canvas());
    await get('result').screenshot({path:resolve(output,`${style}.png`)});
  }
  assert.equal(new Set(modes).size,3);
  await get('file').setInputFiles({name:'structure-fixture.png',mimeType:'image/png',buffer:fixture()});
  await page.waitForFunction(()=>document.getElementById('filename').textContent==='structure-fixture.png');
  await complete('云');
  assert.deepEqual(await get('result').evaluate(el=>[el.width,el.height]),[192,120]);
  const before=await canvas();
  await get('invert').check();
  await page.waitForFunction(previous=>document.getElementById('result').toDataURL()!==previous,before);
  await complete('云');
  assert.notEqual(await canvas(),before);
  const download=page.waitForEvent('download');
  await get('save').click();
  const file=await download;
  assert.equal(file.suggestedFilename(),'structure-fixture-cloud.png');
  const saved=await readFile(await file.path());
  assert.ok(saved.subarray(0,8).equals(pngSignature));
  assert.equal(saved.readUInt32BE(16),192);
  assert.equal(saved.readUInt32BE(20),120);
  const initialOrigin=new URL(process.env.ELEMENTAL_TEST_URL || url).origin;
  assert.ok(requests.every(r=>r.method==='GET' && (!/^https?:/.test(r.url) || new URL(r.url).origin===initialOrigin)));
  assert.deepEqual(errors,[]);
  await page.setViewportSize({width:390,height:844});
  await get('sample').click();
  await page.waitForFunction(()=>document.getElementById('filename').textContent==='手形示例');
  await complete('云');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:resolve(output,'mobile.png'),fullPage:true});
  console.log('Browser checked all 3 models, file upload, aspect ratio, invert, PNG download, local-only requests, and mobile width.');
} finally {
  await browser?.close();
  server.close();
}
