// Compact U-Net inference, matching training/train.py. Planar Float32 channels.
const SPEC = [[1,12,3], [12,16,3], [16,24,3], [24,24,3], [40,16,3], [28,12,3], [12,3,1]];
const WEIGHT_COUNT = SPEC.reduce((n, [a,b,k]) => n + b*(a*k*k+1), 0);
export const STYLES = ['fire', 'wave', 'cloud'];

export function readModel(buffer) {
  if (buffer.byteLength !== 8 + 4*WEIGHT_COUNT) throw Error('权重文件长度不匹配');
  const v = new DataView(buffer);
  if (v.getUint32(0) !== 0x454c4e31 || v.getUint32(4, true) !== WEIGHT_COUNT) {
    throw Error('权重文件版本不匹配');
  }
  const values = new Float32Array(WEIGHT_COUNT);
  for (let i=0; i<values.length; i++) {
    values[i] = v.getFloat32(8+4*i, true);
    if (!Number.isFinite(values[i])) throw Error('权重包含无效数据');
  }
  let pos = 0;
  return SPEC.map(([ic,oc,k]) => {
    const n = ic*oc*k*k;
    const weights = values.subarray(pos, pos+n);
    const bias = values.subarray(pos+n, pos+n+oc);
    pos += n+oc;
    return {ic,oc,k,weights,bias};
  });
}

function conv(x, w, h, layer) {
  const {ic,oc,k,weights,bias} = layer;
  const size = w*h;
  const out = new Float32Array(oc*size);
  for (let co=0; co<oc; co++) {
    const dst = out.subarray(co*size,(co+1)*size);
    dst.fill(bias[co]);
    for (let ci=0; ci<ic; ci++) {
      const ib=ci*size, p=(co*ic+ci)*k*k;
      if (k===1) {
        const a=weights[p];
        for (let q=0;q<size;q++) dst[q] += a*x[ib+q];
        continue;
      }
      const a=weights[p], b=weights[p+1], c=weights[p+2], d=weights[p+3], e=weights[p+4];
      const f=weights[p+5], g=weights[p+6], j=weights[p+7], l=weights[p+8];
      for (let row=1;row<h-1;row++) {
        for (let col=1;col<w-1;col++) {
          const q=row*w+col, t=ib+q;
          dst[q] += a*x[t-w-1]+b*x[t-w]+c*x[t-w+1]+d*x[t-1]+e*x[t]+f*x[t+1]+
            g*x[t+w-1]+j*x[t+w]+l*x[t+w+1];
        }
      }
      for (let col=0;col<w;col++) {
        border(0,col);
        border(h-1,col);
      }
      for (let row=1;row<h-1;row++) {
        border(row,0);
        border(row,w-1);
      }
      function border(row,col) {
        let acc=0;
        for (let dy=Math.max(-1,-row);dy<=Math.min(1,h-1-row);dy++) {
          for (let dx=Math.max(-1,-col);dx<=Math.min(1,w-1-col);dx++) {
            acc += x[ib+(row+dy)*w+col+dx]*weights[p+(dy+1)*3+dx+1];
          }
        }
        dst[row*w+col] += acc;
      }
    }
  }
  return out;
}

function activate(x, residual) {
  for (let i=0;i<x.length;i++) {
    const v=residual ? x[i]+residual[i] : x[i];
    x[i]=v<0 ? .1*v : v;
  }
  return x;
}

function pool(x,w,h,channels) {
  const nw=w/2, nh=h/2, size=w*h, ns=nw*nh;
  const out=new Float32Array(ns*channels);
  for (let c=0;c<channels;c++) {
    for (let y=0;y<nh;y++) {
      for (let a=0;a<nw;a++) {
        const p=c*size+y*2*w+a*2;
        out[c*ns+y*nw+a]=(x[p]+x[p+1]+x[p+w]+x[p+w+1])*.25;
      }
    }
  }
  return out;
}

function expandCat(x,skip,w,h,channels) {
  const nw=w*2, nh=h*2, ns=nw*nh, size=w*h;
  const out=new Float32Array(ns*channels+skip.length);
  for (let c=0;c<channels;c++) {
    for (let y=0;y<nh;y++) {
      for (let a=0;a<nw;a++) out[c*ns+y*nw+a]=x[c*size+(y>>1)*w+(a>>1)];
    }
  }
  out.set(skip, ns*channels);
  return out;
}

export function predict(x,w,h,m) {
  if (w<8 || h<8 || w%4 || h%4 || x.length!==w*h) throw Error('输入尺寸无效');
  const a=activate(conv(x,w,h,m[0]));
  const b=activate(conv(pool(a,w,h,12),w/2,h/2,m[1]));
  const c=activate(conv(pool(b,w/2,h/2,16),w/4,h/4,m[2]));
  const d=activate(conv(c,w/4,h/4,m[3]),c);
  const e=activate(conv(expandCat(d,b,w/4,h/4,24),w/2,h/2,m[4]));
  const f=activate(conv(expandCat(e,a,w/2,h/2,16),w,h,m[5]));
  return conv(f,w,h,m[6]).map(Math.tanh);
}

export function predictVisible(x,w,h,model) {
  // Add context around the photo so internal convolution padding does not paint a rim.
  const pad=16, pw=w+pad*2, ph=h+pad*2, area=pw*ph;
  const reflect=(v,n)=>{
    const period=2*n-2, p=((v%period)+period)%period;
    return p<n?p:period-p;
  };
  const expanded=new Float32Array(area);
  const cols=Array.from({length:pw},(_,i)=>reflect(i-pad,w));
  for (let y=0;y<ph;y++) {
    const row=reflect(y-pad,h)*w;
    for (let col=0;col<pw;col++) expanded[y*pw+col]=x[row+cols[col]];
  }
  const prediction=predict(expanded,pw,ph,model);
  const out=new Float32Array(w*h*3);
  for (let c=0;c<3;c++) {
    for (let y=0;y<h;y++) {
      const start=c*area+(y+pad)*pw+pad;
      out.set(prediction.subarray(start,start+w),c*w*h+y*w);
    }
  }
  return out;
}

export function makeCue(rgba,w,h,detail=65,invert=false) {
  const stride=w+1, sum=new Float64Array(stride*(h+1));
  const hist=new Uint32Array(256);
  for (let y=0;y<h;y++) {
    let running=0;
    for (let x=0;x<w;x++) {
      const i=(y*w+x)*4, alpha=rgba[i+3]/255;
      const v=(rgba[i]*.2126+rgba[i+1]*.7152+rgba[i+2]*.0722)*alpha + 32*(1-alpha);
      running += v;
      sum[(y+1)*stride+x+1]=running+sum[y*stride+x+1];
      hist[Math.round(v)]++;
    }
  }
  const quantile = q => {
    let s=0;
    for (let i=0;i<256;i++) {
      s+=hist[i];
      if (s>=q*w*h) return i;
    }
    return 255;
  };
  const lo=quantile(.03), hi=quantile(.97), range=hi-lo;
  const scale=6+Math.round(Math.min(100,Math.max(0,detail))*.24);
  const bw=Math.min(w,Math.max(3,Math.round(scale*w/Math.max(w,h))));
  const bh=Math.min(h,Math.max(3,Math.round(scale*h/Math.max(w,h))));
  const small=new Float32Array(bw*bh);
  for (let y=0;y<bh;y++) {
    const y0=Math.floor(y*h/bh), y1=Math.ceil((y+1)*h/bh);
    for (let x=0;x<bw;x++) {
      const x0=Math.floor(x*w/bw), x1=Math.ceil((x+1)*w/bw);
      const a=sum[y1*stride+x1]-sum[y1*stride+x0]-sum[y0*stride+x1]+sum[y0*stride+x0];
      const v=a/((x1-x0)*(y1-y0));
      const stretched = range>=32 ? .64*v/255+.36*(v-lo)/range : v/255;
      const t=Math.max(0,Math.min(1,stretched));
      small[y*bw+x]=(invert ? 1-t : t)*2-1;
    }
  }
  const out=new Float32Array(w*h);
  for (let y=0;y<h;y++) {
    const sy=(y+.5)*bh/h-.5, by=Math.floor(sy), dy=sy-by;
    const a=Math.max(0,by)*bw, b=Math.min(bh-1,by+1)*bw;
    for (let x=0;x<w;x++) {
      const sx=(x+.5)*bw/w-.5, bx=Math.floor(sx), dx=sx-bx;
      const c=Math.max(0,bx), d=Math.min(bw-1,bx+1);
      out[y*w+x]=(small[a+c]*(1-dx)+small[a+d]*dx)*(1-dy) +
        (small[b+c]*(1-dx)+small[b+d]*dx)*dy;
    }
  }
  return out;
}

export function renderPixels(pixels,w,h) {
  const n=w*h, out=new Uint8ClampedArray(n*4);
  for (let i=0;i<n;i++) {
    out[4*i]=(pixels[i]+1)*127.5;
    out[4*i+1]=(pixels[n+i]+1)*127.5;
    out[4*i+2]=(pixels[2*n+i]+1)*127.5;
    out[4*i+3]=255;
  }
  return out;
}
