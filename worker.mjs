import {readModel, makeCue, predictVisible, renderPixels, STYLES} from './engine.mjs';
const models = new Map();

self.onmessage = async ({data:{id,style,rgba,w,h,detail,invert}}) => {
  try {
    if (!STYLES.includes(style)) throw Error('未知的风格');
    if (!models.has(style)) {
      const promise = fetch(new URL(`./models/${style}.eln`,import.meta.url)).then(async response => {
        if (!response.ok) throw Error(`模型加载失败（${response.status}）`);
        return readModel(await response.arrayBuffer());
      });
      models.set(style,promise);
      promise.catch(() => models.delete(style));
    }
    const model = await models.get(style);
    const started = performance.now();
    const cue = makeCue(new Uint8ClampedArray(rgba),w,h,detail,invert);
    const pixels = renderPixels(predictVisible(cue,w,h,model),w,h);
    self.postMessage({id,pixels,w,h,ms:Math.round(performance.now()-started)},[pixels.buffer]);
  } catch (e) {
    self.postMessage({id,error:e instanceof Error ? e.message : '生成失败'});
  }
};
