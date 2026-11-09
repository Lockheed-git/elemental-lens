# 元素显影 · Elemental Lens

把一张图重构成火焰、海浪或云。页面加载约 76 KiB 的模型权重，在浏览器本地处理选入的图片，支持调整轮廓、反转明暗和下载 PNG。无需登录、服务器推理或预训练模型。

**[在线体验](https://lockheed-git.github.io/elemental-lens/)** · [原作论文](https://arxiv.org/abs/2003.00902) · [原作视频](https://memo.tv/projects/2017/gloomy-sunday/)

## 运行网页

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory site
```

打开 `http://127.0.0.1:8765`。页面默认加载内置手形示例；也可选入或拖入图片。亮手暗背景适合直接生成，主体比背景暗时可以试试反转明暗。输出最长边 192 像素。

## 复现方法

复现 [Learning to See: You Are What You See](https://arxiv.org/abs/2003.00902) 的训练思路：将目标域图像降为粗略灰度结构，以原图作为监督目标，学会在结构上重构目标域外观。这里用较小的 U-Net 生成器和 PatchGAN 判别器，每个风格独立从随机初始化训练。最终部署每个 19,403 参数的生成器，判别器仅在训练时使用。

三个模型各训练 2,000 步。真实照片来自 Wikimedia Commons（火焰 13 张、海浪 14 张、云 12 张），并混合随机生成的火焰、波纹与云团作为补充。原始照片不随网站分发；[逐张来源及许可](training/data-credits.json)和[训练记录](training/metrics.json)随仓库提供。照片在训练时被缩放、裁切与随机镜像。页面使用的示例手形是仓库内原创 SVG。

这是一套小样本重实现。它学习的是重构整幅画面的外观，输出偏艺术化；不保证指尖位置、局部背景或物理细节都准确。没有使用原作权重。

## 重新训练

安装 [适合本机的 PyTorch](https://pytorch.org/get-started/locally/) 与 Pillow、NumPy。照片下载脚本需要 curl。

```sh
python training/collect_commons.py --restore training/data-credits.json
python training/train.py --steps 2000 --batch 16 --lr 0.00065 --photo-share 0.58 --selection training/selection.json
node training/verify_engine.mjs
```

只有程序合成素材时，也可以去掉 `--selection`，并将 `--photos` 指向空目录。生成器权重导出到 `site/models/`；训练记录、检查点和试验图保存在被 Git 忽略的 `training/runs/`。

部署目录是 `site/`。发布分支 `gh-pages` 只包含该目录的内容，静态资源使用相对路径。
