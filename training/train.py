"""Train compact pix2pix-style generators from target images alone."""
import argparse
import json
import random
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("fire", "wave", "cloud")
LUMA = (.2126, .7152, .0722)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        for name, a, b, k in (
            ("e0", 1, 12, 3), ("e1", 12, 16, 3), ("e2", 16, 24, 3),
            ("mid", 24, 24, 3), ("d1", 40, 16, 3), ("d0", 28, 12, 3),
            ("out", 12, 3, 1),
        ):
            setattr(self, name, nn.Conv2d(a, b, k, padding=k // 2))

    def forward(self, x):
        act = lambda y: F.leaky_relu(y, .1)
        a = act(self.e0(x))
        b = act(self.e1(F.avg_pool2d(a, 2)))
        c = act(self.e2(F.avg_pool2d(b, 2)))
        c = act(self.mid(c) + c)
        c = act(self.d1(torch.cat([F.interpolate(c, scale_factor=2, mode="nearest"), b], 1)))
        c = act(self.d0(torch.cat([F.interpolate(c, scale_factor=2, mode="nearest"), a], 1)))
        return torch.tanh(self.out(c))


def discriminator():
    return nn.Sequential(
        nn.Conv2d(4, 24, 4, 2, 1), nn.LeakyReLU(.2),
        nn.Conv2d(24, 48, 4, 2, 1), nn.InstanceNorm2d(48), nn.LeakyReLU(.2),
        nn.Conv2d(48, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(.2),
        nn.Conv2d(64, 1, 3, padding=1),
    )


def synthetic(style, batch, n, device):
    """Make varied emissive/plume, foam or cloud fields for the bootstrap dataset."""
    def rand(*size):
        return torch.rand(size, device=device)

    def noise(h, w):
        return F.interpolate(rand(batch, 1, h, w) * 2 - 1, (n, n), mode="bicubic",
                             align_corners=False)

    def palette(v, stops, colors):
        out = torch.tensor(colors[0], device=device).reshape(1, 3, 1, 1)
        for i in range(1, len(stops)):
            delta = torch.tensor(np.subtract(colors[i], colors[i-1]), device=device,
                                 dtype=v.dtype).reshape(1, 3, 1, 1)
            out = out + delta * ((v - stops[i-1]) / (stops[i] - stops[i-1])).clamp(0, 1)
        return out

    y = torch.linspace(0, 1, n, device=device)[None, None, :, None]
    x = torch.linspace(0, 1, n, device=device)[None, None, None, :]
    coarse, medium, fine = noise(5, 5), noise(12, 13), noise(28, 31)
    grain = noise(56, 57)
    if style == "fire":
        lean = noise(4, 3) * .16
        freq = 9 + rand(batch, 1, 1, 1) * 13
        threads = torch.sin((x + lean + .04 * medium) * freq + 4 * y + coarse * 3)
        burn = (.62 * noise(5, 13) + .24 * coarse + .16 * noise(10, 28)
                + .09 * fine + .11 * threads + (y - .55) * .47
                + (rand(batch, 1, 1, 1) - .5) * .44)
        ridges = (.25 - (burn - .09).abs()).clamp(0, .25) * .76
        value = (burn * 1.25 + ridges + .038 * grain + .20).clamp(0, 1)
        rgb = palette(value, [0, .17, .34, .56, .77, 1],
                      [[.018,.008,.017], [.15,.018,.016], [.55,.053,.009],
                       [1,.27,.018], [1,.66,.075], [1,.98,.64]])
        smoke = (.5 + coarse + .6 * medium).clamp(0, 1) * (1 - value).square()
        rgb = rgb + smoke * torch.tensor([.060,.052,.062],device=device)[None,:,None,None]
    elif style == "wave":
        warp = y + .21 * coarse + .032 * medium + .12 * torch.sin(5*x + 2*y)
        ripples = torch.sin(warp * (21 + rand(batch,1,1,1)*35) + 2.3 * medium)
        water = (.37 + .32 * coarse + .13 * ripples + .11 * medium + .04 * fine).clamp(0,1)
        rgb = palette(water, [0,.23,.5,.76,1],
                      [[.012,.052,.09],[.02,.15,.25],[.035,.38,.49],[.13,.62,.63],[.37,.78,.8]])
        breaking = ripples * .37 + coarse * .44 + .27 * medium + .30 * fine + .13 * grain
        foam = torch.sigmoid((breaking - (.42 + (rand(batch,1,1,1)-.5)*.50))*28)
        froth = (.82 + .16*fine - .12*grain).clamp(.52,1)
        rgb = rgb * (1-foam) + (froth * foam) * torch.tensor([.93,.985,1.0],device=device)[None,:,None,None]
    else:
        volume = .66 * coarse + .30 * medium + .115 * fine + .046 * grain
        density = volume + .06 + (rand(batch,1,1,1) - .5) * .53
        alpha = torch.sigmoid(density * 24)
        sky = palette((.42 + .43*y + .13*coarse).clamp(0,1),
                      [0,.4,1], [[.07,.22,.47],[.15,.39,.72],[.58,.76,.89]])
        volume_up = F.pad(volume, (0, 0, 2, 0), mode="replicate")[:, :, :-2]
        light = (.81 - density*.32 + .55*(volume-volume_up) + .11*medium + .06*fine).clamp(.30,1)
        clouds = palette(light, [0,.42,.75,.91,1],
                         [[.09,.14,.23],[.38,.45,.57],[.69,.75,.83],[.94,.96,.985],[1,.995,.95]])
        rgb = sky * (1-alpha) + clouds * alpha
    return rgb.clamp(0,1)


def photo_tiles(style, n, root, device, selection):
    tiles = []
    paths = sorted((root / style).glob("*")) if root else []
    if selection is not None:
        ids = {str(value) for value in selection[style]}
        paths = [path for path in paths if path.stem in ids]
        missing = ids - {path.stem for path in paths}
        if missing:
            raise FileNotFoundError(f"{style}: missing curated photos {sorted(missing)}")
    accepted = 0
    for path in paths:
        try:
            with Image.open(path) as file:
                im = ImageOps.exif_transpose(file).convert("RGB")
                if min(im.size) < n:
                    continue
                for i in range(12):
                    width = random.randint(max(n, min(im.size) * 2 // 5), min(im.size))
                    x = random.randint(0, im.width-width)
                    y = random.randint(0, im.height-width)
                    crop = im.crop((x,y,x+width,y+width)).resize((n,n), Image.Resampling.LANCZOS)
                    if i % 2:
                        crop = ImageOps.mirror(crop)
                    tiles.append(np.array(crop))
                accepted += 1
        except (OSError, ValueError):
            continue
    return (torch.as_tensor(np.stack(tiles), device=device).permute(0,3,1,2)
            if tiles else None), accepted


def structure(rgb, blocks, jitter=False):
    luma = rgb.new_tensor(LUMA).reshape(1,3,1,1)
    gray = (rgb * luma).sum(1, keepdim=True)
    gray = F.adaptive_avg_pool2d(gray, (blocks, blocks))
    gray = F.interpolate(gray, rgb.shape[-2:], mode="bilinear", align_corners=False)
    if jitter:
        brightness = torch.rand(gray.size(0),1,1,1,device=gray.device)*.16-.08
        contrast = torch.rand_like(brightness)*.24+.88
        gray = (gray-.5)*contrast+.5+brightness
    return gray.clamp(0,1)*2-1


def save_weights(net, out):
    weights = list(net.state_dict().values())
    out.write_bytes(b"ELN1" + struct.pack("<I", sum(w.numel() for w in weights)) +
                    b"".join(w.detach().cpu().numpy().astype("<f4").tobytes() for w in weights))


def image(tensor):
    return Image.fromarray(((tensor.detach().cpu().permute(1,2,0).numpy()+1)*127.5)
                           .clip(0,255).astype(np.uint8))


def evaluate(net, style, size, device, path):
    g = torch.Generator(device=device).manual_seed(51)
    # A fixed coarse shape checks that the exported network follows input structure.
    luma = torch.rand((1,1,7,7), generator=g, device=device)
    luma = F.interpolate(luma, (size,size), mode="bilinear", align_corners=False)*2-1
    shifted = luma.flip(-1)
    with torch.inference_mode():
        test = torch.cat((luma, shifted))
        prediction = net(test)
        delta = (prediction[0]-prediction[1]).abs().mean().item()
        if not torch.isfinite(prediction).all() or delta < .01:
            raise RuntimeError(f"{style}: output failed structure sensitivity check: {delta}")
    if path:
        src = ROOT / "training" / "fixtures" / "hand-private.jpg"
        if src.is_file():
            with Image.open(src) as im:
                raw = np.array(ImageOps.fit(im.convert("RGB"),(size,size)))
            rgb = torch.as_tensor(raw,device=device).permute(2,0,1)[None].float()/255
            with torch.inference_mode():
                cue = structure(rgb, 19)
                hand = net(cue)[0]
                image(hand).save(path/f"{style}-hand.jpg")
        image(prediction[0]).save(path/f"{style}-probe.jpg")
    return round(delta, 4)


def train(style, args, device):
    start = time.perf_counter()
    net = Generator().to(device=device, memory_format=torch.channels_last)
    disc = discriminator().to(device=device, memory_format=torch.channels_last)
    checkpoint = args.work / f"{style}.pt"
    if args.resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(saved["generator"])
        disc.load_state_dict(saved["discriminator"])
    opt_g = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(.5,.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(.5,.999))
    photos, count = photo_tiles(style, args.size, args.photos, device, args.selection)
    print(f"{style}: {count} photos, {args.steps} steps, {device}", flush=True)
    for i in range(args.steps):
        rgb = synthetic(style, args.batch, args.size, device)
        if photos is not None:
            mix = max(1, min(args.batch, round(args.batch*args.photo_share)))
            rgb[:mix] = photos[torch.randint(len(photos),(mix,),device=device)].float()/255
        x = structure(rgb, random.choice((5,6,8,10,12,15,18,22,26)), True)
        y = rgb*2-1
        fake = net(x)
        opt_d.zero_grad(set_to_none=True)
        if i >= min(45, args.steps//8) or args.resume:
            disc.requires_grad_(True)
            loss_d = (F.relu(1-disc(torch.cat([x,y],1))).mean() +
                      F.relu(1+disc(torch.cat([x,fake.detach()],1))).mean())*.5
            loss_d.backward()
            opt_d.step()
            disc.requires_grad_(False)
            adversarial = -disc(torch.cat([x,fake],1)).mean()
        else:
            adversarial = 0
        opt_g.zero_grad(set_to_none=True)
        reconstruction = (fake-y).abs().mean()
        coarse = (F.avg_pool2d(fake,4)-F.avg_pool2d(y,4)).abs().mean()
        loss_g = 15*reconstruction + 9*coarse + .9*adversarial
        loss_g.backward()
        opt_g.step()
        if i % 100 == 0 or i == args.steps - 1:
            print(f"{style} {i+1}/{args.steps} loss={loss_g.item():.3f}"
                  f" l1={reconstruction.item():.3f} seconds={time.perf_counter()-start:.1f}",flush=True)
    net.eval()
    sensitivity = evaluate(net, style, args.size, device, args.work)
    torch.save({"generator":net.state_dict(),"discriminator":disc.state_dict()}, checkpoint)
    save_weights(net, args.output/f"{style}.eln")
    return {"photos":count,"steps":args.steps,"size":args.size,"params":sum(p.numel() for p in net.parameters()),
            "probe_delta":sensitivity,"seconds":round(time.perf_counter()-start,1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--styles", nargs="+", choices=STYLES, default=list(STYLES))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--lr", type=float, default=.0009)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--work", type=Path, default=ROOT / "training" / "runs")
    ap.add_argument("--output", type=Path, default=ROOT / "site" / "models")
    ap.add_argument("--photos", type=Path, default=ROOT / "training" / "photos")
    ap.add_argument("--selection", type=Path, help="JSON mapping each style to selected photo IDs")
    ap.add_argument("--photo-share", type=float, default=.75)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.selection:
        args.selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if args.size < 64 or args.size % 4 or args.steps < 1 or not 0 <= args.photo_share <= 1:
        ap.error("use a positive step count, photo share in [0,1] and a size >=64 divisible by four")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(6,torch.get_num_threads()))
    if args.device.startswith("cuda"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    report = args.work/"training.json"
    summary = json.loads(report.read_text()) if report.exists() else {}
    for style in args.styles:
        summary[style] = train(style,args,torch.device(args.device))
        report.write_text(json.dumps(summary, indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
