"""Collect training photos and write a reproducible attribution manifest."""
import argparse
import concurrent.futures
import html
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlencode, urlsplit

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
QUERIES = {
    "fire": ["campfire flames at night", "bonfire orange flame close up", "burning wood flames fire",
             "flame closeup dark background", "flames fire black background"],
    "wave": ["breaking ocean wave foam", "sea wave white foam close up", "ocean swell wave crest",
             "surf breaking wave spray", "white water shore wave"],
    "cloud": ["white cumulus clouds blue sky", "cumulus cloud close up sky", "cumulus clouds from aircraft",
              "cotton clouds sky", "cumulus congestus cloud"],
}
AGENT = "ElementalLens/0.1 (small image translation research; GitHub Lockheed-git)"
ALLOWED = re.compile(r"^(cc0(?:-1\.0)?|pd|pdm|public domain|cc-by-(?:2\.0|2\.5|3\.0|4\.0))$", re.I)


def fetch(url, limit):
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl must be installed")
    return subprocess.check_output([
        curl, "-fLsS", "--max-time", "20", "--max-filesize", str(limit),
        "--user-agent", AGENT, url], stderr=subprocess.PIPE)


def metadata(value):
    return html.unescape(re.sub(r"<[^>]*>", "", value)).strip()


def discover(style):
    candidates = {}
    for query in QUERIES[style]:
        params = {
            "action":"query", "generator":"search", "gsrsearch":"filetype:bitmap "+query,
            "gsrnamespace":6, "gsrlimit":40, "prop":"imageinfo",
            "iiprop":"url|mime|extmetadata", "iiurlwidth":480,
            "iiextmetadatafilter":"License|LicenseUrl|LicenseShortName|Artist|Restrictions",
            "formatversion":2, "format":"json",
        }
        try:
            data = json.loads(fetch(API+"?"+urlencode(params),2_000_000))
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"{style}: query {query!r}: {e}",flush=True)
            continue
        for item in sorted(data.get("query",{}).get("pages",[]), key=lambda r:r.get("index",0)):
            inf = (item.get("imageinfo") or [{}])[0]
            if inf.get("mime") not in ("image/jpeg","image/png"):
                continue
            ext = inf.get("extmetadata",{})
            v = lambda key: metadata(ext.get(key,{}).get("value",""))
            license_id = v("License")
            if not (ALLOWED.match(license_id) or v("LicenseShortName") == "Public domain") or v("Restrictions"):
                continue
            pid = str(item["pageid"])
            candidates.setdefault(pid, {
                "file": f"{pid}.jpg", "title":item["title"], "url":inf["descriptionurl"],
                "author":v("Artist"), "license":license_id or v("LicenseShortName"),
                "license_url":v("LicenseUrl"), "image_url":inf["url"].split("?")[0],
                "thumbnail":inf.get("thumburl","").split("?")[0],
            })
    return list(candidates.values())


def download(row, folder):
    path = folder / row["file"]
    if path.exists():
        return True
    original = row["image_url"]
    source = urlsplit(original)
    prefix = "/wikipedia/commons/"
    if source.netloc != "upload.wikimedia.org" or not source.path.startswith(prefix):
        return False
    tail = source.path[len(prefix):]
    fallback_thumb = f"https://thumb.wikimedia.org{prefix}thumb/{tail}/500px-{tail.rsplit('/',1)[-1]}"
    urls = [row.get("thumbnail") or fallback_thumb,original]
    for url in urls:
        if not url:
            continue
        try:
            with Image.open(BytesIO(fetch(url, 12_000_000))) as loaded:
                im = ImageOps.exif_transpose(loaded).convert("RGB")
                if min(im.size) < 180:
                    return False
                im.thumbnail((800,800),Image.Resampling.LANCZOS)
                im.save(path,quality=91,optimize=True)
                return True
        except (subprocess.CalledProcessError, OSError, ValueError):
            pass
    return False


def collect(style, limit, root):
    rows = discover(style)
    print(f"{style}: {len(rows)} licensed candidates", flush=True)
    folder = root/style
    folder.mkdir(parents=True,exist_ok=True)
    selected = []
    for row in rows:
        if download(row,folder):
            selected.append(row)
            if len(selected) % 8 == 0:
                print(f"{style}: downloaded {len(selected)}",flush=True)
            if len(selected) >= limit:
                break
    return style, selected


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit",type=int,default=36)
    ap.add_argument("--output",type=Path,default=ROOT/"training"/"photos")
    ap.add_argument("--restore",type=Path,help="use an existing curated attribution manifest")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.restore:
        index=json.loads(args.restore.read_text(encoding="utf-8"))
        for style,rows in index.items():
            if style not in QUERIES:
                raise ValueError(f"Unknown style: {style}")
            folder=args.output/style
            folder.mkdir(parents=True,exist_ok=True)
            for row in rows:
                if not re.fullmatch(r"[0-9]+\.jpg",row["file"]):
                    raise ValueError(f"Unexpected photo filename: {row['file']}")
                if not download(row,folder):
                    raise RuntimeError(f"Could not download {row['url']}")
        print({k:len(v) for k,v in index.items()})
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(collect,style,args.limit,args.output) for style in QUERIES]
        index=dict(job.result() for job in jobs)
    (args.output/"manifest.json").write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n",
                                             encoding="utf-8")
    print({k:len(v) for k,v in index.items()})


if __name__ == "__main__":
    main()
