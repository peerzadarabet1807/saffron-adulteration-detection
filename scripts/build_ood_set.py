"""Build the out-of-distribution (OOD) photo set for the input check from Wikimedia Commons.

Two steps:

1. ``--discover`` runs the Commons searches listed in CATEGORIES, keeps files whose licence is CC0, public
   domain, CC BY or CC BY-SA, and writes the candidates plus a numbered contact sheet per category to
   ``.work/ood-discover/`` (gitignored). A person looks at the sheets and writes the chosen Commons titles
   into ``docs/ood/selection.json``, with a reason for every candidate that was skipped (for example "shows
   a cow, not dung").
2. Without ``--discover`` the script downloads exactly the titles in ``docs/ood/selection.json``, downscales
   each to at most 640 px on the long side (JPEG quality 85), and writes ``docs/ood/set/<category>/`` and
   the credits in ``docs/ood/credits.json`` (title, author, licence, source URL and the change made).

The committed selection file, not the live search ranking, decides what is in the set, so the set can be
rebuilt byte for byte as long as the Commons files do not change. Licences are re-checked on download.

Usage:
    python scripts/build_ood_set.py --discover            # candidates and contact sheets for review
    python scripts/build_ood_set.py                       # download the committed selection
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOD = ROOT / "docs" / "ood"
SET_DIR = OOD / "set"
SELECTION = OOD / "selection.json"
DISCOVER = ROOT / ".work" / "ood-discover"
UA = {"User-Agent": "saffron-adulteration-detection/1.0 (https://github.com/peerzadarabet1807/saffron-adulteration-detection)"}
API = "https://commons.wikimedia.org/w/api.php"
OPEN_LICENSES = ("CC0", "CC BY", "Public domain", "PD")
LONG_SIDE = 640

#: slug -> (proximity, description, Commons search queries). "near": look-alikes or things a user might
#: photograph instead of a single saffron thread on paper, which the check must reject. "grey": look-alikes that
#: are not among the five training classes and that the spec says to measure and document, not silently decide
#: (safflower, corn silk, loose saffron that is not a single thread on paper). "far": unrelated photos; their
#: titles are the same as the poultry repository's far categories.
CATEGORIES: dict[str, tuple[str, str, list[str]]] = {
    "red_thread_yarn": ("near", "Red thread, yarn or embroidery floss", ["red thread", "red yarn", "red embroidery floss", "red wool yarn", "red string"]),
    "red_fabric": ("near", "Red fabric and textiles close up", ["red fabric", "red textile texture", "red cloth", "red velvet fabric"]),
    "chili_threads": ("near", "Chili threads and dried chili", ["chili threads", "silgochu", "dried chili strips", "red pepper threads", "crushed red pepper"]),
    "hair": ("near", "Hair and loose strands", ["red hair", "hair strand", "loose hair", "single hair macro", "auburn hair"]),
    "saffron_flower": ("near", "Saffron crocus flowers", ["Crocus sativus", "saffron crocus flower", "saffron flower", "saffron harvest flowers"]),
    "saffron_packaged": ("near", "Saffron in jars, packets or tins", ["saffron jar", "saffron packet", "saffron package", "saffron tin", "saffron for sale"]),
    "food": ("near", "Food and dishes, including saffron dishes", ["paella", "saffron rice", "biryani", "risotto alla milanese", "spice market"]),
    "red_objects": ("near", "Small red objects on a light surface", ["red pencil", "red rubber band", "red wire", "red paper clip", "red pen on paper"]),
    "safflower": ("grey", "Safflower (a common saffron substitute)", ["safflower", "Carthamus tinctorius", "safflower petals", "dried safflower"]),
    "corn_silk": ("grey", "Corn silk", ["corn silk", "maize silk", "dried corn silk"]),
    "saffron_loose": ("grey", "Saffron threads in a pile or dish, not a single thread on paper", ["saffron threads", "saffron spice", "saffron stigmas", "saffron pile"]),
    "documents": ("far", "Documents and printed pages", ["document page", "printed page", "newspaper page", "handwritten letter", "receipt"]),
    "screenshots": ("far", "Screenshots of software and websites", ["screenshot", "software screenshot", "website screenshot"]),
    "people": ("far", "People", ["portrait photograph", "people walking", "farmer portrait"]),
    "everyday_objects": ("far", "Everyday objects", ["mug", "keyboard", "chair", "bicycle", "shoes", "car"]),
    "scenes": ("far", "Landscapes, buildings and streets", ["landscape", "city street", "building facade", "beach", "forest"]),
    "other_animals": ("far", "Other animals", ["cat", "dog portrait", "cow field", "horse", "sheep", "songbird"]),
}


#: The original 7 photos (docs/ood/ before the input check) in the categories above. They all go to ood-test.
ORIGINAL_CATEGORY = {
    "red_thread_yarn.jpg": "red_thread_yarn",
    "red_thread_kate_2112.jpg": "red_thread_yarn",
    "thread_amagill.jpg": "red_thread_yarn",
    "spool_of_dark_red_sewing_thread_1.jpg": "red_thread_yarn",
    "handspun_yarn_16211767010.jpg": "red_thread_yarn",
    "backstitch_hand_embroidery.jpg": "red_thread_yarn",
    "red_wool_texture.jpg": "red_fabric",
}


def load_credits() -> list[dict]:
    """docs/ood/credits.json: a list before this script first ran, {"note", "photos"} after."""
    doc = json.loads((OOD / "credits.json").read_text("utf-8"))
    return doc if isinstance(doc, list) else doc["photos"]


def plain(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def http_get(url: str, *, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
        except (urllib.error.URLError, ConnectionError, TimeoutError):  # dropped connections: retry too
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def api(params: dict) -> dict:
    data = http_get(f"{API}?{urllib.parse.urlencode({'format': 'json', **params})}")
    time.sleep(1.0)
    return json.loads(data)


def license_ok(name: str) -> bool:
    return name.startswith(OPEN_LICENSES) and "NC" not in name and "ND" not in name


def describe(info: dict, title: str) -> dict:
    meta = info.get("extmetadata", {})
    return {
        "title": title,
        "author": plain(meta.get("Artist", {}).get("value", "unknown")),
        "license": meta.get("LicenseShortName", {}).get("value", ""),
        "license_url": meta.get("LicenseUrl", {}).get("value", ""),
        "source": info["descriptionurl"],
        "width": info.get("width"),
        "height": info.get("height"),
    }


def discover(limit: int, only: list[str] | None = None) -> int:
    from PIL import Image, ImageDraw

    DISCOVER.mkdir(parents=True, exist_ok=True)
    taken = {p["title"].removeprefix("File:") for p in load_credits()}
    for slug, (_, _, queries) in CATEGORIES.items():
        if only and slug not in only:
            continue
        seen: dict[str, dict] = {}
        per_query: list[list[str]] = []
        for query in queries:
            per_query.append([])
            data = api(
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{query} filetype:bitmap",
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime|size",
                    "iiurlwidth": 330,
                }
            )
            pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
            for page in pages:
                title = page["title"].removeprefix("File:")
                info = (page.get("imageinfo") or [{}])[0]
                if title in seen or title in taken or info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < 300:
                    continue
                d = describe(info, title)
                if license_ok(d["license"]):
                    seen[title] = {**d, "query": query, "thumb": info["thumburl"]}
                    per_query[-1].append(title)
        # round-robin over the queries, so the first query cannot fill the whole list
        order = [t for group in zip(*[q + [None] * (50 - len(q)) for q in per_query], strict=True) for t in group if t]
        cands = [seen[t] for t in dict.fromkeys(order)][:limit]
        (DISCOVER / f"{slug}.json").write_text(json.dumps(cands, indent=2, ensure_ascii=False), "utf-8")
        tiles = []
        for i, c in enumerate(cands):
            try:
                im = Image.open(io.BytesIO(http_get(c["thumb"]))).convert("RGB")
            except Exception:  # noqa: BLE001 (a broken thumbnail just leaves a blank tile)
                im = Image.new("RGB", (200, 200), "grey")
            im.thumbnail((200, 200))
            tile = Image.new("RGB", (200, 220), "white")
            tile.paste(im, ((200 - im.width) // 2, 20))
            ImageDraw.Draw(tile).text((4, 2), str(i), fill="black")
            tiles.append(tile)
            time.sleep(0.3)
        cols = 8
        sheet = Image.new("RGB", (cols * 200, max(1, -(-len(tiles) // cols)) * 220), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 200, (i // cols) * 220))
        sheet.save(DISCOVER / f"{slug}.jpg", quality=80)
        print(f"{slug:<24} {len(cands)} candidates", flush=True)
    return 0


def fetch() -> int:
    from PIL import Image

    selection = json.loads(SELECTION.read_text("utf-8"))
    original = [
        {
            "file": p["file"],
            "category": ORIGINAL_CATEGORY[p["file"]],
            "proximity": CATEGORIES[ORIGINAL_CATEGORY[p["file"]]][0],
            "description": CATEGORIES[ORIGINAL_CATEGORY[p["file"]]][1],
            "title": p["title"].removeprefix("File:"),
            "author": p["author"],
            "license": p["license"],
            "license_url": p.get("license_url", p.get("licenseUrl", "")),
            "source": p["source"],
            "changes": p.get("changes", ""),
            "original": True,
        }
        for p in load_credits()
        if "/" not in p["file"]
    ]
    photos = []
    for slug, entry in selection["categories"].items():
        proximity, description, _ = CATEGORIES[slug]
        out = SET_DIR / slug
        out.mkdir(parents=True, exist_ok=True)
        titles = entry["titles"]
        for i in range(0, len(titles), 25):
            batch = titles[i : i + 25]
            data = api(
                {
                    "action": "query",
                    "titles": "|".join(f"File:{t}" for t in batch),
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|mime|size",
                    "iiurlwidth": 960,
                }
            )
            norm = {n["to"]: n["from"] for n in data["query"].get("normalized", [])}
            by_title = {}
            for page in data["query"]["pages"].values():
                if "imageinfo" not in page:
                    raise LookupError(f"{page['title']}: missing on Commons")
                by_title[norm.get(page["title"], page["title"]).removeprefix("File:")] = page["imageinfo"][0]
            for title in batch:
                info = by_title[title]
                d = describe(info, title)
                if not license_ok(d["license"]):
                    raise ValueError(f"{title}: licence {d['license']} is not open enough")
                im = Image.open(io.BytesIO(http_get(info.get("thumburl") or info["url"]))).convert("RGB")
                scale = LONG_SIDE / max(im.size)
                if scale < 1:
                    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
                stem = re.sub(r"[^a-z0-9]+", "_", Path(title).stem.lower()).strip("_")[:60]
                if len(stem) < 4:  # titles in other scripts: name the file after a hash of the title
                    stem = f"{slug}_{hashlib.sha1(title.encode()).hexdigest()[:10]}"
                path = out / f"{stem}.jpg"
                n = 2
                while path.relative_to(OOD).as_posix() in {q["file"] for q in photos}:
                    path, n = out / f"{stem}_{n}.jpg", n + 1
                im.save(path, quality=85, optimize=True)
                photos.append(
                    {
                        "file": path.relative_to(OOD).as_posix(),
                        "category": slug,
                        "proximity": proximity,
                        "description": description,
                        **{k: d[k] for k in ("title", "author", "license", "license_url", "source")},
                        "changes": f"Downscaled to {max(im.size)} px on the long side, saved as JPEG quality 85.",
                    }
                )
                print(f"{photos[-1]['file']:<70} {d['license']}", flush=True)
                time.sleep(0.5)
    doc = {
        "note": (
            "The original 7 photos (no folder in 'file') were resized to 1024 px on the long side. The photos under "
            "set/ were downscaled as stated in 'changes'. Each photo keeps its own licence; the repository's MIT "
            "licence does not apply to them."
        ),
        "photos": original + photos,
    }
    (OOD / "credits.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n")
    print(f"{len(photos)} photos in set/, {len(original)} original photos kept")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discover", action="store_true", help="write candidates and contact sheets for review")
    ap.add_argument("--limit", type=int, default=40, help="candidates per category with --discover")
    ap.add_argument("--only", action="append", choices=sorted(CATEGORIES), help="with --discover: these only")
    args = ap.parse_args()
    return discover(args.limit, args.only) if args.discover else fetch()


if __name__ == "__main__":
    sys.exit(main())
