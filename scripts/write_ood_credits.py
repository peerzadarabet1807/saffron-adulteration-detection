"""Write docs/ood/CREDITS.md, a readable table of docs/ood/credits.json (every OOD photo, author, licence, source).

Usage: python scripts/write_ood_credits.py
"""
from __future__ import annotations

import json
from pathlib import Path

OOD = Path(__file__).resolve().parents[1] / "docs" / "ood"


def main() -> None:
    doc = json.loads((OOD / "credits.json").read_text(encoding="utf-8"))
    lines = [
        "# Credits: out-of-distribution photos",
        "",
        f"All {len(doc['photos'])} photos in this folder come from Wikimedia Commons. {doc['note']}",
        "",
        "| File | Category | Author | License | Source | Changes |",
        "|---|---|---|---|---|---|",
    ]
    for c in doc["photos"]:
        lic = f"[{c['license']}]({c['license_url']})" if c.get("license_url") else c["license"]
        lines.append(
            f"| `{c['file']}` | {c['category'].replace('_', ' ')} | {c['author'].replace('|', '/')} | {lic} | "
            f"[Wikimedia Commons]({c['source']}) | {c.get('changes') or ''} |"
        )
    (OOD / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(doc['photos'])} photos -> {OOD / 'CREDITS.md'}")


if __name__ == "__main__":
    main()
