"""Télécharge le jeu « BOFiP Impôts, publications en vigueur » (data.gouv.fr) et en extrait un sous-corpus.

Le fichier source n'a pas de schéma documenté : le script détecte lui-même les champs (identifiant BOI,
titre, date, texte), écrit un rapport de schéma, puis un corpus normalisé au format JSON Lines compressé.

Usage :
    python scripts/fetch_bofip.py --out data/bofip_subset.jsonl.gz --series RPPM,ENR,RFPI,PAT
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

SOURCE_URL = "https://www.data.gouv.fr/api/1/datasets/r/aef730a6-de7c-4eb3-baca-d4d699052c35"
BOI_RE = re.compile(r"\bBOI-[A-Z]+(?:-[A-Z0-9]+)*\b")
TAG_RE = re.compile(r"<[^>]+>")


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-fiscal/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def load_records(raw: bytes) -> list[dict]:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8-sig")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
    elif stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:  # JSON Lines
            data = [json.loads(line) for line in io.StringIO(text) if line.strip()]
    else:
        raise SystemExit("Format inattendu : ni JSON ni JSON Lines")
    if isinstance(data, dict):
        for key in ("records", "results", "data", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    records = []
    for rec in data:
        if isinstance(rec, dict) and isinstance(rec.get("fields"), dict):  # export Opendatasoft
            rec = {**rec["fields"], **{k: v for k, v in rec.items() if k != "fields"}}
        records.append(rec)
    return records


def describe(records: list[dict]) -> str:
    keys = Counter(k for r in records for k in r)
    lines = [f"Enregistrements : {len(records)}", "", "Champs (présence, longueur moyenne, exemple) :"]
    for key, count in keys.most_common():
        values = [r[key] for r in records[:2000] if key in r and r[key] not in (None, "")]
        avg = sum(len(str(v)) for v in values) / max(len(values), 1)
        example = str(values[0])[:160].replace("\n", " ") if values else ""
        lines.append(f"- {key} : {count} | {avg:.0f} car. | {example}")
    return "\n".join(lines)


def pick_fields(records: list[dict]) -> dict[str, str]:
    sample = records[:2000]
    keys = {k for r in sample for k in r}

    def avg_len(k: str) -> float:
        vals = [str(r.get(k, "")) for r in sample]
        return sum(map(len, vals)) / max(len(vals), 1)

    def share_boi(k: str) -> float:
        return sum(bool(BOI_RE.fullmatch(str(r.get(k, "")).strip())) for r in sample) / max(len(sample), 1)

    text_field = max(keys, key=avg_len)
    id_field = max(keys, key=share_boi)
    title_field = next((k for k in keys if re.search(r"titre|title", k, re.I)), None)
    date_field = next((k for k in keys if re.search(r"date|validite", k, re.I)), None)
    url_field = next((k for k in keys if re.search(r"url|lien|permalien", k, re.I)), None)
    return {"text": text_field, "id": id_field, "title": title_field or "", "date": date_field or "", "url": url_field or ""}


def html_to_text(value: str) -> str:
    value = re.sub(r"(?i)<\s*(br|/p|/li|/h\d|/tr|/div)\s*/?>", "\n", value)
    value = TAG_RE.sub(" ", value)
    value = re.sub(r"&nbsp;|&#160;", " ", value)
    value = value.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n\n", value)
    return value.strip()


def serie_of(boi_id: str) -> str:
    parts = boi_id.split("-")
    return parts[1] if len(parts) > 1 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bofip_subset.jsonl.gz")
    ap.add_argument("--report", default="data/schema_report.txt")
    ap.add_argument("--series", default="RPPM,ENR,RFPI,PAT")
    ap.add_argument("--max-docs", type=int, default=1500)
    ap.add_argument("--source", default=SOURCE_URL)
    args = ap.parse_args()

    raw = download(args.source) if args.source.startswith("http") else Path(args.source).read_bytes()
    records = load_records(raw)
    fields = pick_fields(records)
    report = describe(records) + "\n\nChamps retenus : " + json.dumps(fields, ensure_ascii=False)

    wanted = {s.strip().upper() for s in args.series.split(",") if s.strip()}
    by_id: dict[str, dict] = {}
    series_count: Counter[str] = Counter()
    for rec in records:
        match = BOI_RE.search(str(rec.get(fields["id"], "")))
        if not match:
            continue
        boi = match.group(0)
        serie = serie_of(boi)
        series_count[serie] += 1
        if wanted and serie not in wanted:
            continue
        text = html_to_text(str(rec.get(fields["text"], "")))
        if not text:
            continue
        doc = by_id.setdefault(boi, {
            "id": boi,
            "title": str(rec.get(fields["title"], "")).strip() if fields["title"] else "",
            "serie": serie,
            "date": str(rec.get(fields["date"], "")).strip()[:10] if fields["date"] else "",
            "url": str(rec.get(fields["url"], "")).strip() if fields["url"] else f"https://bofip.impots.gouv.fr/bofip/{boi}",
            "text": "",
        })
        doc["text"] = (doc["text"] + "\n\n" + text).strip()  # un même BOI peut être réparti sur plusieurs lignes
    docs = sorted((d for d in by_id.values() if len(d["text"]) >= 200), key=lambda d: d["id"])[: args.max_docs]

    report += "\n\nSéries présentes (documents) : " + json.dumps(series_count.most_common(), ensure_ascii=False)
    report += f"\nDocuments retenus : {len(docs)} (séries {sorted(wanted)})"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report + "\n", encoding="utf-8")
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    sys.exit(main())
