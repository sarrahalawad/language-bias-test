"""Step 1 of data ingestion: extract raw text from the source PDFs.

Reads every PDF under the language folders (DE/, EN/) and writes:
  data/processed/manifest.csv  - one row per document (id, language, pages, words, ...)
  data/processed/pages.jsonl   - one record per page with its raw text
  data/processed/text/<LANG>/<doc_id>.txt - full text per document, pages separated by form feeds

Requires poppler's `pdftotext` and `pdfinfo` on PATH.

Usage:
    python ingest/extract_text.py
"""

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ["DE", "EN"]
OUT_DIR = ROOT / "data" / "processed"


def make_doc_id(lang, pdf_path):
    stem = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")
    return f"{lang.lower()}_{stem}"


def page_count(pdf_path):
    info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))


def extract_pages(pdf_path):
    text = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    pages = text.split("\f")
    # pdftotext ends the output with a form feed, leaving one empty trailing chunk
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    with open(OUT_DIR / "pages.jsonl", "w", encoding="utf-8") as pages_out:
        for lang in LANGUAGES:
            text_dir = OUT_DIR / "text" / lang
            text_dir.mkdir(parents=True, exist_ok=True)

            for pdf_path in sorted((ROOT / lang).glob("*.pdf")):
                doc_id = make_doc_id(lang, pdf_path)
                pages = extract_pages(pdf_path)
                n_pages = page_count(pdf_path)
                if len(pages) != n_pages:
                    print(f"WARNING {doc_id}: pdfinfo reports {n_pages} pages, extracted {len(pages)}")

                for page_no, page_text in enumerate(pages, start=1):
                    pages_out.write(json.dumps({
                        "doc_id": doc_id,
                        "language": lang,
                        "page": page_no,
                        "n_words": len(page_text.split()),
                        "text": page_text,
                    }, ensure_ascii=False) + "\n")

                (text_dir / f"{doc_id}.txt").write_text("\f".join(pages), encoding="utf-8")

                word_counts = [len(p.split()) for p in pages]
                manifest_rows.append({
                    "doc_id": doc_id,
                    "language": lang,
                    "source_file": pdf_path.relative_to(ROOT).as_posix(),
                    "n_pages": n_pages,
                    "n_words": sum(word_counts),
                    "empty_pages": sum(1 for c in word_counts if c < 20),
                    "size_bytes": pdf_path.stat().st_size,
                    "sha256": sha256(pdf_path),
                })
                print(f"{doc_id}: {n_pages} pages, {sum(word_counts)} words")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    for lang in LANGUAGES:
        rows = [r for r in manifest_rows if r["language"] == lang]
        print(f"{lang}: {len(rows)} documents, {sum(r['n_pages'] for r in rows)} pages, "
              f"{sum(r['n_words'] for r in rows)} words")


if __name__ == "__main__":
    main()
