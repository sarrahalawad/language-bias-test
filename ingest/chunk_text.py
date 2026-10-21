"""Step 3 of data ingestion: split the cleaned documents into chunks.

Reads data/clean/documents.jsonl (step 2) and writes:
  data/chunks/chunks.jsonl  - one record per chunk (chunk_id, doc_id, language, position, size, text)
  data/chunks/summary.csv   - chunk counts and size statistics per language

Chunks are built from whole paragraphs, packed up to TARGET_WORDS. Paragraphs longer
than that are split at sentence boundaries. Chunks do not overlap, so no text is
counted twice when comparing languages.

Usage:
    python ingest/chunk_text.py
"""

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_DOCS = ROOT / "data" / "clean" / "documents.jsonl"
OUT_DIR = ROOT / "data" / "chunks"

TARGET_WORDS = 250
# a trailing chunk shorter than this is merged into the previous one
MIN_WORDS = 100

# Abbreviations that end with a period but do not end a sentence.
ABBREVIATIONS = {
    "DE": ["z. B.", "u. a.", "v. Chr.", "n. Chr.", "ca.", "bzw.", "vgl.", "Abb.", "Dr.", "Prof.", "S.",
           "Nr.", "Kat.", "Jh.", "Jhs.", "usw.", "d. h.", "s.", "sog.", "inv.", "Inv.", "M. A.", "H.", "Br.",
           "L.", "T.", "St.", "evtl.", "ggf."],
    "EN": ["e.g.", "i.e.", "c.", "ca.", "cf.", "Fig.", "fig.", "figs.", "Dr.", "Prof.", "St.", "Mr.",
           "Mrs.", "no.", "No.", "vol.", "Vol.", "pp.", "p.", "ed.", "eds.", "et al.", "inv.", "op. cit.",
           "Mt.", "vs.", "approx."],
}


def split_sentences(paragraph, lang):
    # protect abbreviations and initials ("A. Spencer") so their periods are not split on
    protected = paragraph
    for abbr in ABBREVIATIONS[lang]:
        protected = protected.replace(abbr, abbr.replace(".", "\0"))
    protected = re.sub(r"\b([A-ZÄÖÜ])\.(?=\s)", "\\1\0", protected)
    parts = re.split(r"(?<=[.!?…])[\"“”’)]?\s+(?=[\"„“(]?[A-ZÄÖÜ0-9])", protected)
    return [p.replace("\0", ".").strip() for p in parts if p.strip()]


def word_count(text):
    return len(text.split())


def units_for(paragraphs, lang):
    """Paragraphs, with long ones broken into sentences."""
    for para in paragraphs:
        if word_count(para) <= TARGET_WORDS:
            yield para
        else:
            yield from split_sentences(para, lang)


def chunk_document(text, lang):
    paragraphs = [p for p in text.split("\n") if p.strip()]
    chunks, current = [], []
    for unit in units_for(paragraphs, lang):
        size = sum(word_count(u) for u in current)
        if current and size + word_count(unit) > TARGET_WORDS and size >= MIN_WORDS:
            chunks.append(current)
            current = []
        current.append(unit)
    if current:
        if chunks and sum(word_count(u) for u in current) < MIN_WORDS:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    return [" ".join(c) for c in chunks]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sizes = {}
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as out:
        for doc in map(json.loads, open(IN_DOCS, encoding="utf-8")):
            lang = doc["language"]
            chunks = chunk_document(doc["text"], lang)
            for i, chunk in enumerate(chunks):
                out.write(json.dumps({
                    "chunk_id": f"{doc['doc_id']}_c{i:03d}",
                    "doc_id": doc["doc_id"],
                    "language": lang,
                    "chunk_index": i,
                    "n_words": word_count(chunk),
                    "n_chars": len(chunk),
                    "text": chunk,
                }, ensure_ascii=False) + "\n")
                sizes.setdefault(lang, []).append((word_count(chunk), len(chunk)))
            print(f"{doc['doc_id']}: {len(chunks)} chunks")

    rows = []
    for lang, values in sorted(sizes.items()):
        words = [w for w, _ in values]
        chars = [c for _, c in values]
        rows.append({
            "language": lang,
            "n_chunks": len(values),
            "total_words": sum(words),
            "total_chars": sum(chars),
            "median_words": statistics.median(words),
            "min_words": min(words),
            "max_words": max(words),
            "median_chars": statistics.median(chars),
        })
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
