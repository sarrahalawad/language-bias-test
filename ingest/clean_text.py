"""Step 2 of data ingestion: clean the extracted text.

Reads data/processed/pages.jsonl (step 1) and writes:
  data/clean/manifest.csv       - one row per document with word counts before/after cleaning
  data/clean/documents.jsonl    - one record per document with its cleaned text
  data/clean/text/<LANG>/<doc_id>.txt - cleaned text per document, one paragraph per line
  data/clean/removed.jsonl      - every removed line/page with the reason, for review

Cleaning steps:
  1. Drop non-content pages (covers, tables of contents, imprints, forms, ads).
  2. Cut trailing sections (acknowledgements, notes, references, foreign-language summary).
  3. Drop footnotes at the bottom of pages.
  4. Drop repeated running headers/footers, page numbers, download notices,
     figure labels, all-caps headings/labels and garbled lines.
  5. Remove footnote reference numbers stuck to words ("Egypt14" -> "Egypt").
  6. Rejoin words hyphenated at line breaks and unwrap lines into paragraphs.

Usage:
    python ingest/clean_text.py
"""

import collections
import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PAGES = ROOT / "data" / "processed" / "pages.jsonl"
OUT_DIR = ROOT / "data" / "clean"

# Per-document rules, chosen by reading each source.
#   drop_pages:     page numbers (1-based) that are not article content
#   cut_from:       regex for a line where trailing back matter starts; it and everything after are dropped
#   footnote_start: regex for the first line of the footnote block at the bottom of a page
#   drop_lines:     regex for extra lines to drop
#   footnote_refs:  strip footnote reference numbers attached to words
DOC_RULES = {
    "de_maat_ausgabe_01_online": {
        # cover, contents, imprint, back cover
        "drop_pages": [1, 2, 38, 39],
    },
    "de_maat_ausgabe_02_online": {
        # cover, membership form, imprint, back cover
        "drop_pages": [1, 43, 44, 45],
    },
    "en_ancient_egyptian_funerary_tools_and_beliefs": {
        "cut_from": r"^Acknowledg(e)?ments$",
        "footnote_start": r"^\d{1,3}$",
        "footnote_refs": True,
    },
    "en_burial_and_the_dead_in_ancient_egyptian_society": {
        "cut_from": r"^Acknowledg(e)?ments$",
        "footnote_refs": True,
    },
    "en_meskell_archaeologieslifedeath_1999": {
        # JSTOR cover sheet
        "drop_pages": [1],
        "footnote_start": r"^\d{1,3}o?\s+\S",
        "drop_lines": r"^(NEW COLLEGE|OXFORD UNIVERSITY|OXFORD OX1 3BN|LYNN\.MESKELL@.*)$",
        "footnote_refs": True,
    },
    "en_mummies_mummification_techniques_and_the": {
        # "Korrektur unklar" is a typesetter's note left in the PDF
        "drop_lines": r"^Korrektur unklar$",
    },
    "en_mummification_in_the_ancient_and_new_wor": {
        # Croatian summary at the end
        "cut_from": r"^Sažetak$",
        "footnote_start": r"^(\d{1,3}|\*)$",
        "drop_lines": r"^(Pregledni rad.*|Review article.*|MUMIFIKACIJA U STAROM I NOVOM SVIJETU|\.H\\ZRUGV:.*)$",
        "footnote_refs": True,
    },
}

BOILERPLATE = [
    r"^This content downloaded from$",
    r"^\d+\.\d+\.\d+\.\d+ on .* UTC$",
    r"^All use subject to https://about\.jstor\.org/terms$",
    r"^Downloaded from$",
    r"^http://jsa\.sagepub\.com$",
    r"^at Oxford University Libraries on .*$",
    r"^Downloaded from http://.*$",
    r"^Page \d+$",
    r"^ISSN:? [\dX-]+.*$",
    r"^www\.\S+$",
]
FIGURE_LABEL = r"^((Abb|Fig|Figure)\.?\s*\d+[a-z]?([–/-]\d+)?\.?\s*)+$"

# A repeated line counts as a running header/footer if it appears on at least
# this share of a document's pages (with digits ignored).
HEADER_MIN_SHARE = 0.25
# Footnote blocks are only looked for in the bottom part of a page.
FOOTNOTE_ZONE = 0.6
# Words that may follow a hyphen at a line end without being joined (German "Kunst- und Kulturgeschichte").
KEEP_HYPHEN_BEFORE = {"und", "oder", "bzw", "sowie", "bis", "noch", "and", "or", "to"}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    # control characters left over from broken font encodings
    return "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch) != "Cc")


def header_key(line):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def find_running_headers(pages):
    counts = collections.Counter()
    for lines in pages.values():
        counts.update({header_key(l) for l in lines if re.search(r"[A-Za-zÄÖÜäöü]", l) and len(l) < 120})
    threshold = max(3, HEADER_MIN_SHARE * len(pages))
    return {k for k, c in counts.items() if c >= threshold}


def is_garbled(line):
    # text set in a font without a proper character map comes out as symbol soup, e.g. "0800,),&$7,21,17+("
    return len(line) >= 8 and " " not in line and re.search(r"[$&+]", line) is not None


def is_caps_line(line):
    # all-caps headings, author bylines and chart labels
    return (re.search(r"[A-ZÄÖÜ]{2}", line) and not re.search(r"[a-zäöüß]", line)
            and len(line.split()) <= 8)


def clean_document(doc_id, pages, removed):
    rules = DOC_RULES.get(doc_id, {})
    drop_pages = set(rules.get("drop_pages", []))
    cut_from = re.compile(rules["cut_from"]) if "cut_from" in rules else None
    footnote_start = re.compile(rules["footnote_start"]) if "footnote_start" in rules else None
    drop_lines = re.compile(rules["drop_lines"]) if "drop_lines" in rules else None
    boilerplate = [re.compile(p) for p in BOILERPLATE]

    def log(page, reason, text):
        removed.append({"doc_id": doc_id, "page": page, "reason": reason, "text": text})

    page_lines = {}
    for page_no, text in pages:
        lines = [l.strip() for l in normalize(text).split("\n")]
        lines = [l for l in lines if l]
        if page_no in drop_pages:
            log(page_no, "non-content page", "\n".join(lines))
            continue
        page_lines[page_no] = lines

    headers = find_running_headers(page_lines)
    kept = []  # (page_no, line)
    cut = False
    for page_no, lines in page_lines.items():
        if cut:
            log(page_no, "back matter", "\n".join(lines))
            continue

        # remove boilerplate first so it doesn't hide the footnote block or count as body
        body = []
        for line in lines:
            if any(p.match(line) for p in boilerplate):
                log(page_no, "boilerplate", line)
            elif header_key(line) in headers:
                log(page_no, "running header/footer", line)
            elif re.fullmatch(r"\d{1,3}", line) and not (footnote_start and footnote_start.match(line)):
                log(page_no, "page number", line)
            else:
                body.append(line)

        if cut_from:
            for i, line in enumerate(body):
                if cut_from.match(line):
                    log(page_no, "back matter", "\n".join(body[i:]))
                    body = body[:i]
                    cut = True
                    break

        if footnote_start:
            zone_start = int(len(body) * (1 - FOOTNOTE_ZONE))
            for i in range(zone_start, len(body)):
                if footnote_start.match(body[i]):
                    log(page_no, "footnotes", "\n".join(body[i:]))
                    body = body[:i]
                    break

        for line in body:
            if re.fullmatch(r"\d{1,3}( \d{1,3})?", line):
                log(page_no, "page number", line)
            elif re.match(FIGURE_LABEL, line):
                log(page_no, "figure label", line)
            elif drop_lines and drop_lines.match(line):
                log(page_no, "document-specific", line)
            elif is_garbled(line):
                log(page_no, "garbled", line)
            elif is_caps_line(line):
                log(page_no, "all-caps heading/label", line)
            else:
                if rules.get("footnote_refs"):
                    line = re.sub(r"([A-Za-z][.,;:)”\"’]{0,2})\d{1,3}(?=\s|$)", r"\1", line)
                kept.append(line)
    return kept


def join_lines(lines, hyphenated_forms):
    """Rejoin hyphenated words and unwrap lines into paragraphs."""
    lengths = sorted(len(l) for l in lines) or [0]
    typical = lengths[int(len(lengths) * 0.75)]
    paragraphs, current = [], ""
    for line in lines:
        if not current:
            current = line
        elif current.endswith("-") and not current.endswith(" -"):
            head = current[:-1]
            first = re.match(r"[^\s]+", line).group(0)
            first_word = re.sub(r"\W+$", "", first)
            prefix = re.search(r"(\S+)$", head).group(1)
            if first_word.lower() in KEEP_HYPHEN_BEFORE or first[:1].isupper() or not first[:1].isalpha():
                current = current + line if first[:1].isupper() else current + " " + line
            elif f"{prefix}-{first_word}".lower() in hyphenated_forms:
                current = current + line
            else:
                current = head + line
        elif re.fullmatch(r"[.,;:]", line):
            current += line
        else:
            current = current + " " + line
        # a short line ending a sentence closes the paragraph
        if re.search(r"[.!?:“”\"]$", line) and len(line) < 0.7 * typical:
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return [re.sub(r"\s+", " ", p).strip() for p in paragraphs]


def main():
    docs = collections.OrderedDict()
    raw_words = collections.Counter()
    language = {}
    for rec in map(json.loads, open(IN_PAGES, encoding="utf-8")):
        docs.setdefault(rec["doc_id"], []).append((rec["page"], rec["text"]))
        raw_words[rec["doc_id"]] += rec["n_words"]
        language[rec["doc_id"]] = rec["language"]

    removed, cleaned = [], {}
    for doc_id, pages in docs.items():
        cleaned[doc_id] = clean_document(doc_id, pages, removed)

    # hyphenated compounds written inside a line somewhere in the same language keep their hyphen
    hyphenated_forms = collections.defaultdict(set)
    for doc_id, lines in cleaned.items():
        for line in lines:
            hyphenated_forms[language[doc_id]].update(
                m.lower() for m in re.findall(r"\w+-\w+", line))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT_DIR / "text", ignore_errors=True)
    manifest = []
    with open(OUT_DIR / "documents.jsonl", "w", encoding="utf-8") as docs_out:
        for doc_id, lines in cleaned.items():
            lang = language[doc_id]
            paragraphs = join_lines(lines, hyphenated_forms[lang])
            text = "\n".join(paragraphs)
            text_dir = OUT_DIR / "text" / lang
            text_dir.mkdir(parents=True, exist_ok=True)
            (text_dir / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")
            n_words = len(text.split())
            docs_out.write(json.dumps({
                "doc_id": doc_id, "language": lang, "n_words": n_words,
                "n_paragraphs": len(paragraphs), "text": text,
            }, ensure_ascii=False) + "\n")
            manifest.append({
                "doc_id": doc_id, "language": lang, "raw_words": raw_words[doc_id],
                "clean_words": n_words, "kept_share": round(n_words / raw_words[doc_id], 3),
                "n_paragraphs": len(paragraphs),
            })
            print(f"{doc_id}: {raw_words[doc_id]} -> {n_words} words")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    with open(OUT_DIR / "removed.jsonl", "w", encoding="utf-8") as f:
        for rec in removed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for lang in sorted(set(language.values())):
        rows = [r for r in manifest if r["language"] == lang]
        print(f"{lang}: {len(rows)} documents, {sum(r['raw_words'] for r in rows)} -> "
              f"{sum(r['clean_words'] for r in rows)} words")
    reasons = collections.Counter(r["reason"] for r in removed)
    print("removed:", dict(reasons))


if __name__ == "__main__":
    main()
