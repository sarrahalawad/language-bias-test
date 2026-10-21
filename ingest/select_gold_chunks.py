"""Step 4 of data ingestion: select the gold chunks that questions will be written for.

Reads data/chunks/chunks.jsonl (step 3) and writes:
  data/gold/gold_chunks.jsonl - the selected chunks with their selection scores

A good gold chunk is running prose (not a caption, list or citation block), contains
specific facts (names, numbers, dates) and is distinctive, i.e. its content is not
repeated in other chunks of the same language. That way a question written from it
can be answered from that chunk only.

The same number of chunks is taken per language, spread over the documents in
proportion to how many eligible chunks each document has. Chunks next to each other
in a document are not both selected. Chunks found unsuitable on reading are listed
in EXCLUDED with the reason.

Usage:
    python ingest/select_gold_chunks.py
"""

import collections
import json
import re
from pathlib import Path

from lexical import cosine, tfidf_vectors, tokenize

ROOT = Path(__file__).resolve().parent.parent
IN_CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = ROOT / "data" / "gold"

GOLD_PER_LANGUAGE = 50
MIN_WORDS, MAX_WORDS = 150, 320
# chunks this similar to another chunk of the same language are not distinctive enough
MAX_SIMILARITY = 0.35

# Chunks rejected after reading them, with the reason.
EXCLUDED = {
    "de_maat_ausgabe_02_online_c067": "list of bibliography abbreviations",
    "de_maat_ausgabe_02_online_c124": "publication catalogue (titles, page counts, prices)",
    "de_maat_ausgabe_02_online_c126": "publication catalogue (titles, page counts, prices)",
    "de_maat_ausgabe_02_online_c123": "publication catalogue (titles, page counts, prices)",
    "de_maat_ausgabe_02_online_c125": "publication catalogue (titles, page counts, prices)",
    "en_meskell_archaeologieslifedeath_1999_c004": "chart labels and words cut off at the column edge of the scan",
    "en_meskell_archaeologieslifedeath_1999_c006": "table text without sentences",
    "en_meskell_archaeologieslifedeath_1999_c014": "table and figure text, words cut off at the column edge of the scan",
    "en_meskell_archaeologieslifedeath_1999_c027": "words cut off at the column edge of the scan",
    "en_meskell_archaeologieslifedeath_1999_c005": "words cut off at the column edge of the scan",
    "en_meskell_archaeologieslifedeath_1999_c013": "words cut off at the column edge of the scan",
    "en_meskell_archaeologieslifedeath_1999_c021": "words cut off at the column edge of the scan",
}


def noise_share(text):
    """Share of the text taken up by citations, figure references, URLs and the like."""
    noise = re.findall(r"\([^()]*\d[^()]*\)|https?://\S+|www\.\S+|\b(?:Abb|Fig|Figure|Kat|Inv|inv)\.?\s*\d+", text)
    return sum(len(n) for n in noise) / len(text)


def fact_density(text):
    """Specific facts per 100 words: numbers/dates and capitalised names not at a sentence start."""
    words = text.split()
    numbers = len(re.findall(r"\b\d[\d.,–-]*\b", text))
    names = sum(1 for prev, w in zip(words, words[1:])
                if w[:1].isupper() and not re.search(r"[.!?:“„\"]$", prev))
    return 100 * (numbers + names) / len(words)


def main():
    chunks = [json.loads(l) for l in open(IN_CHUNKS, encoding="utf-8")]
    gold = []
    for lang in sorted({c["language"] for c in chunks}):
        lang_chunks = [c for c in chunks if c["language"] == lang]
        vectors = tfidf_vectors([tokenize(c["text"], lang) for c in lang_chunks])

        candidates = []
        for i, c in enumerate(lang_chunks):
            max_sim = max(cosine(vectors[i], vectors[j]) for j in range(len(lang_chunks)) if j != i)
            noise = noise_share(c["text"])
            facts = fact_density(c["text"])
            eligible = (MIN_WORDS <= c["n_words"] <= MAX_WORDS and noise < 0.1
                        and max_sim < MAX_SIMILARITY and c["chunk_index"] > 0
                        and c["chunk_id"] not in EXCLUDED)
            if eligible:
                # favour fact-rich and distinctive chunks; on the German side every noun is
                # capitalised, so facts are ranked within each language only
                candidates.append({**c, "fact_density": round(facts, 2), "max_similarity": round(max_sim, 3),
                                   "noise_share": round(noise, 3), "score": facts * (1 - max_sim)})

        by_doc = collections.defaultdict(list)
        for c in candidates:
            by_doc[c["doc_id"]].append(c)
        # proportional quota per document (largest remainder)
        quotas = {d: GOLD_PER_LANGUAGE * len(cs) / len(candidates) for d, cs in by_doc.items()}
        alloc = {d: int(q) for d, q in quotas.items()}
        for d in sorted(quotas, key=lambda d: quotas[d] - alloc[d], reverse=True)[:GOLD_PER_LANGUAGE - sum(alloc.values())]:
            alloc[d] += 1

        for doc_id, cs in sorted(by_doc.items()):
            taken = []
            for c in sorted(cs, key=lambda c: c["score"], reverse=True):
                if len(taken) == alloc[doc_id]:
                    break
                if all(abs(c["chunk_index"] - t["chunk_index"]) > 1 for t in taken):
                    taken.append(c)
            gold.extend(sorted(taken, key=lambda c: c["chunk_index"]))
            print(f"{doc_id}: {len(cs)} eligible, {len(taken)} selected")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "gold_chunks.jsonl", "w", encoding="utf-8") as f:
        for c in gold:
            c.pop("score")
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for lang, n in sorted(collections.Counter(c["language"] for c in gold).items()):
        print(f"{lang}: {n} gold chunks")


if __name__ == "__main__":
    main()
