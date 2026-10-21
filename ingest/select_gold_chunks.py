"""Step 4 of data ingestion: select the candidate gold chunks that questions will be written for.

Reads data/chunks/chunks.jsonl (step 3) and writes:
  data/gold/candidates.jsonl - the candidate chunks, ranked per language (pool_rank 1 = best)

A good gold chunk is running prose (not a caption, list or citation block), contains
specific facts (names, numbers, dates) and is distinctive, i.e. its content is not
repeated in other chunks of the same language. That way a question written from it
can be answered from that chunk only.

Every eligible chunk can become a candidate, except that chunks next to each other in a
document are not both taken (the largest possible set without neighbours is kept). The
pool is larger than the 60 facts per language the experiment needs, so chunks whose
questions fail the checks or the manual review can be replaced. Question generation
works through the pool in pool_rank order. Chunks found unsuitable on reading are
listed in EXCLUDED with the reason.

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
    candidates_all = []
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
                candidates.append({**c, "fact_density": round(facts, 2), "max_similarity": round(max_sim, 3),
                                   "noise_share": round(noise, 3), "score": facts * (1 - max_sim)})

        by_doc = collections.defaultdict(list)
        for c in candidates:
            by_doc[c["doc_id"]].append(c)
        pool = []
        for doc_id, cs in sorted(by_doc.items()):
            # walking through the document in order and skipping direct neighbours gives
            # the largest possible set of chunks without neighbours
            taken = []
            for c in sorted(cs, key=lambda c: c["chunk_index"]):
                if not taken or c["chunk_index"] - taken[-1]["chunk_index"] > 1:
                    taken.append(c)
            pool.extend(taken)
            print(f"{doc_id}: {len(cs)} eligible, {len(taken)} without neighbours")
        # favour fact-rich and distinctive chunks; on the German side every noun is
        # capitalised, so facts are ranked within each language only
        for rank, c in enumerate(sorted(pool, key=lambda c: c["score"], reverse=True), start=1):
            c["pool_rank"] = rank
        candidates_all.extend(sorted(pool, key=lambda c: c["pool_rank"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "candidates.jsonl", "w", encoding="utf-8") as f:
        for c in candidates_all:
            c.pop("score")
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for lang, n in sorted(collections.Counter(c["language"] for c in candidates_all).items()):
        print(f"{lang}: {n} candidate chunks")


if __name__ == "__main__":
    main()
