"""Plan step 5: assign each fact to a condition and build the one shared retrieval index.

Conditions (only where the gold passage of a fact is available differs):
  DE-only - the German version of the gold passage is in the index
  EN-only - the English version is in the index
  Both    - both versions are in the index

Reads
  data/questions/questions.jsonl + review/03_questions.csv   facts, questions (edits in the sheet win)
  data/twins/twins.jsonl + review/04_twins.csv               twins (translated gold chunks)
  data/chunks/chunks.jsonl, data/processed/manifest.csv      all chunks, source files
and writes
  data/experiment/facts.jsonl   one record per fact in the experiment: condition, questions/answers in DE and EN,
                                and the passage ids of the fact that are in the index
  data/experiment/index.jsonl   every passage of the shared index (distractors + fact passages)
  review/05_conditions.csv      the assignment, for checking

Which facts take part: a fact needs a valid twin (its twin was not rejected) and must not be
marked "no" in either review sheet (an empty cell counts as yes). Facts whose twin failed are
left out of every condition, not only the ones that use the twin, so that all cells are drawn
from the same kind of facts.

Assignment: the same number of facts per cell (origin language x condition), as many as the
smaller origin language allows (or MAX_PER_CELL). Within each origin language the facts are
shuffled with a fixed seed and dealt out document by document, so every condition gets a mix
of source documents. Facts that are not drawn are not used; their original chunks stay in the
index as ordinary distractors.

Usage:
    python ingest/assign_conditions.py
"""

import collections
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions" / "questions.jsonl"
QUESTION_REVIEW = ROOT / "review" / "03_questions.csv"
TWINS = ROOT / "data" / "twins" / "twins.jsonl"
TWIN_REVIEW = ROOT / "review" / "04_twins.csv"
CHUNKS = ROOT / "data" / "chunks" / "chunks.jsonl"
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
OUT_DIR = ROOT / "data" / "experiment"
REVIEW_OUT = ROOT / "review" / "05_conditions.csv"

SEED = 2026
MAX_PER_CELL = None  # None = as many as the data allows, equal in every cell
LANGUAGES = ["DE", "EN"]
CONDITIONS = ["DE-only", "EN-only", "Both"]
FIELDS = ["question", "answer", "key_fact"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_sheet(path):
    """Review sheet as {fact_id: row}; reads both ',' and Excel's ';' separated files."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","
        return {row["fact_id"]: row for row in csv.DictReader(f, delimiter=delimiter)}


def rejected_in(sheet, fact_id):
    return sheet.get(fact_id, {}).get("approved", "").strip().lower() in {"no", "n", "nein"}


def passages_for(condition, origin, original_id, twin_id):
    """Passage ids of a fact that go into the index under its condition."""
    by_language = {origin: original_id, ("EN" if origin == "DE" else "DE"): twin_id}
    if condition == "Both":
        return [by_language["DE"], by_language["EN"]]
    return [by_language[condition.split("-")[0]]]


def main():
    questions = {q["fact_id"]: q for q in read_jsonl(QUESTIONS)}
    twins = {t["fact_id"]: t for t in read_jsonl(TWINS)}
    question_sheet = read_sheet(QUESTION_REVIEW)
    twin_sheet = read_sheet(TWIN_REVIEW) if TWIN_REVIEW.exists() else {}
    chunks = read_jsonl(CHUNKS)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    with open(MANIFEST, encoding="utf-8") as f:
        source_file = {row["doc_id"]: row["source_file"] for row in csv.DictReader(f)}

    # 1. facts that can take part
    facts, dropped = [], collections.Counter()
    for fact_id, q in questions.items():
        if rejected_in(question_sheet, fact_id):
            dropped[(q["origin_lang"], "question marked no")] += 1
            continue
        if fact_id not in twins:
            dropped[(q["origin_lang"], "no valid twin")] += 1
            continue
        if rejected_in(twin_sheet, fact_id):
            dropped[(q["origin_lang"], "twin marked no")] += 1
            continue
        twin = twins[fact_id]
        origin = q["origin_lang"]
        target = twin["language"].lower()
        fact = {"fact_id": fact_id, "origin_lang": origin, "doc_id": q["doc_id"],
                "gold_chunk_id": q["chunk_id"], "twin_chunk_id": twin["chunk_id"]}
        for lang in ("de", "en"):
            for field in FIELDS:
                key = f"{field}_{lang}"
                value = q.get(key, "")
                if lang == target and twin.get(key):  # parallel version, possibly rewritten to fit the twin
                    value = twin[key]
                for sheet in (question_sheet, twin_sheet):  # your edits in the review sheets win
                    edited = sheet.get(fact_id, {}).get(key, "").strip()
                    if edited and sheet is question_sheet and lang == target and twin.get(key):
                        continue  # the twin step may have rewritten the parallel version after the question review
                    if edited:
                        value = edited
                fact[key] = value
        facts.append(fact)

    available = collections.Counter(f["origin_lang"] for f in facts)
    per_cell = min(available[lang] for lang in LANGUAGES) // len(CONDITIONS)
    if MAX_PER_CELL is not None:
        per_cell = min(per_cell, MAX_PER_CELL)
    if per_cell == 0:
        raise SystemExit(f"Not enough facts with valid twins: {dict(available)}")

    # 2. random assignment, dealt out document by document
    rng = random.Random(SEED)
    assigned = []
    for lang in LANGUAGES:
        by_doc = collections.defaultdict(list)
        for f in sorted((f for f in facts if f["origin_lang"] == lang), key=lambda f: f["fact_id"]):
            by_doc[f["doc_id"]].append(f)
        docs = sorted(by_doc)
        rng.shuffle(docs)
        for doc in docs:
            rng.shuffle(by_doc[doc])
        # round robin over documents gives an order in which documents alternate
        order = []
        while any(by_doc[d] for d in docs):
            for d in docs:
                if by_doc[d]:
                    order.append(by_doc[d].pop())
        chosen = order[:per_cell * len(CONDITIONS)]
        # deal conditions in a random order per round of three, so position does not decide the condition
        for start in range(0, len(chosen), len(CONDITIONS)):
            round_conditions = CONDITIONS[:]
            rng.shuffle(round_conditions)
            for fact, condition in zip(chosen[start:start + len(CONDITIONS)], round_conditions):
                fact["condition"] = condition
                assigned.append(fact)

    # 3. the shared index
    fact_of_passage = {}
    for f in assigned:
        f["passages_in_index"] = passages_for(f["condition"], f["origin_lang"], f["gold_chunk_id"], f["twin_chunk_id"])
        for pid in f["passages_in_index"]:
            fact_of_passage[pid] = f
    gold_ids = {f["gold_chunk_id"] for f in assigned}

    index = []
    for c in chunks:
        if c["chunk_id"] in gold_ids and c["chunk_id"] not in fact_of_passage:
            continue  # original removed: the condition only allows the other language
        f = fact_of_passage.get(c["chunk_id"])
        index.append({"passage_id": c["chunk_id"], "language": c["language"], "doc_id": c["doc_id"],
                      "source": source_file[c["doc_id"]], "is_twin": False, "twin_of": None,
                      "fact_id": f["fact_id"] if f else None, "condition": f["condition"] if f else None,
                      "text": c["text"]})
    for f in assigned:
        twin = twins[f["fact_id"]]
        if twin["chunk_id"] in fact_of_passage:
            index.append({"passage_id": twin["chunk_id"], "language": twin["language"], "doc_id": twin["doc_id"],
                          "source": source_file[twin["doc_id"]], "is_twin": True, "twin_of": twin["twin_of"],
                          "fact_id": f["fact_id"], "condition": f["condition"], "text": twin["text"]})

    # 4. checks
    ids = [p["passage_id"] for p in index]
    assert len(ids) == len(set(ids)), "a passage appears twice in the index"
    in_index = collections.defaultdict(list)
    for p in index:
        if p["fact_id"]:
            in_index[p["fact_id"]].append(p["language"])
    for f in assigned:
        langs = sorted(in_index[f["fact_id"]])
        expected = {"DE-only": ["DE"], "EN-only": ["EN"], "Both": ["DE", "EN"]}[f["condition"]]
        assert langs == expected, f"{f['fact_id']} ({f['condition']}) has {langs} in the index"
    removed = sum(f["condition"] != f"{f['origin_lang']}-only" and f["condition"] != "Both" for f in assigned)
    distractors = sum(p["fact_id"] is None for p in index)
    assert distractors == len(chunks) - len(assigned), "distractor count does not add up"
    assert len(index) == len(chunks) - removed + sum(p["is_twin"] for p in index)
    cells = collections.Counter((f["origin_lang"], f["condition"]) for f in assigned)
    assert set(cells.values()) == {per_cell} and len(cells) == 6, cells

    # 5. outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "facts.jsonl", "w", encoding="utf-8") as out:
        for f in sorted(assigned, key=lambda f: (f["origin_lang"], CONDITIONS.index(f["condition"]), f["fact_id"])):
            out.write(json.dumps(f, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "index.jsonl", "w", encoding="utf-8") as out:
        for p in index:
            out.write(json.dumps(p, ensure_ascii=False) + "\n")

    columns = ["fact_id", "origin_lang", "condition", "doc_id", "passages_in_index",
               "question_de", "answer_de", "question_en", "answer_en"]
    with open(REVIEW_OUT, "w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for f in sorted(assigned, key=lambda f: (f["origin_lang"], CONDITIONS.index(f["condition"]), f["fact_id"])):
            writer.writerow({**f, "passages_in_index": " | ".join(f["passages_in_index"])})

    # 6. report
    print("Facts with a valid twin:", dict(available))
    for (lang, reason), n in sorted(dropped.items()):
        print(f"  left out ({lang}): {n} - {reason}")
    print(f"Facts per cell: {per_cell}  (total {len(assigned)}; not drawn: {len(facts) - len(assigned)})\n")

    print(f"{'origin':<7}{'condition':<10}{'facts':>6}{'words':>7}  documents")
    for lang in LANGUAGES:
        for condition in CONDITIONS:
            cell = [f for f in assigned if f["origin_lang"] == lang and f["condition"] == condition]
            words = sum(chunk_by_id[f["gold_chunk_id"]]["n_words"] for f in cell) / len(cell)
            docs = collections.Counter(f["doc_id"] for f in cell)
            doc_text = ", ".join(f"{d.split('_', 1)[1][:22]} {n}" for d, n in sorted(docs.items()))
            print(f"{lang:<7}{condition:<10}{len(cell):>6}{words:>7.0f}  {doc_text}")

    by_lang = collections.Counter(p["language"] for p in index)
    print(f"\nIndex: {len(index)} passages ({by_lang['DE']} DE, {by_lang['EN']} EN), "
          f"of which {distractors} distractors, {len(index) - distractors} fact passages "
          f"({sum(p['is_twin'] for p in index)} twins); {removed} originals removed.")
    print("All checks passed.")


if __name__ == "__main__":
    main()
