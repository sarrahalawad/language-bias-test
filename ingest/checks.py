"""Text checks shared by the question and twin notebooks (answer matching, word overlap, language)."""

import re

from lexical import tokenize

OTHER = {"DE": "EN", "EN": "DE"}

FUNCTION_WORDS = {
    "DE": {"der", "die", "das", "und", "wer", "was", "wann", "wo", "welche", "welcher", "welches", "welchem",
           "welchen", "wie", "im", "von", "mit", "den", "dem", "des", "wurde", "wurden", "ist", "zu", "zur",
           "zum", "auf", "für", "ein", "eine", "einen", "einem", "einer", "gehört", "war", "waren", "nicht",
           "sich", "auch", "als", "bei", "aus"},
    "EN": {"the", "and", "who", "what", "when", "where", "which", "how", "of", "with", "was", "were", "is",
           "to", "did", "does", "for", "a", "an", "on", "belongs", "by", "not", "also", "as", "at", "from"},
}


def answer_pattern(answer):
    """Regex for the answer's words in order, ignoring case and punctuation/quotes between them."""
    words = re.findall(r"\w+", answer)
    if not words:
        return None
    return re.compile(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", re.IGNORECASE)


def contains(text, answer):
    pattern = answer_pattern(answer)
    return bool(pattern and pattern.search(text))


def find_span(answer, text):
    """The exact text that matches the answer, or None."""
    pattern = answer_pattern(answer)
    match = pattern.search(text) if pattern else None
    return match.group(0) if match else None


def contains_loosely(text, answer, min_share=0.7):
    """True if most of the answer's words occur in the text, allowing different word endings
    (German inflection: "komplexe Behandlungen" ~ "komplexen Behandlungen")."""
    answer_words = [w.lower() for w in re.findall(r"\w+", answer)]
    text_stems = {w.lower()[:5] for w in re.findall(r"\w+", text)}
    if not answer_words:
        return False
    return sum(w[:5] in text_stems for w in answer_words) / len(answer_words) >= min_share


def word_overlap(text, reference, lang):
    """Share of the text's content words that also occur in the reference text."""
    tokens = tokenize(text, lang)
    reference_tokens = set(tokenize(reference, lang))
    return sum(t in reference_tokens for t in tokens) / max(len(tokens), 1)


def stem_overlap(text, reference, lang, stem=5):
    """Like word_overlap, but words count as equal when their first `stem` letters match,
    so German inflection ("Harz" ~ "Harzes", "verwendet" ~ "verwendeten") does not count as a miss."""
    tokens = tokenize(text, lang)
    reference_stems = {t[:stem] for t in tokenize(reference, lang)}
    return sum(t[:stem] in reference_stems for t in tokens) / max(len(tokens), 1)


def numbers(text):
    return sorted(re.findall(r"\d+", text))


def number_coverage(original, translation):
    """Share of the original's numbers (counted with repeats) that also occur in the translation;
    1.0 if the original has no numbers."""
    original_numbers = numbers(original)
    if not original_numbers:
        return 1.0
    remaining = numbers(translation)
    found = 0
    for n in original_numbers:
        if n in remaining:
            remaining.remove(n)
            found += 1
    return found / len(original_numbers)


def language_counts(text, lang):
    words = re.findall(r"\w+", text.lower())
    own = sum(w in FUNCTION_WORDS[lang] for w in words)
    other = sum(w in FUNCTION_WORDS[OTHER[lang]] for w in words)
    return own, other


def is_language(text, lang):
    """Rough check for short texts: fails only if the other language's function words dominate."""
    own, other = language_counts(text, lang)
    return own >= other


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?…])[\"“”’)]?\s+(?=[\"„“(]?[A-ZÄÖÜ0-9])", text) if s.strip()]
