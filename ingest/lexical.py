"""Tokenization, TF-IDF similarity and BM25 search shared by the gold-chunk and question steps."""

import collections
import math
import re

STOPWORDS = {
    "DE": set("""
        aber alle allem allen aller alles als also am an ander andere anderen anderer anderes auch auf aus bei
        beim bereits bis bisher bzw da dabei damit dann darauf darin das dass dem den denen denn der deren des
        dessen die dies diese diesem diesen dieser dieses doch dort durch ein eine einem einen einer eines er
        es etwa für gab gibt hat hatte hatten hier ihr ihre ihrem ihren ihrer im immer in ins ist ja jedoch
        kann kein keine können konnte man mehr mit nach nicht noch nun nur ob oder ohne sehr seit sein seine
        seinem seinen seiner sich sie sind so sowie über um und uns unter vom von vor war waren warum was
        weil welche welchem welchen welcher welches wenn wer werden wie wieder wird wo wurde wurden zu zum
        zur zwei zwischen wann wem wen wessen wodurch womit woraus worin wozu viele vielen welchen
    """.split()),
    "EN": set("""
        a about after all also an and any are as at be been before being between both but by can could did
        do does during each for from had has have he her his how i if in into is it its may more most much
        must no not of on one or other our over said she should so some such than that the their them then
        there these they this those through to two under up upon was we were what when where which while who
        whom whose why will with would within without
    """.split()),
}


def tokenize(text, lang):
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS[lang] and not t.isdigit() and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in docs) / len(docs)
        df = collections.Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [collections.Counter(d) for d in docs]

    def scores(self, query):
        out = []
        for tf, doc in zip(self.tf, self.docs):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            out.append(sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                           for t in query if t in tf))
        return out


def tfidf_vectors(docs):
    n = len(docs)
    df = collections.Counter(t for d in docs for t in set(d))
    vectors = []
    for d in docs:
        tf = collections.Counter(d)
        v = {t: (1 + math.log(c)) * math.log(n / df[t]) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vectors.append({t: x / norm for t, x in v.items()})
    return vectors


def cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())
