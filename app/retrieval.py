import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
HEADING_RE = re.compile(r"^=+\s*(.+?)\s*=+$")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have he her his how i in is it its "
    "me my of on or she so than that the their them they this to was were what when where which "
    "who whom why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    url: str
    section: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def split_into_chunks(body: str, max_words: int = 180) -> list[tuple[str, str]]:
    """Group paragraphs into ~max_words chunks, never crossing a section heading."""
    out: list[tuple[str, str]] = []
    section = ""
    buf: list[str] = []

    def flush():
        if buf:
            out.append((section, " ".join(buf)))
            buf.clear()

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if m := HEADING_RE.match(line):
            flush()
            section = m.group(1)
            continue
        words = line.split()
        if buf and len(buf) + len(words) > max_words:
            flush()
        while len(words) > max_words:
            out.append((section, " ".join(words[:max_words])))
            words = words[max_words:]
        buf.extend(words)
    flush()
    return out


def load_document(path: Path, max_words: int = 180) -> list[Chunk]:
    # files start with "# Title" and "Source: <url>", see data/
    lines = path.read_text(encoding="utf-8").splitlines()
    title, url, body_start = path.stem.replace("_", " ").title(), "", 0
    if lines and lines[0].startswith("# "):
        title, body_start = lines[0][2:].strip(), 1
    if len(lines) > 1 and lines[1].startswith("Source:"):
        url, body_start = lines[1].removeprefix("Source:").strip(), 2

    body = "\n".join(lines[body_start:])
    return [
        Chunk(path.stem, title, url, section, i, text)
        for i, (section, text) in enumerate(split_into_chunks(body, max_words))
    ]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 1.0
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        # lucene-style idf, never negative even for very common terms
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        result = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            for term in query:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            result.append(s)
        return result


class DocumentIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # title + section go into the indexed text so "Stan Lee early life" can hit
        # a chunk that never repeats the name
        self.bm25 = BM25([tokenize(f"{c.title} {c.section} {c.text}") for c in chunks])

    @classmethod
    def from_dir(cls, path: Path) -> "DocumentIndex":
        files = sorted(path.glob("*.txt"))
        if not files:
            raise RuntimeError(f"no .txt documents found in {path.resolve()}")
        return cls([c for f in files for c in load_document(f)])

    @property
    def titles(self) -> list[str]:
        return sorted({c.title for c in self.chunks})

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        terms = tokenize(query)
        if not terms:
            return []
        scored = sorted(zip(self.bm25.scores(terms), self.chunks), key=lambda x: x[0], reverse=True)
        return [c for s, c in scored[:k] if s > 0]
