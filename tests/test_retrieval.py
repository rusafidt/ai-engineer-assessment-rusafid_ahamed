from pathlib import Path

from app.retrieval import Chunk, DocumentIndex, load_document, split_into_chunks, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Who created the Comics Code?") == ["created", "comics", "code"]


def test_chunks_respect_size_and_sections():
    body = "== Early life ==\n" + "word " * 250 + "\n\nshort paragraph\n== Career ==\nmore text"
    chunks = split_into_chunks(body, max_words=100)

    assert all(len(text.split()) <= 100 for _, text in chunks)
    assert chunks[0][0] == "Early life"
    assert chunks[-1] == ("Career", "more text")


def test_search_ranks_the_relevant_chunk_first(index):
    results = index.search("when was the comics code authority formed")
    assert results[0].doc_id == "cca"


def test_title_and_section_are_searchable():
    index = DocumentIndex(
        [
            Chunk("lee", "Stan Lee", "", "Early life", 0, "He was born in New York City in 1922."),
            Chunk("kirby", "Jack Kirby", "", "Career", 0, "He drew the Fantastic Four."),
        ]
    )
    assert index.search("stan lee early life")[0].doc_id == "lee"


def test_search_with_no_matches_or_only_stopwords(index):
    assert index.search("pasta carbonara") == []
    assert index.search("what is the") == []


def test_load_document_reads_header(tmp_path: Path):
    f = tmp_path / "stan_lee.txt"
    f.write_text("# Stan Lee\nSource: https://en.wikipedia.org/wiki/Stan_Lee\n\nHe wrote comics.\n")
    [chunk] = load_document(f)
    assert chunk.title == "Stan Lee"
    assert chunk.url.endswith("Stan_Lee")
    assert chunk.text == "He wrote comics."


def test_bundled_dataset_loads():
    index = DocumentIndex.from_dir(Path(__file__).parent.parent / "data")
    assert len(index.titles) >= 5
    assert index.search("Comics Code Authority 1954")[0].doc_id == "comics_code_authority"
