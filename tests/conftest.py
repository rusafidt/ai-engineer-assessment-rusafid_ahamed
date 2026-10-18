import json

import pytest

from app.retrieval import Chunk, DocumentIndex
from app.superhero import Hero


class FakeLLM:
    """Returns canned replies in order and records what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeHeroes:
    def __init__(self, db=None, errors=None):
        self.db = db or {}
        self.errors = errors or {}
        self.searched = []

    async def search(self, name):
        self.searched.append(name)
        if name in self.errors:
            raise self.errors[name]
        return self.db.get(name.lower(), [])


def route_json(superhero=False, documents=False, heroes=(), query=""):
    return json.dumps(
        {"superhero": superhero, "documents": documents, "heroes": list(heroes), "query": query}
    )


def make_hero(name="Batman", id="70"):
    return Hero.from_api(
        {
            "id": id,
            "name": name,
            "powerstats": {"intelligence": "100", "strength": "26", "speed": "27"},
            "biography": {"full-name": "Bruce Wayne", "publisher": "DC Comics", "aliases": ["Dark Knight"]},
            "appearance": {"gender": "Male", "race": "Human"},
            "work": {"occupation": "Businessman", "base": "Batcave"},
            "connections": {"group-affiliation": "Justice League"},
        }
    )


@pytest.fixture
def index():
    return DocumentIndex(
        [
            Chunk("cca", "Comics Code Authority", "https://example.org/cca", "Formation", 0,
                  "The Comics Code Authority was formed in 1954 after Senate hearings on juvenile delinquency."),
            Chunk("lee", "Stan Lee", "https://example.org/lee", "Career", 0,
                  "Stan Lee co-created Spider-Man with Steve Ditko in 1962 for Amazing Fantasy."),
            Chunk("kane", "Bob Kane", "https://example.org/kane", "", 0,
                  "Bob Kane and Bill Finger created Batman, who debuted in Detective Comics in 1939."),
        ]
    )
