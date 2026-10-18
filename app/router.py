import json
import logging
import re

from pydantic import BaseModel, ValidationError

from app.llm import LLMClient

log = logging.getLogger(__name__)

MAX_HEROES = 3

SYSTEM_PROMPT = """You route questions for a Q&A bot that has two data sources.

"superhero" - the Superhero API. Looked up by character name only. Has powerstats \
(intelligence, strength, speed, durability, power, combat), real name, aliases, publisher, \
alignment, first appearance, gender, race, occupation, base, team affiliations.

"documents" - encyclopedia articles on comic book history: {titles}. Good for creators, \
publishers, eras, censorship, industry history.

Pick every source needed to answer. Use both when the question needs facts from each. \
Use neither when the question has nothing to do with superheroes or comics.

Reply with JSON only, in this exact shape:
{{"superhero": true/false, "documents": true/false, "heroes": ["names to look up"], "query": "keywords for the article search"}}

Examples:
Q: How fast is the Flash?
{{"superhero": true, "documents": false, "heroes": ["Flash"], "query": ""}}
Q: Why was the Comics Code Authority created?
{{"superhero": false, "documents": true, "heroes": [], "query": "Comics Code Authority formation reasons"}}
Q: Who created Spider-Man and how strong is he?
{{"superhero": true, "documents": true, "heroes": ["Spider-Man"], "query": "Spider-Man created Stan Lee Steve Ditko"}}
Q: Compare Thor and Hulk's strength
{{"superhero": true, "documents": false, "heroes": ["Thor", "Hulk"], "query": ""}}
Q: What's a good pasta recipe?
{{"superhero": false, "documents": false, "heroes": [], "query": ""}}"""


class Route(BaseModel):
    superhero: bool = False
    documents: bool = False
    heroes: list[str] = []
    query: str = ""


def parse_route(raw: str, question: str) -> Route:
    """Validate the router output. Bad output falls back to a document search
    instead of failing the request - it's the source that needs no extracted names."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip()
    try:
        route = Route.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("unusable router output, falling back to documents: %s | %r", e, raw[:200])
        return Route(documents=True, query=question)

    seen, heroes = set(), []
    for name in (h.strip() for h in route.heroes):
        if name and name.lower() not in seen:
            seen.add(name.lower())
            heroes.append(name)
    route.heroes = heroes[:MAX_HEROES]

    if route.superhero and not route.heroes:
        route.superhero = False
    if route.documents and not route.query.strip():
        route.query = question
    return route


class Router:
    def __init__(self, llm: LLMClient, doc_titles: list[str]):
        self.llm = llm
        self.system = SYSTEM_PROMPT.format(titles=", ".join(doc_titles))

    async def route(self, question: str) -> Route:
        raw = await self.llm.complete(
            self.system, question, json_mode=True, temperature=0, max_tokens=200
        )
        return parse_route(raw, question)
