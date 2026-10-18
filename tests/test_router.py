from app.router import Router, parse_route
from tests.conftest import FakeLLM, route_json

Q = "some question"


def test_parses_valid_output():
    route = parse_route(route_json(superhero=True, heroes=["Batman"]), Q)
    assert route.superhero and not route.documents
    assert route.heroes == ["Batman"]


def test_strips_markdown_fences():
    raw = "```json\n" + route_json(documents=True, query="comics code") + "\n```"
    assert parse_route(raw, Q).query == "comics code"


def test_garbage_falls_back_to_documents():
    route = parse_route("sure! here you go: superhero", Q)
    assert route.documents and not route.superhero
    assert route.query == Q


def test_wrong_types_fall_back_too():
    route = parse_route('{"superhero": "maybe", "heroes": "Batman"}', Q)
    assert route.documents and route.query == Q


def test_superhero_without_names_is_dropped():
    route = parse_route(route_json(superhero=True, heroes=[]), Q)
    assert not route.superhero


def test_heroes_are_deduped_and_capped():
    route = parse_route(route_json(superhero=True, heroes=["Thor", "thor ", "Hulk", "Loki", "Odin"]), Q)
    assert route.heroes == ["Thor", "Hulk", "Loki"]


def test_documents_without_query_uses_question():
    assert parse_route(route_json(documents=True, query="  "), Q).query == Q


async def test_router_sends_doc_titles_and_asks_for_json():
    llm = FakeLLM(route_json())
    await Router(llm, ["Stan Lee", "DC Comics"]).route("hello")

    call = llm.calls[0]
    assert "Stan Lee, DC Comics" in call["system"]
    assert call["json_mode"] is True
    assert call["temperature"] == 0
