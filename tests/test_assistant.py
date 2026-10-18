import pytest

from app.assistant import NOTHING_FOUND, OFF_TOPIC, Assistant
from app.errors import UpstreamError
from app.schemas import SourceType
from tests.conftest import FakeHeroes, FakeLLM, make_hero, route_json


def assistant(llm, heroes, index):
    return Assistant(llm, heroes, index, top_k=2)


async def test_superhero_only(index):
    llm = FakeLLM(route_json(superhero=True, heroes=["Batman"]), "Batman's strength is 26 [S1].")
    heroes = FakeHeroes({"batman": [make_hero()]})

    res = await assistant(llm, heroes, index).ask("How strong is Batman?")

    assert res.route == [SourceType.SUPERHERO_API]
    assert [s.type for s in res.sources] == [SourceType.SUPERHERO_API]
    assert "Bruce Wayne" in llm.calls[1]["user"]
    assert "Stan Lee" not in llm.calls[1]["user"]  # documents weren't searched


async def test_documents_only_does_not_touch_the_api(index):
    llm = FakeLLM(route_json(documents=True, query="comics code formed"), "It was formed in 1954 [S1].")
    heroes = FakeHeroes()

    res = await assistant(llm, heroes, index).ask("When was the Comics Code formed?")

    assert heroes.searched == []
    assert res.sources[0].type == SourceType.DOCUMENTS
    assert res.sources[0].title == "Comics Code Authority > Formation"


async def test_both_sources_and_only_cited_ones_returned(index):
    llm = FakeLLM(
        route_json(superhero=True, documents=True, heroes=["Batman"], query="Batman created Bob Kane"),
        "Created by Bob Kane [S2], intelligence 100 [S1].",
    )
    heroes = FakeHeroes({"batman": [make_hero()]})

    res = await assistant(llm, heroes, index).ask("Who created Batman and how smart is he?")

    assert res.route == [SourceType.SUPERHERO_API, SourceType.DOCUMENTS]
    assert {s.id for s in res.sources} == {"S1", "S2"}
    assert {s.type for s in res.sources} == {SourceType.SUPERHERO_API, SourceType.DOCUMENTS}


async def test_no_citations_returns_all_sources(index):
    llm = FakeLLM(route_json(superhero=True, heroes=["Batman"]), "He is very smart.")
    res = await assistant(llm, FakeHeroes({"batman": [make_hero()]}), index).ask("Is Batman smart?")
    assert len(res.sources) == 1


async def test_off_topic_skips_second_llm_call(index):
    llm = FakeLLM(route_json())
    res = await assistant(llm, FakeHeroes(), index).ask("Best pasta recipe?")

    assert res.answer == OFF_TOPIC
    assert res.sources == [] and res.route == []
    assert len(llm.calls) == 1


async def test_unknown_hero_answers_without_calling_llm(index):
    llm = FakeLLM(route_json(superhero=True, heroes=["Zzyzx"]))
    res = await assistant(llm, FakeHeroes(), index).ask("How strong is Zzyzx?")

    assert "no character named 'Zzyzx'" in res.answer
    assert NOTHING_FOUND in res.answer
    assert len(llm.calls) == 1


async def test_api_down_but_documents_still_answer(index):
    llm = FakeLLM(
        route_json(superhero=True, documents=True, heroes=["Batman"], query="Batman Bob Kane"),
        "Bob Kane created him [S1].",
    )
    heroes = FakeHeroes(errors={"Batman": UpstreamError("superhero_api", "timed out", timeout=True)})

    res = await assistant(llm, heroes, index).ask("Who created Batman and how strong is he?")

    assert res.sources[0].type == SourceType.DOCUMENTS
    assert "could not be reached" in llm.calls[1]["user"]


async def test_api_down_with_nothing_else_raises(index):
    llm = FakeLLM(route_json(superhero=True, heroes=["Batman"]))
    heroes = FakeHeroes(errors={"Batman": UpstreamError("superhero_api", "down")})

    with pytest.raises(UpstreamError):
        await assistant(llm, heroes, index).ask("How strong is Batman?")


async def test_api_token_never_leaks_into_sources(index):
    llm = FakeLLM(route_json(superhero=True, heroes=["Batman"]), "[S1]")
    res = await assistant(llm, FakeHeroes({"batman": [make_hero()]}), index).ask("Batman?")
    assert res.sources[0].reference == "superheroapi.com character id 70"
