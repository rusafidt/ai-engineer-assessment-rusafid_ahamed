import json

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from app.errors import UpstreamError
from app.llm import LLMClient
from app.superhero import SuperheroClient

BATMAN = {
    "id": "70",
    "name": "Batman",
    "powerstats": {"intelligence": "100", "strength": "26", "speed": "null"},
    "biography": {"full-name": "Bruce Wayne", "publisher": "DC Comics", "aliases": ["-"]},
    "appearance": {"gender": "Male", "race": "Human"},
    "work": {"occupation": "-", "base": "Batcave"},
    "connections": {},
}


def http_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def heroes_client(handler):
    return SuperheroClient(http_client(handler), "https://superheroapi.com/api.php", "secret-token")


async def test_search_parses_and_puts_exact_match_first():
    def handler(request):
        assert request.url.path == "/api.php/secret-token/search/batman"
        batman_ii = {**BATMAN, "id": "69", "name": "Batman II"}
        return httpx.Response(200, json={"response": "success", "results": [batman_ii, BATMAN]})

    heroes = await heroes_client(handler).search("batman")
    assert [h.name for h in heroes] == ["Batman", "Batman II"]


async def test_context_skips_missing_values():
    def handler(request):
        return httpx.Response(200, json={"response": "success", "results": [BATMAN]})

    [hero] = await heroes_client(handler).search("Batman")
    text = hero.to_context()
    assert "intelligence 100, strength 26" in text
    assert "speed" not in text  # "null" in the API
    assert "Occupation" not in text and "Aliases" not in text  # "-" placeholders


async def test_name_is_url_encoded():
    def handler(request):
        assert request.url.raw_path == b"/api.php/secret-token/search/Iron%20Man"
        return httpx.Response(200, json={"response": "success", "results": []})

    await heroes_client(handler).search("Iron Man")


async def test_not_found_is_empty_not_an_error():
    def handler(request):
        return httpx.Response(200, json={"response": "error", "error": "character with given name not found"})

    assert await heroes_client(handler).search("Zzyzx") == []


async def test_bad_token_is_an_upstream_error():
    def handler(request):
        return httpx.Response(200, json={"response": "error", "error": "access denied"})

    with pytest.raises(UpstreamError, match="access denied"):
        await heroes_client(handler).search("Batman")


async def test_superhero_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(UpstreamError) as e:
        await heroes_client(handler).search("Batman")
    assert e.value.timeout


def llm_client(handler):
    sdk = AsyncOpenAI(
        api_key="key",
        base_url="https://llm.test/v1",
        max_retries=1,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return LLMClient(sdk, "some-model")


def chat_reply(text):
    return httpx2.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "some-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        },
    )


async def test_llm_sends_expected_payload():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return chat_reply("  hi  ")

    assert await llm_client(handler).complete("sys", "user", json_mode=True) == "hi"
    assert seen["auth"] == "Bearer key"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "some-model"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


async def test_llm_omits_response_format_by_default():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return chat_reply("ok")

    await llm_client(handler).complete("s", "u")
    assert "response_format" not in seen["body"]


async def test_llm_retries_once_on_rate_limit():
    responses = [httpx2.Response(429, headers={"retry-after": "0"}), chat_reply("ok")]

    assert await llm_client(lambda r: responses.pop(0)).complete("s", "u") == "ok"


async def test_llm_gives_up_after_retry():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(503, headers={"retry-after": "0"}, text="down")

    with pytest.raises(UpstreamError, match="503"):
        await llm_client(handler).complete("s", "u")
    assert len(calls) == 2


async def test_llm_auth_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(UpstreamError, match="401"):
        await llm_client(handler).complete("s", "u")
    assert len(calls) == 1


async def test_llm_json_mode_rejection_retries_without_it():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.read()))
        if len(bodies) == 1:
            err = {"message": "Failed to generate JSON", "type": "invalid_request_error", "code": "json_validate_failed"}
            return httpx2.Response(400, json={"error": err})
        return chat_reply('{"superhero": false}')

    assert await llm_client(handler).complete("s", "u", json_mode=True) == '{"superhero": false}'
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]


async def test_llm_other_bad_requests_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(400, json={"error": {"message": "bad model", "code": "model_not_found"}})

    with pytest.raises(UpstreamError, match="400"):
        await llm_client(handler).complete("s", "u", json_mode=True)
    assert len(calls) == 1


async def test_llm_timeout():
    def handler(request):
        raise httpx2.ReadTimeout("slow")

    with pytest.raises(UpstreamError) as e:
        await llm_client(handler).complete("s", "u")
    assert e.value.timeout


async def test_llm_connection_failure():
    def handler(request):
        raise httpx2.ConnectError("refused")

    with pytest.raises(UpstreamError, match="connection failed") as e:
        await llm_client(handler).complete("s", "u")
    assert not e.value.timeout
