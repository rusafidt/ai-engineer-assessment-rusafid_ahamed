import pytest
from fastapi.testclient import TestClient

from app.errors import UpstreamError
from app.main import app, get_assistant
from app.schemas import AskResponse, Source, SourceType


class StubAssistant:
    def __init__(self, result):
        self.result = result
        self.questions = []

    async def ask(self, question):
        self.questions.append(question)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def client():
    # no `with` block, so the lifespan (real clients, .env) never runs
    yield TestClient(app)
    app.dependency_overrides.clear()


def use(result):
    stub = StubAssistant(result)
    app.dependency_overrides[get_assistant] = lambda: stub
    return stub


def test_ask_returns_answer_and_sources(client):
    src = Source(id="S1", type=SourceType.DOCUMENTS, title="Stan Lee", reference="data/stan_lee.txt")
    stub = use(AskResponse(answer="Yes [S1]", sources=[src], route=[SourceType.DOCUMENTS]))

    res = client.post("/ask", json={"question": "  Who was Stan Lee?  "})

    assert res.status_code == 200
    assert res.json()["sources"][0]["type"] == "documents"
    assert stub.questions == ["Who was Stan Lee?"]  # whitespace stripped


@pytest.mark.parametrize(
    "body",
    [{}, {"question": ""}, {"question": "   "}, {"question": "hi"}, {"question": "x" * 501}, {"question": 42}],
)
def test_invalid_questions_are_rejected(client, body):
    use(None)
    assert client.post("/ask", json=body).status_code == 422


def test_upstream_failure_is_502(client):
    use(UpstreamError("llm", "status 500"))
    res = client.post("/ask", json={"question": "How strong is Thor?"})
    assert res.status_code == 502
    assert res.json() == {"detail": "llm is unavailable right now, please try again."}


def test_upstream_timeout_is_504(client):
    use(UpstreamError("superhero_api", "timed out", timeout=True))
    assert client.post("/ask", json={"question": "How strong is Thor?"}).status_code == 504
