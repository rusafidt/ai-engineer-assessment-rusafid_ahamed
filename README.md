# Superhero & Comics Q&A

A FastAPI service with one endpoint, `POST /ask`. It answers questions about superheroes (live from the [Superhero API](https://superheroapi.com)) and comic book history (9 Wikipedia articles in `data/`), and can combine both in one answer. Every response lists the sources it used.

How a request flows:

1. **Route.** An LLM call with JSON output decides which sources the question needs (superhero, documents, both or neither). It also extracts the character names and a search query.
2. **Fetch.** Hero lookups run concurrently against the API. Documents are searched with BM25 over ~180-word chunks.
3. **Answer.** A second LLM call answers using only the numbered sources and cites them inline (`[S1]`). The response returns only the sources that were actually cited.

The LLM is Groq (`llama-3.3-70b-versatile`) by default, called through the official `openai` SDK (retries and timeouts come from the SDK). Any OpenAI-compatible endpoint works by changing `LLM_BASE_URL` / `LLM_MODEL`: OpenAI, Groq, Gemini, OpenRouter, Ollama, vLLM and so on.

## Setup

Requires Python 3.12+.

```bash
git clone https://github.com/<you>/ai-engineer-assessment-<name>.git
cd ai-engineer-assessment-<name>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # then fill in both keys
```

- Groq key: https://console.groq.com/keys
- Superhero API token: sign in with GitHub at https://superheroapi.com

## Run

```bash
uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Who created Spider-Man, and how strong is he?"}'
```

```json
{
  "answer": "Spider-Man was created by Stan Lee and Steve Ditko in 1962 [S2]. His strength is rated 55/100 [S1].",
  "sources": [
    {"id": "S1", "type": "superhero_api", "title": "Spider-Man", "reference": "superheroapi.com character id 620"},
    {"id": "S2", "type": "documents", "title": "Stan Lee > Marvel revolution", "reference": "data/stan_lee.txt (chunk 13) - https://en.wikipedia.org/wiki/Stan_Lee"}
  ],
  "route": ["superhero_api", "documents"]
}
```

Interactive docs are at http://localhost:8000/docs.

## Tests

```bash
pytest
```

The tests cover routing and router-output validation, retrieval, both API clients (with mocked HTTP), orchestration across all routing paths and partial failures, and the endpoint's validation and error codes. Nothing in the test suite makes a network call.

## Decisions and trade-offs

- **Explicit router instead of tool calling.** The routing decision is plain JSON that I validate and can unit test. If the output is malformed, the request falls back to a document search instead of failing.
- **BM25 over embeddings.** The corpus is small and keyword-heavy (names, titles, years). BM25 needs no extra API calls or model downloads, and it's deterministic. Embeddings would be the next step for a larger or fuzzier corpus.
- **Partial failure is tolerated.** If the Superhero API is down but documents were also requested, the answer uses the documents and says the API was unreachable. If nothing is usable, the endpoint returns 502 (504 on timeout).
- **No LLM call when there's nothing to say.** Off-topic questions and characters the API doesn't have get a fixed reply, so the model has no chance to make something up.
- **The API token stays out of responses and logs.** The token is part of the Superhero API URL, so source references use the character id and httpx request logging is turned down.
- **Not done:** auth and rate limiting on `/ask`, response caching, conversation memory, streaming, and an eval set to measure routing accuracy.
