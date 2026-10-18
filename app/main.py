import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.assistant import Assistant
from app.config import get_settings
from app.errors import UpstreamError
from app.llm import LLMClient
from app.retrieval import DocumentIndex
from app.schemas import AskRequest, AskResponse
from app.superhero import SuperheroClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
# httpx logs full request URLs at INFO, and the superhero token is part of the URL
logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    index = DocumentIndex.from_dir(settings.data_dir)
    log.info("indexed %d chunks from %d documents", len(index.chunks), len(index.titles))

    # max_retries=1: one retry for rate limits / transient 5xx / connection errors, nothing fancier
    sdk = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.http_timeout,
        max_retries=1,
    )
    async with sdk, httpx.AsyncClient(timeout=settings.http_timeout) as http:
        app.state.assistant = Assistant(
            llm=LLMClient(sdk, settings.llm_model),
            heroes=SuperheroClient(http, settings.superhero_base_url, settings.superhero_api_token),
            index=index,
            top_k=settings.top_k,
        )
        yield


app = FastAPI(title="Superhero & Comics Q&A", lifespan=lifespan)


def get_assistant(request: Request) -> Assistant:
    return request.app.state.assistant


@app.exception_handler(UpstreamError)
async def upstream_error(request: Request, exc: UpstreamError):
    log.error("upstream failure: %s", exc)
    status = 504 if exc.timeout else 502
    return JSONResponse(
        status_code=status,
        content={"detail": f"{exc.service} is unavailable right now, please try again."},
    )


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, assistant: Assistant = Depends(get_assistant)) -> AskResponse:
    return await assistant.ask(body.question)
