import logging

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
)

from app.errors import UpstreamError

log = logging.getLogger(__name__)


class LLMClient:
    """Chat completions through the OpenAI SDK, so any OpenAI-compatible engine works via base_url.

    The SDK handles retries (429 / 5xx / connection errors, honouring Retry-After) and timeouts;
    build the AsyncOpenAI client with max_retries=1 to keep it to one retry.
    """

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def _create(self, system, user, json_mode, temperature, max_tokens):
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        try:
            try:
                resp = await self._create(system, user, json_mode, temperature, max_tokens)
            except BadRequestError as e:
                # Groq validates JSON mode server-side and 400s when the model's output isn't valid
                # JSON. The prompt already asks for JSON, so retry once in plain mode and let the
                # caller's parser deal with whatever comes back.
                if not (json_mode and e.code == "json_validate_failed"):
                    raise
                log.warning("json mode rejected by the engine, retrying without it")
                resp = await self._create(system, user, False, temperature, max_tokens)
        except APITimeoutError:
            raise UpstreamError("llm", "request timed out", timeout=True)
        except APIStatusError as e:
            raise UpstreamError("llm", f"status {e.status_code}: {e.message[:200]}")
        except APIConnectionError as e:
            raise UpstreamError("llm", f"connection failed ({type(e).__name__})")
        except APIError:
            raise UpstreamError("llm", "unexpected response shape")

        try:
            content = resp.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raise UpstreamError("llm", "unexpected response shape")
        return (content or "").strip()
