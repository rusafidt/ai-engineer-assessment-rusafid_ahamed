from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"

    superhero_api_token: str
    # /api 302-redirects here, so skip the extra hop
    superhero_base_url: str = "https://superheroapi.com/api.php"

    data_dir: Path = Path("data")
    http_timeout: float = 20.0
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
