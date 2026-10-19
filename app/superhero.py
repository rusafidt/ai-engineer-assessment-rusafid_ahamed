from dataclasses import dataclass
from urllib.parse import quote

import httpx

from app.errors import UpstreamError

NOT_FOUND = "character with given name not found"


@dataclass(frozen=True)
class Hero:
    id: str
    name: str
    powerstats: dict[str, str]
    biography: dict
    appearance: dict
    work: dict
    connections: dict

    @classmethod
    def from_api(cls, raw: dict) -> "Hero":
        return cls(
            id=str(raw.get("id", "")),
            name=raw.get("name", "unknown"),
            powerstats=raw.get("powerstats") or {},
            biography=raw.get("biography") or {},
            appearance=raw.get("appearance") or {},
            work=raw.get("work") or {},
            connections=raw.get("connections") or {},
        )

    def to_context(self) -> str:
        bio = self.biography
        stats = ", ".join(f"{k} {v}" for k, v in self.powerstats.items() if _known(v))
        fields = [
            ("Name", self.name),
            ("Full name", bio.get("full-name")),
            ("Publisher", bio.get("publisher")),
            ("Alignment", bio.get("alignment")),
            ("First appearance", bio.get("first-appearance")),
            ("Aliases", ", ".join(a for a in bio.get("aliases") or [] if _known(a))),
            ("Powerstats (0-100)", stats or "not available"),
            ("Gender", self.appearance.get("gender")),
            ("Race", self.appearance.get("race")),
            ("Height", _measure(self.appearance.get("height"))),
            ("Weight", _measure(self.appearance.get("weight"))),
            ("Eye colour", self.appearance.get("eye-color")),
            ("Hair colour", self.appearance.get("hair-color")),
            ("Occupation", self.work.get("occupation")),
            ("Base", self.work.get("base")),
            ("Group affiliation", self.connections.get("group-affiliation")),
        ]
        return "\n".join(f"{k}: {v}" for k, v in fields if _known(v))


def _known(value) -> bool:
    return value not in (None, "", "null", "-")


def _measure(values) -> str:
    # height/weight come in two units, e.g. ["6'2", "188 cm"]; unknown ones are "- lb" / "0 kg"
    return " / ".join(v for v in values or [] if _known(v) and not v.startswith(("-", "0 ")))


class SuperheroClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str, token: str, max_results: int = 3):
        self.http = http
        self.base = f"{base_url.rstrip('/')}/{token}"
        self.max_results = max_results

    async def search(self, name: str) -> list[Hero]:
        try:
            resp = await self.http.get(f"{self.base}/search/{quote(name)}")
        except httpx.TimeoutException:
            raise UpstreamError("superhero_api", "request timed out", timeout=True)
        except httpx.HTTPError as e:
            raise UpstreamError("superhero_api", f"connection failed ({type(e).__name__})")

        if resp.status_code != 200:
            raise UpstreamError("superhero_api", f"status {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            raise UpstreamError("superhero_api", "response was not JSON")

        # the API answers 200 for everything and puts errors in the body
        if data.get("response") != "success":
            if data.get("error") == NOT_FOUND:
                return []
            raise UpstreamError("superhero_api", data.get("error", "unknown error"))

        heroes = [Hero.from_api(r) for r in data.get("results", [])]
        # searching "batman" also returns "Batman II" etc - put the exact match first
        heroes.sort(key=lambda h: h.name.lower() != name.lower())
        return heroes[: self.max_results]
