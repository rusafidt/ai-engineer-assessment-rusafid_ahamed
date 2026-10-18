from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]


class AskRequest(BaseModel):
    question: Question


class SourceType(StrEnum):
    SUPERHERO_API = "superhero_api"
    DOCUMENTS = "documents"


class Source(BaseModel):
    id: str
    type: SourceType
    title: str
    reference: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    route: list[SourceType]
