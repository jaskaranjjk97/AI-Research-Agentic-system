"""Data shapes used by the agent."""

from typing import TypedDict

from pydantic import BaseModel


class Question(BaseModel):
    id: str
    text: str
    query: str = ""         # search query to use; a follow-up query replaces it on retry
    attempts: int = 0
    done: bool = False
    answered: bool = False  # done AND enough verified claims


class Source(BaseModel):
    id: int                 # citation number shown in the report: [1], [2] ...
    title: str
    url: str
    content: str


class Claim(BaseModel):
    statement: str
    quote: str              # verbatim text copied from the source
    source_id: int
    question_id: str


class State(TypedDict, total=False):
    """Shared memory that every graph node reads and updates."""

    topic: str
    questions: list[Question]
    sources: list[Source]
    claims: list[Claim]
    current: str            # id of the question being worked on
    batch: list[int]        # source ids returned by the latest search
    searches: int
    errors: list[str]
    report: str