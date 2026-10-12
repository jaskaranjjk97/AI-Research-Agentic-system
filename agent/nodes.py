"""The steps of the agent: plan -> research -> verify -> write.

Each node receives the shared state and returns only the keys it changes.
A failure inside a node is recorded in state["errors"] instead of crashing the run.
"""

import logging

from agent.config import Config
from agent.llm import LLM
from agent.models import Question, State
from agent.search import Searcher

log = logging.getLogger(__name__)


class Nodes:
    def __init__(self, llm: LLM, searcher: Searcher, config: Config):
        self.llm, self.searcher, self.cfg = llm, searcher, config

    # 1. PLAN ---------------------------------------------------------------
    def plan(self, state: State) -> dict:
        topic = state["topic"]
        prompt = (
            "TASK: PLAN\n"
            f"Break this research topic into {self.cfg.n_questions} specific, "
            "non-overlapping questions that can each be answered with a web search.\n"
            f"Topic: {topic}\n"
            'Return JSON: {"questions": ["...", "..."]}'
        )
        errors = list(state.get("errors", []))
        try:
            texts = [str(q).strip() for q in self.llm.json(prompt)["questions"] if str(q).strip()]
        except Exception as exc:
            errors.append(f"Planning failed ({exc}); researching the topic as one question.")
            texts = []
        texts = (texts or [topic])[: self.cfg.n_questions]
        questions = [Question(id=f"q{i}", text=t, query=t) for i, t in enumerate(texts, 1)]
        return {
            "questions": questions,
            "sources": [],
            "claims": [],
            "searches": 0,
            "batch": [],
            "errors": errors,
        }
