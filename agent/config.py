"""All settings in one place, read from environment variables / .env."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    openai_api_key: str = ""
    tavily_api_key: str = ""
    model: str = "gpt-5-mini"
    n_questions: int = 4      # sub-questions the planner creates
    max_searches: int = 8     # hard budget for the whole run (cost + loop guard)
    max_attempts: int = 2     # tries per sub-question before giving up
    min_claims: int = 2       # verified claims needed to call a question "answered"
    max_results: int = 5      # search results per query
    max_chars: int = 6000     # characters kept per source

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import load_dotenv

        load_dotenv()  # reads your .env file into environment variables
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "gpt-5-mini"),
            n_questions=int(os.getenv("N_QUESTIONS", 4)),
            max_searches=int(os.getenv("MAX_SEARCHES", 8)),
            max_attempts=int(os.getenv("MAX_ATTEMPTS", 2)),
            min_claims=int(os.getenv("MIN_CLAIMS", 2)),
        )