"""The steps of the agent: plan -> research -> verify -> write.

Each node receives the shared state and returns only the keys it changes.
A failure inside a node is recorded in state["errors"] instead of crashing the run.
"""

import logging
import re

from agent.config import Config
from agent.llm import LLM
from agent.models import Claim, Question, Source, State
from agent.search import Searcher

log = logging.getLogger(__name__)

# Curly quotes, long dashes and non-breaking spaces become plain characters,
# so a quote still matches when the page uses fancy typography.
_PUNCT = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


def _norm(text: str) -> str:
    """Normalise text so quote matching ignores case, spacing and fancy punctuation."""
    return " ".join(text.translate(_PUNCT).lower().split())


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

    # 2. RESEARCH -----------------------------------------------------------
    def research(self, state: State) -> dict:
        questions = state["questions"]
        question = next(q for q in questions if not q.done)
        sources = list(state["sources"])
        errors = list(state["errors"])
        by_url = {s.url: s for s in sources}
        batch: list[int] = []

        question.attempts += 1
        try:
            for hit in self.searcher.search(question.query or question.text):
                source = by_url.get(hit["url"])
                if source is None:  # new page -> give it the next citation number
                    source = Source(id=len(sources) + 1, **hit)
                    sources.append(source)
                    by_url[source.url] = source
                batch.append(source.id)
        except Exception as exc:
            errors.append(f"Search failed for '{question.text}': {exc}")

        log.info(
            "Researched %s (attempt %d): %d sources",
            question.id,
            question.attempts,
            len(batch),
        )
        return {
            "questions": questions,
            "sources": sources,
            "batch": batch,
            "current": question.id,
            "searches": state["searches"] + 1,
            "errors": errors,
        }

    # 3. VERIFY (extract claims, then check each quote exists in its source) -
    def verify(self, state: State) -> dict:
        questions = state["questions"]
        question = next(q for q in questions if q.id == state["current"])
        claims = list(state["claims"])
        errors = list(state["errors"])
        batch = [s for s in state["sources"] if s.id in state["batch"]]

        if batch:
            try:
                claims += self._extract_claims(question, batch, claims)
            except Exception as exc:
                errors.append(f"Claim extraction failed for '{question.text}': {exc}")

        verified = sum(1 for c in claims if c.question_id == question.id)
        if verified >= self.cfg.min_claims:
            question.done = question.answered = True
        elif question.attempts >= self.cfg.max_attempts:
            question.done = True  # give up; reported as a limitation
        return {"questions": questions, "claims": claims, "errors": errors}

    def _extract_claims(self, question, batch, existing) -> list[Claim]:
        sources_text = "\n\n".join(f"[S{s.id}] {s.title}\n{s.content}" for s in batch)
        prompt = (
            "TASK: EXTRACT\n"
            f"Question: {question.text}\n\n"
            f"Sources:\n{sources_text}\n\n"
            "List facts from the sources that help answer the question. For each fact give:\n"
            "- statement: the fact in your own words\n"
            "- quote: an EXACT sentence copied word-for-word from the source\n"
            "- source_id: the number from [S#]\n"
            "Use only the sources above. If the sources are not enough, also give a better "
            "search query.\n"
            'Return JSON: {"claims": [{"statement": "", "quote": "", "source_id": 1}], '
            '"follow_up_query": ""}'
        )
        data = self.llm.json(prompt)
        by_id = {s.id: s for s in batch}
        seen = {(c.source_id, _norm(c.quote)) for c in existing}
        accepted, rejected = [], 0

        for item in data.get("claims", []):
            try:
                source_id = int(str(item["source_id"]).lstrip("Ss"))
                quote = str(item["quote"]).strip()
                statement = str(item["statement"]).strip()
            except (KeyError, ValueError, TypeError):
                rejected += 1
                continue

            source = by_id.get(source_id)
            grounded = (
                source is not None and len(quote) >= 20 and _norm(quote) in _norm(source.content)
            )
            if not grounded or not statement:
                rejected += 1  # the quote is not in the source -> likely invented
                continue
            if (source_id, _norm(quote)) in seen:
                continue  # same quote already accepted
            seen.add((source_id, _norm(quote)))
            accepted.append(
                Claim(
                    statement=statement,
                    quote=quote,
                    source_id=source_id,
                    question_id=question.id,
                )
            )

        if rejected:
            log.info("Rejected %d ungrounded claims for %s", rejected, question.id)
        follow_up = str(data.get("follow_up_query") or "").strip()
        if follow_up:
            question.query = follow_up  # used if the loop comes back for another try
        return accepted

    # LOOP Decision
    def route(self, state: State) -> str:
        """After Research, open questions exist and budget positive do research again."""

        open_questions = any(not q.done for q in state["questions"])
        if open_questions and state["searches"] < self.cfg.max_searches:
            return "research"
        return "write"

    # Writer to prepare the final report
    def write(self, state: State) -> dict:
        claims = state["claims"]
        questions = state["questions"]
        errors = list(state["errors"])

        if not claims:
            report = f"# {state['topic']}\n\nNo verified findings were found." + self._limitations(
                questions, errors
            )

            return {"report": report, "errors": errors}

        claim_lines = []

        for q in questions:
            question_claims = [c for c in claims if c.question_id == q.id]

            if question_claims:
                claim_lines.append(f"## {q.text}")

                for c in question_claims:
                    claim_lines.append(f"- {c.statement} [{c.source_id}]")

        claim_lines = "\n\n".join(claim_lines)

        prompt = (
            "TASK: WRITE\n"
            f"Write a clear markdown research report on: {state['topic']}\n"
            "Use ONLY the verified findings below. Do not add facts. Keep the citation "
            "numbers like [3] exactly as given, placed right after the sentence they support.\n"
            "Structure: a title, a 3-sentence summary, then one short section per question.\n\n"
            f"Verified findings:\n{claim_lines}"
        )

        try:
            body = self.llm.text(prompt).strip()
        except Exception as exc:
            errors.append(f"Report writing failed ({exc}); showing raw findings.")
            body = f"# {state['topic']}\n\n{claim_lines}"

        # Keep only valid citation numbers
        valid = {c.source_id for c in claims}

        def keep_valid(match):
            return match.group() if int(match.group(1)) in valid else ""

        body = re.sub(r"\[(\d+)\]", keep_valid, body)

        # Find cited sources
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)} or valid

        # Build the source list
        refs = "\n".join(f"[{s.id}] {s.title} - {s.url}" for s in state["sources"] if s.id in cited)

        report = f"{body}\n\n## Sources\n{refs}" + self._limitations(questions, errors)
        return {"report": report, "errors": errors}

    @staticmethod
    def _limitations(questions, errors) -> str:
        gaps = [q.text for q in questions if not q.answered]
        text = ""
        if gaps:
            text += "\n\n## Limitations\nNot enough verified evidence for:\n" + "\n".join(
                f"- {g}" for g in gaps
            )
        if errors:
            text += f"\n\n_{len(errors)} step(s) had errors; see logs._"
        return text
