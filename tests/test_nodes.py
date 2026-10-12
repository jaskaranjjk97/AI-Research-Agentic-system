from agent.config import Config
from agent.nodes import Nodes


class FakeLLM:
    """Stands in for the real LLM: returns whatever we script, no API call."""

    def __init__(self, plan=None, extract=None):
        self.plan_data = plan if plan is not None else {"questions": ["How does solar work?"]}
        self.extract_data = extract  # a function: prompt -> dict

    def json(self, prompt):
        if "TASK: PLAN" in prompt:
            if isinstance(self.plan_data, Exception):
                raise self.plan_data
            return self.plan_data
        if "TASK: EXTRACT" in prompt:
            return self.extract_data(prompt)
        raise AssertionError("unexpected prompt")


def make(llm, searcher=None, **cfg):
    return Nodes(llm, searcher, Config(**cfg))


def test_plan_creates_numbered_questions():
    nodes = make(FakeLLM(plan={"questions": ["What is X?", "Why does X matter?"]}))
    update = nodes.plan({"topic": "X"})

    assert [q.id for q in update["questions"]] == ["q1", "q2"]
    assert update["questions"][0].query == "What is X?"
    assert update["searches"] == 0 and update["sources"] == [] and update["claims"] == []


def test_plan_is_capped_and_blank_questions_are_dropped():
    nodes = make(FakeLLM(plan={"questions": ["a?", "  ", "b?", "c?"]}), n_questions=2)
    update = nodes.plan({"topic": "X"})
    assert [q.text for q in update["questions"]] == ["a?", "b?"]


def test_plan_falls_back_to_topic_when_llm_fails():
    nodes = make(FakeLLM(plan=ValueError("bad json")))
    update = nodes.plan({"topic": "solar power"})

    assert [q.text for q in update["questions"]] == ["solar power"]
    assert any("Planning failed" in e for e in update["errors"])


class FakeSearcher:
    """Stands in for Tavily: returns whatever we script, no internet."""

    def __init__(self, fn):
        self.fn = fn
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.fn(query)


def hit(url, text="some page text"):
    return {"title": f"Title {url}", "url": url, "content": text}


def planned(nodes, topic="X"):
    """Run the planner first, like the real graph does."""
    state = {"topic": topic}
    state.update(nodes.plan(state))
    return state


def test_research_numbers_sources_and_counts_the_search():
    searcher = FakeSearcher(lambda q: [hit("http://a"), hit("http://b")])
    nodes = make(FakeLLM(plan={"questions": ["a?", "b?"]}), searcher)
    state = planned(nodes)

    update = nodes.research(state)

    assert searcher.queries == ["a?"]
    assert [s.id for s in update["sources"]] == [1, 2]
    assert update["batch"] == [1, 2]
    assert update["current"] == "q1"
    assert update["searches"] == 1
    assert update["questions"][0].attempts == 1


def test_research_skips_done_questions_and_uses_follow_up_query():
    searcher = FakeSearcher(lambda q: [hit("http://a")])
    nodes = make(FakeLLM(plan={"questions": ["a?", "b?"]}), searcher)
    state = planned(nodes)
    state["questions"][0].done = True
    state["questions"][1].query = "better query"

    update = nodes.research(state)

    assert searcher.queries == ["better query"]
    assert update["current"] == "q2"


def test_same_url_keeps_the_same_citation_number():
    searcher = FakeSearcher(lambda q: [hit("http://same")])
    nodes = make(FakeLLM(plan={"questions": ["a?", "b?"]}), searcher)
    state = planned(nodes)
    state.update(nodes.research(state))
    state["questions"][0].done = True
    state.update(nodes.research(state))

    assert len(state["sources"]) == 1
    assert state["batch"] == [1]
    assert state["searches"] == 2


def test_search_failure_is_recorded_not_raised():
    def boom(query):
        raise RuntimeError("429 rate limit")

    nodes = make(FakeLLM(), FakeSearcher(boom))
    state = planned(nodes)

    update = nodes.research(state)

    assert update["batch"] == []
    assert update["questions"][0].attempts == 1
    assert any("Search failed" in e for e in update["errors"])


QUOTE = "Solar panels convert sunlight directly into electricity."


def run_verify(extract, page_text=QUOTE, **cfg):
    """plan -> research -> verify on one fake page, like the real graph does."""
    searcher = FakeSearcher(lambda q: [hit("http://a", "Intro. " + page_text + " More.")])
    nodes = make(FakeLLM(extract=extract), searcher, **cfg)
    state = planned(nodes)
    state.update(nodes.research(state))
    state.update(nodes.verify(state))
    return state


def claim(quote=QUOTE, source_id=1):
    return {"statement": "A fact", "quote": quote, "source_id": source_id}


def test_verify_keeps_claim_whose_quote_is_in_the_source():
    state = run_verify(lambda p: {"claims": [claim()]}, min_claims=1)

    assert len(state["claims"]) == 1
    assert state["claims"][0].question_id == "q1"
    assert state["questions"][0].done and state["questions"][0].answered


def test_verify_rejects_an_invented_quote():
    invented = claim("Solar is cheap and everyone knows it.")
    state = run_verify(lambda p: {"claims": [invented]})

    assert state["claims"] == []
    assert not state["questions"][0].done  # one attempt used, one left


def test_quote_matching_ignores_case_spacing_and_curly_quotes():
    page = "It\u2019s clear that solar panels convert sunlight into electricity."
    quote = "it's clear  that SOLAR panels convert sunlight into electricity."
    state = run_verify(lambda p: {"claims": [claim(quote)]}, page_text=page)

    assert len(state["claims"]) == 1


def test_verify_rejects_a_quote_that_is_too_short():
    state = run_verify(lambda p: {"claims": [claim("Solar")]})
    assert state["claims"] == []


def test_verify_rejects_a_claim_citing_a_source_not_in_the_batch():
    state = run_verify(lambda p: {"claims": [claim(source_id=99)]})
    assert state["claims"] == []


def test_verify_drops_duplicate_quotes():
    state = run_verify(lambda p: {"claims": [claim(), claim(QUOTE.upper())]})
    assert len(state["claims"]) == 1


def test_follow_up_query_is_saved_for_the_retry():
    state = run_verify(lambda p: {"claims": [], "follow_up_query": "better query"})
    assert state["questions"][0].query == "better query"


def test_question_gives_up_after_max_attempts():
    state = run_verify(lambda p: {"claims": []}, max_attempts=1)

    assert state["questions"][0].done
    assert not state["questions"][0].answered


def test_extraction_failure_is_recorded_not_raised():
    def boom(prompt):
        raise ValueError("bad json")

    state = run_verify(boom)
    assert state["claims"] == []
    assert any("Claim extraction failed" in e for e in state["errors"])
