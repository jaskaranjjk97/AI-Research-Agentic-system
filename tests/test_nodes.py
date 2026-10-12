from agent.config import Config
from agent.nodes import Nodes


class FakeLLM:
    """Stands in for the real LLM: returns whatever we script, no API call."""

    def __init__(self, plan=None):
        self.plan_data = plan if plan is not None else {"questions": ["How does solar work?"]}

    def json(self, prompt):
        if "TASK: PLAN" in prompt:
            if isinstance(self.plan_data, Exception):
                raise self.plan_data
            return self.plan_data
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
