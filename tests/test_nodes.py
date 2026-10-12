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
