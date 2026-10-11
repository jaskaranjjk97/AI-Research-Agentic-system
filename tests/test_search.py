import pytest

from agent.search import Searcher


def test_searcher_retries_then_succeeds():
    class Flaky:
        calls = 0

        def search(self, **kwargs):
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise ConnectionError("temporary")
            return {"results": [{"url": "http://a", "title": "T",
                                 "raw_content": None, "content": "snippet text"}]}

    results = Searcher("", client=Flaky(), backoff=0).search("q")
    assert Flaky.calls == 3
    assert results[0]["content"] == "snippet text"


def test_searcher_raises_after_all_retries():
    class Down:
        def search(self, **kwargs):
            raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        Searcher("", client=Down(), retries=2, backoff=0).search("q")


def test_results_without_text_are_dropped_and_text_is_cut():
    class Fake:
        def search(self, **kwargs):
            return {"results": [
                {"url": "http://a", "title": "A", "raw_content": "x" * 100},
                {"url": "http://b", "title": "B", "raw_content": "", "content": ""},
            ]}

    results = Searcher("", client=Fake(), max_chars=10, backoff=0).search("q")
    assert len(results) == 1
    assert len(results[0]["content"]) == 10