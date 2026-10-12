"""Web search via Tavily. One call returns results AND page text, so no scraping step."""

import logging
import time

log = logging.getLogger(__name__)


class Searcher:
    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        max_chars: int = 6000,
        retries: int = 3,
        backoff: float = 2.0,
        client=None,
    ):
        if client is None:
            if not api_key:
                raise ValueError("TAVILY_API_KEY is missing. Add it to your .env file.")
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
        self.client = client
        self.max_results = max_results
        self.max_chars = max_chars
        self.retries = retries
        self.backoff = backoff

    def search(self, query: str) -> list[dict]:
        """Return [{title, url, content}], retrying transient failures."""
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.search(
                    query=query, max_results=self.max_results, include_raw_content=True
                )
                break
            except Exception as exc:  # network, rate limit, quota ... all handled the same way
                log.warning("Search attempt %d/%d failed: %s", attempt, self.retries, exc)
                if attempt == self.retries:
                    raise  # out of retries: let the caller decide what to do
                time.sleep(self.backoff**attempt)  # wait 2s, then 4s ...

        results = []
        for item in response.get("results", []):
            text = item.get("raw_content") or item.get("content") or ""
            if item.get("url") and text.strip():
                results.append(
                    {
                        "title": item.get("title") or item["url"],
                        "url": item["url"],
                        "content": text[: self.max_chars],
                    }
                )
        return results
