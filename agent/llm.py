"""Thin OpenAI wrapper: plain text, and JSON that survives messy model output."""

import json
import re


def parse_json(text: str):
    """Parse JSON from model output, tolerating ```fences``` and extra prose."""
    text = (text or "").strip()
    # remove a leading ```json (or ```) and a trailing ```
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back: take everything from the first { to the last } (or [ to ])
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Model did not return valid JSON.")


class LLM:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
        from openai import OpenAI

        # The SDK already retries 429 / 5xx / connection errors with backoff.
        self.client = OpenAI(api_key=api_key, timeout=60, max_retries=3)
        self.model = model

    def text(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text

    def json(self, prompt: str):
        """Ask for JSON; if it can't be parsed, ask once more before failing."""
        raw = self.text(prompt)
        try:
            return parse_json(raw)
        except ValueError:
            return parse_json(self.text(prompt + "\n\nReturn ONLY valid JSON, nothing else."))
