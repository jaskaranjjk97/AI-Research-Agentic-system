import pytest

from agent.llm import parse_json


def test_plain_json():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_code_fences_are_removed():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extra_text_around_json():
    assert parse_json('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}


def test_list_is_supported():
    assert parse_json("[1, 2, 3]") == [1, 2, 3]


def test_garbage_raises():
    with pytest.raises(ValueError):
        parse_json("no json here")
