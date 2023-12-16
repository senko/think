from decimal import Decimal
from unittest.mock import MagicMock

from pydantic import BaseModel
import pytest

from think.ai import ai


def test_ai_requires_docstring():
    def foo():
        pass

    with pytest.raises(ValueError):
        ai(foo)


def test_ai_parses_int_with_align():
    @ai
    def foo(x: int, _retval: int) -> int:
        """Do something: {{ x }}"""
        assert _retval > 0, "X must be positive"
        return _retval

    llm = MagicMock()

    result = foo(llm, x=0)
    assert result == llm.return_value
    llm.assert_called_once()
    assert llm.call_args.args[0].messages == [
        {
            "role": "system",
            "content": (
                "Do something: 0\n\nOutput ONLY the requested output in a single Markdown code block, "
                "without commentary or explanation. Your output must be parsable as Python type int"
            ),
        }
    ]

    align_parser = llm.call_args.kwargs["parser"]

    with pytest.raises(ValueError):
        align_parser("bad output")

    with pytest.raises(ValueError):
        # alignment assertion raises an error
        align_parser("-1")

    assert align_parser("1") == 1


def test_ai_parses_decimal_without_align():
    @ai
    def foo() -> Decimal:
        """Do something"""
        assert False  # body should not be executed since there's no _retval arg

    llm = MagicMock()
    result = foo(llm)

    assert result == llm.return_value
    llm.assert_called_once()

    align_parser = llm.call_args.kwargs["parser"]

    with pytest.raises(ValueError):
        align_parser("bad output")

    assert align_parser("-3.1415") == Decimal("-3.1415")


def test_ai_parses_model_with_align():
    class Foo(BaseModel):
        x: int

    @ai
    def foo(_retval: Foo) -> Foo:
        """Do something"""
        assert _retval.x > 0, "X must be positive"
        return _retval

    llm = MagicMock()

    result = foo(llm)
    assert result == llm.return_value
    llm.assert_called_once()
    assert (
        "Do something\n\nOutput ONLY the requested output in a single Markdown code block, "
        "without commentary or explanation. Your output must obey the following JSON schema: "
    ) in llm.call_args.args[0].messages[0]["content"]

    align_parser = llm.call_args.kwargs["parser"]

    with pytest.raises(ValueError):
        align_parser("bad output")

    with pytest.raises(ValueError):
        # alignment assertion raises an error
        align_parser('{"x": -1}')

    assert align_parser('{"x": 1}') == Foo(x=1)
