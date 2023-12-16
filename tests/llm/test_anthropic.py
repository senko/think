from unittest.mock import patch, MagicMock

import pytest
from anthropic import BadRequestError

from think.llm.anthropic import Claude
from think.chat import Chat


def test_anthropic_client_setup():
    """Test that an Claude object can be created."""

    c = Claude("fake-key", model="claude-2")
    assert c.api_key == "fake-key"
    assert c.model == "claude-2"


def test_anthropic_client_requires_api_key():
    """Test that an Claude object cannot be created without an API key."""

    with pytest.raises(ValueError):
        Claude("")


@patch("think.llm.anthropic.Anthropic")
def test_anthropic_client_call_claude(mock_anthropic):
    """Test that the Claude can call Claude AI."""

    mock_client = mock_anthropic.return_value
    mock_client.completions.create.return_value.completion = "fake-response"

    chat = Chat("system prompt")

    c = Claude("fake-key")
    mock_anthropic.assert_called_once_with(api_key="fake-key")

    response = c(chat)

    assert response == "fake-response"
    mock_client.completions.create.assert_called_once_with(
        model="claude-2",
        prompt="\n\nHuman: system prompt\n\nAssistant:",
        max_tokens_to_sample=1000,
    )


@patch("think.llm.anthropic.Anthropic")
def test_anthropic_client_call_claude_handles_error(mock_anthropic, caplog):
    mock_client = mock_anthropic.return_value
    mock_client.completions.create.side_effect = BadRequestError(
        "fake-error", response=MagicMock(), body=None
    )

    c = Claude("fake-key")
    response = c(Chat())

    assert response is None
    assert "Error calling Claude: fake-error" in caplog.text
