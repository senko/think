from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core.exceptions import NotFound

from think.llm.base import ConfigError
from think.llm.chat import Chat
from think.llm.google import GoogleClient


@pytest.mark.asyncio
async def test_no_retry_on_not_found():
    """A NotFound (unknown model) is non-transient and must not be retried."""
    generate = AsyncMock(side_effect=NotFound("unknown model"))

    with patch("think.llm.google.genai") as mock_genai:
        mock_genai.GenerativeModel.return_value = MagicMock(
            generate_content_async=generate
        )
        with patch("think.llm.base.asyncio.sleep", new=AsyncMock()):
            client = GoogleClient(model="gemini-pro", api_key="fake")
            chat = Chat("system").user("hi")
            with pytest.raises(ConfigError):
                await client(chat, max_retries=5)

    assert generate.call_count == 1
