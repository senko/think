from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from think.llm.bedrock import BedrockClient
from think.llm.chat import Chat


@pytest.mark.asyncio
async def test_max_retries_passed_to_botocore_config():
    """BedrockClient should pass max_retries directly as botocore max_attempts.

    Botocore's `max_attempts` already means "retries after the initial request",
    so the value must NOT be incremented by 1 here.
    """
    # Build a fake bedrock-runtime client whose .converse returns a minimal
    # valid response.
    fake_runtime = MagicMock()
    fake_runtime.converse = AsyncMock(
        return_value={
            "output": {"message": {"role": "assistant", "content": [{"text": "Hi!"}]}}
        }
    )

    # The session.client(...) returns an async-context-manager.
    runtime_cm = MagicMock()
    runtime_cm.__aenter__ = AsyncMock(return_value=fake_runtime)
    runtime_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("think.llm.bedrock.Session") as MockSession,
        patch("think.llm.bedrock.Config") as MockConfig,
    ):
        MockSession.return_value.client = MagicMock(return_value=runtime_cm)

        client = BedrockClient(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1",
        )
        chat = Chat("system").user("hi")
        await client(chat, max_retries=4)

        MockConfig.assert_called_once_with(
            retries={"max_attempts": 4, "mode": "standard"}
        )
