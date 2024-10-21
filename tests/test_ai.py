from unittest.mock import AsyncMock

import pytest

from think import LLMQuery, ask
from think.llm.chat import Chat, ContentPart, ContentType, Message, Role


@pytest.mark.asyncio
async def test_ask_basic():
    llm = AsyncMock(return_value="Hi!")
    result = await ask(llm, "Hello, {{ name }}!", name="world")

    assert result == "Hi!"

    llm.assert_awaited_once()

    chat: Chat = llm.await_args[0][0]
    assert chat.messages == [
        Message(
            role=Role.user,
            content=[ContentPart(type=ContentType.text, text="Hello, world!")],
        )
    ]


@pytest.mark.asyncio
async def test_llm_query():
    class TestQuery(LLMQuery):
        """Prompt with {{ text }}"""

        msg: str

    llm = AsyncMock(return_value=TestQuery(msg="Hi!"))

    result = await TestQuery.run(llm, text="some text")
    assert isinstance(result, TestQuery)
    assert result.msg == "Hi!"

    chat: Chat = llm.await_args[0][0]
    assert chat.messages[0].content[0].text.startswith("Prompt with some text\n")
