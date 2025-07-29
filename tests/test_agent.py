from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from think.agent import BaseAgent, RAGMixin, SimpleRAGAgent, tool
from think.llm.base import LLM
from think.llm.chat import Role
from think.llm.tool import ToolKit


class TestBaseAgent:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock(spec=LLM)

    def test_init_with_docstring_system_prompt(self, mock_llm):
        class TestAgent(BaseAgent):
            """You are a helpful assistant. Today is {{today}}."""

            pass

        agent = TestAgent(mock_llm, today="Monday")

        assert len(agent.chat.messages) == 1
        assert agent.chat.messages[0].role == Role.system
        assert agent.chat.messages[0].content[0].text is not None
        assert "Today is Monday" in agent.chat.messages[0].content[0].text

    def test_init_with_string_system_prompt(self, mock_llm):
        class TestAgent(BaseAgent):
            """Original docstring"""

            pass

        agent = TestAgent(mock_llm, system="Custom system prompt")

        assert len(agent.chat.messages) == 1
        assert agent.chat.messages[0].role == Role.system
        assert agent.chat.messages[0].content[0].text == "Custom system prompt"

    def test_init_with_no_system_prompt(self, mock_llm):
        class TestAgent(BaseAgent):
            """ """  # Empty docstring that will be stripped to empty

        agent = TestAgent(mock_llm)

        assert len(agent.chat.messages) == 0

    def test_init_with_empty_docstring(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        agent = TestAgent(mock_llm)

        assert len(agent.chat.messages) == 0

    def test_init_with_file_system_prompt(self, mock_llm, tmp_path):
        class TestAgent(BaseAgent):
            """"""

        # Create a real temporary file
        system_file = tmp_path / "system.txt"
        system_file.write_text("Hello {{name}}!")

        agent = TestAgent(mock_llm, system=system_file, name="World")

        assert len(agent.chat.messages) == 1
        assert agent.chat.messages[0].role == Role.system
        assert "Hello World!" in agent.chat.messages[0].content[0].text  # type: ignore

    def test_init_with_nonexistent_file_raises_error(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        system_file = Path("/fake/nonexistent.txt")
        with pytest.raises(ValueError, match="does not exist"):
            TestAgent(mock_llm, system=system_file)

    def test_init_with_decorated_tools(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

            @tool
            def tool1(self, arg: str) -> str:
                """Tool 1"""
                return arg

            @tool(name="custom_tool")
            def tool2(self, arg: int) -> int:
                """Tool 2"""
                return arg * 2

        agent = TestAgent(mock_llm)

        assert "tool1" in agent.toolkit.tools
        assert "custom_tool" in agent.toolkit.tools
        assert len(agent.toolkit.tools) == 2

    def test_init_with_class_tools_attribute(self, mock_llm):
        def external_tool(arg: str) -> str:
            """External tool"""
            return arg

        class TestAgent(BaseAgent):
            """"""

            tools = [external_tool]

        agent = TestAgent(mock_llm)

        assert "external_tool" in agent.toolkit.tools
        assert len(agent.toolkit.tools) == 1

    def test_init_with_constructor_tools_list(self, mock_llm):
        def external_tool(arg: str) -> str:
            """External tool"""
            return arg

        class TestAgent(BaseAgent):
            """"""

        agent = TestAgent(mock_llm, tools=[external_tool])

        assert "external_tool" in agent.toolkit.tools
        assert len(agent.toolkit.tools) == 1

    def test_init_with_constructor_toolkit(self, mock_llm):
        def external_tool(arg: str) -> str:
            """External tool"""
            return arg

        class TestAgent(BaseAgent):
            """"""

        toolkit = ToolKit([external_tool])
        agent = TestAgent(mock_llm, tools=toolkit)

        assert agent.toolkit is toolkit
        assert "external_tool" in agent.toolkit.tools

    def test_init_combines_all_tool_sources(self, mock_llm):
        def class_tool(arg: str) -> str:
            """Class tool"""
            return arg

        def constructor_tool(arg: str) -> str:
            """Constructor tool"""
            return arg

        class TestAgent(BaseAgent):
            """"""

            tools = [class_tool]

            @tool
            def decorated_tool(self, arg: str) -> str:
                """Decorated tool"""
                return arg

        agent = TestAgent(mock_llm, tools=[constructor_tool])

        assert len(agent.toolkit.tools) == 3
        assert "class_tool" in agent.toolkit.tools
        assert "constructor_tool" in agent.toolkit.tools
        assert "decorated_tool" in agent.toolkit.tools

    def test_add_tool(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        def new_tool(arg: str) -> str:
            """New tool"""
            return arg

        agent = TestAgent(mock_llm)
        agent.add_tool("my_tool", new_tool)

        assert "my_tool" in agent.toolkit.tools
        assert agent.toolkit.tools["my_tool"].func is new_tool

    def test_add_tool_duplicate_name_raises_error(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

            @tool
            def existing_tool(self, arg: str) -> str:
                """Existing tool"""
                return arg

        def new_tool(arg: str) -> str:
            """New tool"""
            return arg

        agent = TestAgent(mock_llm)

        with pytest.raises(ValueError, match="already added"):
            agent.add_tool("existing_tool", new_tool)

    def test_add_tool_non_callable_raises_error(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        agent = TestAgent(mock_llm)

        with pytest.raises(ValueError, match="must be a callable"):
            agent.add_tool("not_tool", "not a function")  # type: ignore

    @pytest.mark.asyncio
    async def test_invoke_basic(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        result = await agent.invoke("Test query")

        assert result == "Response"
        mock_llm.assert_called_once()
        args = mock_llm.call_args
        chat = args[0][0]
        assert len(chat.messages) == 1
        assert chat.messages[0].role == Role.user
        assert chat.messages[0].content[0].text == "Test query"

    @pytest.mark.asyncio
    async def test_invoke_with_images(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        # Use valid base64 image data
        valid_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVR4AWNgAAAAAgABc3UBGAAAAABJRU5ErkJggg=="
        result = await agent.invoke("Test query", images=[valid_image])

        assert result == "Response"
        mock_llm.assert_called_once()
        args = mock_llm.call_args
        chat = args[0][0]
        assert len(chat.messages) == 1
        assert len(chat.messages[0].content) == 2  # text + image

    @pytest.mark.asyncio
    async def test_invoke_with_parser(self, mock_llm):
        class TestModel(BaseModel):
            text: str

        class TestAgent(BaseAgent):
            """"""

        mock_response = TestModel(text="Response")
        mock_llm.return_value = mock_response
        agent = TestAgent(mock_llm)

        # The invoke method currently doesn't pass parser to LLM, so we test the current behavior
        result = await agent.invoke("Test query", parser=TestModel)

        assert result is mock_response
        mock_llm.assert_called_once()
        # Note: Currently invoke() doesn't pass parser to LLM, this tests current behavior
        args, kwargs = mock_llm.call_args
        assert "parser" not in kwargs

    @pytest.mark.asyncio
    async def test_invoke_empty(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        result = await agent.invoke()

        assert result == "Response"
        mock_llm.assert_called_once()
        args = mock_llm.call_args
        chat = args[0][0]
        # Should still call LLM but without adding user message
        assert len(chat.messages) == 0

    @pytest.mark.asyncio
    async def test_invoke_passes_tools(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

            @tool
            def my_tool(self, arg: str) -> str:
                """My tool"""
                return arg

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        await agent.invoke("Test query")

        mock_llm.assert_called_once()
        args = mock_llm.call_args
        assert args[1]["tools"] is agent.toolkit

    @pytest.mark.asyncio
    async def test_interact_default_returns_none(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

        agent = TestAgent(mock_llm)
        result = await agent.interact("response")

        assert result is None

    @pytest.mark.asyncio
    async def test_run_single_interaction(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.interaction_count = 0

            async def interact(self, response):
                self.interaction_count += 1
                return None  # End interaction

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        await agent.run("Initial query")

        assert agent.interaction_count == 1
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_multiple_interactions(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.interaction_count = 0

            async def interact(self, response):
                self.interaction_count += 1
                if self.interaction_count < 3:
                    return f"Follow-up {self.interaction_count}"
                return None

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        await agent.run("Initial query")

        assert agent.interaction_count == 3
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_run_without_initial_query(self, mock_llm):
        class TestAgent(BaseAgent):
            """"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.interaction_count = 0

            async def interact(self, response):
                self.interaction_count += 1
                return None

        mock_llm.return_value = "Response"
        agent = TestAgent(mock_llm)

        await agent.run()

        assert agent.interaction_count == 1
        mock_llm.assert_called_once()


class TestRAGMixin:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock(spec=LLM)

    @pytest.fixture
    def mock_rag(self):
        rag = AsyncMock()
        rag.return_value = "RAG response"
        return rag

    def test_rag_init_single_source(self, mock_llm, mock_rag):
        class TestAgent(RAGMixin, BaseAgent):
            """"""

        agent = TestAgent(mock_llm)
        agent.rag_init({"movie": mock_rag})

        assert agent.rag_sources == {"movie": mock_rag}
        assert "lookup_movie" in agent.toolkit.tools

    def test_rag_init_multiple_sources(self, mock_llm):
        mock_rag1 = AsyncMock()
        mock_rag2 = AsyncMock()

        class TestAgent(RAGMixin, BaseAgent):
            """"""

        agent = TestAgent(mock_llm)
        agent.rag_init({"movie": mock_rag1, "person": mock_rag2})

        assert agent.rag_sources == {"movie": mock_rag1, "person": mock_rag2}
        assert "lookup_movie" in agent.toolkit.tools
        assert "lookup_person" in agent.toolkit.tools

    def test_rag_init_updates_docstring(self, mock_llm, mock_rag):
        class TestAgent(RAGMixin, BaseAgent):
            """Original docstring"""

        agent = TestAgent(mock_llm)
        agent.rag_init({"movie": mock_rag})

        assert "{name}" in agent.__doc__  # type: ignore  # The code uses literal {name}, not formatted
        assert "Original docstring" in agent.__doc__  # type: ignore


class TestSimpleRAGAgent:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock(spec=LLM)

    @pytest.fixture
    def mock_rag(self):
        rag = AsyncMock()
        rag.return_value = "RAG response"
        return rag

    def test_init_with_rag_name(self, mock_llm, mock_rag):
        class TestRAGAgent(SimpleRAGAgent):
            """Test RAG agent"""

            rag_name = "movie"

        agent = TestRAGAgent(mock_llm, mock_rag)

        assert agent.rag_sources == {"movie": mock_rag}
        assert "lookup_movie" in agent.toolkit.tools

    def test_init_without_rag_name_raises_error(self, mock_llm, mock_rag):
        class TestRAGAgent(SimpleRAGAgent):
            """Test RAG agent"""  # No rag_name defined

        with pytest.raises(ValueError, match="rag_name must be set"):
            TestRAGAgent(mock_llm, mock_rag)

    def test_init_with_empty_rag_name_raises_error(self, mock_llm, mock_rag):
        class TestRAGAgent(SimpleRAGAgent):
            """Test RAG agent"""

            rag_name = ""

        with pytest.raises(ValueError, match="rag_name must be set"):
            TestRAGAgent(mock_llm, mock_rag)


class TestAgentIntegration:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock(spec=LLM)

    @pytest.mark.asyncio
    async def test_agent_with_tools_end_to_end(self, mock_llm):
        class TestAgent(BaseAgent):
            """You are a helpful assistant."""

            @tool
            def get_weather(self, city: str) -> str:
                """Get weather for a city"""
                return f"Sunny in {city}"

            @tool
            def calculate(self, expression: str) -> str:
                """Calculate a mathematical expression"""
                return f"Result of {expression} is 42"

        mock_llm.return_value = "The weather is sunny and the calculation result is 42"
        agent = TestAgent(mock_llm)

        result = await agent.invoke("What's the weather in NYC and what's 2+2?")

        assert result == "The weather is sunny and the calculation result is 42"
        mock_llm.assert_called_once()

        # Verify tools were passed to LLM
        args = mock_llm.call_args
        assert args[1]["tools"] is agent.toolkit
        assert len(agent.toolkit.tools) == 2
        assert "get_weather" in agent.toolkit.tools
        assert "calculate" in agent.toolkit.tools

    @pytest.mark.asyncio
    async def test_agent_template_rendering(self, mock_llm):
        class TestAgent(BaseAgent):
            """You are a {{role}} assistant. Today is {{day}}."""

        mock_llm.return_value = "Hello!"
        agent = TestAgent(mock_llm, role="helpful", day="Monday")

        await agent.invoke("Hi")

        # Check that system message was properly templated
        assert len(agent.chat.messages) == 2  # system + user
        system_msg = agent.chat.messages[0]
        assert system_msg.role == Role.system
        assert "helpful assistant" in system_msg.content[0].text  # type: ignore
        assert "Today is Monday" in system_msg.content[0].text  # type: ignore

    def test_agent_tool_naming_priority(self, mock_llm):
        """Test that @tool(name="...") takes precedence over method name"""

        class TestAgent(BaseAgent):
            """"""

            @tool(name="custom_name")
            def original_name(self, arg: str) -> str:
                """Tool with custom name"""
                return arg

        agent = TestAgent(mock_llm)

        assert "custom_name" in agent.toolkit.tools
        assert "original_name" not in agent.toolkit.tools
        assert agent.toolkit.tools["custom_name"].name == "custom_name"
