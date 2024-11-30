import logging
from os import getenv

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from think import LLM
from think.llm.base import BadRequestError, ConfigError
from think.llm.chat import Chat

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

TEST_IMG = """
iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAAAAADmVT4XAAABW0lEQVR42u1ay5LDIAyzMvz/L3sv
6U623RQztXBDpHuwkB/BBjNBEARBuDvAWNQH1gfTesQEW4GuFYyvjVQK2/Erd1poeYRAanA/Uzjb
HMblzY0EEFM8xIBJIMSASiDCYLNicBUISEAm0Gewugv6EpQrsL4Lej5oOb/fC7tgm+9kH3IBls8C
ESgn0D76OqFKyAWfuQALKFD+N1QQ6ky4vAtu35hU94a3b8+rJyTVM6LcKdn44Ss4LG2l2+dlQXxW
fLlpea51+4IbE0FAerep5lQQBGFSX5BWs1WKBUEQaGdgD67OJWAG8/dWGllhL1Pgj3UUEbCe/nb5
W7NgnvkejCgh0AnGVlsFuDHw+4Zxf92I+UEYyMNGi7wXt+PfszRRgV1497cFsZEz8OgFL6iEj1ro
p+bolbC3w1mF6JTLpOt7qDesjQGoPRcBERABERABEaiaD9hhUPK1CvwAlD5AuPT/3ngAAAAASUVO
RK5CYII=
"""

if getenv("INTEGRATION_TESTS", "").lower() not in ["true", "yes", "1", "on"]:
    pytest.skip("Skipping integration tests", allow_module_level=True)


def model_urls() -> list[str]:
    """
    Returns a list of models to test with, based on available API keys.

    :return: A list of model URLs based on the available API keys.
    """
    retval = []
    if getenv("OPENAI_API_KEY"):
        retval.append("openai:///gpt-4o-mini")
    if getenv("ANTHROPIC_API_KEY"):
        retval.append("anthropic:///claude-3-haiku-20240307")
    if getenv("GEMINI_API_KEY"):
        retval.append("google:///gemini-1.5-pro-latest")
    if getenv("GROQ_API_KEY"):
        retval.append("groq:///llama-3.2-90b-vision-preview")
    if getenv("OLLAMA_MODEL"):
        retval.append(f"ollama:///{getenv('OLLAMA_MODEL')}")
    if retval == []:
        raise RuntimeError("No LLM API keys found in environment")
    return retval


def api_model_urls() -> list[str]:
    return [url for url in model_urls() if not url.startswith("ollama:")]


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_basic_request(url):
    c = Chat("You're a friendly assistant").user("Tell me a joke")
    llm = LLM.from_url(url)
    resp = await llm(c)
    assert resp != ""


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_streaming(url):
    c = Chat("You're a friendly assistant").user("Tell me a joke")
    llm = LLM.from_url(url)
    text = ""
    async for chunk in llm.stream(c):
        text += chunk
    assert text != ""
    assert text == c.messages[-1].content[0].text


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_tool_use(url):
    tool_called = False

    def get_temperature(city: str) -> str:
        """Returns the real-time temperature in a city.

        :param city: The city to get the temperature for.
        :return: The temperature in the city.
        """
        assert "new york" in city.lower()
        nonlocal tool_called
        tool_called = True
        return "70F"

    c = Chat(
        "You're a friendly assistant, and you have access to the "
        "'get_temperature' function"
    ).user("What's the temperature in New York?")

    llm = LLM.from_url(url)
    resp = await llm(c, tools=[get_temperature])

    assert "70" in resp, f"Expected 70F in LLM response, got {resp}"
    assert tool_called, "Expected 'get_temperature' tool to be called"


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_vision(url):
    c = Chat("You're a friendly assistant").user(
        "Describe the image in detail",
        images=[TEST_IMG],
    )
    llm = LLM.from_url(url)
    resp = await llm(c)
    assert resp != ""


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_pydantic_parser(url):
    class CityInfo(BaseModel):
        name: str
        country: str
        population: int
        latitude: float
        longitude: float

    c = Chat(
        "You're a friendly assistant. "
        "You always output your answers in requested format"
    ).user(
        """
        Give me basic information about New York.

        The response should be in the following JSON format:
        {
            "name": "...",
            "country": "...",
            "population": ...,
            "latitude": ...,
            "longitude": ...
        }
        """
    )
    llm = LLM.from_url(url)
    resp = await llm(c, parser=CityInfo)
    assert isinstance(resp, CityInfo), f"Expected CityInfo, got `{type(resp).__name__}`"


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_custom_parser(url):
    c = Chat(
        "You're a friendly assistant. "
        "You always output your answers in requested format"
    ).user(
        """
        Output the speed of light in vacuum, in km/s.

        Output ONLY the number, without the unit suffix or any other comments.
        """
    )
    llm = LLM.from_url(url)
    resp = await llm(c, parser=float)
    assert isinstance(resp, float), f"Expected a number, got `{type(resp).__name__}`"


@pytest.mark.parametrize("url", api_model_urls())
@pytest.mark.asyncio
async def test_auth_error(url):
    c = Chat("You're a friendly assistant").user("Tell me a joke")
    invalid_key_url = url.replace("///", "//testing-incorrect-key@/")
    llm = LLM.from_url(invalid_key_url)

    with pytest.raises(ConfigError):
        await llm(c)

    with pytest.raises(ConfigError):
        async for _ in llm.stream(c):
            pass


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_model_error(url):
    c = Chat("You're a friendly assistant").user("Tell me a joke")
    invalid_model_url = url + "-invalid"
    llm = LLM.from_url(invalid_model_url)

    with pytest.raises(ConfigError):
        await llm(c)

    with pytest.raises(ConfigError):
        async for _ in llm.stream(c):
            pass


@pytest.mark.parametrize("url", model_urls())
@pytest.mark.asyncio
async def test_chat_error(url):
    c = Chat("You're a friendly assistant").user("Tell me a joke")
    llm = LLM.from_url(url)

    class FakeAdapter:
        spec = None

        def __init__(self, *args, **kwargs):
            pass

        def dump_chat(self, chat: Chat):
            return "", {"messages": "invalid"}

    llm.adapter_class = FakeAdapter

    with pytest.raises(BadRequestError):
        await llm(c)

    with pytest.raises(BadRequestError):
        async for _ in llm.stream(c):
            pass
