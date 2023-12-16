import json
import sys
from pydantic import BaseModel
import click

sys.path.append(".")

from think.llm.openai import ChatGPT  # noqa E402
from think.chat import Chat  # noqa E402
from think.parser import JSONParser  # noqa E402


class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    latitude: float
    longitude: float


@click.command()
@click.option("--api-key", "-k", default=None)
@click.argument("city", default="Zagreb", required=False)
def main(city, api_key=None):
    """
    Ask GPT-3 to answer information about a city in a structured format.

    API key, if not provided, will be read from OPENAI_API_KEY environment variable.
    """
    llm = ChatGPT(model="gpt-3.5-turbo-16k")
    parser = JSONParser(spec=CityInfo)
    chat = Chat(
        "You are a hepful assistant. Your task is to answer questions about cities, "
        "to the best of your knowledge. Your output must be a valid JSON conforming to "
        "this JSON schema:\n" + json.dumps(parser.schema)
    ).user(city)

    answer = llm(chat, parser=parser)
    print(
        f"{answer.name} is a city in {answer.country} with {answer.population} inhabitants."
    )
    print(
        f"It is located at {answer.latitude:.2f}° latitude and {answer.longitude:.2f}° longitude."
    )


if __name__ == "__main__":
    main()
