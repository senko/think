# example: structured.py
from asyncio import run

from think import LLM, LLMQuery

llm = LLM.from_url("openai:///gpt-4o-mini")


class CityInfo(LLMQuery):
    """
    Give me basic information about {{ city }}.
    """

    name: str
    country: str
    population: int
    latitude: float
    longitude: float


async def city_info(city):
    return await CityInfo.run(llm, city=city)


info = run(city_info("Paris"))
print(f"{info.name} is a city in {info.country} with {info.population} inhabitants.")
