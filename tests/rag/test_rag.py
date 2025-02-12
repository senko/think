from dotenv import load_dotenv
from os import getenv
import pytest
import time


from think import LLM
from think.rag.base import RAG, RagDocument

from conftest import first_model_url

load_dotenv()

if getenv("INTEGRATION_TESTS", "").lower() not in ["true", "yes", "1", "on"]:
    pytest.skip("Skipping integration tests", allow_module_level=True)


LLM_URL = first_model_url()

MOVIES = [
    "Titanic (1997): A sweeping romantic epic set against the backdrop of the "
    "ill-fated Titanic, following the love story of Jack and Rose as they "
    "navigate class divisions and impending disaster.",
    "The Godfather (1972): A gripping mafia saga that follows the Corleone "
    "crime family, led by patriarch Vito and his reluctant son Michael, as "
    "they navigate power, loyalty, and betrayal.",
    "Schindler's List (1993): A harrowing and deeply moving Holocaust drama "
    "about Oskar Schindler, a businessman who saves over a thousand Jewish "
    "lives by employing them in his factory.",
    "The Lord of the Rings: The Return of the King (2003): The epic conclusion "
    "to the fantasy trilogy, featuring the final battle for Middle-earth and "
    "Frodo's journey to destroy the One Ring.",
    "Ben-Hur (1959): A monumental historical drama following Judah Ben-Hur, "
    "a nobleman betrayed into slavery who seeks revenge, culminating in a "
    "legendary chariot race.",
    "Forrest Gump (1994): A heartwarming tale of a simple man whose accidental "
    "presence in key historical events shows how love and kindness shape his "
    "extraordinary life.",
    "Casablanca (1942): A timeless romance set in WWII, following Rick Blaine, "
    "a cynical American expatriate, as he must choose between love and aiding "
    "the resistance.",
    "One Flew Over the Cuckoo's Nest (1975): A rebellious patient shakes up a "
    "rigid mental institution, challenging authority and inspiring fellow "
    "inmates to reclaim their dignity.",
    "Gladiator (2000): A revenge-driven historical epic where a betrayed Roman "
    "general fights as a gladiator to avenge his family and bring justice to a "
    "corrupt emperor.",
    "Gone with the Wind (1939): A sweeping Civil War-era romance that follows "
    "the tumultuous life of the headstrong Scarlett O'Hara and her relationship "
    "with Rhett Butler.",
    "The Silence of the Lambs (1991): A chilling psychological thriller in which "
    "young FBI agent Clarice Starling seeks the help of imprisoned cannibal "
    "Hannibal Lecter to catch a serial killer.",
    "No Country for Old Men (2007): A tense neo-Western thriller about a hunter "
    "who stumbles upon a drug deal gone wrong, pursued by a relentless assassin "
    "and a weary sheriff.",
    "Parasite (2019): A sharp social satire about a poor family infiltrating a "
    "wealthy household, exposing deep class divides through dark humor and "
    "shocking twists.",
    "Amadeus (1984): A dramatic retelling of the rivalry between composers "
    "Mozart and Salieri, exploring genius, jealousy, and the price of artistic "
    "brilliance.",
    "Braveheart (1995): A brutal and inspiring historical epic about William "
    "Wallace leading the Scottish rebellion against English tyranny in the "
    "13th century.",
    "12 Years a Slave (2013): A gut-wrenching true story of Solomon Northup, "
    "a free Black man abducted and sold into slavery, depicting the horrors "
    "and resilience of the human spirit.",
    "The Shape of Water (2017): A unique fantasy romance between a mute woman "
    "and a mysterious amphibious creature, set in Cold War-era America.",
    "The Departed (2006): A gritty crime thriller about undercover agents on "
    "opposite sides of the law, entangled in a dangerous game of deception and "
    "survival.",
    "Slumdog Millionaire (2008): A rags-to-riches tale of a Mumbai slum boy whose "
    "life experiences help him succeed on a game show while searching for his "
    "lost love.",
    "The Artist (2011): A silent film homage about a 1920s movie star "
    "struggling to adapt to the rise of talkies, capturing the magic and "
    "pain of Hollywood's transition.",
]

PINECONE_INDEX_NAME = "think-test"


@pytest.fixture
def pinecone_index():
    api_key = getenv("PINECONE_API_KEY")
    if api_key is None:
        yield None
        return

    try:
        from pinecone.grpc import PineconeGRPC as Pinecone
    except ImportError:
        yield None
        return

    pc = Pinecone(api_key=api_key)
    indices = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in indices:
        yield None
        return

    index = pc.Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True)
    except:  # noqa
        pass

    yield PINECONE_INDEX_NAME

    try:
        index.delete(delete_all=True)
    except:  # noqa
        pass


@pytest.mark.asyncio
async def test_txtai_integration():
    llm = LLM.from_url(LLM_URL)
    rag_class = RAG.for_provider("txtai")

    rag: RAG = rag_class(llm)

    data = [RagDocument(id=str(i), text=text) for i, text in enumerate(MOVIES)]
    await rag.add_documents(data)

    n_docs = await rag.count()
    assert n_docs == len(MOVIES)

    query = "A movie about a ship that sinks"
    result = await rag(query)
    assert "titanic" in result.lower()

    await rag.remove_documents([doc["id"] for doc in data])

    no_docs = await rag.count()
    assert no_docs == 0


@pytest.mark.asyncio
async def test_chroma_integration(tmpdir):
    llm = LLM.from_url(LLM_URL)
    rag_class = RAG.for_provider("chroma")

    rag: RAG = rag_class(llm, collection="test", path=tmpdir)

    data = [RagDocument(id=str(i), text=text) for i, text in enumerate(MOVIES)]
    await rag.add_documents(data)

    n_docs = await rag.count()
    assert n_docs == len(MOVIES)

    query = "A movie about a ship that sinks"
    result = await rag(query)
    assert "titanic" in result.lower()

    await rag.remove_documents([doc["id"] for doc in data])

    no_docs = await rag.count()
    assert no_docs == 0


@pytest.mark.asyncio
@pytest.mark.skip("Pinecone is flaky/slow, can't reliably count docs after insert")
async def test_pinecone_integration(tmpdir, pinecone_index):
    if pinecone_index is None:
        pytest.skip("Pinecone index not available")

    llm = LLM.from_url(LLM_URL)
    rag_class = RAG.for_provider("pinecone")

    rag: RAG = rag_class(llm, index_name=pinecone_index)

    data = [RagDocument(id=str(i), text=text) for i, text in enumerate(MOVIES)]
    await rag.add_documents(data)

    for i in range(60):
        time.sleep(1)
        n_docs = await rag.count()
        if n_docs == len(MOVIES):
            break
    else:
        assert False, "Failed to add documents to Pinecone index after 30s"

    query = "A movie about a ship that sinks"
    result = await rag(query)
    assert "titanic" in result.lower()

    await rag.remove_documents([doc["id"] for doc in data])

    no_docs = await rag.count()
    assert no_docs == 0
