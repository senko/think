from os import getenv


def model_urls(vision: bool = False) -> list[str]:
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
        retval.append("google:///gemini-2.0-flash-lite-preview-02-05")
    if getenv("GROQ_API_KEY"):
        retval.append("groq:///llama-3.2-90b-vision-preview")
    if getenv("OLLAMA_MODEL"):
        if vision:
            retval.append(f"ollama:///{getenv('OLLAMA_VISION_MODEL')}")
        else:
            retval.append(f"ollama:///{getenv('OLLAMA_MODEL')}")
    if getenv("AWS_SECRET_ACCESS_KEY"):
        retval.append("bedrock:///amazon.nova-lite-v1:0?region=us-east-1")
    if retval == []:
        raise RuntimeError("No LLM API keys found in environment")
    return retval


def api_model_urls() -> list[str]:
    return [url for url in model_urls() if not url.startswith("ollama:")]


def first_model_url() -> str:
    urls = model_urls()
    if urls:
        return urls[0]
    raise RuntimeError("No LLM API keys found in environment")
