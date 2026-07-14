"""Creation of the OpenRouter API client."""

from openai import OpenAI


def get_openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
