import anthropic
from functools import lru_cache
from src.core.config import CONFIG, get_anthropic_api_key


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_anthropic_api_key())


def call_llm(system_prompt: str, user_message: str, conversation_history: list | None = None) -> str:
    client = get_client()
    cfg = CONFIG["llm"]

    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text
