import anthropic
from functools import lru_cache
from src.core.config import CONFIG, get_anthropic_api_key


def resolve_model_name(model_name: str) -> str:
    aliases = {
        "claude-3-haiku": "claude-opus-4-1",
        "claude-sonnet-4-6": "claude-opus-4-1",
        "claude-3-5-haiku-latest": "claude-opus-4-1",
        "claude-3-7-sonnet-latest": "claude-opus-4-1",
    }
    return aliases.get(model_name, model_name)


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

    try:
        response = client.messages.create(
            model=resolve_model_name(cfg["model"]),
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text
    except Exception:
        return (
            "I’m sorry, but I’m unable to generate a response right now. "
            "Please try again in a moment."
        )
