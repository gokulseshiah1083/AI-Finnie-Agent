from unittest.mock import patch

from src.core.llm_client import call_llm, resolve_model_name


def test_resolve_model_name_maps_legacy_aliases():
    assert resolve_model_name("claude-3-haiku") == "claude-3-5-haiku-latest"
    assert resolve_model_name("claude-sonnet-4-6") == "claude-3-5-sonnet-latest"
    assert resolve_model_name("claude-3-5-haiku-latest") == "claude-3-5-haiku-latest"


def test_call_llm_returns_fallback_when_api_call_fails():
    class FakeClient:
        class Messages:
            @staticmethod
            def create(*args, **kwargs):
                raise RuntimeError("boom")

        messages = Messages()

    with patch("src.core.llm_client.get_client", return_value=FakeClient()):
        response = call_llm("system", "hello")

    assert "unable to generate a response" in response.lower()
