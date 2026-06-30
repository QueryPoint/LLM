import asyncio

import pytest

from assistant_service.services.gemini_client import GeminiClient


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeModels:
    def __init__(self, stream_fragments: tuple[str | None, ...]) -> None:
        self._stream_fragments = stream_fragments

    async def generate_content(self, **kwargs: object) -> FakeResponse:
        return FakeResponse("  Generated answer.  ")

    async def generate_content_stream(self, **kwargs: object) -> object:
        async def stream() -> object:
            for text in self._stream_fragments:
                yield FakeResponse(text)

        return stream()


class FakeAioClient:
    def __init__(self, stream_fragments: tuple[str | None, ...]) -> None:
        self.models = FakeModels(stream_fragments)

    async def aclose(self) -> None:
        return None


class FakeSdkClient:
    def __init__(self, stream_fragments: tuple[str | None, ...]) -> None:
        self.aio = FakeAioClient(stream_fragments)


def _client(
    stream_fragments: tuple[str | None, ...] = ("", "  first  ", "   ", "second"),
) -> GeminiClient:
    client = GeminiClient(
        api_key="placeholder",
        model="gemini-2.5-flash",
        timeout_seconds=60,
    )
    client._client = FakeSdkClient(stream_fragments)
    return client


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError):
        GeminiClient(api_key="   ", model="gemini-2.5-flash", timeout_seconds=60)


def test_constructor_rejects_empty_model_and_invalid_timeout() -> None:
    invalid_values = (
        ("   ", 60),
        ("gemini-2.5-flash", 0),
    )

    for model, timeout_seconds in invalid_values:
        with pytest.raises(ValueError):
            GeminiClient(
                api_key="placeholder",
                model=model,
                timeout_seconds=timeout_seconds,
            )


def test_generate_text_rejects_empty_required_parameters() -> None:
    invalid_values = (
        ("   ", "Prompt"),
        ("System", "   "),
    )

    for system_instruction, prompt in invalid_values:
        with pytest.raises(ValueError):
            asyncio.run(
                _client().generate_text(
                    system_instruction=system_instruction,
                    prompt=prompt,
                )
            )


def test_generate_text_returns_stripped_text_from_fake_sdk_response() -> None:
    response = asyncio.run(
        _client().generate_text(
            system_instruction="System",
            prompt="Prompt",
        )
    )

    assert response == "Generated answer."


def test_stream_text_skips_empty_chunks_and_yields_non_empty_fragments() -> None:
    async def collect() -> list[str]:
        return [
            fragment
            async for fragment in _client().stream_text(
                system_instruction="System",
                prompt="Prompt",
            )
        ]

    assert asyncio.run(collect()) == ["  first  ", "second"]


def test_stream_text_preserves_fragment_boundaries() -> None:
    async def collect() -> list[str]:
        return [
            fragment
            async for fragment in _client(
                stream_fragments=("Привет ", "мир", "!", "   ")
            ).stream_text(
                system_instruction="System",
                prompt="Prompt",
            )
        ]

    assert asyncio.run(collect()) == ["Привет ", "мир", "!"]
