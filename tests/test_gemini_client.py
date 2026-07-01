import asyncio

import pytest

from assistant_service.services.gemini_client import GeminiClient, GeminiRequestLimitError


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("status error")
        self.status_code = status_code


class FakeModels:
    def __init__(
        self,
        stream_fragments: tuple[str | None, ...],
        generate_results: list[object] | None = None,
    ) -> None:
        self._stream_fragments = stream_fragments
        self._generate_results = generate_results or [FakeResponse("  Generated answer.  ")]
        self.generate_content_calls = 0

    async def generate_content(self, **kwargs: object) -> FakeResponse:
        self.generate_content_calls += 1
        result = self._generate_results.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, FakeResponse):
            return result
        return FakeResponse(str(result))

    async def generate_content_stream(self, **kwargs: object) -> object:
        async def stream() -> object:
            for text in self._stream_fragments:
                yield FakeResponse(text)

        return stream()


class FakeAioClient:
    def __init__(
        self,
        stream_fragments: tuple[str | None, ...],
        generate_results: list[object] | None = None,
    ) -> None:
        self.models = FakeModels(stream_fragments, generate_results)

    async def aclose(self) -> None:
        return None


class FakeSdkClient:
    def __init__(
        self,
        stream_fragments: tuple[str | None, ...],
        generate_results: list[object] | None = None,
    ) -> None:
        self.aio = FakeAioClient(stream_fragments, generate_results)


def _client(
    stream_fragments: tuple[str | None, ...] = ("", "  first  ", "   ", "second"),
    generate_results: list[object] | None = None,
    max_prompt_chars: int = 16_000,
    max_response_chars: int = 12_000,
    retry_max_attempts: int = 3,
) -> GeminiClient:
    client = GeminiClient(
        api_key="placeholder",
        model="gemini-2.5-flash",
        timeout_seconds=60,
        retry_max_attempts=retry_max_attempts,
        retry_initial_delay_seconds=0.5,
        retry_max_delay_seconds=4.0,
        max_user_prompt_chars=4_000,
        max_prompt_chars=max_prompt_chars,
        max_response_chars=max_response_chars,
    )
    client._client = FakeSdkClient(stream_fragments, generate_results)
    return client


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError):
        GeminiClient(
            api_key="   ",
            model="gemini-2.5-flash",
            timeout_seconds=60,
            retry_max_attempts=3,
            retry_initial_delay_seconds=0.5,
            retry_max_delay_seconds=4.0,
            max_user_prompt_chars=4_000,
            max_prompt_chars=16_000,
            max_response_chars=12_000,
        )


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
                retry_max_attempts=3,
                retry_initial_delay_seconds=0.5,
                retry_max_delay_seconds=4.0,
                max_user_prompt_chars=4_000,
                max_prompt_chars=16_000,
                max_response_chars=12_000,
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


def test_generate_text_retries_rate_limit_then_returns_response() -> None:
    client = _client(
        generate_results=[
            FakeStatusError(429),
            FakeStatusError(429),
            "  Retry success  ",
        ]
    )
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    client._sleep = fake_sleep

    response = asyncio.run(
        client.generate_text(system_instruction="System", prompt="Prompt")
    )

    assert response == "Retry success"
    assert client._client.aio.models.generate_content_calls == 3
    assert sleep_delays == [0.5, 1.0]


def test_generate_text_enforces_prompt_and_response_limits() -> None:
    prompt_limit_client = _client(max_prompt_chars=10)

    with pytest.raises(GeminiRequestLimitError):
        asyncio.run(
            prompt_limit_client.generate_text(
                system_instruction="System",
                prompt="Prompt longer than limit",
            )
        )
    assert prompt_limit_client._client.aio.models.generate_content_calls == 0

    response_limit_client = _client(
        generate_results=["response longer than limit"],
        max_response_chars=5,
    )
    with pytest.raises(GeminiRequestLimitError):
        asyncio.run(
            response_limit_client.generate_text(
                system_instruction="System",
                prompt="Prompt",
            )
        )


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
