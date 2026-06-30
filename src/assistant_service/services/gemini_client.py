import logging
from collections.abc import AsyncIterator
from numbers import Real

from google import genai
from google.genai import types

from assistant_service.core.config import Settings

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Raised when Gemini generation cannot be completed safely."""


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: int) -> None:
        normalized_api_key = api_key.strip()
        normalized_model = model.strip()

        if normalized_api_key == "":
            raise ValueError("api_key must not be empty")
        if normalized_model == "":
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._model = normalized_model
        self._timeout_seconds = timeout_seconds
        self._client = genai.Client(
            api_key=normalized_api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )

    async def generate_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        normalized_system_instruction, normalized_prompt = (
            self._validate_generation_parameters(
                system_instruction=system_instruction,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        )
        generation_config = self._build_generation_config(
            system_instruction=normalized_system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=normalized_prompt,
                config=generation_config,
            )
            return self._extract_text(response)
        except GeminiClientError:
            raise
        except Exception as exc:
            self._log_generation_exception("generate_text", exc)
            raise GeminiClientError("Gemini text generation failed.") from exc

    async def stream_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        normalized_system_instruction, normalized_prompt = (
            self._validate_generation_parameters(
                system_instruction=system_instruction,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        )
        generation_config = self._build_generation_config(
            system_instruction=normalized_system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        emitted_fragments = 0

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=normalized_prompt,
                config=generation_config,
            )

            async for chunk in stream:
                text = self._extract_optional_text(chunk)
                if text is None:
                    continue

                emitted_fragments += 1
                yield text
        except GeminiClientError:
            raise
        except Exception as exc:
            self._log_generation_exception("stream_text", exc)
            raise GeminiClientError("Gemini streaming generation failed.") from exc

        if emitted_fragments == 0:
            raise GeminiClientError("Gemini streaming response did not contain text.")

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    def _validate_generation_parameters(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float,
    ) -> tuple[str, str]:
        normalized_system_instruction = system_instruction.strip()
        normalized_prompt = prompt.strip()

        if normalized_system_instruction == "":
            raise ValueError("system_instruction must not be empty")
        if normalized_prompt == "":
            raise ValueError("prompt must not be empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if not isinstance(temperature, Real) or not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

        return normalized_system_instruction, normalized_prompt

    @staticmethod
    def _build_generation_config(
        *,
        system_instruction: str,
        max_output_tokens: int,
        temperature: float,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        text = GeminiClient._extract_optional_text(response)
        if text is None:
            raise GeminiClientError("Gemini response did not contain text.")
        return text

    @staticmethod
    def _extract_optional_text(response: object) -> str | None:
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return None

        stripped_text = text.strip()
        return stripped_text or None

    def _log_generation_exception(self, operation: str, exc: Exception) -> None:
        logger.exception(
            "Gemini generation failed: operation=%s model=%s "
            "timeout_seconds=%s error_type=%s",
            operation,
            self._model,
            self._timeout_seconds,
            type(exc).__name__,
        )


def create_gemini_client_from_settings(settings: Settings) -> GeminiClient:
    if settings.gemini_api_key is None or settings.gemini_api_key.strip() == "":
        raise ValueError("GEMINI_API_KEY is required to create GeminiClient.")

    return GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
