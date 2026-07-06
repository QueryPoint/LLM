import asyncio
import logging
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from numbers import Real

from google import genai
from google.genai import types

from assistant_service.core.config import Settings

logger = logging.getLogger(__name__)


class GeminiClientError(RuntimeError):
    """Raised when Gemini generation cannot be completed safely."""


class GeminiRequestLimitError(GeminiClientError):
    """Локальный лимит prompt/chunks/response превышен."""


class GeminiRateLimitError(GeminiClientError):
    """Gemini вернул rate limit / quota exhaustion."""


class GeminiTransientError(GeminiClientError):
    """Временная ошибка Gemini после исчерпания retry."""


class GeminiPermanentError(GeminiClientError):
    """Неповторяемая Gemini API ошибка."""


RATE_LIMIT_STATUS_CODE = 429
TRANSIENT_STATUS_CODES = frozenset({408, 500, 502, 503, 504})
PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 413, 422})
GEMINI_FILE_READY_STATES = frozenset({"ACTIVE", "SUCCEEDED"})
GEMINI_FILE_FAILED_STATES = frozenset({"FAILED", "ERROR"})
GEMINI_FILE_PROCESSING_POLL_SECONDS = 2


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int,
        *,
        retry_max_attempts: int,
        retry_initial_delay_seconds: float,
        retry_max_delay_seconds: float,
        max_user_prompt_chars: int,
        max_prompt_chars: int,
        max_response_chars: int,
    ) -> None:
        normalized_api_key = api_key.strip()
        normalized_model = model.strip()

        if normalized_api_key == "":
            raise ValueError("api_key must not be empty")
        if normalized_model == "":
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retry_max_attempts <= 0:
            raise ValueError("retry_max_attempts must be positive")
        if retry_initial_delay_seconds <= 0:
            raise ValueError("retry_initial_delay_seconds must be positive")
        if retry_max_delay_seconds <= 0:
            raise ValueError("retry_max_delay_seconds must be positive")
        if retry_max_delay_seconds < retry_initial_delay_seconds:
            raise ValueError(
                "retry_max_delay_seconds must be greater than or equal to "
                "retry_initial_delay_seconds"
            )
        if max_user_prompt_chars <= 0:
            raise ValueError("max_user_prompt_chars must be positive")
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")
        if max_response_chars <= 0:
            raise ValueError("max_response_chars must be positive")

        self._model = normalized_model
        self._timeout_seconds = timeout_seconds
        self._retry_max_attempts = retry_max_attempts
        self._retry_initial_delay_seconds = retry_initial_delay_seconds
        self._retry_max_delay_seconds = retry_max_delay_seconds
        self._max_user_prompt_chars = max_user_prompt_chars
        self._max_prompt_chars = max_prompt_chars
        self._max_response_chars = max_response_chars
        self._sleep = asyncio.sleep
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

        async def request() -> str:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=normalized_prompt,
                config=generation_config,
            )
            return self._extract_text(response)

        return await self._run_with_retries("generate_text", request)

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
        total_response_chars = 0

        for attempt_index in range(self._retry_max_attempts):
            try:
                stream = await self._client.aio.models.generate_content_stream(
                    model=self._model,
                    contents=normalized_prompt,
                    config=generation_config,
                )

                async for chunk in stream:
                    text = self._extract_optional_stream_text(chunk)
                    if text is None:
                        continue

                    if total_response_chars + len(text) > self._max_response_chars:
                        raise GeminiRequestLimitError(
                            "Gemini response exceeds configured character limit."
                        )

                    emitted_fragments += 1
                    total_response_chars += len(text)
                    yield text
                break
            except GeminiClientError:
                raise
            except Exception as exc:
                if emitted_fragments > 0:
                    self._log_generation_failure("stream_text", exc)
                    raise GeminiTransientError(
                        "Gemini streaming generation failed after partial response."
                    ) from exc

                if not await self._handle_retryable_exception(
                    operation="stream_text",
                    exc=exc,
                    attempt_index=attempt_index,
                ):
                    raise
        else:
            raise GeminiTransientError("Gemini streaming generation failed.")

        if emitted_fragments == 0:
            raise GeminiClientError("Gemini streaming response did not contain text.")

    async def summarize_pdf_document(
        self,
        *,
        data: bytes,
        prompt: str,
        max_output_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> str:
        normalized_prompt = prompt.strip()
        if data == b"":
            raise GeminiRequestLimitError("Gemini document input is empty.")
        if normalized_prompt == "":
            raise ValueError("prompt must not be empty")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if not isinstance(temperature, Real) or not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")

        uploaded_file: object | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temporary_file:
                temporary_file.write(data)
                temporary_file.flush()
                uploaded_file = await self._client.aio.files.upload(
                    file=temporary_file.name,
                    config=types.UploadFileConfig(mime_type="application/pdf"),
                )

            uploaded_file = await self._wait_until_file_ready(uploaded_file)
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[normalized_prompt, uploaded_file],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            return self._extract_text(response)
        except GeminiClientError:
            raise
        except Exception as exc:
            self._log_generation_failure("pdf_document_summary", exc)
            if self._extract_status_code(exc) == RATE_LIMIT_STATUS_CODE:
                raise GeminiRateLimitError("Gemini rate limit exceeded.") from exc
            if self._is_transient_exception(exc, self._extract_status_code(exc)):
                raise GeminiTransientError("Gemini document summary failed.") from exc
            raise GeminiPermanentError("Gemini document summary failed.") from exc
        finally:
            if uploaded_file is not None:
                await self._delete_uploaded_file(uploaded_file)

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def _wait_until_file_ready(self, uploaded_file: object) -> object:
        name = getattr(uploaded_file, "name", None)
        if not isinstance(name, str) or name == "":
            return uploaded_file

        deadline = time.monotonic() + self._timeout_seconds
        current_file = uploaded_file
        while time.monotonic() < deadline:
            state = self._file_state(current_file)
            if state is None or state in GEMINI_FILE_READY_STATES:
                return current_file
            if state in GEMINI_FILE_FAILED_STATES:
                raise GeminiTransientError("Gemini file processing failed.")

            await self._sleep(GEMINI_FILE_PROCESSING_POLL_SECONDS)
            current_file = await self._client.aio.files.get(name=name)

        raise GeminiTransientError("Gemini file processing timed out.")

    async def _delete_uploaded_file(self, uploaded_file: object) -> None:
        name = getattr(uploaded_file, "name", None)
        if not isinstance(name, str) or name == "":
            return

        try:
            await self._client.aio.files.delete(name=name)
        except Exception as exc:
            logger.warning(
                "Gemini file cleanup failed: operation=%s error_type=%s",
                "files_api_delete",
                type(exc).__name__,
            )

    @staticmethod
    def _file_state(file_object: object) -> str | None:
        state = getattr(file_object, "state", None)
        if state is None:
            return None
        value = getattr(state, "value", state)
        return str(value).upper()

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
        if (
            len(normalized_prompt) > self._max_prompt_chars
            or len(normalized_system_instruction) + len(normalized_prompt)
            > self._max_prompt_chars
        ):
            raise GeminiRequestLimitError(
                "Gemini prompt exceeds configured character limit."
            )

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

    def _extract_text(self, response: object) -> str:
        text = GeminiClient._extract_optional_text(response)
        if text is None:
            raise GeminiClientError("Gemini response did not contain text.")
        if len(text) > self._max_response_chars:
            raise GeminiRequestLimitError(
                "Gemini response exceeds configured character limit."
            )
        return text

    @staticmethod
    def _extract_optional_text(response: object) -> str | None:
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return None

        stripped_text = text.strip()
        return stripped_text or None

    @staticmethod
    def _extract_optional_stream_text(response: object) -> str | None:
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            return None

        if text.strip() == "":
            return None

        return text

    async def _run_with_retries(
        self,
        operation: str,
        request: Callable[[], Awaitable[str]],
    ) -> str:
        for attempt_index in range(self._retry_max_attempts):
            try:
                return await request()
            except GeminiClientError:
                raise
            except Exception as exc:
                if not await self._handle_retryable_exception(
                    operation=operation,
                    exc=exc,
                    attempt_index=attempt_index,
                ):
                    raise

        raise GeminiTransientError("Gemini generation failed.")

    async def _handle_retryable_exception(
        self,
        *,
        operation: str,
        exc: Exception,
        attempt_index: int,
    ) -> bool:
        status_code = self._extract_status_code(exc)
        is_rate_limit = status_code == RATE_LIMIT_STATUS_CODE
        is_transient = is_rate_limit or self._is_transient_exception(exc, status_code)

        if not is_transient:
            self._log_generation_failure(operation, exc)
            raise GeminiPermanentError("Gemini request failed permanently.") from exc

        if attempt_index >= self._retry_max_attempts - 1:
            self._log_generation_failure(operation, exc)
            if is_rate_limit:
                raise GeminiRateLimitError("Gemini rate limit exceeded.") from exc
            raise GeminiTransientError("Gemini request failed after retries.") from exc

        logger.warning(
            "Gemini request retry: operation=%s attempt=%s status_code=%s "
            "error_type=%s",
            operation,
            attempt_index + 1,
            status_code,
            type(exc).__name__,
        )
        await self._sleep(self._retry_delay(attempt_index))
        return True

    def _retry_delay(self, retry_index: int) -> float:
        return min(
            self._retry_initial_delay_seconds * (2**retry_index),
            self._retry_max_delay_seconds,
        )

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        for attribute_name in ("code", "status_code"):
            value = getattr(exc, attribute_name, None)
            if isinstance(value, int):
                return value

        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
        return None

    @staticmethod
    def _is_transient_exception(exc: Exception, status_code: int | None) -> bool:
        if status_code in TRANSIENT_STATUS_CODES:
            return True
        if status_code in PERMANENT_STATUS_CODES:
            return False
        if isinstance(exc, asyncio.TimeoutError):
            return True

        error_type = type(exc).__name__.lower()
        return any(
            marker in error_type
            for marker in ("timeout", "connecterror", "connectionerror", "network")
        )

    def _log_generation_failure(self, operation: str, exc: Exception) -> None:
        logger.warning(
            "Gemini request failed: operation=%s status_code=%s error_type=%s",
            operation,
            self._extract_status_code(exc),
            type(exc).__name__,
        )


def create_gemini_client_from_settings(settings: Settings) -> GeminiClient:
    if settings.gemini_api_key is None or settings.gemini_api_key.strip() == "":
        raise ValueError("GEMINI_API_KEY is required to create GeminiClient.")

    return GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
        retry_max_attempts=settings.gemini_retry_max_attempts,
        retry_initial_delay_seconds=settings.gemini_retry_initial_delay_seconds,
        retry_max_delay_seconds=settings.gemini_retry_max_delay_seconds,
        max_user_prompt_chars=settings.gemini_max_user_prompt_chars,
        max_prompt_chars=settings.gemini_max_prompt_chars,
        max_response_chars=settings.gemini_max_response_chars,
    )
