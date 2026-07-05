import argparse
import asyncio
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Protocol

from google import genai
from google.genai import types
from minio import Minio

from assistant_service.core.config import Settings

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """
Составь краткое структурированное изложение документа:
1. тема;
2. ключевые тезисы;
3. основные термины;
4. итог.
Не добавляй факты, которых нет в документе.
""".strip()
SUPPORTED_FORMATS = frozenset({"pdf", "docx"})
INLINE_PDF_MAX_BYTES = 8 * 1024 * 1024
GEMINI_FILE_READY_STATES = frozenset({"ACTIVE", "SUCCEEDED"})
GEMINI_FILE_FAILED_STATES = frozenset({"FAILED", "ERROR"})
GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS = 60
GEMINI_FILE_PROCESSING_POLL_SECONDS = 2


class ObjectStorageClient(Protocol):
    async def download_object(self, *, bucket: str, object_key: str) -> bytes:
        ...

    async def presigned_get_url(
        self,
        *,
        bucket: str,
        object_key: str,
        expires_seconds: int,
    ) -> str:
        ...


class DocumentGeminiProbe(Protocol):
    async def summarize_with_files_api(
        self,
        *,
        data: bytes,
        document_format: str,
    ) -> "MethodProbeResult":
        ...

    async def summarize_inline_pdf(self, *, data: bytes) -> "MethodProbeResult":
        ...

    async def summarize_with_presigned_url(
        self,
        *,
        url: str,
        document_format: str,
    ) -> "MethodProbeResult":
        ...


@dataclass(frozen=True, slots=True)
class MethodProbeResult:
    method: str
    success: bool
    reason: str
    elapsed_ms: int = 0
    cleanup_success: bool | None = None


@dataclass(frozen=True, slots=True)
class DocumentProbeResult:
    document_format: str
    file_size_bytes: int
    minio_download: str
    files_api: MethodProbeResult
    inline_pdf: MethodProbeResult
    presigned_url: MethodProbeResult
    recommendation: str

    def to_safe_lines(self) -> list[str]:
        return [
            f"format={self.document_format}",
            f"file_size_bytes={self.file_size_bytes}",
            f"minio_download={self.minio_download}",
            f"files_api={_status(self.files_api)}",
            f"inline_pdf={_status(self.inline_pdf)}",
            f"presigned_url={_status(self.presigned_url)}",
            f"recommendation={self.recommendation}",
        ]


class MinioObjectStorageClient:
    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool,
    ) -> None:
        normalized_endpoint = self._normalize_endpoint(endpoint)
        self._client = Minio(
            normalized_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    async def download_object(self, *, bucket: str, object_key: str) -> bytes:
        return await asyncio.to_thread(
            self._download_object_sync,
            bucket,
            object_key,
        )

    async def presigned_get_url(
        self,
        *,
        bucket: str,
        object_key: str,
        expires_seconds: int,
    ) -> str:
        return await asyncio.to_thread(
            self._client.presigned_get_object,
            bucket,
            object_key,
            timedelta(seconds=expires_seconds),
        )

    def _download_object_sync(self, bucket: str, object_key: str) -> bytes:
        response = self._client.get_object(bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        parsed = urlparse(endpoint.strip())
        if parsed.scheme and parsed.netloc:
            return parsed.netloc
        return endpoint.strip().rstrip("/")


class GeminiDocumentProbe:
    def __init__(self, *, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def summarize_with_files_api(
        self,
        *,
        data: bytes,
        document_format: str,
    ) -> MethodProbeResult:
        started_at = time.monotonic()
        uploaded_file: object | None = None
        cleanup_success: bool | None = None
        success = False
        reason = "success"
        suffix = f".{document_format}"
        mime_type = _mime_type(document_format)

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary_file:
                temporary_file.write(data)
                temporary_file.flush()
                uploaded_file = await self._client.aio.files.upload(
                    file=temporary_file.name,
                    config=types.UploadFileConfig(mime_type=mime_type),
                )

            uploaded_file = await self._wait_until_file_ready(uploaded_file)
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[SUMMARY_PROMPT, uploaded_file],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=800),
            )
            if _extract_text(response) is None:
                reason = "empty_gemini_response"
            else:
                success = True
        except Exception as exc:
            logger.warning(
                "Document probe method failed: operation=%s format=%s error_type=%s",
                "files_api",
                document_format,
                type(exc).__name__,
            )
            reason = type(exc).__name__

        if uploaded_file is not None:
            cleanup_success = await self._delete_uploaded_file(uploaded_file)

        if success:
            return _succeeded("files_api", started_at, cleanup_success)
        return _failed("files_api", reason, started_at, cleanup_success)
    async def summarize_inline_pdf(self, *, data: bytes) -> MethodProbeResult:
        started_at = time.monotonic()
        if len(data) > INLINE_PDF_MAX_BYTES:
            return _failed("inline_pdf", "file_too_large_for_inline_pdf", started_at)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[
                    SUMMARY_PROMPT,
                    types.Part.from_bytes(data=data, mime_type="application/pdf"),
                ],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=800),
            )
            if _extract_text(response) is None:
                return _failed("inline_pdf", "empty_gemini_response", started_at)
            return _succeeded("inline_pdf", started_at)
        except Exception as exc:
            logger.warning(
                "Document probe method failed: operation=%s format=%s error_type=%s",
                "inline_pdf",
                "pdf",
                type(exc).__name__,
            )
            return _failed("inline_pdf", type(exc).__name__, started_at)

    async def summarize_with_presigned_url(
        self,
        *,
        url: str,
        document_format: str,
    ) -> MethodProbeResult:
        started_at = time.monotonic()
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=[
                    SUMMARY_PROMPT,
                    types.Part.from_uri(
                        file_uri=url,
                        mime_type=_mime_type(document_format),
                    ),
                ],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=800),
            )
            if _extract_text(response) is None:
                return _failed("presigned_url", "empty_gemini_response", started_at)
            return _succeeded("presigned_url", started_at)
        except Exception as exc:
            logger.warning(
                "Document probe method failed: operation=%s format=%s error_type=%s",
                "presigned_url",
                document_format,
                type(exc).__name__,
            )
            return _failed("presigned_url", type(exc).__name__, started_at)

    async def _wait_until_file_ready(self, uploaded_file: object) -> object:
        name = getattr(uploaded_file, "name", None)
        if not isinstance(name, str) or name == "":
            return uploaded_file

        deadline = time.monotonic() + GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS
        current_file = uploaded_file
        while time.monotonic() < deadline:
            state = _file_state(current_file)
            if state is None or state in GEMINI_FILE_READY_STATES:
                return current_file
            if state in GEMINI_FILE_FAILED_STATES:
                raise RuntimeError("Gemini file processing failed.")

            await asyncio.sleep(GEMINI_FILE_PROCESSING_POLL_SECONDS)
            current_file = await self._client.aio.files.get(name=name)

        raise TimeoutError("Gemini file processing timed out.")

    async def _delete_uploaded_file(self, uploaded_file: object) -> bool:
        name = getattr(uploaded_file, "name", None)
        if not isinstance(name, str) or name == "":
            return False

        try:
            await self._client.aio.files.delete(name=name)
            return True
        except Exception as exc:
            logger.warning(
                "Document probe cleanup failed: operation=%s error_type=%s",
                "files_api_delete",
                type(exc).__name__,
            )
            return False


async def run_document_probe(
    *,
    document_uid: str,
    object_key: str,
    storage_client: ObjectStorageClient,
    gemini_probe: DocumentGeminiProbe,
    bucket: str,
    presigned_url_enabled: bool,
) -> DocumentProbeResult:
    document_format = _detect_format(object_key)
    if document_format not in SUPPORTED_FORMATS:
        skipped = MethodProbeResult("skipped", False, "unsupported_file_extension")
        return DocumentProbeResult(
            document_format=document_format or "unknown",
            file_size_bytes=0,
            minio_download="skipped",
            files_api=skipped,
            inline_pdf=skipped,
            presigned_url=skipped,
            recommendation="unsupported_extension",
        )

    try:
        data = await storage_client.download_object(bucket=bucket, object_key=object_key)
    except Exception as exc:
        logger.warning(
            "Document probe MinIO operation failed: operation=%s format=%s error_type=%s",
            "download",
            document_format,
            type(exc).__name__,
        )
        failed = MethodProbeResult("skipped", False, "minio_download_failed")
        return DocumentProbeResult(
            document_format=document_format,
            file_size_bytes=0,
            minio_download="failed",
            files_api=failed,
            inline_pdf=failed,
            presigned_url=failed,
            recommendation="manual_probe_failed",
        )

    if data == b"":
        failed = MethodProbeResult("skipped", False, "empty_object")
        return DocumentProbeResult(
            document_format=document_format,
            file_size_bytes=0,
            minio_download="success",
            files_api=failed,
            inline_pdf=failed,
            presigned_url=failed,
            recommendation="empty_object",
        )

    files_api = await gemini_probe.summarize_with_files_api(
        data=data,
        document_format=document_format,
    )
    inline_pdf = (
        await gemini_probe.summarize_inline_pdf(data=data)
        if document_format == "pdf"
        else MethodProbeResult("inline_pdf", False, "not_pdf")
    )
    presigned_url = await _run_presigned_url_probe(
        storage_client=storage_client,
        gemini_probe=gemini_probe,
        bucket=bucket,
        object_key=object_key,
        document_format=document_format,
        enabled=presigned_url_enabled,
    )

    return DocumentProbeResult(
        document_format=document_format,
        file_size_bytes=len(data),
        minio_download="success",
        files_api=files_api,
        inline_pdf=inline_pdf,
        presigned_url=presigned_url,
        recommendation=_recommend(document_format, files_api, inline_pdf),
    )


async def run_from_environment(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, list[str]]:
    env = environ or os.environ
    if env.get("MINIO_SPIKE_ENABLED", "").strip().lower() != "true":
        return 2, ["reason=spike_disabled"]

    document_uid = env.get("SPIKE_DOCUMENT_UID", "").strip()
    if document_uid == "":
        return 2, ["reason=missing_document_uid"]

    object_key = env.get("SPIKE_OBJECT_KEY", "").strip()
    if object_key == "":
        return 2, ["reason=object_key_convention_missing"]

    settings = Settings()
    missing_settings = _missing_minio_settings(settings)
    if missing_settings:
        return 2, [f"reason=missing_{missing_settings[0]}"]
    if settings.gemini_api_key is None:
        return 2, ["reason=missing_gemini_api_key"]

    storage_client = MinioObjectStorageClient(
        endpoint=settings.minio_endpoint or "",
        access_key=settings.minio_access_key or "",
        secret_key=settings.minio_secret_key or "",
        secure=settings.minio_secure,
    )
    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=settings.gemini_timeout_seconds * 1000),
    )
    try:
        result = await run_document_probe(
            document_uid=document_uid,
            object_key=object_key,
            storage_client=storage_client,
            gemini_probe=GeminiDocumentProbe(client=client, model=settings.gemini_model),
            bucket=settings.minio_bucket or "",
            presigned_url_enabled=env.get("SPIKE_PRESIGNED_URL_ENABLED", "").strip().lower()
            == "true",
        )
        return 0 if result.files_api.success or result.inline_pdf.success else 1, result.to_safe_lines()
    finally:
        await client.aio.aclose()


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="document-probe",
        description="Run an isolated MinIO/Gemini document summary probe.",
    )


def main() -> int:
    build_parser().parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    exit_code, lines = asyncio.run(run_from_environment())
    for line in lines:
        print(line)
    return exit_code


async def _run_presigned_url_probe(
    *,
    storage_client: ObjectStorageClient,
    gemini_probe: DocumentGeminiProbe,
    bucket: str,
    object_key: str,
    document_format: str,
    enabled: bool,
) -> MethodProbeResult:
    if not enabled:
        return MethodProbeResult("presigned_url", False, "not_enabled")

    try:
        url = await storage_client.presigned_get_url(
            bucket=bucket,
            object_key=object_key,
            expires_seconds=300,
        )
    except Exception as exc:
        logger.warning(
            "Document probe MinIO operation failed: operation=%s format=%s error_type=%s",
            "presigned_url",
            document_format,
            type(exc).__name__,
        )
        return MethodProbeResult("presigned_url", False, type(exc).__name__)

    if "minio:9000" in url or "localhost" in url:
        return MethodProbeResult("presigned_url", False, "not_externally_reachable")

    return await gemini_probe.summarize_with_presigned_url(
        url=url,
        document_format=document_format,
    )


def _missing_minio_settings(settings: Settings) -> list[str]:
    missing_settings: list[str] = []
    for name in (
        "minio_endpoint",
        "minio_bucket",
        "minio_access_key",
        "minio_secret_key",
    ):
        if getattr(settings, name) is None:
            missing_settings.append(name)
    return missing_settings


def _detect_format(object_key: str) -> str:
    suffix = Path(object_key).suffix.lower().removeprefix(".")
    return suffix


def _mime_type(document_format: str) -> str:
    if document_format == "pdf":
        return "application/pdf"
    if document_format == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    raise ValueError(f"Unsupported document format: {document_format}")


def _file_state(file_object: object) -> str | None:
    state = getattr(file_object, "state", None)
    if state is None:
        return None
    value = getattr(state, "value", state)
    return str(value).upper()


def _extract_text(response: object) -> str | None:
    text = getattr(response, "text", None)
    if not isinstance(text, str):
        return None
    stripped_text = text.strip()
    return stripped_text or None


def _succeeded(
    method: str,
    started_at: float,
    cleanup_success: bool | None = None,
) -> MethodProbeResult:
    return MethodProbeResult(
        method=method,
        success=True,
        reason="success",
        elapsed_ms=_elapsed_ms(started_at),
        cleanup_success=cleanup_success,
    )


def _failed(
    method: str,
    reason: str,
    started_at: float,
    cleanup_success: bool | None = None,
) -> MethodProbeResult:
    return MethodProbeResult(
        method=method,
        success=False,
        reason=reason,
        elapsed_ms=_elapsed_ms(started_at),
        cleanup_success=cleanup_success,
    )


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def _status(result: MethodProbeResult) -> str:
    if result.success:
        return "success"
    return result.reason


def _recommend(
    document_format: str,
    files_api: MethodProbeResult,
    inline_pdf: MethodProbeResult,
) -> str:
    if document_format == "pdf" and inline_pdf.success and not files_api.success:
        return "inline_pdf_for_small_pdf_only"
    if files_api.success:
        return f"files_api_for_{document_format}"
    return "do_not_enable_full_document_summary"


if __name__ == "__main__":
    raise SystemExit(main())
