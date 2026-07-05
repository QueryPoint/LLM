import asyncio
import logging
from urllib.parse import urlparse

from minio import Minio
from minio.error import S3Error

from assistant_service.core.config import Settings

logger = logging.getLogger(__name__)


class DocumentStorageError(RuntimeError):
    """Raised when document bytes cannot be loaded safely."""


class DocumentStorageUnavailableError(DocumentStorageError):
    """Object storage is unavailable or not configured."""


class DocumentStorageObjectNotFoundError(DocumentStorageError):
    """Requested document object was not found."""


class DocumentStorageInvalidObjectError(DocumentStorageError):
    """Requested document object is empty or too large."""


class MinioDocumentStorageClient:
    def __init__(
        self,
        *,
        client: Minio | None,
        bucket: str | None,
        max_file_bytes: int,
    ) -> None:
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")

        self._client = client
        self._bucket = bucket.strip() if isinstance(bucket, str) else None
        self._max_file_bytes = max_file_bytes

    async def download_pdf_for_summary(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> bytes:
        if self._client is None or not self._bucket:
            raise DocumentStorageUnavailableError("Document storage is not configured.")

        object_key = self._build_pdf_object_key(
            user_id=user_id,
            document_id=document_id,
        )
        return await asyncio.to_thread(self._download_object_sync, object_key)

    def _download_object_sync(self, object_key: str) -> bytes:
        assert self._client is not None
        assert self._bucket is not None

        try:
            response = self._client.get_object(self._bucket, object_key)
        except S3Error as exc:
            self._log_storage_error("download", exc)
            if exc.code in {"NoSuchKey", "NoSuchBucket", "NoSuchObject"}:
                raise DocumentStorageObjectNotFoundError(
                    "Document object was not found."
                ) from exc
            raise DocumentStorageUnavailableError(
                "Document storage operation failed."
            ) from exc
        except Exception as exc:
            self._log_storage_error("download", exc)
            raise DocumentStorageUnavailableError(
                "Document storage operation failed."
            ) from exc

        try:
            data = response.read(self._max_file_bytes + 1)
        finally:
            response.close()
            response.release_conn()

        if data == b"":
            raise DocumentStorageInvalidObjectError("Document object is empty.")
        if len(data) > self._max_file_bytes:
            raise DocumentStorageInvalidObjectError("Document object is too large.")

        return data

    @staticmethod
    def _build_pdf_object_key(*, user_id: str, document_id: str) -> str:
        normalized_user_id = user_id.strip()
        normalized_document_id = document_id.strip()
        if normalized_user_id == "" or normalized_document_id == "":
            raise DocumentStorageObjectNotFoundError("Document key cannot be built.")
        return f"{normalized_user_id}/{normalized_document_id}.pdf"

    @staticmethod
    def normalize_endpoint(endpoint: str) -> str:
        parsed = urlparse(endpoint.strip())
        if parsed.scheme and parsed.netloc:
            return parsed.netloc
        return endpoint.strip().rstrip("/")

    @staticmethod
    def _log_storage_error(operation: str, exc: Exception) -> None:
        logger.warning(
            "Document storage operation failed: operation=%s error_type=%s",
            operation,
            type(exc).__name__,
        )


def create_minio_document_storage_from_settings(
    settings: Settings,
) -> MinioDocumentStorageClient:
    if (
        settings.minio_endpoint is None
        or settings.minio_bucket is None
        or settings.minio_access_key is None
        or settings.minio_secret_key is None
    ):
        return MinioDocumentStorageClient(
            client=None,
            bucket=settings.minio_bucket,
            max_file_bytes=settings.document_summary_max_file_bytes,
        )

    client = Minio(
        MinioDocumentStorageClient.normalize_endpoint(settings.minio_endpoint),
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    return MinioDocumentStorageClient(
        client=client,
        bucket=settings.minio_bucket,
        max_file_bytes=settings.document_summary_max_file_bytes,
    )
