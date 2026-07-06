import asyncio

import pytest

from assistant_service.services.minio_storage import (
    DocumentStorageInvalidObjectError,
    MinioDocumentStorageClient,
)


class FakeObject:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.close_calls = 0
        self.release_calls = 0

    def read(self, size: int | None = None) -> bytes:
        if size is None:
            return self._data
        return self._data[:size]

    def close(self) -> None:
        self.close_calls += 1

    def release_conn(self) -> None:
        self.release_calls += 1


class FakeMinioClient:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls: list[tuple[str, str]] = []

    def get_object(self, bucket: str, object_key: str) -> FakeObject:
        self.calls.append((bucket, object_key))
        return FakeObject(self.data)


def test_endpoint_normalization_and_download_size_limits() -> None:
    assert MinioDocumentStorageClient.normalize_endpoint(
        "http://localhost:9000"
    ) == "localhost:9000"
    assert MinioDocumentStorageClient.normalize_endpoint("localhost:9000") == "localhost:9000"
    assert MinioDocumentStorageClient.normalize_endpoint("http://minio:9000") == "minio:9000"

    empty_client = MinioDocumentStorageClient(
        client=FakeMinioClient(b""),
        bucket="bucket",
        max_file_bytes=5,
    )
    with pytest.raises(DocumentStorageInvalidObjectError):
        asyncio.run(
            empty_client.download_pdf_for_summary(
                user_id="user-123",
                document_id="document-123",
            )
        )

    oversized_client = MinioDocumentStorageClient(
        client=FakeMinioClient(b"abcdef"),
        bucket="bucket",
        max_file_bytes=5,
    )
    with pytest.raises(DocumentStorageInvalidObjectError):
        asyncio.run(
            oversized_client.download_pdf_for_summary(
                user_id="user-123",
                document_id="document-123",
            )
        )
