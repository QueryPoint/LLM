import io
import os
from uuid import uuid4

import pytest
from minio import Minio

from assistant_service.core.config import Settings
from assistant_service.services.minio_storage import MinioDocumentStorageClient


@pytest.mark.integration
@pytest.mark.requires_docker
def test_minio_roundtrip_uses_temporary_test_object_only() -> None:
    if os.getenv("RUN_INTEGRATION_TESTS", "").strip().lower() != "true":
        pytest.skip("integration tests are disabled")
    if os.getenv("RUN_MINIO_INTEGRATION_TESTS", "").strip().lower() != "true":
        pytest.skip("RUN_MINIO_INTEGRATION_TESTS is not set")

    settings = Settings()
    if (
        settings.minio_endpoint is None
        or settings.minio_bucket is None
        or settings.minio_access_key is None
        or settings.minio_secret_key is None
    ):
        pytest.skip("MinIO settings are not configured")

    endpoint = MinioDocumentStorageClient.normalize_endpoint(settings.minio_endpoint)
    client = Minio(
        endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    bucket = settings.minio_bucket
    object_key = f"integration/{uuid4().hex}.pdf"
    data = b"%PDF-1.4 integration test"
    downloaded = b""

    try:
        if not client.bucket_exists(bucket):
            pytest.skip("MinIO bucket is not available")

        client.put_object(
            bucket,
            object_key,
            io.BytesIO(data),
            len(data),
            content_type="application/pdf",
        )
        response = client.get_object(bucket, object_key)
        try:
            downloaded = response.read()
        finally:
            response.close()
            response.release_conn()
    except Exception as exc:
        pytest.skip(f"MinIO integration unavailable: {type(exc).__name__}")
    finally:
        try:
            client.remove_object(bucket, object_key)
        except Exception:
            pass

    assert downloaded == data
