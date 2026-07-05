import asyncio

from assistant_service.tools.document_probe import (
    GeminiDocumentProbe,
    MethodProbeResult,
    MinioObjectStorageClient,
    run_document_probe,
    run_from_environment,
)


class FakeStorageClient:
    def __init__(self, data: bytes = b"fake pdf") -> None:
        self.download_calls = 0
        self.presigned_calls = 0
        self._data = data

    async def download_object(self, *, bucket: str, object_key: str) -> bytes:
        self.download_calls += 1
        return self._data

    async def presigned_get_url(
        self,
        *,
        bucket: str,
        object_key: str,
        expires_seconds: int,
    ) -> str:
        self.presigned_calls += 1
        return "https://example.invalid/document"


class FakeGeminiProbe:
    def __init__(self) -> None:
        self.files_api_calls = 0
        self.inline_pdf_calls = 0
        self.presigned_url_calls = 0

    async def summarize_with_files_api(
        self,
        *,
        data: bytes,
        document_format: str,
    ) -> MethodProbeResult:
        self.files_api_calls += 1
        return MethodProbeResult("files_api", True, "success")

    async def summarize_inline_pdf(self, *, data: bytes) -> MethodProbeResult:
        self.inline_pdf_calls += 1
        return MethodProbeResult("inline_pdf", True, "success")

    async def summarize_with_presigned_url(
        self,
        *,
        url: str,
        document_format: str,
    ) -> MethodProbeResult:
        self.presigned_url_calls += 1
        return MethodProbeResult("presigned_url", True, "success")


class FakeUploadedFile:
    name = "files/test"
    state = "ACTIVE"


class FakeFiles:
    def __init__(self) -> None:
        self.delete_calls: list[str] = []

    async def upload(self, *, file: str, config: object = None) -> FakeUploadedFile:
        return FakeUploadedFile()

    async def get(self, *, name: str) -> FakeUploadedFile:
        return FakeUploadedFile()

    async def delete(self, *, name: str) -> None:
        self.delete_calls.append(name)


class FakeModels:
    async def generate_content(self, **kwargs: object) -> object:
        raise RuntimeError("generation failed")


class FakeAioClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.models = FakeModels()


class FakeGeminiClient:
    def __init__(self) -> None:
        self.aio = FakeAioClient()


def test_document_probe_disabled_refuses_before_network_calls() -> None:
    exit_code, lines = asyncio.run(run_from_environment({"MINIO_SPIKE_ENABLED": "false"}))

    assert exit_code == 2
    assert lines == ["reason=spike_disabled"]


def test_minio_endpoint_accepts_scheme_and_host_form() -> None:
    assert MinioObjectStorageClient._normalize_endpoint("http://localhost:9000") == "localhost:9000"
    assert MinioObjectStorageClient._normalize_endpoint("localhost:9000") == "localhost:9000"


def test_document_probe_unsupported_extension_skips_minio_and_gemini() -> None:
    storage = FakeStorageClient()
    gemini = FakeGeminiProbe()

    result = asyncio.run(
        run_document_probe(
            document_uid="test-uid",
            object_key="documents/test.txt",
            storage_client=storage,
            gemini_probe=gemini,
            bucket="bucket",
            presigned_url_enabled=True,
        )
    )

    assert result.recommendation == "unsupported_extension"
    assert storage.download_calls == 0
    assert gemini.files_api_calls == 0
    assert gemini.inline_pdf_calls == 0
    assert gemini.presigned_url_calls == 0


def test_document_probe_files_api_failure_deletes_uploaded_file_best_effort() -> None:
    fake_client = FakeGeminiClient()
    probe = GeminiDocumentProbe(client=fake_client, model="gemini-test")  # type: ignore[arg-type]

    result = asyncio.run(
        probe.summarize_with_files_api(data=b"%PDF-1.4", document_format="pdf")
    )

    assert result.success is False
    assert result.reason == "RuntimeError"
    assert fake_client.aio.files.delete_calls == ["files/test"]
