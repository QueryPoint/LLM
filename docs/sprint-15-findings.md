# Sprint-15: MinIO/Gemini Full-Document Summary Spike

## Scope

Sprint-15 adds an isolated manual probe for checking whether a test document can
be downloaded from MinIO and passed to Gemini for a full-document summary. The
probe is not wired into `TaskOrchestrator`, RabbitMQ, Redis retrieval cache or
the production `summarize_document` flow.

Production summary behavior remains unchanged:

```text
Для подготовки краткого изложения нужен полный текст выбранного документа.
```

## Current Infrastructure Findings

The LLM-service repository does not currently define:

```text
MINIO endpoint
bucket name
authentication values
uid -> MinIO object key convention
document extension source outside Elasticsearch file_name
```

Because the object key convention is not defined, the manual probe requires an
explicit `SPIKE_OBJECT_KEY` for a known test object. The tool does not guess an
object path from `uid`.

No access key, secret key, presigned URL, object key, UID, document text or
Gemini output should be written to logs or reports.

## Probe Methods

The implemented manual probe can check:

| Method | Status | Notes |
| --- | --- | --- |
| Gemini Files API | implemented, unverified with real object | Downloads MinIO bytes to a temporary local file, uploads it to Gemini Files API, waits for readiness and deletes the remote Gemini file best effort. |
| Inline PDF bytes | implemented, unverified with real object | Limited to small PDFs. It sends `application/pdf` bytes directly through `Part.from_bytes`. |
| Presigned URL | implemented as controlled experiment | Disabled unless explicitly enabled. Internal `minio:9000` or localhost URLs are treated as not externally reachable. |

## Manual Command

```bash
MINIO_SPIKE_ENABLED=true \
SPIKE_DOCUMENT_UID=<test-uid> \
SPIKE_OBJECT_KEY=<test-object-key> \
uv run python -m assistant_service.tools.document_probe
```

The command refuses to run when:

```text
MINIO_SPIKE_ENABLED != true
SPIKE_DOCUMENT_UID is missing
SPIKE_OBJECT_KEY is missing
MinIO settings are incomplete
GEMINI_API_KEY is missing
```

## PDF Result

| format | file size | MinIO download success | Gemini input method | Gemini request success | summary quality | cleanup success | recommended for production | failure reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pdf | not tested | not tested | Files API / inline bytes / presigned URL | not tested | not evaluated | not tested | no | no test object and no uid-to-object-key convention |

## DOCX Result

| format | file size | MinIO download success | Gemini input method | Gemini request success | summary quality | cleanup success | recommended for production | failure reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docx | not tested | not tested | Files API / presigned URL | not tested | not evaluated | not tested | no | no test object and no uid-to-object-key convention |

## Recommendation

Chosen option for now:

```text
D. Не внедрять Gemini full-file summary пока нет надёжного способа
```

This is a conservative decision based on the actual Sprint-15 state: the probe
code exists, but real PDF/DOCX checks were not executed because the repository
does not define MinIO connection values or the `uid` to object key rule.

Before Sprint-16, backend/Infra must provide:

```text
1. MinIO endpoint reachable from LLM-service in Docker.
2. Local MinIO endpoint for manual runs.
3. Bucket name.
4. Safe credentials delivery through environment variables.
5. Exact uid -> object key rule or metadata lookup contract.
6. One explicit test PDF object.
7. One explicit test DOCX object, if DOCX summary remains in scope.
```
