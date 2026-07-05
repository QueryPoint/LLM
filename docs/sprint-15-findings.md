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

The implemented manual probe checks:

| Method | Status | Notes |
| --- | --- | --- |
| Gemini Files API | validated for PDF, failed for DOCX | Downloads MinIO bytes to a temporary local file, uploads it to Gemini Files API, waits for readiness and deletes the remote Gemini file best effort. |
| Inline PDF bytes | validated for PDF, not applicable for DOCX | Limited to small PDFs. It sends `application/pdf` bytes directly through `Part.from_bytes`. |
| Presigned URL | not enabled | Kept disabled for this spike and not used for the recommendation. |

## Manual Command

The manual probe was run against a test PDF and a test DOCX object selected from the local MinIO-backed backend flow. The command refused to print secrets, object keys or document text.

## PDF Result

| format | file size | MinIO download success | Gemini input method | Gemini request success | summary quality | cleanup success | recommended for production | failure reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pdf | < 1 MB | success | Files API | success | acceptable for this spike | best effort | yes | none |

## DOCX Result

| format | file size | MinIO download success | Gemini input method | Gemini request success | summary quality | cleanup success | recommended for production | failure reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docx | < 1 MB | success | Files API | failed with `ClientError` | not evaluated | best effort | no | Gemini Files API was not stable for DOCX in this validation |

## Recommendation

Chosen option:

```text
B. Files API only for PDF; DOCX through backend text extraction / conversion to PDF.
```

Sprint-16 can use the confirmed temporary backend storage convention for PDF:

```text
<user_id>/<doc_id>.pdf
```

This remains a technical limitation of the current LLM-service implementation.
A future production contract should replace derived object keys with one stable
internal file-access contract for an authorized `uid`:

```text
1. Preferred contract: internal backend endpoint returning file bytes/stream.
2. Acceptable contract: metadata lookup returning storage_key and mime_type.
```
