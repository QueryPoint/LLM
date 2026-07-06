# 06. Full-document PDF summary

## 1. Назначение

Полное краткое изложение предназначено для выбранного PDF-документа. Этот путь отличается от обычного chunk-based question answering: вместо ограниченного набора Elasticsearch chunks сервис скачивает исходный PDF из MinIO и передаёт файл Gemini Files API.

## 2. Preconditions

Для summary должны быть доступны:

- valid `prompt` с `doc`;
- metadata в Elasticsearch, найденная по `user_id + doc_id`;
- `file_name` с расширением `.pdf` без учёта регистра;
- MinIO endpoint/bucket/credentials;
- объект по соглашению `<user_id>/<document_id>.pdf`;
- PDF размером не больше `DOCUMENT_SUMMARY_MAX_FILE_BYTES`;
- Gemini API key и доступ к Files API.

## 3. Алгоритм

```text
1. IntentAgent выбирает summarize_document.
2. Оркестратор запрашивает metadata выбранного документа.
3. Если metadata отсутствует -> neutral summary unavailable fallback.
4. Если расширение не PDF -> PDF-only fallback.
5. MinioDocumentStorageClient читает PDF с лимитом max_file_bytes + 1.
6. Пустой или слишком большой объект отклоняется.
7. GeminiClient создаёт временный .pdf файл.
8. Файл upload в Gemini Files API.
9. Клиент ждёт состояния ACTIVE/SUCCEEDED, опрашивая Files API.
10. Gemini получает summary prompt + uploaded file.
11. В finally выполняется best-effort delete временного Gemini file.
12. Пользователь получает один response.
```

## 4. MinIO storage convention

Текущая реализация строит object key так:

```text
<user_id>/<document_id>.pdf
```

Это временное соглашение LLM-service. Оно должно совпадать с фактическим способом сохранения PDF на стороне системы. Не публикуйте object key в логах, UI и документации с реальными данными.

## 5. Защита MinIO download

`MinioDocumentStorageClient`:

- не создаёт клиента, если обязательные MinIO settings отсутствуют;
- нормализует endpoint, позволяя передавать `http://host:port` или `host:port`;
- закрывает и освобождает response connection;
- ограничивает read по `DOCUMENT_SUMMARY_MAX_FILE_BYTES + 1`;
- различает not found, unavailable и invalid object;
- не логирует bucket/object key/user/document identifiers.

## 6. Gemini Files API

`GeminiClient.summarize_pdf_document()`:

- пишет bytes во временный PDF;
- upload-ит файл с MIME `application/pdf`;
- ожидает готовность файла до `GEMINI_TIMEOUT_SECONDS`;
- обрабатывает terminal states `FAILED`/`ERROR`;
- удаляет uploaded file best effort даже при generation failure;
- классифицирует errors в rate limit, transient и permanent.

## 7. Пользовательские fallback-ответы

| Условие | Ответ |
|---|---|
| не выбран документ | `Для подготовки краткого изложения нужен полный текст выбранного документа.` |
| выбран не PDF | `Краткое изложение пока доступно только для PDF-документов.` |
| metadata/MinIO/Gemini summary недоступны | `Не удалось подготовить краткое изложение документа. Попробуйте позже.` |
| Gemini quota/rate limit | `Сервис временно перегружен. Подождите немного и повторите запрос.` |

## 8. Что пока не поддерживается

- DOCX full-document summary;
- summary из presigned URL в production flow;
- прямой inline PDF в production flow;
- summary по объединению всех Elasticsearch chunks;
- vector retrieval.

Manual `document_probe` предназначен для изолированной проверки методов интеграции, а не для RabbitMQ worker path. Не используйте реальные пользовательские документы для probe без разрешения и без безопасной очистки тестовых объектов.
