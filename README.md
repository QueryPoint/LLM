# assistant-service

LLM-service для проекта «Интеллектуальная база знаний университета».

Сервис получает запросы из RabbitMQ, определяет намерение пользователя, ищет
релевантные chunks в Elasticsearch и вызывает Gemini только для grounded ответа
по найденному контексту.

## RabbitMQ Contract

Входящие события:

```text
prompt
delete
```

Исходящие события:

```text
think
response
```

`think.data` всегда содержит человекочитаемую строку, например:

```json
{
  "type": "think",
  "user_id": "user-123",
  "data": "Ищу подходящие материалы..."
}
```

`response` является единственным финальным событием:

```json
{
  "type": "response",
  "user_id": "user-123",
  "data": "Ответ, подготовленный только по найденным материалам.",
  "warning": 34
}
```

Внешний contract не содержит `sync`, `token`, `delta`, `done`, `error`,
`sources`, `request_id` или `session_id`.

## Prompt Payload

Backend отправляет минимальный payload:

```json
{
  "type": "prompt",
  "user_id": "user-123",
  "prompt": "Объясни нормальные формы баз данных",
  "uid": null
}
```

Для выбранного документа:

```json
{
  "type": "prompt",
  "user_id": "user-123",
  "prompt": "Какие нормальные формы описаны в документе?",
  "uid": "document-uid-123"
}
```

Backend не передаёт `mode` и `document_context`. Поле `uid` соответствует
Elasticsearch полю `documents.doc_id`.

## Intent

LLM-service сам классифицирует запрос во внутренний `IntentDecision`:

```json
{
  "task_type": "explain_topic",
  "requires_retrieval": true,
  "requires_full_document": false,
  "keywords": ["нормальные формы", "нормализация баз данных"]
}
```

Allowed internal task types:

```text
answer_question
explain_topic
document_search
summarize_document
unsupported
```

Intent result не является RabbitMQ event и не отправляется frontend.

## Elasticsearch Retrieval

Сервис ищет chunks в index:

```text
documents
```

Подтверждённые поля mapping:

```text
chunk_id
doc_id
file_name
page_number
text
user_id
```

`user_id` не используется как Elasticsearch filter в текущем sprint. Проверку
доступа пользователя выполняет backend до отправки RabbitMQ message.

Если `uid` отсутствует, выполняется global search по `text`:

```json
{
  "query": {
    "match": {
      "text": {
        "query": "нормализация базы данных"
      }
    }
  }
}
```

Если `uid` передан, поиск ограничивается документом через точный filter:

```json
{
  "query": {
    "bool": {
      "filter": [
        {
          "term": {
            "doc_id": "document-uid-123"
          }
        }
      ],
      "must": [
        {
          "match": {
            "text": {
              "query": "нормализация базы данных"
            }
          }
        }
      ]
    }
  }
}
```

Query text строится только из `IntentDecision.keywords`. Raw user prompt не
используется как Elasticsearch query.

Найденные `SearchResult` кэшируются в Redis с TTL 900 секунд. Пустые
результаты не кэшируются, а повреждённый cache entry удаляется best effort и
обрабатывается как cache miss.

## Grounded Answer Flow

Для `answer_question` и `explain_topic` Gemini вызывается только после того,
как Elasticsearch вернул валидные chunks и `ContextAgent` подготовил `FOUND`
context.

Для `document_search` Gemini не вызывается: response строится
детерминированно по найденным материалам.

Для `summarize_document` полный summary поддерживается только для выбранного
PDF-документа. Сервис получает `file_name` из Elasticsearch, скачивает PDF из
MinIO и передаёт файл в Gemini через Files API. DOCX и другие форматы пока не
поддерживаются.

Если `uid` не передан:

```text
Для подготовки краткого изложения нужен полный текст выбранного документа.
```

Если выбранный документ не PDF:

```text
Краткое изложение пока доступно только для PDF-документов.
```

Если документ не удалось скачать или обработать:

```text
Не удалось подготовить краткое изложение документа. Попробуйте позже.
```

Если материалы не найдены:

```text
Не удалось найти материалы для подготовки ответа. Попробуйте уточнить запрос или выбрать документ.
```

Если Elasticsearch недоступен:

```text
Не удалось выполнить поиск по базе знаний. Попробуйте позже.
```

## Warning

`response.warning` — целый процент использованного prompt/context budget:

```text
floor(used_prompt_chars / max_prompt_chars * 100)
```

Для grounded answer расчёт выполняется по фактическому generation prompt,
который передаётся Gemini: system instruction + user prompt markup +
подготовленный context. Если фактический размер достигает или превышает лимит,
Gemini не вызывается и возвращается context-too-large fallback.

## Configuration

Elasticsearch settings:

```env
ELASTICSEARCH_URL=http://elasticsearch:9200
ELASTICSEARCH_INDEX=documents
ELASTICSEARCH_TIMEOUT_SECONDS=5
ELASTICSEARCH_MAX_RESULTS=8
RETRIEVAL_CACHE_TTL_SECONDS=900
```

При локальном запуске с Mac можно переопределить:

```env
ELASTICSEARCH_URL=http://localhost:9200
```

MinIO/Gemini document probe settings:

```env
MINIO_ENDPOINT=
MINIO_BUCKET=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_SECURE=false
MINIO_SPIKE_ENABLED=false
DOCUMENT_SUMMARY_MAX_FILE_BYTES=20971520
```

`DOCUMENT_SUMMARY_MAX_FILE_BYTES` ограничивает размер PDF для production
full-document summary. Текущий MinIO storage convention является временным
техническим ограничением LLM-service: объект PDF собирается как
`<user_id>/<uid>.pdf` после проверки `file_name` из Elasticsearch.

The document probe is an isolated manual tool and is not part of the RabbitMQ
worker flow:

```bash
MINIO_SPIKE_ENABLED=true \
SPIKE_DOCUMENT_UID=<test-uid> \
SPIKE_OBJECT_KEY=<test-object-key> \
uv run python -m assistant_service.tools.document_probe
```

`SPIKE_OBJECT_KEY` is required until backend/Infra defines the mapping from
`uid` to MinIO object key. The command prints only aggregate status lines and
does not print secrets, URLs, object keys, document text or Gemini output.

## Manual Retrieval Check

Index `documents` может быть пустым. Для ручной E2E проверки backend должен:

1. Загрузить PDF/DOCX.
2. Извлечь текст.
3. Нарезать документ на chunks.
4. Записать chunks в index `documents`.
5. Сделать `docs.count > 0`.

После этого можно вручную проверить поиск:

```bash
curl -X POST "http://localhost:9200/documents/_search?pretty" \
  -H "Content-Type: application/json" \
  -d '{
    "size": 3,
    "query": {
      "match": {
        "text": {
          "query": "нормализация баз данных"
        }
      }
    }
  }'
```

Sprint-16 добавляет PDF full-document summary через MinIO и Gemini Files API.
Presigned URLs, DOCX summary, `chunk_order` support и vector search по-прежнему
не реализованы.
