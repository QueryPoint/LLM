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

## Grounded Answer Flow

Для `answer_question` и `explain_topic` Gemini вызывается только после того,
как Elasticsearch вернул валидные chunks и `ContextAgent` подготовил `FOUND`
context.

Для `document_search` Gemini не вызывается: response строится
детерминированно по найденным материалам.

Для `summarize_document` полный summary временно не выполняется, потому что в
index нет надёжного `chunk_order`. Сервис возвращает:

```text
Для подготовки краткого изложения нужен полный текст выбранного документа.
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
```

При локальном запуске с Mac можно переопределить:

```env
ELASTICSEARCH_URL=http://localhost:9200
```

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

Sprint-13 намеренно не реализует Redis retrieval cache, MinIO/S3, presigned
URLs, full-document summary, `chunk_order` support и vector search.
