# 05. Retrieval, кэш и tenant isolation

## 1. Ожидаемый Elasticsearch mapping

LLM-service читает index `documents`. Нужны поля:

```json
{
  "chunk_id": {"type": "keyword"},
  "doc_id": {"type": "keyword"},
  "file_name": {"type": "keyword"},
  "page_number": {"type": "integer"},
  "text": {"type": "text"},
  "user_id": {"type": "keyword"}
}
```

Сервис не использует `chunk_order`, vector fields или full-document reconstruction по набору chunks.

## 2. Tenant-safe query policy

Каждый retrieval ограничивается `user_id`. Это обязательный defence-in-depth слой, а не замена Backend access control.

### Generic retrieval (`doc = null`)

```json
{
  "bool": {
    "filter": [{"term": {"user_id": "<user>"}}],
    "must": [{"match": {"text": {"query": "<keywords>"}}}]
  }
}
```

### Retrieval по выбранному документу (`doc != null`)

```json
{
  "bool": {
    "filter": [
      {"term": {"user_id": "<user>"}},
      {"term": {"doc_id": "<document>"}}
    ],
    "must": [{"match": {"text": {"query": "<keywords>"}}}]
  }
}
```

### Metadata lookup для PDF summary

Metadata lookup тоже использует `user_id + doc_id`. Чужой документ должен быть неотличим от несуществующего: не возвращайте «access denied» и не подтверждайте факт его существования.

## 3. Обработка пустых значений

- Пустой `user_id` не запускает Elasticsearch запрос и возвращает пустой retrieval result / `None` metadata.
- Пустой `document_id` для metadata не запускает Elasticsearch запрос.
- Запрещён fallback, который убирает user filter и превращает запрос в global search.

## 4. Query text

Elasticsearch получает не raw prompt, а нормализованные `IntentDecision.keywords`:

- trim;
- lower-case;
- collapse whitespace;
- dedupe;
- пустые keywords исключаются.

Это снижает риск случайной передачи большого пользовательского текста в search body.

## 5. Context preparation

`ContextAgent`:

1. пропускает пустые chunks;
2. удаляет дубликаты по `chunk_id` и нормализованному text;
3. выбирает chunks в исходном порядке результатов;
4. ограничивает число chunks и суммарные символы;
5. строит `ContextDecision` со статусом `FOUND` или `NOT_FOUND`.

При отсутствии достаточного валидного context Gemini не вызывается для grounded answer.

## 6. Retrieval cache в Redis

Кэш работает по модели cache-aside:

```text
Request -> build retrieval key -> Redis GET
  hit -> deserialize SearchResult[] -> return
  miss -> Elasticsearch search -> Redis SET only if non-empty
```

Ключ имеет префикс `retrieval:v1:` и SHA-256 digest. В fingerprint входят:

```text
index name
normalized keywords
hash(normalized user_id)
document_id или marker global
```

Сырой `user_id` не попадает в key. Один и тот же query разных пользователей создаёт разные cache keys.

Кэш payload содержит schema version. Повреждённый/устаревший payload рассматривается как cache miss; удаление выполняется best effort. Пустые search results не кэшируются.

## 7. Ошибки Elasticsearch

| Ошибка | Внутреннее исключение | Пользовательское поведение |
|---|---|---|
| index отсутствует | `ElasticsearchIndexNotFoundError` | нейтральный retrieval unavailable fallback |
| connection/timeout | `ElasticsearchUnavailableError` | нейтральный retrieval unavailable fallback |
| invalid API/serialization/transport response | `ElasticsearchResponseError` | нейтральный retrieval unavailable fallback |

Логи должны содержать только operation/error type, а не query body, prompt, `user_id`, document ID или найденный текст.
