# 01. Архитектура и потоки обработки

## 1. Компонентная схема

```text
                    +---------------------------+
                    |        RabbitMQ            |
                    | back_to_llm / llm_to_back  |
                    +-------------+-------------+
                                  |
                                  v
+------------------------------------------------------------------+
|                        assistant-service                         |
|                                                                  |
| RabbitMQWorker -> TaskOrchestrator                               |
|                    |                                             |
|                    +-> IntentAgent                               |
|                    +-> RetrievalService -> RedisStateStore       |
|                    |                     -> ElasticsearchClient  |
|                    +-> ContextAgent                              |
|                    +-> AnswerAgent -> GeminiClient               |
|                    +-> PDF summary -> MinIO -> GeminiClient      |
+------------------------------------------------------------------+
```

## 2. Runtime composition

Точка входа: `src/assistant_service/main.py`.

`run_worker()` создаёт и связывает:

- `RabbitMQWorker`;
- `GeminiClient`;
- `RedisStateStore`;
- `ElasticsearchClient`;
- `MinioDocumentStorageClient`;
- `IntentAgent`;
- `ContextAgent`;
- `AnswerAgent`;
- `DocumentSummaryAgent`;
- `PdfDocumentSummaryService`;
- `TaskOrchestrator`.

При завершении процесса worker останавливается, а клиенты Gemini, Redis и Elasticsearch закрываются best effort.

## 3. Стандартный путь prompt

```text
1. RabbitMQWorker получает persistent AMQP message.
2. JSON декодируется и валидируется Pydantic-моделью.
3. TaskOrchestrator проверяет размер prompt и prompt budget.
4. Публикуется think: «Определяю тип запроса...»
5. IntentAgent возвращает IntentDecision.
6. Для retrieval-задач публикуется think: «Ищу подходящие материалы...»
7. RetrievalService читает Redis cache или обращается к Elasticsearch.
8. ContextAgent удаляет дубли, ограничивает размер context и собирает sources.
9. Для answer/explain публикуется think: «Формирую ответ...»
10. AnswerAgent вызывает Gemini stream и собирает fragments в один текст.
11. Публикуется единственный response.
12. В RabbitMQ message выполняется ack.
```

## 4. Путь `document_search`

`document_search` использует тот же retrieval и context preparation, но Gemini не вызывается. `response_builder` формирует текст вида:

```text
Нашёл подходящие фрагменты:

1. <file_name>, стр. <page>
<chunk text>
```

Это уменьшает стоимость и обеспечивает предсказуемый ответ для поиска материалов.

## 5. Путь PDF summary

```text
prompt(doc != null)
  -> IntentAgent: summarize_document
  -> think: «Определяю выбранный документ.»
  -> Elasticsearch metadata lookup: user_id + doc_id
  -> проверка расширения .pdf
  -> think: «Подготавливаю документ к обработке.»
  -> MinIO download: <user_id>/<document_id>.pdf
  -> think: «Формирую краткое изложение.»
  -> Gemini Files API upload -> readiness polling -> generate -> delete
  -> response
```

Если metadata не найдена, документ другого пользователя, файл не PDF, MinIO недоступен или Gemini не может завершить summary, пользователю возвращается нейтральный fallback без раскрытия внутренних причин и наличия чужого документа.

## 6. Delete-событие

`delete` содержит `type` и `user_id`. Оркестратор очищает пользовательское состояние Redis и публикует `think` с текстом «История диалога очищена.». Полный контракт описан в [04_rabbitmq_contract.md](04_rabbitmq_contract.md).

## 7. Ответственность классов

| Компонент | Ответственность |
|---|---|
| `RabbitMQWorker` | connection, queue declaration, validation, ack/retry/reject |
| `TaskOrchestrator` | контроль сценария, fallback, progress events |
| `IntentAgent` | Gemini JSON classification + rule-based fallback |
| `RetrievalService` | cache-aside retrieval, cache validation, metadata delegation |
| `ElasticsearchClient` | query body, tenant filters, response normalization |
| `ContextAgent` | dedupe и ограничения chunks/context |
| `AnswerAgent` | building grounded prompt и answer generation |
| `GeminiClient` | Gemini calls, retry, limits, Files API cleanup |
| `MinioDocumentStorageClient` | безопасное чтение PDF и file-size control |
| `RedisStateStore` | TTL storage, locks, fail-open Redis operations |
