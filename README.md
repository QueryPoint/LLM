# LLM

Assistant service для проекта «Интеллектуальная база знаний университета».

## RabbitMQ contract

Входящие события:

- `prompt`
- `delete`

Исходящие события:

- `think`
- `response`

Backend отправляет минимальный `prompt` payload:

```json
{
  "type": "prompt",
  "user_id": "user-123",
  "prompt": "Объясни нормализацию баз данных",
  "uid": null
}
```

Если запрос относится к выбранному документу, backend передаёт UID документа:

```json
{
  "type": "prompt",
  "user_id": "user-123",
  "prompt": "Сделай краткое содержание документа",
  "uid": "document-uid-123"
}
```

Backend больше не передаёт `mode` и `document_context`. Поле документа называется
строго `uid`.

`think.data` всегда содержит человекочитаемую строку, например:

```json
{
  "type": "think",
  "user_id": "user-123",
  "data": "Определяю тип запроса..."
}
```

Финальный ответ приходит одним событием:

```json
{
  "type": "response",
  "user_id": "user-123",
  "data": "Не удалось найти материалы для подготовки ответа. Попробуйте уточнить запрос или выбрать документ.",
  "warning": 12
}
```

На текущем этапе Elasticsearch/retrieval и MinIO/S3 ещё не подключены. Поэтому
Gemini не вызывается без найденного реального контекста, а сервис возвращает
контролируемый fallback.

`warning` — целый процент использованного prompt/context budget. Если фактический
размер prompt/context достигает лимита или превышает его, generation блокируется.
