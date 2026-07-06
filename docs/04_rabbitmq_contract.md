# 04. RabbitMQ contract

## 1. Очереди

| Направление | Очередь | Кто публикует | Кто потребляет |
|---|---|---|---|
| запрос к LLM | `back_to_llm` | Backend | `assistant-service` |
| результат LLM | `llm_to_back` | `assistant-service` | Backend |

Обе очереди объявляются durable. Исходящие сообщения публикуются как persistent JSON (`application/json`).

## 2. Входящие события

Поддерживаются только два типа:

```text
prompt
delete
```

### 2.1 `prompt`

```json
{
  "type": "prompt",
  "user_id": "user-123",
  "prompt": "Объясни нормальные формы",
  "doc": "document-id-or-null"
}
```

| Поле | Тип | Обязательно | Правило |
|---|---|---:|---|
| `type` | literal `prompt` | да | discriminator |
| `user_id` | string | да | после trim не может быть пустым |
| `prompt` | string | да | после trim не может быть пустым |
| `doc` | string/null | нет | после trim пустая строка нормализуется в `null` |

`doc` соответствует Elasticsearch `doc_id` и используется как выбранный документ.

### 2.2 `delete`

```json
{
  "type": "delete",
  "user_id": "user-123"
}
```

Команда очищает Redis user-state и публикует прогресс-сообщение.

## 3. Строгая валидация

Все модели имеют `extra="forbid"`. Сообщения с лишними полями отвергаются до обработки. В частности, внешний prompt не должен содержать:

```text
uid
mode
document_context
doc_uid
request_id
session_id
sync
token
delta
done
error
sources
```

`uid` не поддерживается как alias для `doc`.

## 4. Исходящие события

### 4.1 `think`

```json
{
  "type": "think",
  "user_id": "user-123",
  "data": "Ищу подходящие материалы..."
}
```

`data` — обычная русскоязычная строка для отображения текущего этапа. Типовые сообщения:

```text
Определяю тип запроса...
Ищу подходящие материалы...
Формирую ответ...
Определяю выбранный документ.
Подготавливаю документ к обработке.
Формирую краткое изложение.
История диалога очищена.
```

### 4.2 `response`

```json
{
  "type": "response",
  "user_id": "user-123",
  "data": "...",
  "warning": 34
}
```

`response` — единственное финальное событие каждого успешного или controlled-fallback сценария. `warning` — целое число, отражающее процент использованного prompt budget.

## 5. Ack, reject и retry

1. Invalid JSON, invalid UTF-8 или Pydantic validation failure -> `reject(requeue=False)`.
2. Успешная обработка -> `ack()`.
3. Неожиданное исключение обработчика -> исходное сообщение republish в `back_to_llm` с header `x-assistant-processing-attempt`, затем `ack()` исходника.
4. Если retry republish не удалось -> пауза `RABBITMQ_REQUEUE_DELAY_SECONDS`, затем `nack(requeue=True)`.
5. Когда исчерпан `RABBITMQ_MAX_PROCESSING_ATTEMPTS` -> `reject(requeue=False)`.

Важно: controlled fallback внутри `TaskOrchestrator` обычно завершает обработку без исключения и поэтому приводит к обычному `ack()`.

## 6. Совместимость

Изменения внешнего RabbitMQ contract должны выполняться одновременно с contract tests. Нельзя молча добавлять альтернативные названия полей, потому что это создаёт неявные два контракта и усложняет поддержку.
