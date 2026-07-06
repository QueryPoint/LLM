# 08. Эксплуатация и диагностика

## 1. Контейнеры и логи

```bash
docker compose ps
docker compose logs -f --tail=200 assistant-service
```

Основной container name:

```text
querypoint_assistant_service
```

Проверка общего состояния инфраструктуры:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -fsS http://localhost:9200/_cluster/health
docker exec querypoint_rabbitmq rabbitmq-diagnostics -q ping
docker exec querypoint_redis sh -c 'redis-cli -a "$REDIS_PASSWORD" ping'
curl -fsS http://localhost:9000/minio/health/live
```

## 2. Типовые симптомы

### Worker не стартует

Проверить:

1. `querypoint_network` существует.
2. RabbitMQ/Redis/Elasticsearch resolvable из Docker network.
3. `RABBITMQ_USER`, `RABBITMQ_PASSWORD`, `RABBITMQ_VHOST`, `REDIS_PASSWORD` определены для compose substitution.
4. `RABBITMQ_URL` не использует `localhost` внутри контейнера.
5. `.env` не содержит пробелов/неполных URL.

### В очереди растут сообщения

Проверить:

```bash
docker exec querypoint_rabbitmq rabbitmqctl list_queues name messages consumers
```

Дальше изучить безопасные логи worker-а:

- invalid message rejected;
- message handler error;
- retry attempt;
- Gemini/Elasticsearch/MinIO error type.

Не публикуйте реальные payload body в issue или чат.

### Пользователь получает retrieval unavailable

Вероятные причины:

- Elasticsearch недоступен;
- index `documents` отсутствует;
- transport/API ответ некорректен;
- нет доступных chunks текущего user scope;
- metadata selected document не соответствует `user_id + doc_id`.

### PDF summary unavailable

Проверить:

- document действительно PDF;
- ES metadata существует для конкретного user/document;
- object convention совпадает с `<user_id>/<document_id>.pdf`;
- MinIO env configured;
- размер PDF не превышает лимит;
- Gemini API quota/availability.

## 3. Безопасные логи

Допустимые диагностические значения:

- event type;
- task type;
- has_document boolean;
- cache hit/miss;
- error type;
- число chunks;
- число символов;
- retry attempt;
- status.

Запрещённые значения:

```text
GEMINI_API_KEY, passwords, full URLs with credentials
prompt
user_id
document_id/doc
file_name
bucket/object key
retrieved chunk text
summary/Gemini answer
full Elasticsearch body
```

## 4. Перезапуск

```bash
# пересборка только LLM-service
docker compose up -d --build assistant-service

# просмотр последних ошибок
docker compose logs --tail=200 assistant-service
```

Не используйте destructive Docker commands в процессе диагностики без отдельного решения команды.

## 5. Monitoring ideas

Текущие логи подходят для базовой диагностики. Следующий безопасный уровень observability:

- counters: RabbitMQ messages accepted/rejected/retried;
- cache hit/miss;
- retrieval success/failure;
- PDF-summary success/failure;
- latency ES/Redis/MinIO/Gemini;
- fallback reason categories.

Метрики не должны включать raw user or document identifiers.
