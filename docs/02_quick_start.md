# 02. Быстрый запуск

## 1. Предварительные условия

Нужны:

- Docker Desktop / Docker Engine с `docker compose`;
- доступная внешняя Docker network `querypoint_network`;
- поднятые инфраструктурные сервисы Query Point: RabbitMQ, Redis, Elasticsearch, MinIO;
- корректно заполненный `.env` в корне LLM-репозитория;
- Gemini API key только в локальном secret-хранилище или `.env`, который не попадает в Git.

Проект рассчитан на Python `>=3.14`; зависимости управляются `uv`.

## 2. Подготовка `.env`

```bash
cd ~/PycharmProjects/LLM
cp .env.example .env
```

Заполните реальные значения только локально. Не коммитьте `.env`.

Для запуска контейнера в compose URL RabbitMQ и Redis переопределяются на Docker DNS names:

```text
querypoint_rabbitmq
querypoint_redis
querypoint_elasticsearch
minio
```

## 3. Запуск через Docker Compose

```bash
cd ~/PycharmProjects/LLM

docker compose config
docker compose up -d --build
docker compose ps
docker compose logs -f --tail=200 assistant-service
```

Compose содержит вспомогательные `*-check` контейнеры. Они ожидают TCP-доступность RabbitMQ, Redis и Elasticsearch в сети `querypoint_network`, затем завершаются. Это ожидаемое поведение.

Основной контейнер:

```text
querypoint_assistant_service
```

## 4. Проверка инфраструктурной доступности

Перед запуском assistant-service инфраструктура должна быть доступна:

```bash
curl -fsS http://localhost:9200/_cluster/health
curl -fsS http://localhost:9000/minio/health/live

docker exec querypoint_rabbitmq rabbitmq-diagnostics -q ping
docker exec querypoint_redis sh -c 'redis-cli -a "$REDIS_PASSWORD" ping'
```

Ожидается ответ `PONG` для Redis. Elasticsearch в одноузловом dev-окружении может иметь статус `yellow` только из-за неразмещённой реплики; если primary shards active, это не блокирует retrieval.

## 5. Локальный запуск без Docker

Локальный запуск полезен для разработки, но endpoint-ы должны указывать на хостовые адреса, например `localhost`, а не Docker service names.

```bash
cd ~/PycharmProjects/LLM
uv sync --group dev
uv run python -m assistant_service.main
```

Типичная host-конфигурация:

```env
RABBITMQ_URL=amqp://<user>:<password>@localhost:5672/
REDIS_URL=redis://:<password>@localhost:6379/0
ELASTICSEARCH_URL=http://localhost:9200
MINIO_ENDPOINT=localhost:9000
```

Не используйте один и тот же `.env` бездумно для host и Docker: Docker DNS names не резолвятся с хоста.

## 6. Остановка

```bash
docker compose down
```

Не используйте `docker compose down -v`, если не намерены удалить данные подключённой инфраструктуры.

## 7. Smoke checks

```bash
uv run python -m assistant_service.main --help
uv run python -m assistant_service.tools.document_probe --help
uv run pytest -q
```

Gemini не вызывается командами `--help` и обычными unit/contract tests.
