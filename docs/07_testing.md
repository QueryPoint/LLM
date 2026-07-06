# 07. Тестирование

## 1. Структура

```text
tests/
├── contract/rabbitmq/     # внешний JSON contract RabbitMQ
├── fixtures/              # builders и общие тестовые сущности
├── integration/           # opt-in тесты реальных сервисов
├── unit/agents/           # Intent, Context, Answer
├── unit/messaging/        # RabbitMQ worker
└── unit/services/         # ES, Redis, MinIO, Gemini, orchestrator
```

Некоторые старые тестовые файлы могут присутствовать в корне `tests/`; при изменении тестов ориентируйтесь на фактическую структуру и не создавайте дубликаты.

## 2. Базовые команды

```bash
cd ~/PycharmProjects/LLM
uv sync --group dev
uv run pytest -q
uv run pytest --collect-only -q
uv run python -m assistant_service.main --help
uv run python -m assistant_service.tools.document_probe --help
```

Обычный запуск должен быть быстрым и не должен выполнять реальный вызов Gemini.

## 3. Что проверяют unit tests

- Intent classification и rule-based fallback;
- strict RabbitMQ models и outgoing event shape;
- ack/retry/reject logic worker-а;
- context dedupe и budget limits;
- ES query bodies и response normalization;
- user-scoped retrieval и metadata lookup;
- tenant-safe retrieval cache key;
- Redis fail-open behavior;
- MinIO size/empty/not-found safeguards;
- Gemini error classification и Files API cleanup;
- controlled fallbacks orchestrator-а.

## 4. Contract tests

Contract tests должны защищать совместимость с Backend. Минимально проверяются:

- valid `prompt` с `type`, `user_id`, `prompt`, `doc`;
- `doc: null`;
- rejection legacy `uid`;
- rejection лишних `mode`, `document_context`, `request_id`, `session_id`;
- допустимые outgoing events: только `think`, `response`;
- `data` — строка, response включает допустимый `warning`.

## 5. Integration tests

Integration tests не должны быть частью ежедневного default run, если они требуют Docker или ручных безопасных переменных.

Рекомендуемый запуск:

```bash
RUN_INTEGRATION_TESTS=true uv run pytest -m integration -q
```

Для MinIO дополнительно:

```bash
RUN_INTEGRATION_TESTS=true \
RUN_MINIO_INTEGRATION_TESTS=true \
uv run pytest -m integration -q
```

Elasticsearch metadata integration test требует test-only значения:

```text
INTEGRATION_USER_ID
INTEGRATION_DOCUMENT_ID
```

Тест должен читать только заранее разрешённый тестовый документ. Нельзя использовать production/user documents.

## 6. Что нельзя делать в integration tests

```text
FLUSHALL / очистка Redis целиком
удаление Elasticsearch index
изменение ES mapping
использование пользовательских MinIO objects
логирование secret values
логирование prompt/user_id/document_id/object key
реальный Gemini call в pytest по умолчанию
```

## 7. Проверки перед commit

```bash
uv run pytest -q
uv run pytest --collect-only -q
git diff --check
git status --short
```

Для изменённого RabbitMQ contract обязательно добавить/обновить contract tests. Для изменения retrieval обязательно проверить user-scoping и cache-key separation.
