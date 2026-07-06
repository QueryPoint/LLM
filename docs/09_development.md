# 09. Руководство разработчика

## 1. Layout исходного кода

```text
src/assistant_service/
├── agents/
│   ├── intent_agent.py
│   ├── context_agent.py
│   ├── answer_agent.py
│   └── document_summary_agent.py
├── core/
│   ├── config.py
│   └── enums.py
├── messaging/
│   ├── contracts.py
│   └── rabbitmq.py
├── services/
│   ├── elasticsearch_client.py
│   ├── retrieval_service.py
│   ├── retrieval_cache.py
│   ├── redis_state.py
│   ├── gemini_client.py
│   ├── minio_storage.py
│   ├── pdf_document_summary.py
│   ├── prompt_budget.py
│   ├── response_builder.py
│   └── task_orchestrator.py
├── tools/
│   └── document_probe.py
└── main.py
```

## 2. Добавление нового intent

1. Добавьте значение в `IntentTaskType`.
2. Определите `requires_retrieval` и `requires_full_document` semantics.
3. Обновите Gemini intent instruction и rule-based fallback.
4. Добавьте mapping в `TaskOrchestrator` или отдельный handler.
5. Определите, нужен ли Gemini, deterministic response или новый service.
6. Добавьте unit tests intent + orchestrator tests.
7. Убедитесь, что внешние RabbitMQ events не меняются без согласования.

## 3. Изменение Elasticsearch query

Каждое изменение retrieval должно сохранять:

- `user_id` `term` filter всегда;
- `doc_id` `term` filter при выбранном документе;
- отсутствие global fallback при пустом user scope;
- cache key isolation между users;
- отсутствие raw user/document values в логах.

Изменение mapping, index templates или lifecycle policies не должно выполняться из этого repository без согласования с владельцем Infra/Backend.

## 4. Изменение Gemini usage

Перед добавлением нового Gemini call ответьте на вопросы:

- Есть ли trusted/grounded source material?
- Есть ли локальный limit по input и output?
- Есть ли controlled fallback для rate limit, transient и permanent error?
- Не попадут ли sensitive values в logs?
- Нужна ли cleanup логика для uploaded files?
- Можно ли покрыть flow fake client-ом без live quota?

## 5. Изменение RabbitMQ contract

Нельзя делать alias-поддержку «на всякий случай» (`uid` и `doc` одновременно). Изменение contract должно содержать:

1. единый согласованный payload;
2. обновление strict Pydantic models;
3. contract tests;
4. обновление README/contract docs;
5. синхронизацию с producer/consumer владельцами.

## 6. Code quality

Перед commit:

```bash
uv run pytest -q
uv run pytest --collect-only -q
git diff --check
```

Рекомендуемый стиль изменений:

- маленький scoped diff;
- явные keyword arguments для `user_id`/`document_id`;
- no secrets in code/docs/tests;
- no broad exception swallowing without neutral fallback;
- no live Gemini test in default pytest;
- no destructive integration test actions.

## 7. Известные технические задачи

- раздельные host/Docker env profiles для integration testing;
- зарегистрированные pytest markers и снижение warning noise;
- healthcheck основного LLM-container;
- безопасные metrics/reason codes;
- DOCX strategy для full-document summary;
- review использования `DocumentSummaryAgent` относительно текущего PDF-only production path.
