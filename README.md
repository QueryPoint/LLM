# Документация LLM-service Query Point

Этот пакет подготовлен по актуальному исходному коду `assistant-service` из архива проекта. Его можно перенести в корень репозитория: файл `README.md` использовать как основу для корневой документации, а каталог `docs/` — скопировать в существующий каталог документации.

## Содержание

| Документ | Для чего нужен |
|---|---|
| [00_overview.md](docs/00_overview.md) | назначение сервиса, возможности, границы ответственности |
| [01_architecture.md](docs/01_architecture.md) | компоненты, потоки данных и жизненный цикл запроса |
| [02_quick_start.md](docs/02_quick_start.md) | запуск через Docker и локальный запуск |
| [03_configuration.md](docs/03_configuration.md) | все переменные окружения и безопасная настройка |
| [04_rabbitmq_contract.md](docs/04_rabbitmq_contract.md) | строгий контракт RabbitMQ, очереди, retry и ack |
| [05_retrieval_security.md](docs/05_retrieval_security.md) | Elasticsearch, tenant isolation, Redis retrieval cache |
| [06_pdf_summary.md](docs/06_pdf_summary.md) | PDF-summary через MinIO и Gemini Files API |
| [07_testing.md](docs/07_testing.md) | структура тестов, команды и правила integration testing |
| [08_operations.md](docs/08_operations.md) | диагностика, логи, типовые сбои, безопасная эксплуатация |
| [09_development.md](docs/09_development.md) | устройство исходного кода и правила внесения изменений |

## Быстрый ориентир

`assistant-service` — асинхронный Python worker. Он принимает команды из RabbitMQ, определяет тип пользовательского запроса, при необходимости извлекает разрешённый контекст из Elasticsearch, обращается к Gemini только для grounded-ответов и отправляет события обратно через RabbitMQ.

```text
Backend -> RabbitMQ(back_to_llm) -> assistant-service
                                   |-> Redis
                                   |-> Elasticsearch
                                   |-> MinIO (только PDF summary)
                                   `-> Gemini API
assistant-service -> RabbitMQ(llm_to_back) -> Backend
```

## Что важно не менять без согласования

- Входящий RabbitMQ `prompt` использует поле `doc`, а не `uid`.
- Внешние исходящие события ограничены `think` и `response`.
- Retrieval и metadata lookup обязаны быть ограничены `user_id`.
- `doc` соответствует полю Elasticsearch `doc_id`.
- PDF-summary использует временное соглашение MinIO: `<user_id>/<document_id>.pdf`.
- В логах нельзя печатать prompt, `user_id`, `document_id`, object key, текст документа, summary, ключи и пароли.

Не копируйте `.env` и не добавляйте реальные secrets в Markdown-файлы.
