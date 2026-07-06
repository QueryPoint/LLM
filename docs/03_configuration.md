# 03. Конфигурация

Настройки загружаются классом `Settings` из `src/assistant_service/core/config.py`. Используется `pydantic-settings`, `.env` читается из рабочего каталога, неизвестные переменные игнорируются.

## 1. Общие настройки

| Переменная | Default | Назначение |
|---|---:|---|
| `APP_NAME` | `assistant-service` | имя приложения |
| `APP_ENV` | `development` | окружение для логического разделения конфигурации |
| `LOG_LEVEL` | `INFO` | уровень Python logging |

## 2. RabbitMQ

| Переменная | Default | Назначение |
|---|---:|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | AMQP URL worker-а |
| `RABBITMQ_BACK_TO_LLM_QUEUE` | `back_to_llm` | очередь входящих команд |
| `RABBITMQ_LLM_TO_BACK_QUEUE` | `llm_to_back` | очередь исходящих событий |
| `RABBITMQ_PREFETCH_COUNT` | `1` | максимум непodтверждённых сообщений consumer-а |
| `RABBITMQ_REQUEUE_DELAY_SECONDS` | `1.0` | задержка перед `nack(requeue=True)` при сбое retry republish |
| `RABBITMQ_MAX_PROCESSING_ATTEMPTS` | `3` | максимум попыток обработки через republish |

В Docker Compose URL строится с `RABBITMQ_USER`, `RABBITMQ_PASSWORD`, `RABBITMQ_VHOST`; эти значения должны быть определены в локальном `.env`, но не документироваться с реальными значениями.

## 3. Gemini

| Переменная | Default | Назначение |
|---|---:|---|
| `GEMINI_API_KEY` | пусто | обязательный ключ для реального LLM runtime |
| `GEMINI_MODEL` | `gemini-2.5-flash` | имя модели |
| `GEMINI_TIMEOUT_SECONDS` | `60` | HTTP timeout и предел ожидания Files API readiness |
| `GEMINI_RETRY_MAX_ATTEMPTS` | `3` | максимум попыток retry обычной генерации |
| `GEMINI_RETRY_INITIAL_DELAY_SECONDS` | `0.5` | начальная задержка exponential backoff |
| `GEMINI_RETRY_MAX_DELAY_SECONDS` | `4.0` | верхняя граница backoff |
| `GEMINI_MAX_USER_PROMPT_CHARS` | `4000` | лимит raw пользовательского prompt |
| `GEMINI_MAX_CHUNK_CHARS` | `12000` | лимит отдельного chunk |
| `GEMINI_MAX_CHUNKS_PER_REQUEST` | `32` | лимит числа chunks |
| `GEMINI_MAX_PROMPT_CHARS` | `16000` | общий лимит generation prompt |
| `GEMINI_MAX_RESPONSE_CHARS` | `12000` | клиентский лимит streamed output |

## 4. Redis

| Переменная | Default | Назначение |
|---|---:|---|
| `REDIS_URL` | `redis://localhost:6379/0` | URL Redis |
| `REDIS_ANSWER_CACHE_TTL_SECONDS` | `900` | TTL answer cache |
| `REDIS_INTENT_CACHE_TTL_SECONDS` | `900` | TTL intent cache |
| `REDIS_TASK_STATUS_TTL_SECONDS` | `3600` | TTL статуса задачи |
| `REDIS_SESSION_SUMMARY_TTL_SECONDS` | `21600` | TTL session summary |
| `REDIS_ANSWER_LOCK_TTL_SECONDS` | `90` | TTL answer lock |
| `RETRIEVAL_CACHE_TTL_SECONDS` | `900` | TTL retrieval cache |

Redis ошибки должны быть fail-open: отказ кэша не должен блокировать retrieval/answer flow.

## 5. Elasticsearch

| Переменная | Default | Назначение |
|---|---:|---|
| `ELASTICSEARCH_URL` | `http://elasticsearch:9200` | endpoint Elasticsearch |
| `ELASTICSEARCH_INDEX` | `documents` | индекс chunks |
| `ELASTICSEARCH_TIMEOUT_SECONDS` | `5` | timeout клиента |
| `ELASTICSEARCH_MAX_RESULTS` | `8` | максимум search hits |

Не меняйте mapping из LLM-repository. Ожидаемые поля описаны в [05_retrieval_security.md](05_retrieval_security.md).

## 6. MinIO и PDF summary

| Переменная | Default | Назначение |
|---|---:|---|
| `MINIO_ENDPOINT` | пусто | endpoint MinIO, например `minio:9000` в Docker |
| `MINIO_BUCKET` | пусто | bucket исходных документов |
| `MINIO_ACCESS_KEY` | пусто | access key |
| `MINIO_SECRET_KEY` | пусто | secret key |
| `MINIO_SECURE` | `false` | использовать HTTPS для MinIO SDK |
| `MINIO_SPIKE_ENABLED` | `false` | включает только manual document probe |
| `DOCUMENT_SUMMARY_MAX_FILE_BYTES` | `20971520` | лимит PDF bytes для production summary |

Если один из обязательных MinIO параметров отсутствует, runtime не падает при старте, но PDF summary возвращает controlled unavailable fallback.

## 7. Правила secrets

- `.env` должен быть в `.gitignore`.
- Не храните `GEMINI_API_KEY`, MinIO secrets, AMQP password или Redis password в `README.md`, тестах и tracked config.
- В отчётах и логах допускается только статус `configured/missing`, но не значение секрета.
- После случайного раскрытия ключа его нужно отозвать и выпустить новый.
