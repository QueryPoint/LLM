import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextAgent
from assistant_service.agents.document_summary_agent import DocumentSummaryAgent
from assistant_service.agents.intent_agent import IntentAgent
from assistant_service.core.config import settings
from assistant_service.messaging.rabbitmq import RabbitMQWorker
from assistant_service.services.elasticsearch_client import (
    ElasticsearchClient,
    create_elasticsearch_client_from_settings,
)
from assistant_service.services.gemini_client import (
    GeminiClient,
    create_gemini_client_from_settings,
)
from assistant_service.services.redis_state import (
    RedisStateStore,
    create_redis_state_store_from_settings,
)
from assistant_service.services.retrieval_service import RetrievalService
from assistant_service.services.task_orchestrator import TaskOrchestrator

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="assistant-service",
        description="Run the assistant-service RabbitMQ worker.",
    )


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, stop_event.set)
        except NotImplementedError:
            signal.signal(signal_number, lambda *_: stop_event.set())


async def run_worker() -> None:
    worker = RabbitMQWorker(
        settings=settings,
        max_processing_attempts=settings.rabbitmq_max_processing_attempts,
    )
    gemini_client: GeminiClient | None = None
    redis_state: RedisStateStore | None = None
    elasticsearch_client: ElasticsearchClient | None = None
    context_agent = ContextAgent()
    try:
        gemini_client = create_gemini_client_from_settings(settings)
        redis_state = create_redis_state_store_from_settings(settings)
        elasticsearch_client = create_elasticsearch_client_from_settings(settings)
        retrieval_service = RetrievalService(search_client=elasticsearch_client)
        intent_agent = IntentAgent(text_generator=gemini_client)
        answer_agent = AnswerAgent(text_generator=gemini_client)
        document_summary_agent = DocumentSummaryAgent(
            text_generator=gemini_client,
            max_chunk_chars=settings.gemini_max_chunk_chars,
            max_prompt_chars=settings.gemini_max_prompt_chars,
        )
        orchestrator = TaskOrchestrator(
            publisher=worker,
            intent_agent=intent_agent,
            context_agent=context_agent,
            answer_agent=answer_agent,
            document_summary_agent=document_summary_agent,
            retrieval_service=retrieval_service,
            redis_state=redis_state,
            max_user_prompt_chars=settings.gemini_max_user_prompt_chars,
            max_chunk_chars=settings.gemini_max_chunk_chars,
            max_chunks_per_request=settings.gemini_max_chunks_per_request,
            max_prompt_chars=settings.gemini_max_prompt_chars,
        )
        worker.set_message_handler(orchestrator.handle)

        stop_event = asyncio.Event()
        _install_signal_handlers(stop_event)

        await worker.start()
        logger.info("Assistant service worker started")

        await stop_event.wait()
        logger.info("Shutdown signal received")
    finally:
        await worker.stop()
        if gemini_client is not None:
            try:
                await gemini_client.aclose()
            except Exception:
                logger.exception("Gemini client close failed")
        if redis_state is not None:
            try:
                await redis_state.aclose()
            except Exception:
                logger.exception("Redis state close failed")
        if elasticsearch_client is not None:
            try:
                await elasticsearch_client.aclose()
            except Exception:
                logger.exception("Elasticsearch client close failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)

    logging.basicConfig(level=settings.log_level.upper())

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
        return 0
    except Exception:
        logger.exception("Worker failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
