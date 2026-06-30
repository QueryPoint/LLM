import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextAgent
from assistant_service.agents.intent_agent import IntentAgent
from assistant_service.core.config import settings
from assistant_service.messaging.rabbitmq import RabbitMQWorker
from assistant_service.services.gemini_client import (
    GeminiClient,
    create_gemini_client_from_settings,
)
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
    worker = RabbitMQWorker(settings=settings)
    gemini_client: GeminiClient | None = None
    intent_agent = IntentAgent()
    context_agent = ContextAgent()
    try:
        gemini_client = create_gemini_client_from_settings(settings)
        answer_agent = AnswerAgent(text_generator=gemini_client)
        orchestrator = TaskOrchestrator(
            publisher=worker,
            intent_agent=intent_agent,
            context_agent=context_agent,
            answer_agent=answer_agent,
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
