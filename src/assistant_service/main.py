import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence

from assistant_service.core.config import settings
from assistant_service.messaging.rabbitmq import RabbitMQWorker
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
    orchestrator = TaskOrchestrator(publisher=worker)
    worker.set_message_handler(orchestrator.handle)

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    await worker.start()
    logger.info("Assistant service worker started")

    try:
        await stop_event.wait()
        logger.info("Shutdown signal received")
    finally:
        await worker.stop()


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
