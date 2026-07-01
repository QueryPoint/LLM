import logging
from collections.abc import Awaitable, Callable
from typing import Protocol
from uuid import UUID

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextAgent, ContextDecision
from assistant_service.agents.document_summary_agent import DocumentSummaryAgent
from assistant_service.agents.intent_agent import IntentAgent, IntentDecision
from assistant_service.core.enums import (
    AssistantMode,
    LLMStatus,
    OutgoingEventType,
    RetrievalStatus,
)
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    IncomingMessage,
    OutgoingEvent,
    PromptRequestMessage,
    ResponseEvent,
    ThinkEvent,
)
from assistant_service.services.response_builder import (
    build_summary_requires_complete_document_response,
    build_response_text,
    get_response_kind,
)
from assistant_service.services.cache_keys import (
    build_answer_cache_key,
    build_answer_lock_key,
    build_intent_cache_key,
)
from assistant_service.services.redis_state import RedisStateStore

logger = logging.getLogger(__name__)

STATUS_MESSAGES = {
    LLMStatus.QUEUED: "Запрос добавлен в очередь...",
    LLMStatus.PROCESSING: "Приняли запрос в обработку...",
    LLMStatus.THINKING: "Анализируем запрос...",
    LLMStatus.SEARCHING: "Ищем релевантные материалы...",
    LLMStatus.FOUND: "Нашли подходящие материалы...",
    LLMStatus.GENERATING: "Готовим дальнейшую обработку...",
}
INTENT_DETECTED_TEXT = "Определили тип запроса..."
CONTEXT_NOT_FOUND_TEXT = "Подходящие материалы не найдены."
CONTEXT_INSUFFICIENT_TEXT = "Найденных материалов недостаточно для подготовки ответа."
DELETE_THINK_TEXT = "История диалога очищена."
SAFE_ERROR_RESPONSE = "Не удалось обработать запрос. Попробуйте ещё раз."


class EventPublisher(Protocol):
    async def publish_event(self, event: OutgoingEvent) -> None:
        ...


class TaskOrchestrator:
    def __init__(
        self,
        publisher: EventPublisher,
        intent_agent: IntentAgent,
        context_agent: ContextAgent,
        answer_agent: AnswerAgent,
        document_summary_agent: DocumentSummaryAgent,
        redis_state: RedisStateStore,
    ) -> None:
        self._publisher = publisher
        self._intent_agent = intent_agent
        self._context_agent = context_agent
        self._answer_agent = answer_agent
        self._document_summary_agent = document_summary_agent
        self._redis_state = redis_state

    async def handle(self, message: IncomingMessage) -> None:
        try:
            if isinstance(message, PromptRequestMessage):
                await self._handle_prompt(message)
                return

            if isinstance(message, DeleteRequestMessage):
                await self._handle_delete(message)
                return

        except Exception:
            logger.exception(
                "Task orchestrator processing failed: user_id=%s message_type=%s",
                message.user_id,
                message.type,
            )
            await self._redis_state.set_task_status(
                user_id=message.user_id,
                status="failed",
            )
            try:
                await self._publish_response(
                    user_id=message.user_id,
                    data=SAFE_ERROR_RESPONSE,
                    mark_completed=False,
                )
                return
            except Exception:
                logger.exception(
                    "Task orchestrator failed to publish safe response: "
                    "user_id=%s message_type=%s",
                    message.user_id,
                    message.type,
                )
                raise

        raise ValueError(f"Unsupported incoming message type: {message.type}")

    async def _handle_prompt(self, message: PromptRequestMessage) -> None:
        logger.info(
            "Task orchestrator received prompt: user_id=%s",
            message.user_id,
        )
        await self._publish_prompt_status(message.user_id, LLMStatus.PROCESSING)
        await self._publish_prompt_status(message.user_id, LLMStatus.THINKING)

        intent_decision = await self._detect_intent(message)
        logger.info(
            "Intent detected: user_id=%s selected_mode=%s "
            "selection_source=%s has_document=%s",
            message.user_id,
            intent_decision.mode.value,
            intent_decision.source,
            message.doc is not None,
        )

        await self._publish_think(user_id=message.user_id, data=INTENT_DETECTED_TEXT)
        await self._publish_prompt_status(message.user_id, LLMStatus.SEARCHING)

        context_decision = self._context_agent.prepare(
            message.document_context,
            mode=intent_decision.mode,
        )
        logger.info(
            "Context prepared: user_id=%s retrieval_status=%s selected_chunks=%s "
            "total_chars=%s sources=%s",
            message.user_id,
            context_decision.status.value,
            len(context_decision.chunks),
            context_decision.total_chars,
            len(context_decision.sources),
        )
        await self._publish_context_status(
            user_id=message.user_id,
            retrieval_status=context_decision.status,
        )
        await self._publish_prompt_status(message.user_id, LLMStatus.GENERATING)

        if self._should_use_summary_incomplete_fallback(
            mode=intent_decision.mode,
            retrieval_status=context_decision.status,
        ):
            response_text = build_summary_requires_complete_document_response()
            logger.info(
                "Complete document summary required: user_id=%s mode=%s "
                "retrieval_status=%s",
                message.user_id,
                intent_decision.mode.value,
                context_decision.status.value,
            )
        elif self._should_use_response_builder(
            mode=intent_decision.mode,
            retrieval_status=context_decision.status,
        ):
            response_text = build_response_text(
                mode=intent_decision.mode,
                context_decision=context_decision,
            )
            response_kind = get_response_kind(
                mode=intent_decision.mode,
                context_decision=context_decision,
            )
            logger.info(
                "Response selected without LLM: user_id=%s mode=%s "
                "retrieval_status=%s response_kind=%s",
                message.user_id,
                intent_decision.mode.value,
                context_decision.status.value,
                response_kind,
            )
        elif intent_decision.mode == AssistantMode.SUMMARIZE_DOCUMENT:
            logger.info(
                "Document summary selected: user_id=%s mode=%s "
                "retrieval_status=%s selected_chunks=%s",
                message.user_id,
                intent_decision.mode.value,
                context_decision.status.value,
                len(context_decision.chunks),
            )
            response_text = await self._get_or_generate_answer(
                user_id=message.user_id,
                mode=intent_decision.mode,
                user_prompt=message.prompt,
                context_decision=context_decision,
                generator=lambda: self._document_summary_agent.summarize(
                    user_prompt=message.prompt,
                    context_decision=context_decision,
                ),
            )
        else:
            logger.info(
                "Answer generation selected: user_id=%s mode=%s "
                "retrieval_status=%s selected_chunks=%s",
                message.user_id,
                intent_decision.mode.value,
                context_decision.status.value,
                len(context_decision.chunks),
            )
            response_text = await self._get_or_generate_answer(
                user_id=message.user_id,
                mode=intent_decision.mode,
                user_prompt=message.prompt,
                context_decision=context_decision,
                generator=lambda: self._answer_agent.answer(
                    mode=intent_decision.mode,
                    user_prompt=message.prompt,
                    context_decision=context_decision,
                ),
            )

        await self._publish_response(user_id=message.user_id, data=response_text)

    async def _handle_delete(self, message: DeleteRequestMessage) -> None:
        logger.info(
            "Task orchestrator received delete: user_id=%s",
            message.user_id,
        )
        await self._redis_state.clear_user_state(user_id=message.user_id)
        await self._publish_think(user_id=message.user_id, data=DELETE_THINK_TEXT)

    async def _detect_intent(self, message: PromptRequestMessage) -> IntentDecision:
        if message.mode is not None:
            return self._intent_agent.detect(
                prompt=message.prompt,
                document_id=message.doc,
                requested_mode=message.mode,
            )

        intent_cache_key = build_intent_cache_key(
            prompt=message.prompt,
            document_id=message.doc,
            requested_mode=None,
        )
        cached_mode = await self._redis_state.get_intent_mode(intent_cache_key)
        if cached_mode is not None:
            return IntentDecision(mode=cached_mode, source="rule_based")

        decision = self._intent_agent.detect(
            prompt=message.prompt,
            document_id=message.doc,
            requested_mode=None,
        )
        await self._redis_state.set_intent_mode(intent_cache_key, decision.mode)
        return decision

    async def _get_or_generate_answer(
        self,
        *,
        user_id: UUID,
        mode: AssistantMode,
        user_prompt: str | None,
        context_decision: ContextDecision,
        generator: Callable[[], Awaitable[str]],
    ) -> str:
        answer_cache_key = build_answer_cache_key(
            user_id=user_id,
            mode=mode,
            user_prompt=user_prompt,
            context_decision=context_decision,
        )
        cached_answer = await self._redis_state.get_answer(answer_cache_key)
        if cached_answer is not None:
            logger.info(
                "Answer cache hit: user_id=%s mode=%s",
                user_id,
                mode.value,
            )
            return cached_answer

        lock = await self._redis_state.acquire_answer_lock(
            lock_key=build_answer_lock_key(answer_cache_key)
        )
        try:
            answer = await generator()
            await self._redis_state.set_answer(answer_cache_key, answer)
            return answer
        finally:
            if lock is not None:
                await self._redis_state.release_answer_lock(lock)

    async def _publish_think(
        self,
        user_id: UUID,
        data: str,
        status: LLMStatus | None = None,
    ) -> None:
        await self._publisher.publish_event(
            ThinkEvent(
                type=OutgoingEventType.THINK,
                user_id=user_id,
                data=data,
            )
        )
        logger.info(
            "Task orchestrator published think event: user_id=%s status=%s",
            user_id,
            status.value if status is not None else None,
        )
        if status is not None:
            await self._redis_state.set_task_status(
                user_id=user_id,
                status=status.value.lower(),
            )

    async def _publish_prompt_status(self, user_id: UUID, status: LLMStatus) -> None:
        await self._publish_think(
            user_id=user_id,
            data=STATUS_MESSAGES[status],
            status=status,
        )

    async def _publish_context_status(
        self,
        user_id: UUID,
        retrieval_status: RetrievalStatus,
    ) -> None:
        if retrieval_status == RetrievalStatus.FOUND:
            await self._publish_prompt_status(user_id, LLMStatus.FOUND)
            return

        if retrieval_status == RetrievalStatus.NOT_FOUND:
            await self._publish_think(user_id=user_id, data=CONTEXT_NOT_FOUND_TEXT)
            await self._redis_state.set_task_status(
                user_id=user_id,
                status="not_found",
            )
            return

        await self._publish_think(user_id=user_id, data=CONTEXT_INSUFFICIENT_TEXT)
        await self._redis_state.set_task_status(
            user_id=user_id,
            status="insufficient",
        )

    @staticmethod
    def _should_use_response_builder(
        *,
        mode: AssistantMode,
        retrieval_status: RetrievalStatus,
    ) -> bool:
        return (
            mode == AssistantMode.DOCUMENT_SEARCH
            or retrieval_status != RetrievalStatus.FOUND
        )

    @staticmethod
    def _should_use_summary_incomplete_fallback(
        *,
        mode: AssistantMode,
        retrieval_status: RetrievalStatus,
    ) -> bool:
        return (
            mode == AssistantMode.SUMMARIZE_DOCUMENT
            and retrieval_status == RetrievalStatus.INSUFFICIENT
        )

    async def _publish_response(
        self,
        user_id: UUID,
        data: str,
        *,
        mark_completed: bool = True,
    ) -> None:
        await self._publisher.publish_event(
            ResponseEvent(
                type=OutgoingEventType.RESPONSE,
                user_id=user_id,
                data=data,
                warning=0,
            )
        )
        if mark_completed:
            await self._redis_state.set_task_status(
                user_id=user_id,
                status="completed",
            )
