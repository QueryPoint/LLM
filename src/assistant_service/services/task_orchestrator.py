import logging
from typing import Protocol

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextAgent
from assistant_service.agents.document_summary_agent import DocumentSummaryAgent
from assistant_service.agents.intent_agent import (
    IntentAgent,
    IntentDecision,
    IntentTaskType,
)
from assistant_service.core.enums import LLMStatus, OutgoingEventType
from assistant_service.messaging.contracts import (
    DeleteRequestMessage,
    IncomingMessage,
    OutgoingEvent,
    PromptRequestMessage,
    ResponseEvent,
    ThinkEvent,
)
from assistant_service.services.prompt_budget import (
    calculate_prompt_budget_warning,
    prompt_budget_exceeded,
)
from assistant_service.services.response_builder import (
    build_context_too_large_response,
    build_materials_not_found_response,
    build_request_too_large_response,
    build_unsupported_request_response,
)
from assistant_service.services.redis_state import RedisStateStore

logger = logging.getLogger(__name__)

DETERMINE_INTENT_TEXT = "Определяю тип запроса..."
DELETE_THINK_TEXT = "История диалога очищена."
SAFE_ERROR_RESPONSE = "Не удалось обработать запрос. Попробуйте ещё раз."
NO_RETRIEVAL_PROMPT_MARKUP = "ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n\nКОНТЕКСТ:\n"


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
        max_user_prompt_chars: int,
        max_chunk_chars: int,
        max_chunks_per_request: int,
        max_prompt_chars: int,
    ) -> None:
        if max_user_prompt_chars <= 0:
            raise ValueError("max_user_prompt_chars must be positive")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")
        if max_chunks_per_request <= 0:
            raise ValueError("max_chunks_per_request must be positive")
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")

        self._publisher = publisher
        self._intent_agent = intent_agent
        self._context_agent = context_agent
        self._answer_agent = answer_agent
        self._document_summary_agent = document_summary_agent
        self._redis_state = redis_state
        self._max_user_prompt_chars = max_user_prompt_chars
        self._max_chunk_chars = max_chunk_chars
        self._max_chunks_per_request = max_chunks_per_request
        self._max_prompt_chars = max_prompt_chars

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
        warning = self._calculate_warning(user_prompt=message.prompt)
        if self._is_user_prompt_too_large(message.prompt):
            logger.info(
                "Prompt rejected by local limit: user_id=%s prompt_chars=%s",
                message.user_id,
                len(message.prompt),
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_request_too_large_response(),
                warning=warning,
            )
            return

        if self._prompt_budget_exceeded(user_prompt=message.prompt):
            logger.info(
                "Prompt rejected by budget limit: user_id=%s warning=%s",
                message.user_id,
                warning,
            )
            await self._publish_response(
                user_id=message.user_id,
                data=build_context_too_large_response(),
                warning=warning,
            )
            return

        logger.info(
            "Task orchestrator received prompt: user_id=%s has_uid=%s",
            message.user_id,
            message.uid is not None,
        )
        await self._redis_state.set_task_status(
            user_id=message.user_id,
            status=LLMStatus.PROCESSING.value.lower(),
        )
        await self._publish_think(
            user_id=message.user_id,
            data=DETERMINE_INTENT_TEXT,
            status=LLMStatus.THINKING,
        )

        intent_decision = await self._detect_intent(message)
        logger.info(
            "Intent detected without retrieval: user_id=%s task_type=%s "
            "selection_source=%s has_uid=%s",
            message.user_id,
            intent_decision.task_type.value,
            intent_decision.source,
            message.uid is not None,
        )
        if intent_decision.task_type == IntentTaskType.UNSUPPORTED:
            await self._publish_response(
                user_id=message.user_id,
                data=build_unsupported_request_response(),
                warning=warning,
            )
            return

        await self._redis_state.set_task_status(
            user_id=message.user_id,
            status="not_found",
        )
        await self._publish_response(
            user_id=message.user_id,
            data=build_materials_not_found_response(),
            warning=warning,
        )

    async def _handle_delete(self, message: DeleteRequestMessage) -> None:
        logger.info(
            "Task orchestrator received delete: user_id=%s",
            message.user_id,
        )
        await self._redis_state.clear_user_state(user_id=message.user_id)
        await self._publish_think(user_id=message.user_id, data=DELETE_THINK_TEXT)

    async def _detect_intent(self, message: PromptRequestMessage) -> IntentDecision:
        return await self._intent_agent.detect(prompt=message.prompt, uid=message.uid)

    async def _publish_think(
        self,
        user_id: str,
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

    def _is_user_prompt_too_large(self, prompt: str) -> bool:
        return len(prompt) > self._max_user_prompt_chars

    def _calculate_warning(self, *, user_prompt: str) -> int:
        return calculate_prompt_budget_warning(
            max_prompt_chars=self._max_prompt_chars,
            user_prompt=user_prompt,
            prompt_markup=NO_RETRIEVAL_PROMPT_MARKUP,
        )

    def _prompt_budget_exceeded(self, *, user_prompt: str) -> bool:
        return prompt_budget_exceeded(
            max_prompt_chars=self._max_prompt_chars,
            user_prompt=user_prompt,
            prompt_markup=NO_RETRIEVAL_PROMPT_MARKUP,
        )

    async def _publish_response(
        self,
        user_id: str,
        data: str,
        *,
        warning: int = 0,
        mark_completed: bool = True,
    ) -> None:
        await self._publisher.publish_event(
            ResponseEvent(
                type=OutgoingEventType.RESPONSE,
                user_id=user_id,
                data=data,
                warning=warning,
            )
        )
        if mark_completed:
            await self._redis_state.set_task_status(
                user_id=user_id,
                status="completed",
            )
