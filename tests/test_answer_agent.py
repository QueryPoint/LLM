import asyncio
from uuid import UUID

import pytest

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextDecision
from assistant_service.agents.intent_agent import IntentDecision
from assistant_service.core.enums import (
    AssistantMode,
    OutgoingEventType,
    RetrievalStatus,
)
from assistant_service.messaging.contracts import (
    IncomingMessage,
    OutgoingEvent,
    RetrievedChunk,
    incoming_message_adapter,
)
from assistant_service.services.gemini_client import GeminiClientError
from assistant_service.services.task_orchestrator import TaskOrchestrator


USER_ID = "00000000-0000-0000-0000-000000000002"
DOCUMENT_ID = "00000000-0000-0000-0000-000000000003"
CHUNK_ID = "00000000-0000-0000-0000-000000000004"
MAX_USER_PROMPT_CHARS = 4_000
MAX_CHUNK_CHARS = 12_000
MAX_CHUNKS_PER_REQUEST = 32


class FakeTextGenerator:
    def __init__(
        self,
        stream_fragments: tuple[str, ...] = ("Ответ ", "на основе контекста."),
    ) -> None:
        self.generate_text_calls: list[dict[str, object]] = []
        self.stream_text_calls: list[dict[str, object]] = []
        self._stream_fragments = stream_fragments

    async def generate_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        self.generate_text_calls.append(
            {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        return "generate_text must not be used"

    async def stream_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> object:
        self.stream_text_calls.append(
            {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        for fragment in self._stream_fragments:
            yield fragment


class FakePublisher:
    def __init__(self) -> None:
        self.events: list[OutgoingEvent] = []

    async def publish_event(self, event: OutgoingEvent) -> None:
        self.events.append(event)


class FakeIntentAgent:
    def detect(
        self,
        prompt: str | None,
        document_id: UUID | None,
        requested_mode: AssistantMode | None,
    ) -> IntentDecision:
        return IntentDecision(mode=AssistantMode.ANSWER_QUESTION, source="rule_based")


class FakeContextAgent:
    def __init__(self, decision: ContextDecision) -> None:
        self._decision = decision

    def prepare(
        self,
        document_context: object,
        mode: AssistantMode | None = None,
    ) -> ContextDecision:
        return self._decision


class FakeAnswerAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[AssistantMode, str | None, ContextDecision]] = []

    async def answer(
        self,
        *,
        mode: AssistantMode,
        user_prompt: str | None,
        context_decision: ContextDecision,
    ) -> str:
        self.calls.append((mode, user_prompt, context_decision))
        return "Ответ AnswerAgent"


class FakeDocumentSummaryAgent:
    async def summarize(
        self,
        *,
        user_prompt: str | None,
        context_decision: ContextDecision,
    ) -> str:
        return "Summary"


class FakeRedisState:
    async def get_answer(self, key: str) -> None:
        return None

    async def set_answer(self, key: str, answer: str) -> None:
        return None

    async def acquire_answer_lock(self, *, lock_key: str) -> None:
        return None

    async def release_answer_lock(self, lock: object) -> None:
        return None

    async def get_intent_mode(self, key: str) -> None:
        return None

    async def set_intent_mode(self, key: str, mode: AssistantMode) -> None:
        return None

    async def set_task_status(self, *, user_id: UUID, status: str) -> None:
        return None

    async def clear_user_state(self, *, user_id: UUID) -> None:
        return None


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(CHUNK_ID),
        document_id=UUID(DOCUMENT_ID),
        file_name="lecture.pdf",
        page=5,
        text="Нормализация уменьшает избыточность данных.",
        score=12.5,
    )


def _context_decision(
    *,
    status: RetrievalStatus = RetrievalStatus.FOUND,
    chunks: tuple[RetrievedChunk, ...] | None = None,
) -> ContextDecision:
    selected_chunks = (_chunk(),) if chunks is None else chunks
    return ContextDecision(
        status=status,
        chunks=selected_chunks,
        sources=(),
        total_chars=sum(len(chunk.text) for chunk in selected_chunks),
    )


def _prompt_message() -> IncomingMessage:
    return incoming_message_adapter.validate_python(
        {
            "type": "prompt",
            "user_id": USER_ID,
            "prompt": "Что такое нормализация?",
            "doc": DOCUMENT_ID,
        }
    )


def test_answer_question_uses_grounded_prompt_without_internal_identifiers() -> None:
    generator = FakeTextGenerator(
        stream_fragments=("Ответ ", "сформирован", " корректно.")
    )
    agent = AnswerAgent(text_generator=generator)

    answer = asyncio.run(
        agent.answer(
            mode=AssistantMode.ANSWER_QUESTION,
            user_prompt=" Что такое нормализация? ",
            context_decision=_context_decision(),
        )
    )

    assert answer == "Ответ сформирован корректно."
    assert generator.generate_text_calls == []
    assert len(generator.stream_text_calls) == 1
    call = generator.stream_text_calls[0]
    assert "Отвечай только на основе материалов" in str(call["system_instruction"])
    assert "Ответь прямо на вопрос пользователя." in str(call["system_instruction"])
    prompt = str(call["prompt"])
    assert "ЗАПРОС ПОЛЬЗОВАТЕЛЯ:" in prompt
    assert "КОНТЕКСТ:" in prompt
    assert "lecture.pdf" in prompt
    assert "Страница: 5" in prompt
    assert "Нормализация уменьшает избыточность данных." in prompt
    assert CHUNK_ID not in prompt
    assert DOCUMENT_ID not in prompt
    assert "chunk_id" not in prompt
    assert "document_id" not in prompt
    assert "score" not in prompt
    assert "UUID" not in prompt


def test_explain_topic_uses_educational_instruction() -> None:
    generator = FakeTextGenerator()
    agent = AnswerAgent(text_generator=generator)

    asyncio.run(
        agent.answer(
            mode=AssistantMode.EXPLAIN_TOPIC,
            user_prompt="Объясни нормализацию",
            context_decision=_context_decision(),
        )
    )

    assert "Объясни тему простым учебным языком." in str(
        generator.stream_text_calls[0]["system_instruction"]
    )


def test_summarize_document_uses_summary_instruction_and_token_limit() -> None:
    generator = FakeTextGenerator()
    agent = AnswerAgent(text_generator=generator)

    asyncio.run(
        agent.answer(
            mode=AssistantMode.SUMMARIZE_DOCUMENT,
            user_prompt=None,
            context_decision=_context_decision(),
        )
    )

    assert "Сделай краткое структурированное изложение" in str(
        generator.stream_text_calls[0]["system_instruction"]
    )
    assert generator.stream_text_calls[0]["max_output_tokens"] == 1200
    assert "Сформируй ответ по переданным материалам." in str(
        generator.stream_text_calls[0]["prompt"]
    )


def test_empty_buffered_stream_raises_client_error() -> None:
    generator = FakeTextGenerator(stream_fragments=("", "   "))
    agent = AnswerAgent(text_generator=generator)

    with pytest.raises(GeminiClientError):
        asyncio.run(
            agent.answer(
                mode=AssistantMode.ANSWER_QUESTION,
                user_prompt="Prompt",
                context_decision=_context_decision(),
            )
        )


def test_invalid_generation_paths_raise_without_calling_generator() -> None:
    generator = FakeTextGenerator()
    agent = AnswerAgent(text_generator=generator)
    invalid_values = (
        (
            AssistantMode.DOCUMENT_SEARCH,
            _context_decision(status=RetrievalStatus.FOUND),
        ),
        (
            AssistantMode.ANSWER_QUESTION,
            _context_decision(status=RetrievalStatus.NOT_FOUND, chunks=()),
        ),
        (
            AssistantMode.EXPLAIN_TOPIC,
            _context_decision(status=RetrievalStatus.FOUND, chunks=()),
        ),
    )

    for mode, context_decision in invalid_values:
        with pytest.raises(ValueError):
            asyncio.run(
                agent.answer(
                    mode=mode,
                    user_prompt="Prompt",
                    context_decision=context_decision,
                )
            )

    assert generator.generate_text_calls == []
    assert generator.stream_text_calls == []


def test_orchestrator_uses_answer_agent_for_answer_question_with_found_context() -> None:
    publisher = FakePublisher()
    answer_agent = FakeAnswerAgent()
    orchestrator = TaskOrchestrator(
        publisher=publisher,
        intent_agent=FakeIntentAgent(),
        context_agent=FakeContextAgent(_context_decision()),
        answer_agent=answer_agent,
        document_summary_agent=FakeDocumentSummaryAgent(),
        redis_state=FakeRedisState(),
        max_user_prompt_chars=MAX_USER_PROMPT_CHARS,
        max_chunk_chars=MAX_CHUNK_CHARS,
        max_chunks_per_request=MAX_CHUNKS_PER_REQUEST,
    )

    message = _prompt_message()
    asyncio.run(orchestrator.handle(message))

    assert len(answer_agent.calls) == 1
    assert answer_agent.calls[0][0] == AssistantMode.ANSWER_QUESTION
    assert answer_agent.calls[0][1] == "Что такое нормализация?"
    assert publisher.events[-1].type == OutgoingEventType.RESPONSE
    assert publisher.events[-1].data == "Ответ AnswerAgent"
