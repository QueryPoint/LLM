import asyncio

import pytest

from assistant_service.agents.answer_agent import AnswerAgent
from assistant_service.agents.context_agent import ContextChunk, ContextDecision
from assistant_service.core.enums import (
    AssistantMode,
    RetrievalStatus,
)
from assistant_service.services.gemini_client import GeminiClientError


DOCUMENT_ID = "00000000-0000-0000-0000-000000000003"
CHUNK_ID = "00000000-0000-0000-0000-000000000004"


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


def _chunk() -> ContextChunk:
    return ContextChunk(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        file_name="lecture.pdf",
        page=5,
        text="Нормализация уменьшает избыточность данных.",
        score=12.5,
    )


def _context_decision(
    *,
    status: RetrievalStatus = RetrievalStatus.FOUND,
    chunks: tuple[ContextChunk, ...] | None = None,
) -> ContextDecision:
    selected_chunks = (_chunk(),) if chunks is None else chunks
    return ContextDecision(
        status=status,
        chunks=selected_chunks,
        sources=(),
        total_chars=sum(len(chunk.text) for chunk in selected_chunks),
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
