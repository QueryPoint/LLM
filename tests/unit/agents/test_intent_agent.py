import asyncio

import pytest

from assistant_service.agents.intent_agent import (
    IntentAgent,
    IntentClassificationError,
    IntentTaskType,
)
from assistant_service.services.gemini_client import GeminiTransientError


class FakeTextGenerator:
    def __init__(self, response: str | Exception) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response

    async def generate_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            }
        )
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_gemini_json_response_becomes_valid_intent_decision() -> None:
    # Реальный контракт: Gemini присылает только task_type и keywords (см.
    # INTENT_SYSTEM_INSTRUCTION) — requires_retrieval/requires_full_document
    # вычисляются из task_type в _build_decision, модель их не возвращает.
    generator = FakeTextGenerator(
        """
        {
          "task_type": "explain_topic",
          "keywords": [" нормальные формы ", "нормализация баз данных"]
        }
        """
    )
    agent = IntentAgent(text_generator=generator)

    decision = asyncio.run(
        agent.detect(prompt="Объясни нормальные формы баз данных", document_id=None)
    )

    assert decision.task_type == IntentTaskType.EXPLAIN_TOPIC
    assert decision.requires_retrieval is True
    assert decision.requires_full_document is False
    assert decision.keywords == ["нормальные формы", "нормализация баз данных"]
    assert decision.source == "gemini_json"
    assert len(generator.calls) == 1
    assert "Верни только один JSON-объект" in str(
        generator.calls[0]["system_instruction"]
    )
    assert "document_id: absent" in str(generator.calls[0]["prompt"])


def test_keywords_are_trimmed_deduplicated_and_invalid_shape_is_rejected() -> None:
    # Регрессионный тест: реальный Gemini часто оборачивает JSON в
    # ```json ... ``` и никогда не возвращает requires_retrieval/
    # requires_full_document — раньше это валило AI-классификацию в 100%
    # случаев и всегда откатывалось на rule_based, никто не замечал.
    agent = IntentAgent()
    decision = agent.parse_classification_response(
        """
        ```json
        {
          "task_type": "answer_question",
          "keywords": [" SQL ", "sql", "СУБД"]
        }
        ```
        """,
        document_id=None,
    )

    assert decision.task_type == IntentTaskType.ANSWER_QUESTION
    assert decision.requires_retrieval is True
    assert decision.requires_full_document is False
    assert decision.source == "gemini_json"
    assert decision.keywords == ["SQL", "СУБД"]

    with pytest.raises(IntentClassificationError):
        agent.parse_classification_response(
            """
            {
              "task_type": "answer_question",
              "keywords": "SQL, СУБД"
            }
            """,
            document_id=None,
        )


def test_gemini_response_missing_or_invalid_task_type_is_rejected() -> None:
    agent = IntentAgent()

    with pytest.raises(IntentClassificationError):
        agent.parse_classification_response('{"keywords": ["SQL"]}', document_id=None)

    with pytest.raises(IntentClassificationError):
        agent.parse_classification_response(
            '{"task_type": "not_a_real_type", "keywords": ["SQL"]}',
            document_id=None,
        )


def test_invalid_json_or_gemini_error_uses_rule_based_fallback() -> None:
    invalid_json_agent = IntentAgent(text_generator=FakeTextGenerator("not json"))
    gemini_error_agent = IntentAgent(
        text_generator=FakeTextGenerator(GeminiTransientError("temporary"))
    )

    invalid_json_decision = asyncio.run(
        invalid_json_agent.detect(prompt="Объясни нормализацию", document_id=None)
    )
    gemini_error_decision = asyncio.run(
        gemini_error_agent.detect(prompt="Найди лекцию про SQL", document_id=None)
    )

    assert invalid_json_decision.task_type == IntentTaskType.EXPLAIN_TOPIC
    assert invalid_json_decision.source == "rule_based"
    assert gemini_error_decision.task_type == IntentTaskType.DOCUMENT_SEARCH
    assert gemini_error_decision.source == "rule_based"


def test_rule_based_fallback_can_return_unsupported() -> None:
    decision = asyncio.run(
        IntentAgent().detect(prompt="Создай сайт интернет-магазина", document_id=None)
    )

    assert decision.task_type == IntentTaskType.UNSUPPORTED
    assert decision.requires_retrieval is False
    assert decision.requires_full_document is False
    assert decision.keywords == []
