import asyncio

import pytest

from assistant_service.agents.intent_agent import (
    IntentAgent,
    IntentClassificationError,
    IntentTaskType,
)
from assistant_service.services.gemini_client import GeminiTransientError


DOCUMENT_UID = "document-uid-123"


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
    generator = FakeTextGenerator(
        """
        {
          "task_type": "explain_topic",
          "requires_retrieval": false,
          "requires_full_document": true,
          "keywords": [" нормальные формы ", "нормализация баз данных"]
        }
        """
    )
    agent = IntentAgent(text_generator=generator)

    decision = asyncio.run(
        agent.detect(prompt="Объясни нормальные формы баз данных", uid=None)
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


def test_keywords_are_trimmed_deduplicated_and_invalid_shape_is_rejected() -> None:
    agent = IntentAgent()
    decision = agent.parse_classification_response(
        """
        ```json
        {
          "task_type": "answer_question",
          "requires_retrieval": true,
          "requires_full_document": false,
          "keywords": [" SQL ", "sql", "СУБД"]
        }
        ```
        """,
        uid=None,
    )

    assert decision.keywords == ["SQL", "СУБД"]

    with pytest.raises(IntentClassificationError):
        agent.parse_classification_response(
            """
            {
              "task_type": "answer_question",
              "requires_retrieval": true,
              "requires_full_document": false,
              "keywords": "SQL, СУБД"
            }
            """,
            uid=None,
        )


def test_invalid_json_or_gemini_error_uses_rule_based_fallback() -> None:
    invalid_json_agent = IntentAgent(text_generator=FakeTextGenerator("not json"))
    gemini_error_agent = IntentAgent(
        text_generator=FakeTextGenerator(GeminiTransientError("temporary"))
    )

    invalid_json_decision = asyncio.run(
        invalid_json_agent.detect(prompt="Объясни нормализацию", uid=None)
    )
    gemini_error_decision = asyncio.run(
        gemini_error_agent.detect(prompt="Найди лекцию про SQL", uid=None)
    )

    assert invalid_json_decision.task_type == IntentTaskType.EXPLAIN_TOPIC
    assert invalid_json_decision.source == "rule_based"
    assert gemini_error_decision.task_type == IntentTaskType.DOCUMENT_SEARCH
    assert gemini_error_decision.source == "rule_based"


def test_rule_based_fallback_can_return_unsupported() -> None:
    decision = asyncio.run(
        IntentAgent().detect(prompt="Создай сайт интернет-магазина", uid=None)
    )

    assert decision.task_type == IntentTaskType.UNSUPPORTED
    assert decision.requires_retrieval is False
    assert decision.requires_full_document is False
    assert decision.keywords == []
