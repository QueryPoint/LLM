from dataclasses import dataclass
from typing import Literal

from assistant_service.core.enums import AssistantMode

DecisionSource = Literal["rule_based"]

SUMMARIZE_KEYWORDS = (
    "кратко",
    "краткое содержание",
    "сделай краткое содержание",
    "суммируй",
    "суммаризируй",
    "подведи итог",
    "выдели главное",
    "основные мысли",
)
DOCUMENT_SEARCH_KEYWORDS = (
    "найди документ",
    "найди файл",
    "покажи документ",
    "покажи файл",
    "покажи конспект",
    "где находится",
    "в каком документе",
    "в каком файле",
    "поиск по документам",
    "найди материалы",
    "найди конспект",
)
EXPLAIN_KEYWORDS = (
    "объясни",
    "объясните",
    "расскажи",
    "расскажите",
    "что такое",
    "как работает",
    "почему",
    "в чем разница",
    "в чём разница",
    "разбери",
    "поясни",
    "поясните",
)


@dataclass(frozen=True, slots=True)
class IntentDecision:
    mode: AssistantMode
    source: DecisionSource


class IntentAgent:
    def detect(
        self,
        prompt: str,
        uid: str | None,
    ) -> IntentDecision:
        normalized_prompt = self._normalize_prompt(prompt)
        has_document = uid is not None

        if has_document and (
            normalized_prompt == ""
            or self._contains_keyword(normalized_prompt, SUMMARIZE_KEYWORDS)
        ):
            return IntentDecision(
                mode=AssistantMode.SUMMARIZE_DOCUMENT,
                source="rule_based",
            )

        if self._contains_keyword(normalized_prompt, DOCUMENT_SEARCH_KEYWORDS):
            return IntentDecision(
                mode=AssistantMode.DOCUMENT_SEARCH,
                source="rule_based",
            )

        if self._contains_keyword(normalized_prompt, EXPLAIN_KEYWORDS):
            return IntentDecision(
                mode=AssistantMode.EXPLAIN_TOPIC,
                source="rule_based",
            )

        return IntentDecision(
            mode=AssistantMode.ANSWER_QUESTION,
            source="rule_based",
        )

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return " ".join(prompt.strip().lower().split())

    @staticmethod
    def _contains_keyword(prompt: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in prompt for keyword in keywords)
