import json
import logging
from collections.abc import Mapping
from enum import Enum
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

from assistant_service.services.gemini_client import GeminiClientError

logger = logging.getLogger(__name__)

DecisionSource = Literal["gemini_json", "rule_based"]
MAX_KEYWORDS = 8
MAX_KEYWORD_CHARS = 140
INTENT_MAX_OUTPUT_TOKENS = 256
INTENT_TEMPERATURE = 0.0

INTENT_SYSTEM_INSTRUCTION = """
Ты внутренний классификатор запросов университетского ассистента.
Не отвечай пользователю и не объясняй решение.
Верни только один JSON-объект без Markdown, комментариев и дополнительного текста.
Определи task_type только из:
answer_question, explain_topic, document_search, summarize_document, unsupported.
Верни keywords как JSON-массив коротких строк.
Не включай персональные данные, выдуманные факты, ответы на вопрос или инструкции пользователю.
Если запрос не относится к поиску, объяснению, ответу по университетским материалам или краткому изложению документа, выбери unsupported.
""".strip()

SUMMARIZE_KEYWORDS = (
    "кратко",
    "краткое содержание",
    "сделай краткое содержание",
    "суммируй",
    "суммаризируй",
    "подведи итог",
    "выдели главное",
    "основные мысли",
    "перескажи",
    "резюмируй",
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
    "найди лекцию",
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
QUESTION_PREFIXES = (
    "кто",
    "что",
    "когда",
    "где",
    "как",
    "почему",
    "зачем",
    "какой",
    "какая",
    "какие",
    "сколько",
)
UNSUPPORTED_KEYWORDS = (
    "напиши код",
    "создай сайт",
    "telegram-бот",
    "телеграм-бот",
    "инвестиционный совет",
    "медицинскую консультацию",
    "юридическую консультацию",
    "напиши диплом",
    "курсовую за меня",
    "сделай домашку",
)


class TextGenerator(Protocol):
    async def generate_text(
        self,
        *,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        ...


class IntentTaskType(str, Enum):
    ANSWER_QUESTION = "answer_question"
    EXPLAIN_TOPIC = "explain_topic"
    DOCUMENT_SEARCH = "document_search"
    SUMMARIZE_DOCUMENT = "summarize_document"
    UNSUPPORTED = "unsupported"


class IntentClassificationError(RuntimeError):
    """Raised when structured intent classification cannot be trusted."""


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: IntentTaskType
    requires_retrieval: bool
    requires_full_document: bool
    keywords: list[str]
    source: DecisionSource = "gemini_json"

    @field_validator("keywords", mode="before")
    @classmethod
    def _validate_keywords_shape(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("keywords must be a list")
        if not all(isinstance(keyword, str) for keyword in value):
            raise ValueError("keywords must contain only strings")
        return value

    @field_validator("keywords")
    @classmethod
    def _normalize_keywords(cls, value: list[str]) -> list[str]:
        normalized_keywords: list[str] = []
        seen_keywords: set[str] = set()

        for keyword in value:
            if not isinstance(keyword, str):
                raise ValueError("keywords must contain only strings")

            normalized_keyword = " ".join(keyword.strip().split())
            if normalized_keyword == "":
                continue
            if len(normalized_keyword) > MAX_KEYWORD_CHARS:
                raise ValueError("keyword is too long")

            dedupe_key = normalized_keyword.lower()
            if dedupe_key in seen_keywords:
                continue

            seen_keywords.add(dedupe_key)
            normalized_keywords.append(normalized_keyword)
            if len(normalized_keywords) >= MAX_KEYWORDS:
                break

        return normalized_keywords

    @model_validator(mode="after")
    def _normalize_flags_and_keywords(self) -> "IntentDecision":
        self.requires_retrieval = self.task_type != IntentTaskType.UNSUPPORTED
        self.requires_full_document = self.task_type == IntentTaskType.SUMMARIZE_DOCUMENT

        if self.task_type == IntentTaskType.UNSUPPORTED:
            self.keywords = []

        return self


class IntentAgent:
    def __init__(self, text_generator: TextGenerator | None = None) -> None:
        self._text_generator = text_generator

    async def detect(
        self,
        prompt: str,
        document_id: str | None,
    ) -> IntentDecision:
        if self._text_generator is None:
            return self.detect_rule_based(prompt=prompt, document_id=document_id)

        try:
            raw_response = await self._text_generator.generate_text(
                system_instruction=INTENT_SYSTEM_INSTRUCTION,
                prompt=self._build_classification_prompt(
                    prompt=prompt,
                    document_id=document_id,
                ),
                max_output_tokens=INTENT_MAX_OUTPUT_TOKENS,
                temperature=INTENT_TEMPERATURE,
            )
            return self.parse_classification_response(
                raw_response,
                document_id=document_id,
            )
        except (GeminiClientError, IntentClassificationError) as exc:
            logger.warning(
                "Intent classification fallback selected: error_type=%s",
                type(exc).__name__,
            )
            return self.detect_rule_based(prompt=prompt, document_id=document_id)

    def detect_rule_based(
        self,
        prompt: str,
        document_id: str | None,
    ) -> IntentDecision:
        normalized_prompt = self._normalize_prompt(prompt)
        has_document = document_id is not None

        if self._contains_keyword(normalized_prompt, UNSUPPORTED_KEYWORDS):
            return self._build_decision(
                task_type=IntentTaskType.UNSUPPORTED,
                keywords=[],
                source="rule_based",
            )

        if has_document and (
            normalized_prompt == ""
            or self._contains_keyword(normalized_prompt, SUMMARIZE_KEYWORDS)
        ):
            return self._build_decision(
                task_type=IntentTaskType.SUMMARIZE_DOCUMENT,
                keywords=[],
                source="rule_based",
            )

        if self._contains_keyword(normalized_prompt, SUMMARIZE_KEYWORDS):
            return self._build_decision(
                task_type=IntentTaskType.SUMMARIZE_DOCUMENT,
                keywords=self._fallback_keywords(normalized_prompt),
                source="rule_based",
            )

        if self._contains_keyword(normalized_prompt, DOCUMENT_SEARCH_KEYWORDS):
            return self._build_decision(
                task_type=IntentTaskType.DOCUMENT_SEARCH,
                keywords=self._fallback_keywords(normalized_prompt),
                source="rule_based",
            )

        if self._contains_keyword(normalized_prompt, EXPLAIN_KEYWORDS):
            return self._build_decision(
                task_type=IntentTaskType.EXPLAIN_TOPIC,
                keywords=self._fallback_keywords(normalized_prompt),
                source="rule_based",
            )

        if normalized_prompt.endswith("?") or normalized_prompt.startswith(
            QUESTION_PREFIXES
        ):
            return self._build_decision(
                task_type=IntentTaskType.ANSWER_QUESTION,
                keywords=self._fallback_keywords(normalized_prompt),
                source="rule_based",
            )

        return self._build_decision(
            task_type=IntentTaskType.UNSUPPORTED,
            keywords=[],
            source="rule_based",
        )

    def parse_classification_response(
        self,
        raw_response: str,
        document_id: str | None,
    ) -> IntentDecision:
        try:
            payload = json.loads(self._normalize_json_response(raw_response))
        except json.JSONDecodeError as exc:
            raise IntentClassificationError(
                "Intent classifier returned invalid JSON."
            ) from exc

        if not isinstance(payload, Mapping):
            raise IntentClassificationError(
                "Intent classifier response must be an object."
            )

        # Gemini is only ever asked for `task_type` and `keywords` (see
        # INTENT_SYSTEM_INSTRUCTION); `requires_retrieval`/`requires_full_document`
        # are derived from task_type, same as the rule-based path — they are not
        # part of the model's JSON contract and must not be required from it here.
        try:
            task_type = IntentTaskType(payload.get("task_type"))
        except ValueError as exc:
            raise IntentClassificationError(
                "Intent classifier returned invalid task_type."
            ) from exc

        keywords = payload.get("keywords")
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) for keyword in keywords
        ):
            raise IntentClassificationError(
                "Intent classifier returned invalid keywords."
            )

        try:
            decision = self._build_decision(
                task_type=task_type,
                keywords=keywords,
                source="gemini_json",
            )
        except ValidationError as exc:
            raise IntentClassificationError(
                "Intent classifier response failed validation."
            ) from exc

        if (
            self._keywords_required(decision.task_type, document_id)
            and not decision.keywords
        ):
            raise IntentClassificationError("Intent classifier returned empty keywords.")

        return decision

    @staticmethod
    def _build_classification_prompt(
        *,
        prompt: str,
        document_id: str | None,
    ) -> str:
        document_id_state = "present" if document_id is not None else "absent"
        return (
            "КЛАССИФИЦИРУЙ ЗАПРОС.\n"
            f"document_id: {document_id_state}\n"
            "prompt:\n"
            f"{prompt}"
        )

    @staticmethod
    def _build_decision(
        *,
        task_type: IntentTaskType,
        keywords: list[str],
        source: DecisionSource,
    ) -> IntentDecision:
        return IntentDecision(
            task_type=task_type,
            requires_retrieval=task_type != IntentTaskType.UNSUPPORTED,
            requires_full_document=task_type == IntentTaskType.SUMMARIZE_DOCUMENT,
            keywords=keywords,
            source=source,
        )

    @staticmethod
    def _normalize_json_response(raw_response: str) -> str:
        normalized_response = raw_response.strip()
        if normalized_response.startswith("```json") and normalized_response.endswith(
            "```"
        ):
            return (
                normalized_response.removeprefix("```json")
                .removesuffix("```")
                .strip()
            )
        if normalized_response.startswith("```") and normalized_response.endswith("```"):
            return normalized_response.removeprefix("```").removesuffix("```").strip()
        return normalized_response

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        return " ".join(prompt.strip().lower().split())

    @staticmethod
    def _contains_keyword(prompt: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in prompt for keyword in keywords)

    @staticmethod
    def _fallback_keywords(normalized_prompt: str) -> list[str]:
        if normalized_prompt == "":
            return []
        return [normalized_prompt[:MAX_KEYWORD_CHARS]]

    @staticmethod
    def _keywords_required(
        task_type: IntentTaskType,
        document_id: str | None,
    ) -> bool:
        return task_type not in {
            IntentTaskType.UNSUPPORTED,
            IntentTaskType.SUMMARIZE_DOCUMENT,
        } or (
            task_type == IntentTaskType.SUMMARIZE_DOCUMENT and document_id is None
        )
