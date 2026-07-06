from enum import Enum


class IncomingMessageType(str, Enum):
    PROMPT = "prompt"
    DELETE = "delete"


class OutgoingEventType(str, Enum):
    THINK = "think"
    RESPONSE = "response"


class AssistantMode(str, Enum):
    ANSWER_QUESTION = "answer_question"
    DOCUMENT_SEARCH = "document_search"
    SUMMARIZE_DOCUMENT = "summarize_document"
    EXPLAIN_TOPIC = "explain_topic"


class RetrievalStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    INSUFFICIENT = "insufficient"


class LLMStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    THINKING = "THINKING"
    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    GENERATING = "GENERATING"
