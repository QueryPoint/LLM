from enum import Enum


class IncomingMessageType(str, Enum):
    PROMPT = "prompt"
    DELETE = "delete"


class OutgoingEventType(str, Enum):
    THINK = "think"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


class AssistantMode(str, Enum):
    ANSWER_QUESTION = "answer_question"
    DOCUMENT_SEARCH = "document_search"
    SUMMARIZE_DOCUMENT = "summarize_document"
    EXPLAIN_TOPIC = "explain_topic"


class RetrievalStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    INSUFFICIENT = "insufficient"


class ThinkStage(str, Enum):
    QUEUED = "QUEUED"
    RECEIVED = "RECEIVED"
    VALIDATING_REQUEST = "VALIDATING_REQUEST"
    DETECTING_INTENT = "DETECTING_INTENT"
    CHECKING_CONTEXT = "CHECKING_CONTEXT"
    PREPARING_CONTEXT = "PREPARING_CONTEXT"
    CONTEXT_READY = "CONTEXT_READY"
    GENERATING = "GENERATING"
    STREAMING = "STREAMING"
    FINALIZING = "FINALIZING"
    NO_CONTEXT = "NO_CONTEXT"
    CLEARING_SESSION = "CLEARING_SESSION"
