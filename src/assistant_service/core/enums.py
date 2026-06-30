from enum import Enum


class IncomingMessageType(str, Enum):
    PROMPT = "prompt"
    DELETE = "delete"


class OutgoingEventType(str, Enum):
    THINK = "think"
    RESPONSE = "response"


class LLMStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    THINKING = "THINKING"
    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    GENERATING = "GENERATING"
