from enum import Enum


class IncomingMessageType(str, Enum):
    PROMPT = "prompt"
    DELETE = "delete"


class OutgoingEventType(str, Enum):
    SYNC = "sync"
    RESPONSE = "response"
