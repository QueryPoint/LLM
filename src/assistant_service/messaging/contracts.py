from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from assistant_service.core.enums import IncomingMessageType, OutgoingEventType


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.PROMPT]
    user_id: UUID
    prompt: str | None = None
    doc: UUID | None = None


class DeleteRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.DELETE]
    user_id: UUID


IncomingMessage = Annotated[
    PromptRequestMessage | DeleteRequestMessage,
    Field(discriminator="type"),
]


class SyncEvent(StrictBaseModel):
    type: Literal[OutgoingEventType.SYNC]
    user_id: UUID
    data: str


class ResponseEvent(StrictBaseModel):
    type: Literal[OutgoingEventType.RESPONSE]
    user_id: UUID
    data: str
    warning: int = Field(default=0, ge=0, le=100)


OutgoingEvent = Annotated[
    SyncEvent | ResponseEvent,
    Field(discriminator="type"),
]


incoming_message_adapter = TypeAdapter(IncomingMessage)
outgoing_event_adapter = TypeAdapter(OutgoingEvent)
