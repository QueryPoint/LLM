from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from assistant_service.core.enums import (
    AssistantMode,
    IncomingMessageType,
    OutgoingEventType,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.PROMPT]
    user_id: UUID
    prompt: str | None = None
    doc: UUID | None = None
    mode: AssistantMode | None = None


class DeleteRequestMessage(StrictBaseModel):
    type: Literal[IncomingMessageType.DELETE]
    user_id: UUID


IncomingMessage = Annotated[
    PromptRequestMessage | DeleteRequestMessage,
    Field(discriminator="type"),
]


class ThinkEvent(StrictBaseModel):
    type: Literal[OutgoingEventType.THINK]
    user_id: UUID
    data: str = Field(min_length=1)


class ResponseEvent(StrictBaseModel):
    type: Literal[OutgoingEventType.RESPONSE]
    user_id: UUID
    data: str = Field(min_length=1)
    warning: int = Field(default=0, ge=0, le=100)


OutgoingEvent = Annotated[
    ThinkEvent | ResponseEvent,
    Field(discriminator="type"),
]


incoming_message_adapter = TypeAdapter(IncomingMessage)
outgoing_event_adapter = TypeAdapter(OutgoingEvent)
