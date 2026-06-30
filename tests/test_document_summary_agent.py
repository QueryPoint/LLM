import asyncio
from uuid import UUID

from assistant_service.agents.context_agent import ContextDecision
from assistant_service.agents.document_summary_agent import (
    FINAL_SUMMARY_MAX_OUTPUT_TOKENS,
    MAP_MAX_OUTPUT_TOKENS,
    DocumentSummaryAgent,
)
from assistant_service.core.enums import RetrievalStatus
from assistant_service.messaging.contracts import RetrievedChunk

DOCUMENT_ID = "00000000-0000-0000-0000-000000000004"
CHUNK_ID_1 = "00000000-0000-0000-0000-000000000101"
CHUNK_ID_2 = "00000000-0000-0000-0000-000000000102"
CHUNK_ID_3 = "00000000-0000-0000-0000-000000000103"


class FakeTextGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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

        if "промежуточное краткое изложение" in system_instruction:
            return f"Map summary {len(self.calls)}"
        return f"Final summary {len(self.calls)}"


def _chunk(chunk_id: str, text: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=UUID(chunk_id),
        document_id=UUID(DOCUMENT_ID),
        file_name="lecture.pdf",
        page=1,
        text=text,
        score=score,
    )


def _context(chunks: tuple[RetrievedChunk, ...]) -> ContextDecision:
    return ContextDecision(
        status=RetrievalStatus.FOUND,
        chunks=chunks,
        sources=(),
        total_chars=sum(len(chunk.text) for chunk in chunks),
    )


def test_small_document_uses_single_direct_summary_call() -> None:
    generator = FakeTextGenerator()
    agent = DocumentSummaryAgent(text_generator=generator)

    response = asyncio.run(
        agent.summarize(
            user_prompt="Сделай краткое изложение",
            context_decision=_context((_chunk(CHUNK_ID_1, "Small document text"),)),
        )
    )

    assert response == "Final summary 1"
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["max_output_tokens"] == FINAL_SUMMARY_MAX_OUTPUT_TOKENS
    prompt = str(call["prompt"])
    assert "Small document text" in prompt
    assert CHUNK_ID_1 not in prompt
    assert DOCUMENT_ID not in prompt
    assert "chunk_id" not in prompt
    assert "document_id" not in prompt
    assert "score" not in prompt


def test_large_document_uses_map_reduce_in_original_group_order() -> None:
    generator = FakeTextGenerator()
    agent = DocumentSummaryAgent(text_generator=generator)
    chunks = (
        _chunk(CHUNK_ID_1, "A" * 6_000),
        _chunk(CHUNK_ID_2, "B" * 6_000),
        _chunk(CHUNK_ID_3, "C" * 6_000),
    )

    response = asyncio.run(
        agent.summarize(
            user_prompt=None,
            context_decision=_context(chunks),
        )
    )

    assert response == "Final summary 4"
    assert len(generator.calls) == 4
    assert [call["max_output_tokens"] for call in generator.calls[:3]] == [
        MAP_MAX_OUTPUT_TOKENS,
        MAP_MAX_OUTPUT_TOKENS,
        MAP_MAX_OUTPUT_TOKENS,
    ]
    assert generator.calls[3]["max_output_tokens"] == FINAL_SUMMARY_MAX_OUTPUT_TOKENS
    assert "A" * 20 in str(generator.calls[0]["prompt"])
    assert "B" * 20 in str(generator.calls[1]["prompt"])
    assert "C" * 20 in str(generator.calls[2]["prompt"])
    assert "Map summary 1" in str(generator.calls[3]["prompt"])
    assert "Map summary 2" in str(generator.calls[3]["prompt"])
    assert "Map summary 3" in str(generator.calls[3]["prompt"])
