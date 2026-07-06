from collections.abc import Sequence
from dataclasses import dataclass

from assistant_service.core.enums import AssistantMode, RetrievalStatus
from assistant_service.services.elasticsearch_client import SearchResult


@dataclass(frozen=True, slots=True)
class ContextSource:
    chunk_id: str
    document_id: str
    file_name: str
    page: int


@dataclass(frozen=True, slots=True)
class ContextChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page: int
    text: str
    score: float
    highlights: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextDecision:
    status: RetrievalStatus
    chunks: tuple[ContextChunk, ...]
    sources: tuple[ContextSource, ...]
    total_chars: int


@dataclass(frozen=True, slots=True)
class _RankedChunk:
    index: int
    chunk: ContextChunk


class ContextAgent:
    def __init__(self, max_context_chars: int = 12_000, max_chunks: int = 8) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        self._max_context_chars = max_context_chars
        self._max_chunks = max_chunks

    def prepare(
        self,
        search_results: Sequence[SearchResult] | None,
        mode: AssistantMode | None = None,
    ) -> ContextDecision:
        if search_results is None:
            return self._empty_decision(RetrievalStatus.NOT_FOUND)

        ranked_chunks = self._deduplicate(search_results)
        selected_chunks = self._select_chunks(ranked_chunks)
        sources = tuple(self._build_source(chunk) for chunk in selected_chunks)
        total_chars = sum(len(chunk.text) for chunk in selected_chunks)

        status = RetrievalStatus.FOUND if selected_chunks else RetrievalStatus.NOT_FOUND

        return ContextDecision(
            status=status,
            chunks=tuple(selected_chunks),
            sources=sources,
            total_chars=total_chars,
        )

    @staticmethod
    def _empty_decision(status: RetrievalStatus) -> ContextDecision:
        return ContextDecision(status=status, chunks=(), sources=(), total_chars=0)

    def _deduplicate(self, search_results: Sequence[SearchResult]) -> list[_RankedChunk]:
        ranked_chunks: list[_RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()

        for index, result in enumerate(search_results):
            chunk = self._build_chunk(result)
            if chunk is None:
                continue

            normalized_text = self._normalize_text(chunk.text)
            if chunk.chunk_id in seen_chunk_ids or normalized_text in seen_texts:
                continue

            ranked_chunks.append(_RankedChunk(index=index, chunk=chunk))
            seen_chunk_ids.add(chunk.chunk_id)
            seen_texts.add(normalized_text)

        return ranked_chunks

    @staticmethod
    def _build_chunk(result: SearchResult) -> ContextChunk | None:
        normalized_text = result.text.strip()
        if normalized_text == "":
            return None

        return ContextChunk(
            chunk_id=result.chunk_id,
            document_id=result.doc_id,
            file_name=result.file_name,
            page=result.page_number,
            text=normalized_text,
            score=result.score,
            highlights=result.highlights,
        )

    def _select_chunks(self, ranked_chunks: list[_RankedChunk]) -> list[ContextChunk]:
        selected_chunks: list[ContextChunk] = []
        remaining_chars = self._max_context_chars

        for ranked_chunk in ranked_chunks:
            if len(selected_chunks) >= self._max_chunks:
                break

            chunk = ranked_chunk.chunk
            text_length = len(chunk.text)
            if text_length > remaining_chars:
                continue

            selected_chunks.append(chunk)
            remaining_chars -= text_length

        return selected_chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _build_source(chunk: ContextChunk) -> ContextSource:
        return ContextSource(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            page=chunk.page,
        )
