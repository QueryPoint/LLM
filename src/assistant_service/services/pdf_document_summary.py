from typing import Protocol


PDF_SUMMARY_PROMPT = """
Составь краткое структурированное изложение документа:
1. тема;
2. ключевые тезисы;
3. основные термины;
4. итог.
Не добавляй факты, которых нет в документе.
""".strip()


class PdfSummaryTextGenerator(Protocol):
    async def summarize_pdf_document(
        self,
        *,
        data: bytes,
        prompt: str,
        max_output_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> str:
        ...


class PdfDocumentSummaryService:
    def __init__(self, *, text_generator: PdfSummaryTextGenerator) -> None:
        self._text_generator = text_generator

    async def summarize_pdf(self, *, pdf_bytes: bytes) -> str:
        return await self._text_generator.summarize_pdf_document(
            data=pdf_bytes,
            prompt=PDF_SUMMARY_PROMPT,
            max_output_tokens=1200,
            temperature=0.1,
        )
