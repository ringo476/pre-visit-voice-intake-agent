"""The document extraction "router" — a plain function, not a service:
decide by mime type whether to pull text straight out of a PDF (no network
call) or run OCR on an image. Deliberately not an LLM call: the agent must
never be the thing that both reads a document and decides what it says, or
its cited evidence stops being independently checkable (same reasoning as
keeping speech-to-text separate from the reasoning model).

`pdf_reader`/`ocr` are injectable so the router logic and the PDF-text path
are fully unit-testable with no credentials; only the real Vision call goes
untested locally, same as voice/stt_client.py and voice/tts_client.py.
"""

import io
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pypdf import PdfReader

SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

ExtractionMethod = Literal["pdf_text", "pdf_text_empty", "vision_ocr"]

PdfReaderFn = Callable[[bytes], str]
OcrFn = Callable[[bytes], str]


class UnsupportedDocumentTypeError(Exception):
    pass


@dataclass
class ExtractTextResult:
    text: str
    method: ExtractionMethod


_cached_vision_client = None


def _get_vision_client():
    global _cached_vision_client
    if _cached_vision_client is None:
        from google.cloud import vision

        _cached_vision_client = vision.ImageAnnotatorClient()
    return _cached_vision_client


def _default_pdf_reader(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _default_ocr(data: bytes) -> str:
    from google.cloud import vision

    client = _get_vision_client()
    image = vision.Image(content=data)
    response = client.document_text_detection(image=image)
    return response.full_text_annotation.text if response.full_text_annotation else ""


def extract_text(
    data: bytes,
    mime_type: str,
    pdf_reader: Optional[PdfReaderFn] = None,
    ocr: Optional[OcrFn] = None,
) -> ExtractTextResult:
    pdf_reader = pdf_reader or _default_pdf_reader
    ocr = ocr or _default_ocr

    if mime_type == "application/pdf":
        text = pdf_reader(data).strip()
        if not text:
            # Rasterizing a scanned PDF page-by-page for OCR is out of scope
            # for this MVP — report plainly rather than guessing.
            return ExtractTextResult(text="", method="pdf_text_empty")
        return ExtractTextResult(text=text, method="pdf_text")

    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        text = ocr(data).strip()
        return ExtractTextResult(text=text, method="vision_ocr")

    raise UnsupportedDocumentTypeError(f"Unsupported document type: {mime_type}")
