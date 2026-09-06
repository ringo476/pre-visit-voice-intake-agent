import io

import pytest
from reportlab.pdfgen import canvas

from app.documents.document_ingest import UnsupportedDocumentTypeError, extract_text


def make_text_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(300, 150))
    c.drawString(20, 100, text)
    c.save()
    return buf.getvalue()


def make_blank_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(300, 150))
    c.showPage()
    c.save()
    return buf.getvalue()


def test_extracts_real_embedded_text_from_a_text_pdf_with_no_network_call():
    pdf = make_text_pdf("Amoxicillin 500mg twice daily")
    result = extract_text(pdf, "application/pdf")
    assert result.method == "pdf_text"
    assert "Amoxicillin 500mg twice daily" in result.text


def test_reports_pdf_text_empty_for_a_pdf_with_no_embedded_text():
    pdf = make_blank_pdf()
    result = extract_text(pdf, "application/pdf")
    assert result.method == "pdf_text_empty"
    assert result.text == ""


def test_routes_images_to_the_injected_ocr_function():
    result = extract_text(b"fake-image-bytes", "image/jpeg", ocr=lambda data: "Ibuprofen 200mg")
    assert result.method == "vision_ocr"
    assert result.text == "Ibuprofen 200mg"


@pytest.mark.parametrize("mime_type", ["image/png", "image/webp"])
def test_supports_png_and_webp_through_the_same_ocr_path(mime_type):
    result = extract_text(b"x", mime_type, ocr=lambda data: "extracted")
    assert result.method == "vision_ocr"


def test_rejects_an_unsupported_document_type():
    with pytest.raises(UnsupportedDocumentTypeError):
        extract_text(b"x", "text/plain")
