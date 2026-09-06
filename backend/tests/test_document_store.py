import os
import uuid

from app.documents.document_store import add_document, create_uploaded_document, retrieve_from_documents

os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)


def test_returns_no_results_when_nothing_uploaded():
    session_id = f"s-{uuid.uuid4()}"
    assert retrieve_from_documents(session_id, "anything") == []


def test_retrieves_the_most_relevant_uploaded_document():
    session_id = f"s-{uuid.uuid4()}"
    add_document(session_id, create_uploaded_document("prescription.pdf", "application/pdf", "Amoxicillin 500mg twice daily for 7 days"))
    add_document(session_id, create_uploaded_document("lab.pdf", "application/pdf", "White blood cell count within normal range"))

    results = retrieve_from_documents(session_id, "Amoxicillin dose and frequency", k=1)
    assert results[0][0].metadata["filename"] == "prescription.pdf"
    assert "Amoxicillin" in results[0][0].page_content
