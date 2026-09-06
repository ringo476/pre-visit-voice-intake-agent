import os
import tempfile

from app.rag.embeddings import HashingEmbeddings
from app.rag.retriever import build_index, retrieve, retrieve_follow_up_guidance, retrieve_prior_chart

# Ensure the hashing fallback is used regardless of the host environment.
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

RESPIRATORY = "respiratory-intake"
LEG_INJURY = "musculoskeletal-leg-injury"


def test_build_index_and_retrieve_ranks_most_relevant_first():
    store = build_index(
        texts=[
            "the patient has a history of asthma and uses an inhaler",
            "the patient smokes half a pack a day",
            "no known drug allergies on file",
        ],
        metadatas=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
        collection_name="test-generic",
        persist_directory=tempfile.mkdtemp(),
        embeddings=HashingEmbeddings(),
    )
    results = retrieve(store, "does the patient have an inhaler for asthma", k=1)
    assert results[0][0].metadata["id"] == "a"


def test_retrieve_prior_chart_surfaces_inhaler_note():
    results = retrieve_prior_chart(RESPIRATORY, "does the patient have a prior inhaler prescription", k=1)
    assert results[0][0].metadata["id"] == "chart_1"


def test_retrieve_prior_chart_surfaces_allergy_note():
    results = retrieve_prior_chart(RESPIRATORY, "any known medication allergies documented", k=1)
    assert results[0][0].metadata["id"] == "chart_2"


def test_retrieve_follow_up_guidance_wheeze_inhaler():
    results = retrieve_follow_up_guidance(RESPIRATORY, "patient mentioned wheezing and an old inhaler", k=1)
    assert results[0][0].metadata["id"] == "guidance_wheeze_inhaler"


def test_retrieve_follow_up_guidance_allergy_reaction():
    results = retrieve_follow_up_guidance(RESPIRATORY, "patient says they have a medication allergy", k=1)
    assert results[0][0].metadata["id"] == "guidance_allergy_reaction"


def test_protocols_have_independent_rag_stores():
    """Proves the fix: a second protocol's reference data is isolated from the first's, not mixed together."""
    leg_results = retrieve_prior_chart(LEG_INJURY, "did the patient sprain this ankle before", k=1)
    assert leg_results[0][0].metadata["id"] == "chart_1"
    assert "ankle" in leg_results[0][0].page_content.lower()

    guidance_results = retrieve_follow_up_guidance(LEG_INJURY, "patient can't bear any weight on the leg", k=1)
    assert guidance_results[0][0].metadata["id"] == "guidance_weight_bearing"
