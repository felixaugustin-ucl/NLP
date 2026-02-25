from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_refs import extract_explicit_references


def flatten_article_dsts(refs):
    dsts = []
    for ref in refs:
        if ref.ref_type == "ARTICLE":
            dsts.extend(ref.dst_node_ids)
    return dsts


def test_single_article_with_paragraph_reference():
    text = "This applies in accordance with Article 6(1) and subject to safeguards."
    refs = extract_explicit_references(text)
    assert any(r.evidence_span == "Article 6(1)" for r in refs)
    assert "Article 6" in flatten_article_dsts(refs)


def test_mixed_single_range_single():
    text = "only Article 6(1), Articles 102 to 109 and Article 112 apply. Article 57 applies ..."
    refs = extract_explicit_references(text)
    spans = [r.evidence_span for r in refs]
    assert "Article 6(1)" in spans
    assert "Articles 102 to 109" in spans
    assert "Article 112" in spans
    assert "Article 57" in spans
    article_dsts = set(flatten_article_dsts(refs))
    for n in range(102, 110):
        assert f"Article {n}" in article_dsts


def test_article_list_extraction():
    text = "See Articles 5, 10 and 12 before proceeding."
    refs = extract_explicit_references(text)
    assert any(r.evidence_span == "Articles 5, 10 and 12" for r in refs)
    assert set(flatten_article_dsts(refs)) == {"Article 5", "Article 10", "Article 12"}


def test_annex_reference_extraction():
    text = "products covered by the Union harmonisation legislation listed in Section B of Annex I."
    refs = extract_explicit_references(text)
    annex_refs = [r for r in refs if r.ref_type == "ANNEX"]
    assert len(annex_refs) == 1
    assert annex_refs[0].evidence_span == "Annex I"
    assert annex_refs[0].dst_node_ids == ("ANNEX I",)
