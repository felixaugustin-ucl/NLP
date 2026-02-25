from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


ROMAN_ANNEX_RE = re.compile(r"\bAnnex\s+([IVXLCDM]+)\b", re.IGNORECASE)
ARTICLE_RANGE_RE = re.compile(
    r"\bArticles\s+(\d+)\s*(?:to|-)\s*(\d+)\b",
    re.IGNORECASE,
)
ARTICLE_LIST_RE = re.compile(
    r"\bArticles\s+((?:\d+[A-Za-z]?(?:\(\d+\))?\s*,\s*)+\d+[A-Za-z]?(?:\(\d+\))?"
    r"(?:\s*,?\s*(?:and|or)\s+\d+[A-Za-z]?(?:\(\d+\))?)?)",
    re.IGNORECASE,
)
ARTICLE_SINGLE_RE = re.compile(
    r"\bArticle\s+(\d+[A-Za-z]?)(?:\((\d+)\))?(?!\w)",
    re.IGNORECASE,
)
ARTICLE_ID_RE = re.compile(r"\d+[A-Za-z]?(?:\(\d+\))?")


@dataclass(frozen=True)
class RefMatch:
    ref_type: str  # "ARTICLE" | "ANNEX"
    dst_node_ids: tuple[str, ...]
    evidence_span: str
    start: int
    end: int


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def _normalize_article_dst(article_token: str) -> str:
    # Ignore referenced paragraph in destination mapping; keep only article id.
    article_base = article_token.split("(")[0]
    return f"Article {article_base}"


def _expand_article_range(start_num: str, end_num: str) -> list[str]:
    a = int(start_num)
    b = int(end_num)
    if a <= b:
        return [f"Article {n}" for n in range(a, b + 1)]
    return [f"Article {n}" for n in range(a, b - 1, -1)]


def extract_explicit_references(text: str) -> list[RefMatch]:
    """
    Extract explicit legal cross-references and preserve exact matched spans.

    Parsing order is range -> list -> single -> annex to avoid overlapping double-counts.
    """
    refs: list[RefMatch] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in ARTICLE_RANGE_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied_spans):
            continue
        dsts = tuple(_expand_article_range(match.group(1), match.group(2)))
        refs.append(
            RefMatch(
                ref_type="ARTICLE",
                dst_node_ids=dsts,
                evidence_span=match.group(0),
                start=start,
                end=end,
            )
        )
        occupied_spans.append((start, end))

    for match in ARTICLE_LIST_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied_spans):
            continue
        tokens = ARTICLE_ID_RE.findall(match.group(1))
        dsts = tuple(_normalize_article_dst(tok) for tok in tokens)
        if not dsts:
            continue
        refs.append(
            RefMatch(
                ref_type="ARTICLE",
                dst_node_ids=dsts,
                evidence_span=match.group(0),
                start=start,
                end=end,
            )
        )
        occupied_spans.append((start, end))

    for match in ARTICLE_SINGLE_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied_spans):
            continue
        dst = _normalize_article_dst(match.group(1))
        refs.append(
            RefMatch(
                ref_type="ARTICLE",
                dst_node_ids=(dst,),
                evidence_span=match.group(0),
                start=start,
                end=end,
            )
        )
        occupied_spans.append((start, end))

    for match in ROMAN_ANNEX_RE.finditer(text):
        start, end = match.span()
        if _overlaps(start, end, occupied_spans):
            continue
        annex_id = f"ANNEX {match.group(1).upper()}"
        refs.append(
            RefMatch(
                ref_type="ANNEX",
                dst_node_ids=(annex_id,),
                evidence_span=match.group(0),
                start=start,
                end=end,
            )
        )

    refs.sort(key=lambda r: (r.start, r.end))
    return refs
