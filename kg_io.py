from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd


WS_RE = re.compile(r"\s+")
ARTICLE_HEADING_RE = re.compile(r"^Article\s+(\d+[A-Za-z]?)$")
ANNEX_HEADING_RE = re.compile(r"^ANNEX\s+([IVXLCDM]+)$")
PARAGRAPH_RE = re.compile(r"^(?:\((\d+)\)|(\d+)\.)\s*(.*)$")
PAGE_ARTIFACT_RE = re.compile(r"^\d+/\d+$")

DEFAULT_SOURCE_TEXT_NAME = "EU_AI_ACT.txt"
SOURCE_ARTIFACT_LINES = {
    "Official Journal",
    "of the European Union",
    "EN",
    "L series",
    "OJ L, 12.7.2024",
}


@dataclass(frozen=True)
class SourceRow:
    seq: int
    kind: str
    locator: str
    text: str


def normalize_ws(text: str) -> str:
    return WS_RE.sub(" ", (text or "")).strip()


def _clean_source_line(line: str) -> str:
    line = line.replace("\x0c", "").replace("`", "").strip()
    return normalize_ws(line)


def _is_source_artifact(line: str) -> bool:
    return (
        not line
        or line in SOURCE_ARTIFACT_LINES
        or line == "2024/1689"
        or line == "12.7.2024"
        or line.startswith("ELI: http://data.europa.eu/eli/")
        or PAGE_ARTIFACT_RE.match(line) is not None
    )


def _read_source_lines(text_path: str | Path) -> list[str]:
    raw_lines = Path(text_path).read_text(encoding="utf-8").splitlines()
    return [line for raw in raw_lines if not _is_source_artifact(line := _clean_source_line(raw))]


def _consume_section(lines: list[str], start: int) -> tuple[int, list[str]]:
    idx = start
    chunk: list[str] = []
    while idx < len(lines):
        line = lines[idx]
        if ARTICLE_HEADING_RE.match(line) or ANNEX_HEADING_RE.match(line):
            break
        chunk.append(line)
        idx += 1
    return idx, chunk


def _parse_article(lines: list[str], start: int, seq_start: int) -> tuple[int, list[SourceRow], int]:
    heading = lines[start]
    match = ARTICLE_HEADING_RE.match(heading)
    if match is None:
        raise ValueError(f"Expected article heading at line {start}: {heading!r}")

    article_id = f"Article {match.group(1)}"
    idx = start + 1
    title = ""
    if idx < len(lines) and not ARTICLE_HEADING_RE.match(lines[idx]) and not ANNEX_HEADING_RE.match(lines[idx]):
        title = lines[idx]
        idx += 1

    idx, content_lines = _consume_section(lines, idx)

    rows = [SourceRow(seq=seq_start, kind="ARTICLE", locator=article_id, text=title)]
    paragraph_number: str | None = None
    paragraph_parts: list[str] = []
    intro_parts: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_number, paragraph_parts
        if paragraph_number is None:
            return
        text = normalize_ws(" ".join(paragraph_parts))
        if text:
            rows.append(
                SourceRow(
                    seq=seq_start + len(rows),
                    kind="PARAGRAPH",
                    locator=f"{article_id}({paragraph_number})",
                    text=text,
                )
            )
        paragraph_number = None
        paragraph_parts = []

    for line in content_lines:
        paragraph_match = PARAGRAPH_RE.match(line)
        if paragraph_match:
            flush_paragraph()
            paragraph_number = paragraph_match.group(1) or paragraph_match.group(2)
            remainder = paragraph_match.group(3).strip()
            paragraph_parts = [remainder] if remainder else []
            continue
        if paragraph_number is None:
            intro_parts.append(line)
        else:
            paragraph_parts.append(line)

    flush_paragraph()
    article_text = normalize_ws(" ".join(part for part in [title, *intro_parts] if part))
    rows[0] = SourceRow(seq=seq_start, kind="ARTICLE", locator=article_id, text=article_text)
    return idx, rows, seq_start + len(rows)


def _parse_annex(lines: list[str], start: int, seq_start: int) -> tuple[int, SourceRow, int]:
    heading = lines[start]
    match = ANNEX_HEADING_RE.match(heading)
    if match is None:
        raise ValueError(f"Expected annex heading at line {start}: {heading!r}")

    annex_id = f"ANNEX {match.group(1)}"
    idx, content_lines = _consume_section(lines, start + 1)
    annex_text = normalize_ws(" ".join(content_lines))
    return idx, SourceRow(seq=seq_start, kind="ANNEX", locator=annex_id, text=annex_text), seq_start + 1


def extract_source_rows(text_path: str | Path) -> list[SourceRow]:
    lines = _read_source_lines(text_path)
    start_idx = next((idx for idx, line in enumerate(lines) if line == "Article 1"), None)
    if start_idx is None:
        raise ValueError("Could not find 'Article 1' in source text.")

    rows: list[SourceRow] = []
    idx = start_idx
    seq = 0
    while idx < len(lines):
        line = lines[idx]
        if ARTICLE_HEADING_RE.match(line):
            idx, article_rows, seq = _parse_article(lines, idx, seq)
            rows.extend(article_rows)
            continue
        if ANNEX_HEADING_RE.match(line):
            idx, annex_row, seq = _parse_annex(lines, idx, seq)
            rows.append(annex_row)
            continue
        idx += 1
    return rows


def build_source_dataframe(text_path: str | Path) -> pd.DataFrame:
    rows = extract_source_rows(text_path)
    return pd.DataFrame(
        [{"seq": row.seq, "kind": row.kind, "locator": row.locator, "text": row.text} for row in rows],
        columns=["seq", "kind", "locator", "text"],
    )


def write_source_tsv(text_path: str | Path, tsv_path: str | Path) -> Path:
    tsv_path = Path(tsv_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    build_source_dataframe(text_path).to_csv(tsv_path, sep="\t", header=False, index=False)
    return tsv_path


def resolve_source_text_path(tsv_path: str | Path, source_text_path: str | Path | None = None) -> Path | None:
    if source_text_path is not None:
        candidate = Path(source_text_path)
        return candidate if candidate.exists() else None

    tsv_path = Path(tsv_path)
    candidates = [
        tsv_path.parent / DEFAULT_SOURCE_TEXT_NAME,
        Path(__file__).resolve().with_name(DEFAULT_SOURCE_TEXT_NAME),
        Path(DEFAULT_SOURCE_TEXT_NAME),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def ensure_source_tsv(tsv_path: str | Path, source_text_path: str | Path | None = None) -> tuple[Path, bool]:
    tsv_path = Path(tsv_path)
    if tsv_path.exists():
        return tsv_path, False

    resolved_source = resolve_source_text_path(tsv_path, source_text_path)
    if resolved_source is None:
        raise FileNotFoundError(
            f"Input TSV not found at '{tsv_path}' and no fallback {DEFAULT_SOURCE_TEXT_NAME} source file was found."
        )

    return write_source_tsv(resolved_source, tsv_path), True


def read_tsv(tsv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["seq", "kind", "locator", "text"],
        dtype={"seq": "Int64", "kind": "string", "locator": "string", "text": "string"},
        keep_default_na=False,
    )
    df["text"] = df["text"].astype(str).map(normalize_ws)
    df["locator"] = df["locator"].astype(str).map(str.strip)
    return df


def _is_trivial_article_header(locator: str, text: str) -> bool:
    return text.strip().lower() == locator.strip().lower()


def merge_duplicate_nodes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    duplicate_count = 0

    for locator, group in df.groupby("locator", sort=False):
        if len(group) > 1:
            duplicate_count += len(group) - 1

        group = group.sort_values("seq", kind="stable")
        best_row = group.iloc[0].to_dict()
        best_text = str(best_row["text"])

        for _, row in group.iloc[1:].iterrows():
            candidate_text = str(row["text"])
            if _should_replace_text(locator, best_text, candidate_text):
                best_text = candidate_text
                best_row["kind"] = row["kind"]
            best_row["seq"] = min(int(best_row["seq"]), int(row["seq"]))

        best_row["node_id"] = locator
        best_row["text"] = best_text
        rows.append(
            {
                "node_id": best_row["node_id"],
                "kind": str(best_row["kind"]),
                "text": str(best_row["text"]),
                "seq": int(best_row["seq"]),
            }
        )

    nodes_df = pd.DataFrame(rows, columns=["node_id", "kind", "text", "seq"])
    nodes_df = nodes_df.sort_values("seq", kind="stable").reset_index(drop=True)
    return nodes_df, duplicate_count


def _should_replace_text(locator: str, current_text: str, candidate_text: str) -> bool:
    if not current_text:
        return True
    if not candidate_text:
        return False

    current_trivial = _is_trivial_article_header(locator, current_text)
    candidate_trivial = _is_trivial_article_header(locator, candidate_text)

    if current_trivial and not candidate_trivial:
        return True
    if candidate_trivial and not current_trivial:
        return False
    return len(candidate_text) > len(current_text)


def write_table(df: pd.DataFrame, outdir: str | Path, base_name: str) -> str:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if os.getenv("KG_TABLE_FORMAT", "").lower() == "parquet":
        path = outdir / f"{base_name}.parquet"
        try:
            df.to_parquet(path, index=False)
            return path.name
        except Exception:
            pass

    path = outdir / f"{base_name}.csv"
    df.to_csv(path, index=False)
    return path.name
