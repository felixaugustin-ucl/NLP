from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return WS_RE.sub(" ", (text or "")).strip()


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

    try:
        import pyarrow  # noqa: F401

        path = outdir / f"{base_name}.parquet"
        df.to_parquet(path, index=False)
        return path.name
    except Exception:
        path = outdir / f"{base_name}.csv"
        df.to_csv(path, index=False)
        return path.name

