from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from build_kg import write_node_embeddings
from kg_io import extract_source_rows


def test_extract_source_rows_parses_articles_paragraphs_and_annexes(tmp_path):
    text_path = tmp_path / "EU_AI_ACT.txt"
    text_path.write_text(
        "\n".join(
            [
                "Official Journal",
                "EN",
                "Article 1",
                "Subject matter`",
                "1.",
                "This Regulation applies.",
                "2.",
                "See Annex I.",
                "45/144",
                "ELI: http://data.europa.eu/eli/reg/2024/1689/oj",
                "Article 2",
                "Definitions",
                "For the purposes of this Regulation, the following definitions apply:",
                "(1) 'AI system' means a system referred to in Article 1.",
                "ANNEX I",
                "Supporting material",
                "Section 1",
                "Additional context.",
            ]
        ),
        encoding="utf-8",
    )

    rows = extract_source_rows(text_path)
    locators = [row.locator for row in rows]
    assert locators == ["Article 1", "Article 1(1)", "Article 1(2)", "Article 2", "Article 2(1)", "ANNEX I"]
    assert rows[0].text == "Subject matter"
    assert rows[2].text == "See Annex I."
    assert rows[3].text == "Definitions For the purposes of this Regulation, the following definitions apply:"
    assert rows[4].text == "'AI system' means a system referred to in Article 1."
    assert rows[5].text == "Supporting material Section 1 Additional context."


def test_build_kg_generates_missing_tsv_from_source_text(tmp_path):
    text_path = tmp_path / "EU_AI_ACT.txt"
    text_path.write_text(
        "\n".join(
            [
                "Article 1",
                "Subject matter",
                "1.",
                "This Regulation applies.",
                "Article 2",
                "Scope",
                "1.",
                "This paragraph refers to Article 1 and Annex I.",
                "ANNEX I",
                "List of things",
            ]
        ),
        encoding="utf-8",
    )

    tsv_path = tmp_path / "generated.tsv"
    outdir = tmp_path / "kg_out"
    script_path = Path(__file__).resolve().with_name("build_kg.py")
    fake_st_path = tmp_path / "sentence_transformers.py"
    fake_st_path.write_text(
        "\n".join(
            [
                "class SentenceTransformer:",
                "    def __init__(self, model_name):",
                "        self.model_name = model_name",
                "    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):",
                "        return [[float(i), float(len(text))] for i, text in enumerate(texts)]",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script_path), "--tsv", str(tsv_path), "--outdir", str(outdir)],
        capture_output=True,
        text=True,
        check=True,
        env={**dict(os.environ), "PYTHONPATH": str(tmp_path)},
    )

    assert "Generated missing TSV" in result.stdout
    assert tsv_path.exists()
    assert (outdir / "nodes.csv").exists()
    assert (outdir / "edges.csv").exists()
    assert (outdir / "edges.parquet").exists()
    assert (outdir / "node_embeddings.parquet").exists()

    nodes = pd.read_csv(outdir / "nodes.csv")
    edges = pd.read_csv(outdir / "edges.csv")
    parquet_edges = pd.read_parquet(outdir / "edges.parquet")
    embeddings = pd.read_parquet(outdir / "node_embeddings.parquet")
    assert "ANNEX I" in set(nodes["node_id"])
    assert len(parquet_edges) == len(edges)
    assert set(embeddings["node_id"]) == set(nodes["node_id"])
    assert ("Article 2(1)", "REFERS_TO", "Article 1") in {
        (row.src, row.rel, row.dst) for row in edges.itertuples(index=False)
    }


def test_write_node_embeddings_writes_parquet(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            assert self.model_name == "all-MiniLM-L6-v2"
            assert normalize_embeddings is True
            assert show_progress_bar is False
            return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    class FakeModule:
        SentenceTransformer = FakeModel

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeModule)

    nodes = pd.DataFrame(
        {
            "node_id": ["Article 1", "Article 1(1)"],
            "text": ["Subject matter", ""],
        }
    )
    filename = write_node_embeddings(nodes, tmp_path)

    embeddings = pd.read_parquet(tmp_path / filename)
    assert filename == "node_embeddings.parquet"
    records = embeddings.assign(embedding=embeddings["embedding"].map(list)).to_dict(orient="records")
    assert records == [
        {"node_id": "Article 1", "embedding": [0.0, 14.0]},
        {"node_id": "Article 1(1)", "embedding": [1.0, 12.0]},
    ]
