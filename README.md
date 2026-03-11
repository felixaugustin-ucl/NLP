# NLP

Small EU AI Act knowledge-graph pipeline.

- `build_kg.py` is the only entrypoint.
- `kg_io.py` handles source ingestion, TSV generation, node normalization, and table output.
- `parse_refs.py` extracts explicit legal cross-references.
- `build_kg.py` also writes `node_embeddings.parquet` with `all-MiniLM-L6-v2` node embeddings.
- Tests live in `test_parse_refs.py` and `test_kg_io.py`.
