"""
Grounded Answer Generation.

Generates answers using ONLY the retrieved legal provisions
as evidence. Answers include citations and explicit uncertainty
when evidence is insufficient.

This is the anti-hallucination mechanism.

Usage:
    python -m src.qa.generate_answer --query "What AI systems are prohibited?"
"""

import logging

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_evidence(retrieved_nodes: list[dict]) -> str:
    """
    Format retrieved nodes into a structured evidence string.

    Each node is presented with its ID and full text.
    """
    evidence_parts = []
    for node in retrieved_nodes:
        node_id = node.get("node_id", "unknown")
        text = node.get("text", "")
        node_type = node.get("type", "")
        evidence_parts.append(
            f"[{node_id}] ({node_type})\n{text}"
        )

    return "\n\n---\n\n".join(evidence_parts)


def build_prompt(
    question: str,
    evidence: str,
    template: str = None,
) -> str:
    """
    Build the QA prompt with question and evidence.

    The prompt instructs the model to:
    - Answer only from provided evidence
    - Cite specific article/node IDs
    - State uncertainty when evidence is insufficient
    """
    if template is None:
        template = """You are a legal assistant specialising in the EU AI Act.
Answer the following question using ONLY the provided legal provisions.
Cite the specific article, paragraph, or section for each claim.
If the evidence is insufficient, say so explicitly.

Question: {question}

Evidence:
{evidence}

Answer:"""

    return template.format(question=question, evidence=evidence)


def generate_answer(
    question: str,
    retrieved_nodes: list[dict],
    config: dict,
) -> str:
    """
    Generate a grounded answer from retrieved evidence.

    Args:
        question: user's natural language question
        retrieved_nodes: list of retrieved node dicts with text
        config: QA configuration

    Returns:
        Generated answer string with citations
    """
    evidence = format_evidence(retrieved_nodes)
    prompt = build_prompt(
        question, evidence,
        template=config.get("qa", {}).get("prompt_template"),
    )

    # TODO: Choose your LLM backend:
    # Option 1: OpenAI API
    # Option 2: Local model (e.g., HuggingFace)
    # Option 3: Ollama or other local inference

    logger.info("Answer generation - TODO: integrate LLM backend")
    logger.info(f"Prompt length: {len(prompt)} characters")

    # Placeholder
    answer = (
        f"[Answer generation not yet integrated]\n\n"
        f"Question: {question}\n\n"
        f"Evidence nodes: {len(retrieved_nodes)}\n\n"
        f"Prompt preview:\n{prompt[:500]}..."
    )

    return answer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    config = load_config()

    # TODO: integrate with retriever to get actual nodes
    # For now, placeholder
    logger.info(f"Query: {args.query}")
    logger.info("Connect this to the retrieval pipeline to generate answers.")


if __name__ == "__main__":
    main()
