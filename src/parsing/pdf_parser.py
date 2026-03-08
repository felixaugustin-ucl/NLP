"""
PDF Parser for the EU AI Act.

Converts the EU AI Act PDF into machine-readable text,
split by page and section.

Usage:
    python -m src.parsing.pdf_parser
"""

import json
import logging
from pathlib import Path

import pdfplumber
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from PDF, page by page.

    Returns:
        List of dicts with keys: page_number, text
    """
    pages = []
    logger.info(f"Extracting text from: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_number": i + 1,
                    "text": text.strip(),
                })

    logger.info(f"Extracted {len(pages)} pages")
    return pages


def clean_text(text: str) -> str:
    """
    Clean extracted text.

    - Remove excessive whitespace
    - Fix common OCR/extraction issues
    - Normalize line breaks
    """
    import re

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Fix common artifacts
    text = text.replace('\x00', '')

    return text.strip()


def save_parsed_text(pages: list[dict], output_path: str) -> None:
    """Save parsed text to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved parsed text to: {output_path}")


def main():
    config = load_config()
    pdf_path = config["paths"]["raw_pdf"]
    output_path = config["paths"]["parsed_text"]

    # Extract
    pages = extract_text_from_pdf(pdf_path)

    # Clean
    for page in pages:
        page["text"] = clean_text(page["text"])

    # Save
    save_parsed_text(pages, output_path)


if __name__ == "__main__":
    main()
