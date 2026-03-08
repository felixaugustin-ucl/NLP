"""
Node Builder for the EU AI Act Graph.

Splits parsed text into legal chunk nodes:
- Articles, Paragraphs, Sub-paragraphs
- Recitals, Annexes, Annex items
- Definitions

Output: data/processed/nodes.csv

Usage:
    python -m src.graph_construction.node_builder
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_parsed_text(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_page_number(pages: list[dict], char_offset: int, page_boundaries: list[int]) -> int:
    """Find which page a character offset falls on."""
    for i, boundary in enumerate(page_boundaries):
        if char_offset < boundary:
            return pages[i]["page_number"] if i < len(pages) else -1
    return pages[-1]["page_number"] if pages else -1


def _build_chapter_map(full_text: str) -> dict[int, str]:
    """
    Build a mapping from article number to chapter number.

    Returns: {article_number: "I", "II", ...}
    """
    chapter_pattern = r'CHAPTER ([IVX]+)\n([^\n]+)'
    article_pattern = r'Article (\d+)\n'

    chapters = list(re.finditer(chapter_pattern, full_text))
    articles = list(re.finditer(article_pattern, full_text))

    # Assign each article to the chapter that precedes it
    article_to_chapter = {}
    for art_match in articles:
        art_num = int(art_match.group(1))
        art_pos = art_match.start()

        current_chapter = None
        for ch_match in chapters:
            if ch_match.start() <= art_pos:
                current_chapter = ch_match.group(1)
            else:
                break

        if current_chapter:
            article_to_chapter[art_num] = current_chapter

    return article_to_chapter


def extract_recitals(full_text: str, pages: list[dict], page_boundaries: list[int]) -> list[dict]:
    """
    Extract recital nodes from the preamble (before Article 1).

    Recitals are numbered: (1) ..., (2) ..., etc.
    """
    nodes = []

    # Find preamble (everything before first Article)
    art1_match = re.search(r'Article 1\n', full_text)
    if not art1_match:
        logger.warning("Could not find Article 1 — skipping recitals")
        return nodes

    preamble = full_text[:art1_match.start()]

    # Find all recital-style numbered items: (N) followed by text
    # Use a forward-looking approach: each recital starts at (N) and ends before (N+1)
    recital_pattern = r'\((\d+)\)\s+'
    matches = list(re.finditer(recital_pattern, preamble))

    # Deduplicate: keep only the first occurrence of each recital number
    # (footnote numbers can also match the pattern)
    seen_numbers = set()
    unique_matches = []
    for m in matches:
        num = int(m.group(1))
        if num not in seen_numbers and 1 <= num <= 200:
            seen_numbers.add(num)
            unique_matches.append(m)

    for i, match in enumerate(unique_matches):
        recital_num = int(match.group(1))
        start_pos = match.start()

        # End position: start of next recital or end of preamble
        if i + 1 < len(unique_matches):
            end_pos = unique_matches[i + 1].start()
        else:
            end_pos = len(preamble)

        text = preamble[start_pos:end_pos].strip()

        # Clean up: remove page headers like "EN\nOJ L, 12.7.2024"
        text = re.sub(r'\nEN\nOJ L,? \d+\.\d+\.\d+', '', text)
        text = text.strip()

        page = _find_page_number(pages, start_pos, page_boundaries)

        nodes.append({
            "node_id": f"recital_{recital_num}",
            "title": f"Recital ({recital_num})",
            "type": "recital",
            "text": text,
            "page": page,
            "article_number": None,
            "chapter": None,
            "annex": None,
            "parent_id": None,
        })

    logger.info(f"Extracted {len(nodes)} recital nodes")
    return nodes


def extract_articles(full_text: str, pages: list[dict], page_boundaries: list[int]) -> list[dict]:
    """
    Extract article-level nodes from the full text.

    Returns list of node dicts with:
        node_id, title, type, text, article_number, chapter, parent_id
    """
    nodes = []
    chapter_map = _build_chapter_map(full_text)

    # Pattern: "Article N\nTitle text"
    article_pattern = r'Article (\d+)\n([^\n]+)'
    matches = list(re.finditer(article_pattern, full_text))

    # Only keep articles in the main body (after preamble, before annexes)
    annex_start = full_text.find('ANNEX I\n')
    if annex_start == -1:
        annex_start = len(full_text)

    for i, match in enumerate(matches):
        art_num = int(match.group(1))
        title = match.group(2).strip()
        start_pos = match.start()

        # Skip articles that appear before the actual main body
        # (e.g., references to "Article 16 TFEU" in the preamble)
        art1_pos = full_text.find('Article 1\nSubject matter')
        if art1_pos > 0 and start_pos < art1_pos and art_num != 1:
            continue

        # Skip if past annexes
        if start_pos >= annex_start:
            continue

        # End position: start of next article or start of annexes
        end_pos = annex_start
        for j in range(i + 1, len(matches)):
            next_pos = matches[j].start()
            if next_pos > start_pos and next_pos < annex_start:
                end_pos = next_pos
                break

        text = full_text[start_pos:end_pos].strip()
        text = re.sub(r'\nEN\nOJ L,? \d+\.\d+\.\d+', '', text)
        text = text.strip()

        page = _find_page_number(pages, start_pos, page_boundaries)

        nodes.append({
            "node_id": f"article_{art_num}",
            "title": f"Article {art_num}: {title}",
            "type": "article",
            "text": text,
            "page": page,
            "article_number": art_num,
            "chapter": chapter_map.get(art_num),
            "annex": None,
            "parent_id": None,
        })

    logger.info(f"Extracted {len(nodes)} article nodes")
    return nodes


def extract_paragraphs(article_nodes: list[dict]) -> list[dict]:
    """
    Extract paragraph-level nodes from articles.

    Each numbered paragraph (1., 2., 3., ...) becomes a child node.
    """
    nodes = []

    for article in article_nodes:
        art_text = article["text"]
        art_id = article["node_id"]
        art_num = article["article_number"]

        # Find numbered paragraphs: lines starting with "N. "
        para_pattern = r'(?:^|\n)(\d+)\.\s+'
        para_matches = list(re.finditer(para_pattern, art_text))

        if not para_matches:
            continue

        for i, match in enumerate(para_matches):
            para_num = int(match.group(1))
            start_pos = match.start()

            # End at next paragraph or end of article
            if i + 1 < len(para_matches):
                end_pos = para_matches[i + 1].start()
            else:
                end_pos = len(art_text)

            text = art_text[start_pos:end_pos].strip()

            nodes.append({
                "node_id": f"article_{art_num}_para_{para_num}",
                "title": f"Article {art_num}, Paragraph {para_num}",
                "type": "paragraph",
                "text": text,
                "page": article["page"],
                "article_number": art_num,
                "chapter": article["chapter"],
                "annex": None,
                "parent_id": art_id,
            })

    logger.info(f"Extracted {len(nodes)} paragraph nodes")
    return nodes


def extract_definitions(full_text: str, pages: list[dict], page_boundaries: list[int]) -> list[dict]:
    """
    Extract definition nodes from Article 3 (Definitions).

    Definitions use smart quotes: \u2018term\u2019 means ...
    """
    nodes = []

    # Find Article 3
    art3_start = full_text.find('Article 3\nDefinitions')
    if art3_start == -1:
        logger.warning("Could not find Article 3 — skipping definitions")
        return nodes

    # Find Article 4
    art4_match = re.search(r'Article 4\n', full_text[art3_start + 100:])
    if art4_match:
        art3_end = art3_start + 100 + art4_match.start()
    else:
        art3_end = art3_start + 20000  # fallback

    art3_text = full_text[art3_start:art3_end]

    # Pattern: (N) \u2018term\u2019 means ...
    def_pattern = r'\((\d+)\)\s+\u2018([^\u2019]+)\u2019\s+means\s+'
    def_matches = list(re.finditer(def_pattern, art3_text))

    for i, match in enumerate(def_matches):
        def_num = int(match.group(1))
        term = match.group(2).strip()
        start_pos = match.start()

        # End at next definition or end of article
        if i + 1 < len(def_matches):
            end_pos = def_matches[i + 1].start()
        else:
            end_pos = len(art3_text)

        text = art3_text[start_pos:end_pos].strip()
        text = re.sub(r'\nEN\nOJ L,? \d+\.\d+\.\d+', '', text)
        text = text.strip()

        page = _find_page_number(pages, art3_start + start_pos, page_boundaries)

        nodes.append({
            "node_id": f"definition_{def_num}",
            "title": f"Definition: {term}",
            "type": "definition",
            "text": text,
            "page": page,
            "article_number": 3,
            "chapter": "I",
            "annex": None,
            "parent_id": "article_3",
        })

    logger.info(f"Extracted {len(nodes)} definition nodes")
    return nodes


def extract_annexes(full_text: str, pages: list[dict], page_boundaries: list[int]) -> list[dict]:
    """Extract annex-level nodes and their items."""
    nodes = []

    # Find all ANNEX headers
    annex_pattern = r'ANNEX ([IVX]+)\n([^\n]+)'
    annex_matches = list(re.finditer(annex_pattern, full_text))

    for i, match in enumerate(annex_matches):
        annex_num = match.group(1)
        annex_title = match.group(2).strip()
        start_pos = match.start()

        # End at next annex or end of document
        if i + 1 < len(annex_matches):
            end_pos = annex_matches[i + 1].start()
        else:
            end_pos = len(full_text)

        text = full_text[start_pos:end_pos].strip()
        text = re.sub(r'\nEN\nOJ L,? \d+\.\d+\.\d+', '', text)
        text = text.strip()

        page = _find_page_number(pages, start_pos, page_boundaries)

        annex_node_id = f"annex_{annex_num}"
        nodes.append({
            "node_id": annex_node_id,
            "title": f"Annex {annex_num}: {annex_title[:80]}",
            "type": "annex",
            "text": text,
            "page": page,
            "article_number": None,
            "chapter": None,
            "annex": annex_num,
            "parent_id": None,
        })

        # Extract numbered items within the annex (e.g., "1.", "2.", etc.)
        item_pattern = r'(?:^|\n)(\d+)\.\s+'
        item_matches = list(re.finditer(item_pattern, text))

        for j, item_match in enumerate(item_matches):
            item_num = int(item_match.group(1))
            item_start = item_match.start()

            if j + 1 < len(item_matches):
                item_end = item_matches[j + 1].start()
            else:
                item_end = len(text)

            item_text = text[item_start:item_end].strip()

            # Only create item nodes for substantial items (not just numbers in text)
            if len(item_text) > 30:
                nodes.append({
                    "node_id": f"annex_{annex_num}_item_{item_num}",
                    "title": f"Annex {annex_num}, Item {item_num}",
                    "type": "annex_item",
                    "text": item_text,
                    "page": page,
                    "article_number": None,
                    "chapter": None,
                    "annex": annex_num,
                    "parent_id": annex_node_id,
                })

    logger.info(f"Extracted {len(nodes)} annex/annex_item nodes")
    return nodes


def build_all_nodes(parsed_pages: list[dict]) -> pd.DataFrame:
    """
    Main node construction pipeline.

    Combines all extracted nodes into a single DataFrame.
    """
    # Combine all page text and compute page boundaries
    full_text = "\n\n".join(page["text"] for page in parsed_pages)

    page_boundaries = []
    running_length = 0
    for page in parsed_pages:
        running_length += len(page["text"]) + 2  # +2 for "\n\n"
        page_boundaries.append(running_length)

    all_nodes = []

    # Extract each node type
    recital_nodes = extract_recitals(full_text, parsed_pages, page_boundaries)
    all_nodes.extend(recital_nodes)

    article_nodes = extract_articles(full_text, parsed_pages, page_boundaries)
    all_nodes.extend(article_nodes)

    paragraph_nodes = extract_paragraphs(article_nodes)
    all_nodes.extend(paragraph_nodes)

    definition_nodes = extract_definitions(full_text, parsed_pages, page_boundaries)
    all_nodes.extend(definition_nodes)

    annex_nodes = extract_annexes(full_text, parsed_pages, page_boundaries)
    all_nodes.extend(annex_nodes)

    df = pd.DataFrame(all_nodes)
    logger.info(f"Total nodes constructed: {len(df)}")
    return df


def save_nodes(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved nodes to: {output_path}")


def main():
    config = load_config()
    parsed_pages = load_parsed_text(config["paths"]["parsed_text"])

    nodes_df = build_all_nodes(parsed_pages)
    save_nodes(nodes_df, config["paths"]["nodes_csv"])


if __name__ == "__main__":
    main()
