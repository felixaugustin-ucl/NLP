from retrieve_embedding import retrieve
from deepseek_reasoning import ask_deepseek
from graph_retrieval import expand_graph


def main():

    query = input("\nAsk about EU AI Act:\n")

    # Step 1: Retrieval
    results = retrieve(query)

    context = ""
    graph_context = ""

    print("\nQUESTION")
    print(query)

    print("\nTOP RELEVANT ARTICLES")
    print("------------------------------------------------")

    for i, r in enumerate(results[:5], start=1):

        article_id = r["article"]
        print(f"\n{i}. {article_id}")
        print(f"Similarity Score: {round(r['score'], 3)}\n")

       

        print("\n------------------------------------------------")

        context += f"\n{article_id}\n{r['text']}\n"

        # =========================
        # Graph Expansion
        # =========================

        try:

            article_number = int(article_id.replace("Article", "").strip())

            connections = expand_graph(article_number)

            for c in connections:

                graph_context += f"{article_id} {c['relation']} {c['concept']}\n"

        except:
            pass

    # Combine contexts
    full_context = f"""
Relevant Articles:
{context}

Knowledge Graph Relations:
{graph_context}
"""

    # Step 2: Reasoning
    answer = ask_deepseek(query, full_context)

    print("\nLEGAL EXPLANATION")
    print("------------------------------------------------\n")

    print(answer)


if __name__ == "__main__":
    main()