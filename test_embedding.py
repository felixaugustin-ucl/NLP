from retrieve_embedding import retrieve

query = "What AI systems are prohibited?"

results = retrieve(query)

for r in results:

    print("\n", r["article"])
    print("score:", r["score"])
    print(r["text"])