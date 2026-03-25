def build_prompt(question, docs):
    context = "\n\n".join([
        d["text"]
        for d in docs
    ])

    prompt = f"""
You are a legal assistant.

Answer the question based ONLY on the provided text.

---------------------
REASONING (internal, do NOT output):

Step 1 — Identify nodes:
- Identify the main concept in the question
- Identify key legal nodes from the text:
  • concepts (e.g., systems, models, categories)
  • legal provisions (Articles, Chapters, Recitals)
  • obligations or prohibitions

Step 2 — Identify edges (relationships):
Prioritise relationships in this order:
1. Classification → what the concept IS or BELONGS TO
2. Legal status → defined / not defined / indirectly covered
3. Regulation → what rules or obligations apply
4. Ignore implementation details unless they are central

Step 3 — Build structure:
Construct the answer in this logical order:
→ legal status → classification → applicable rules

---------------------
OUTPUT REQUIREMENTS:

- Write ONE paragraph (3–5 sentences)
- The FIRST sentence MUST directly answer the question with a clear legal conclusion

- If the question involves definition / scope / regulation:
  you MUST explicitly state:
  • explicitly defined OR
  • not explicitly defined OR
  • indirectly covered

- You MUST explain:
  • what category the concept belongs to (if mentioned)
  • how it is regulated within the framework

- Include legal references (Articles / Chapters) if present in the text

- Focus ONLY on core legal structure:
  DO NOT focus on technical or implementation details 
  (e.g., APIs, source code access, monitoring mechanisms)
  UNLESS they are the main subject of the question

- Use wording grounded in the provided text
- Do NOT introduce external knowledge

- Do NOT use bullet points or headings
- The second sentence MUST explain or support the conclusion

---------------------
QUALITY CHECK (internal):

Before answering, ensure:
- Legal status is clearly stated
- Classification is identified (if applicable)
- Regulatory rules are included (if present)

If any are missing, revise internally before answering.

---------------------
Question:
{question}

Text:
{context}
"""
    return prompt