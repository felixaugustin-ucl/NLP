from openai import OpenAI

# 初始化 DeepSeek client
client = OpenAI(
    api_key="sk-0d60415485924bb386b33ff88cfe978d",
    base_url="https://api.deepseek.com"
)

def ask_deepseek(question, context):

    prompt = f"""
You are an expert in the EU AI Act.

Answer the question using the provided legal articles.

Structure your answer as follows:

1. Short legal conclusion
2. Relevant articles cited
3. Key regulatory points (bullet points)

Articles:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a legal expert in EU AI Act regulation."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content