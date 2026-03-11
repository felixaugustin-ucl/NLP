import fitz
import re
import pandas as pd

doc = fitz.open("pdf/eu_ai_act.pdf")

text = ""

for page in doc:
    text += page.get_text()

# 只匹配真正的 Article 标题
pattern = r'\nArticle\s+(\d+)\n'

matches = list(re.finditer(pattern, text))

articles = []

for i in range(len(matches)):

    start = matches[i].start()

    if i < len(matches) - 1:
        end = matches[i+1].start()
    else:
        end = len(text)

    article_id = "Article " + matches[i].group(1)

    article_text = text[start:end]

    articles.append({
        "Article_ID": article_id,
        "Text": article_text.strip()
    })

df = pd.DataFrame(articles)

df.to_csv("data/eu_law_dataset.csv", index=False)

print("Articles extracted:", len(df))
print(df.head())
import os
import json

os.makedirs("data", exist_ok=True)

articles_list = df.to_dict(orient="records")

with open("data/articles.json", "w", encoding="utf-8") as f:
    json.dump(articles_list, f, ensure_ascii=False, indent=2)

print("Saved to data/articles.json")