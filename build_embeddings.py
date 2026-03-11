import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 读取文章
with open("data/articles.json","r",encoding="utf-8") as f:
    articles = json.load(f)

texts = [a["Text"].replace("\n"," ") for a in articles]

print("Total articles:",len(texts))

# 生成 embedding
embeddings = model.encode(texts, show_progress_bar=True)

# 创建文件夹
os.makedirs("embeddings",exist_ok=True)

# 保存
np.save("embeddings/article_embeddings.npy", embeddings)

print("Embeddings saved.")