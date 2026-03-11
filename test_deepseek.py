import requests

# 你的 DeepSeek API key
API_KEY = "sk-0d60415485924bb386b33ff88cfe978d"

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user",
            "content": "Explain the EU AI Act in simple terms."
        }
    ]
}

response = requests.post(url, headers=headers, json=data)

print(response.json()["choices"][0]["message"]["content"])