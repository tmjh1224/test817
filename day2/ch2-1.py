import httpx
from openai import OpenAI

CONFIG = {
    "base_url": "https://163.17.136.119:8591/v1",
    "api_key": "sk-9o0cj_Q6aJWWbSLODEPKBQ",
    "model": "gemma-4-E4B-it"
}

# 關閉 SSL 憑證驗證
http_client = httpx.Client(verify=False)

client = OpenAI(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    http_client=http_client
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    response = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": "你是一個繁體中文的各領域專業人士，請根據知道的回答"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=256
    )
    
    print(f"AI  : {response.choices[0].message.content}")