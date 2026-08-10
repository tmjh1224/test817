from openai import OpenAI

CONFIG = {
    "base_url": "https://163.17.136.119:8591/v1", 
    "api_key": "sk-9o0cj_Q6aJWWbSLODEPKBQ",
    "model": "gemma-4-E4B-it"
}

client = OpenAI(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    response = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔答覆"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=256
    )
    
    print(f"AI  : {response.choices[0].message.content}")