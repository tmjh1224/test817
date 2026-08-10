from openai import OpenAI

CONFIG = {
    "base_url": "https://ws-02.wade0426.me/v1", 
    "api_key": "vllm-token",
    "model": "Qwen/Qwen3-VL-2B-Instruct"
}

client = OpenAI(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
)


history = [
    {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔答覆"}
]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break
    history.append({"role": "user", "content": user_input})
    try:
        print("AI  : (思考中...)", end="\r")
        
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=history, 
            temperature=0.7,
            max_tokens=256
        )
        full_reply = response.choices[0].message.content
        print(f"AI  : {full_reply}\n")
        history.append({"role": "assistant", "content": full_reply})
    except Exception as e:
        print(f"Error: {e}")
