from openai import OpenAI

CONFIG = {
    # vLLM 預設 port 是 8000，路徑標準為 /v1
    "base_url": "https://win4090p8000.huannago.com/v1/",#"http://localhost:8000/v1", 
    # vLLM 通常不驗證 API Key，但 client 端需要填一個字串，隨便填即可
    "api_key": "vllm-token",
    # 這裡必須跟你啟動 vLLM 時下的 --model 參數完全一致
    "model": "Qwen/Qwen2.5-1.5B-Instruct"
}

# 初始化客戶端
client = OpenAI(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
)

prompt = "請用100字形容『人工智慧』。"
temps = [0.1, 1.5] # 0.1 很冷靜, 1.5 很發散

for t in temps:
    print(f"\n➡️  測試 Temperature = {t} ...")
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=t,
            max_tokens=100 # 限制長度方便觀察
        )
        print(f"🤖 回覆: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")