import httpx
import urllib3
from openai import OpenAI

# 關閉不安全連線警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONFIG = {
    "base_url": "https://163.17.136.119:8591/v1",
    "api_key": "sk-9o0cj_Q6aJWWbSLODEPKBQ",
    # 修改為伺服器上實際存在的模型 ID
    "model": "gemma-4-E4B-it"
}

# 關閉 SSL 憑證驗證
http_client = httpx.Client(verify=False)

client = OpenAI(
    base_url=CONFIG["base_url"],
    api_key=CONFIG["api_key"],
    http_client=http_client
)

prompt = "你是一位專精於數學的資深研究員，請你對離散數學這一主題進行講解。輸出必須使用較學的語氣（Formal Tone）。"
temps = [0.001, 1.5]

for t in temps:
    print(f"\n➡️  測試 Temperature = {t} ...")
    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=t,
            max_tokens=1000
        )
        print(f"🤖 回覆: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")