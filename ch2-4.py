import json
from openai import OpenAI

client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
)

user_input = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

system_prompt = """你是一個資料提取助手。
請從使用者的文字中提取以下資訊，並嚴格以 JSON 格式回傳。
需要的欄位: name, phone, product, quantity, address"""

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ],
    temperature=0.1,
)

json_content = response.choices[0].message.content

if not json_content: 
    raise ValueError("Empty response")

clean_json = json_content.replace("```json", "").replace("```", "").strip()

decision = json.loads(clean_json)

print(json.dumps(decision, ensure_ascii=False, indent=2))