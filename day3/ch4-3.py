import json
import traceback
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# 1. 初始化 LLM
http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0
)

# 2. 定義摘要 Chain (不依賴 native tool calling)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個資深的科技主編。請將輸入的技術文章內容，精簡地歸納出 3 個關鍵重點 (Key Takeaways)。請用繁體中文輸出。"),
    ("user", "{text}")
])
summary_chain = summary_prompt | llm | StrOutputParser()

# 3. 定義 Router Prompt (使用 JSON 判斷意圖)
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一個智慧意圖路由助手。請評估使用者的輸入內容：
1. 若內容屬於「科技」、「程式設計」、「AI」、「軟體工程」或「IT 技術」領域，請回應 JSON：{{"is_tech": true}}
2. 若內容為閒聊、食譜、日常訂單等非技術內容，請直接回覆使用者回答。

只需輸出 JSON 或對非技術問題的回應，不要輸出額外解釋。"""),
    ("user", "{input}")
])

router_chain = router_prompt | llm | StrOutputParser()

# 4. 主執行迴圈
while True:
    user_input = input("\nUser: ").strip()
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    if not user_input:
        continue

    try:
        response = router_chain.invoke({"input": user_input})
        
        # 嘗試解析是否需要呼叫科技摘要
        is_tech = False
        try:
            res_json = json.loads(response.strip())
            if isinstance(res_json, dict) and res_json.get("is_tech") is True:
                is_tech = True
        except json.JSONDecodeError:
            is_tech = False

        if is_tech:
            print("✅ [決策] 判斷為科技文章")
            final_result = summary_chain.invoke({"text": user_input})
            print(f"📄 [執行結果]:\n{final_result}")
        else:
            print("❌ [決策] 判斷為閒聊/非科技文章")
            print(f"💬 [AI 回應]: {response}")

    except Exception as e:
        print(f"⚠️ [發生錯誤]: {e}")
        traceback.print_exc()