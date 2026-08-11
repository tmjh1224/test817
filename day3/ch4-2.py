from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

# ================= 1. 定義工具 (Tool Definition) =================
# 對應原程式：System Prompt 裡的 "需要的欄位: name, phone..."
# 對應原程式：JsonOutputParser 的格式化能力

@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """

    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }

# ================= 2. 設定 LLM (Agent Brain) =================


# 關鍵步驟：綁定工具
llm_with_tools = llm.bind_tools([extract_order_data])

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("user", "{user_input}")
])

def extract_tool_args(ai_message):
    if ai_message.tool_calls:
        return ai_message.tool_calls[0]['args']
    else:
        return ai_message.content

chain = prompt | llm_with_tools | extract_tool_args

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    result = chain.invoke({"user_input": user_input})

    print(json.dumps(result, ensure_ascii=False, indent=2))