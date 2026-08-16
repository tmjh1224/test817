import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json







http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
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

# 定義資料結構
class OrderData(BaseModel):
    name: str = Field(description="姓名")
    phone: str = Field(description="電話")
    product: str = Field(description="商品")
    quantity: int = Field(description="數量")
    address: str = Field(description="地址")

# 綁定結構化輸出
structured_llm = llm.with_structured_output(OrderData)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("user", "{user_input}")
])

chain = prompt | structured_llm

user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

result = chain.invoke({"user_input": user_text})
print(result.model_dump_json(indent=2))