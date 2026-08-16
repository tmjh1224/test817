import httpx
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0
)

# 1. 定義資料結構 (請確保 description 明確)
class OrderData(BaseModel):
    name: str = Field(description="客戶姓名，例如：陳大明")
    phone: str = Field(description="電話號碼")
    product: str = Field(description="商品名稱")
    quantity: int = Field(description="商品數量")
    address: str = Field(description="送貨地址")

# 2. 強制使用 json_mode 避開原本後端 Tool Calling 的相容性問題與警告
structured_llm = llm.with_structured_output(OrderData, method="json_mode")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請務必根據用戶輸入提取正確的訂單資訊。特別注意：姓名欄位必須是人的姓名。"),
    ("user", "{user_input}")
])

chain = prompt | structured_llm

user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

result = chain.invoke({"user_input": user_text})

# 3. 輸出結果
print(result.model_dump_json(indent=2))