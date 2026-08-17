import httpx
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. 定義 Pydantic 資料結構
class OrderData(BaseModel):
    name: str = Field(description="訂購人姓名")
    phone: str = Field(description="聯絡電話")
    product: str = Field(description="商品名稱")
    quantity: int = Field(description="購買數量")
    address: str = Field(description="送貨地址")

# 2. 建立解析器
parser = PydanticOutputParser(pydantic_object=OrderData)

http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}  # 確保啟用 JSON Mode
)

# 3. 將 format_instructions 注入 System Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。\n{format_instructions}"),
    ("user", "{user_input}")
]).partial(format_instructions=parser.get_format_instructions())

# 4. 組裝 Chain (LLM 輸出後直接由 parser 解析成 Pydantic 物件)
chain = prompt | llm | parser

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break

    result = chain.invoke({"user_input": user_input})
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))