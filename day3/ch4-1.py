import httpx
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0
)

# 1. 定義 Pydantic 結構
class OrderData(BaseModel):
    name: str = Field(description="客戶姓名")
    phone: str = Field(description="電話號碼")
    product: str = Field(description="商品名稱")
    quantity: int = Field(description="商品數量")
    address: str = Field(description="送貨地址")

# 2. 初始化 JsonOutputParser
parser = JsonOutputParser(pydantic_object=OrderData)

# 3. Prompt 加入 format_instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員。請嚴格遵守以下 JSON 格式輸出，不要包裹額外外層 key，也不要翻譯 key 名稱：\n{format_instructions}"),
    ("user", "{user_input}")
]).partial(format_instructions=parser.get_format_instructions())

# 4. 建立 Chain
chain = prompt | llm | parser

user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

result = chain.invoke({"user_input": user_text})

print("✅ 提取成功:")
print(json.dumps(result, ensure_ascii=False, indent=2))