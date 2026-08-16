import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
import json

# 1. 設定 LLM (對接 vLLM)
http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client
)

# 2. 設定 Parser (LangChain 的強項)
parser = JsonOutputParser()

system_prompt = """你是一個資料提取助手。
{format_instructions}
需要的欄位: name, phone, product, quantity, address"""

# 3. 設定 Prompt Template (將 Parser 的格式指令注入 Prompt)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{text}")
])

# 4. 建立 Chain (LCEL 語法: Prompt -> LLM -> Parser)
chain = prompt | llm | parser

user_input = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

try:
    result = chain.invoke({
        "text": user_input,
        "format_instructions": parser.get_format_instructions() # 自動生成 "請回傳 JSON..." 的指令
    })

    print(json.dumps(result, ensure_ascii=False, indent=2))

except Exception as e:
    print(f"❌ Chain 執行錯誤: {e}")