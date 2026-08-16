import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys # 為了更順暢的輸出控制

# ================= 配置區 =================
http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0.3,
    max_tokens=256
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個專業的科技文章編輯。請將使用者提供的文章內容，歸納出 3 個重點，並以繁體中文條列式輸出。"),
    ("human", "{article_content}") 
])

parser = StrOutputParser()

# Chain 的定義完全不用變
chain = prompt | llm | parser

tech_article = """
LangChain 是一個開源框架，旨在簡化使用大型語言模型（LLM）開發應用程式的過程。
它提供了一套工具和介面，讓開發者能夠將 LLM 與其他資料來源（如網際網路或個人檔案）連接起來。
LangChain 的核心概念包括 Chain（鏈）、Agent（代理）和 Memory（記憶）。
透過 LCEL 語法，開發者可以輕鬆地將不同的組件串聯在一起，構建複雜的 AI 應用。
最近，LangChain 推出了 LangGraph，專門用於構建具備循環邏輯的有狀態代理。
"""

print("--- 開始生成摘要 (串流模式) ---")
for chunk in chain.stream({"article_content": tech_article}):
    print(chunk, end="", flush=True)