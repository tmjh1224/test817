from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
import json

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="/models/gpt-oss-120b",
    temperature=0
)

@tool
def generate_tech_summary(article_content: str):
    """
    科技文章專用摘要生成工具。
    【判斷邏輯】：
    1. 只有當輸入內容屬於「科技」、「程式設計」、「AI」、「軟體工程」或「IT 技術」領域時，才使用此工具。
    2. 如果內容是「閒聊」、「食譜」、「天氣」、「日常日記」等非技術內容，請勿使用此工具。
    
    功能：將輸入的技術文章歸納出 3 個重點。
    """

    # 2. 定義摘要專用的 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個資深的科技主編。請將輸入的技術文章內容，精簡地歸納出 3 個關鍵重點 (Key Takeaways)。請用繁體中文輸出。"),
        ("user", "{text}")
    ])

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"text": article_content})
    
    return result


llm_with_tools = llm.bind_tools([generate_tech_summary])

router_prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break
    
    chain = router_prompt | llm_with_tools
    ai_msg = chain.invoke({"input": user_input})
    if ai_msg.tool_calls:
        print(f"✅ [決策] 判斷為科技文章")
        tool_args = ai_msg.tool_calls[0]['args']

        final_result = generate_tech_summary.invoke(tool_args)
        
        print(f"📄 [執行結果]:\n{final_result}")
        
    else:
        print(f"❌ [決策] 判斷為閒聊/非科技文章，直接回覆。")
        print(f"💬 [AI 回應]: {ai_msg.content}")