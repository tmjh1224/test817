import json
import warnings
import httpx
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode  # 官方標準工具節點

# 忽略關閉 SSL 驗證產生的警告
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# ================= 配置區 =================
http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0
)

# ================= 1. 定義工具 (Tools) =================
@tool
def get_weather(city: str):
    """查詢指定城市的天氣。輸入參數 city 必須是城市名稱。"""
    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    elif "台中" in city:
        return "台中晴天，氣溫 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"

tools = [get_weather]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================= 2. 系統提示詞（用來模擬 Tool Calling） =================
SYSTEM_PROMPT = """你是一個天氣查詢助手。
若使用者詢問特定城市的天氣，你必須呼叫工具 `get_weather`。
請嚴格僅輸出以下格式的 JSON 來調用工具，不要包含任何額外說明或 Markdown 標籤：
{
  "tool": "get_weather",
  "args": {
    "city": "城市名稱"
  }
}
若使用者只是打招呼或詢問一般問題，請正常用文字回應即可。
"""

# ================= 3. 定義節點 (Nodes) =================
def chatbot_node(state: AgentState):
    """思考節點：負責呼叫 LLM，並將回傳結果解析為標準 tool_calls"""
    messages = state["messages"]
    prompt_messages = [HumanMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(prompt_messages)
    
    content = response.content.strip()
    ai_msg = AIMessage(content=content)
    
    # 嘗試解析模型輸出的 JSON 是否為工具調用
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("tool") == "get_weather":
            ai_msg.tool_calls = [{
                "name": "get_weather",
                "args": data.get("args", {}),
                "id": "call_weather_1"
            }]
    except Exception:
        pass

    return {"messages": [ai_msg]}

# 定義工具節點 (使用 LangGraph 內建的 ToolNode)
tool_node_executor = ToolNode(tools)

# ================= 4. 定義邊 (Edges & Router) =================
def router(state: AgentState) -> Literal["tools", "end"]:
    """路由邏輯：決定下一步是執行工具還是結束"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 檢查最後一則訊息是否有 tool_calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# ================= 5. 組裝 Graph =================
workflow = StateGraph(AgentState)

# (1) 加入節點
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)

# (2) 設定入口
workflow.set_entry_point("agent")

# (3) 設定條件邊 (Conditional Edge)
workflow.add_conditional_edges(
    "agent",     # 從 agent 出發
    router,      # 經過 router 判斷
    {
        "tools": "tools",  # 如果 router 回傳 "tools"，走向 tools 節點
        "end": END         # 如果 router 回傳 "end"，走向結束
    }
)

# (4) 設定普通邊 (Normal Edge)
workflow.add_edge("tools", "agent")

# (5) 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())

# ================= 6. 執行測試 =================
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "q"]: break

        final_state = app.invoke({"messages": [HumanMessage(content=user_input)]})

        for msg in final_state["messages"]:
            role = msg.type
            content = msg.content
            if role == "ai" and getattr(msg, "tool_calls", None):
                print(f"[AI 呼叫工具]: {msg.tool_calls}")
            elif role == "tool":
                print(f"[工具回傳]: {content}")
            else:
                print(f"[{role}]: {content}")