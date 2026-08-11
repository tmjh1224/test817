import operator
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode # 這是官方提供的標準工具節點，省得自己寫執行邏輯

# ================= 配置區 =================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
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
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================= 3. 定義節點 (Nodes) =================

def chatbot_node(state: AgentState):
    """思考節點：負責呼叫 LLM"""
    # 傳入目前的對話紀錄，LLM 決定要回話還是呼叫工具
    response = llm_with_tools.invoke(state["messages"])
    
    # 回傳的 dict 會自動合併進 State
    return {"messages": [response]}

# 定義工具節點 (使用 LangGraph 內建的 ToolNode)
# 它會自動讀取上一則訊息的 tool_calls，執行對應工具，並產生 ToolMessage
tool_node_executor = ToolNode(tools)

# ================= 4. 定義邊 (Edges & Router) =================

def router(state: AgentState) -> Literal["tools", "end"]:
    """路由邏輯：決定下一步是執行工具還是結束"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 檢查最後一則訊息是否有 tool_calls
    if last_message.tool_calls:
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
    "agent",       # 從 agent 出發
    router,        # 經過 router 判斷
    {
        "tools": "tools",  # 如果 router 回傳 "tools"，走向 tools 節點
        "end": END         # 如果 router 回傳 "end"，走向結束
    }
)

# (4) 設定普通邊 (Normal Edge)
# 工具執行完，一定要把結果丟回給 Agent，讓 Agent 消化後再回答使用者
# 這就是 "Loop" 的由來
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
            if role == "ai" and msg.tool_calls:
                print(f"[AI 呼叫工具]: {msg.tool_calls}")
            elif role == "tool":
                print(f"[工具回傳]: {content}")
            else:
                print(f"[{role}]: {content}")