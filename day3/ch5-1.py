import json
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
# LangGraph 必要元件
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

# --- 配置區 (維持原樣) ---
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

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


# 綁定工具
llm_with_tools = llm.bind_tools([extract_order_data])

# ================= 1. 元件一：State (狀態) =================
# 這是 Agent 的記憶體，add_messages 確保訊息是「疊加」而非覆蓋
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================= 2. 元件二：Nodes (節點) =================
# Node A: 思考節點 (負責呼叫 LLM)
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Node B: 工具節點 (LangGraph 內建的 ToolNode，省去自己寫執行邏輯)
tool_node = ToolNode([extract_order_data])

# ================= 3. 元件三：Edges (邊與決策) =================
# 判斷邏輯：LLM 有呼叫工具 -> 走工具節點；沒有 -> 結束
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- 組裝 Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model) # 加入節點 A
workflow.add_node("tools", tool_node)  # 加入節點 B

workflow.set_entry_point("agent")      # 設定起點

# 設定條件邊 (Conditional Edge)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

# 設定循環邊 (Loop): 工具做完後，把結果丟回給 Agent 再看一次
workflow.add_edge("tools", "agent")

app = workflow.compile()
print(app.get_graph().draw_ascii())
# ================= 測試執行 =================
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "q"]: break
        
        # 使用 stream 看過程，或 invoke 直接拿結果
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                # 這裡簡單印出最後一條訊息內容觀測
                print(value["messages"][-1].content or value["messages"][-1].tool_calls)
