import json
import warnings
import httpx
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
# LangGraph 必要元件
from langgraph.graph import StateGraph, END, add_messages

# 忽略關閉 SSL 驗證產生的警告
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# --- 配置區 ---
http_client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0
)

# --- 定義工具 ---
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

# ================= 1. 元件一：State (狀態) =================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ================= 2. 元件二：Nodes (節點) =================
# System Prompt 指導模型在需要調用工具時輸出特定格式 JSON
SYSTEM_PROMPT = """你是一個訂單處理助手。
若使用者提供訂單資訊，你必須呼叫工具 `extract_order_data`。
調用工具時，請**嚴格僅輸出**如下格式的 JSON，不要包含任何額外說明或 Markdown 標籤：
{
  "tool": "extract_order_data",
  "args": {
    "name": "姓名",
    "phone": "電話",
    "product": "商品名稱",
    "quantity": 數量數字,
    "address": "地址"
  }
}
若使用者只是打招呼或詢問一般問題，請正常回應即可。
"""

def call_model(state: AgentState):
    messages = state["messages"]
    prompt_messages = [HumanMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(prompt_messages)
    
    # 解析模型輸出是否包含工具調用格式
    content = response.content.strip()
    ai_msg = AIMessage(content=content)
    
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("tool") == "extract_order_data":
            ai_msg.tool_calls = [{
                "name": "extract_order_data",
                "args": data.get("args", {}),
                "id": "call_manual_1"
            }]
    except Exception:
        pass

    return {"messages": [ai_msg]}

def custom_tool_node(state: AgentState):
    """自訂工具執行節點，執行 extract_order_data 並將結果包裝回傳"""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    args = tool_call["args"]
    
    # 執行工具
    result = extract_order_data.invoke(args)
    
    tool_msg = ToolMessage(
        content=json.dumps(result, ensure_ascii=False, indent=2),
        tool_call_id=tool_call["id"]
    )
    return {"messages": [tool_msg]}

# ================= 3. 元件三：Edges (邊與決策) =================
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# --- 組裝 Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", custom_tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

workflow.add_edge("tools", "agent")

app = workflow.compile()
print(app.get_graph().draw_ascii())

# ================= 測試執行 =================
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "q"]: 
            break
        
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                last_msg = value["messages"][-1]
                print(last_msg.content)