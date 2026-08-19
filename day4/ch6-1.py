import json
import os
import random
import re
import warnings
from typing import Annotated, Literal, TypedDict, Union

import httpx
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

# 忽略關閉 SSL 驗證產生的警告
try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# ================= 配置區 =================
http_client = httpx.Client(verify=False, timeout=30.0)
llm = ChatOpenAI(
    base_url="https://163.17.136.119:8591/v1",
    api_key="sk-9o0cj_Q6aJWWbSLODEPKBQ",
    model="gemma-4-E4B-it",
    http_client=http_client,
    temperature=0,
)

# ================= 1. 定義工具 (Tools) =================
@tool
def get_weather(city: str):
    """查詢指定城市的天氣。"""
    # 模擬 100% 失敗以測試重試與熔斷機制
    if random.random() < 0.5:
        return "系統錯誤：天氣資料庫連線失敗，請再試一次。"

    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    elif "台中" in city:
        return "台中晴天，氣溫 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"


tools = [get_weather]
KNOWN_CITIES = ["台北", "台中", "高雄"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ================= 2. JSON / 城市名 擷取輔助函式 =================

def try_parse_tool_json(content: str):
    """
    多層次嘗試把模型輸出解析成 {"tool": "get_weather", "args": {"city": ...}}
    比原版更寬容：允許前後夾雜文字、單引號、全形符號等。
    解析失敗回傳 None。
    """
    candidates = []

    if "```json" in content:
        candidates.append(content.split("```json")[1].split("```")[0])
    if "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            candidates.append(parts[1])

    # 抓內容中「第一個完整的 {...}」區塊
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(content[start : end + 1])

    candidates.append(content)

    for raw in candidates:
        raw = raw.strip()
        if not raw:
            continue
        # 容錯：把中文全形引號、單引號都換成標準雙引號再試一次
        normalized = raw.replace("「", '"').replace("」", '"').replace("'", '"')
        for text in (raw, normalized):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and data.get("tool") == "get_weather":
                    return data.get("args", {}) or {}
            except Exception:
                continue
    return None


def extract_city_from_text(text: str):
    """從任意文字中用關鍵字比對出城市名，找不到回傳 None。"""
    for city in KNOWN_CITIES:
        if city in text:
            return city
    return None


def find_target_city(messages: list[BaseMessage]):
    """
    最後一道防線：往回找整個對話歷史（含使用者原始問題），
    只要出現過已知城市名稱，就用它來強制呼叫工具。
    """
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        city = extract_city_from_text(content)
        if city:
            return city
    return "台北"  # 真的完全找不到時的預設值


# ================= 3. 定義節點 (Nodes) =================

def chatbot_node(state: AgentState):
    """思考節點：具備多層安全網，確保模型輸出格式不穩定時工具仍能被觸發"""
    messages = state["messages"]
    last_msg = messages[-1] if messages else None
    is_tool_error = isinstance(last_msg, ToolMessage) and "系統錯誤" in last_msg.content

    # ★ 關鍵修正：偵測「剛從 fallback 熔斷節點回來」這個狀態。
    #   一旦是這個狀態，代表流程已經決定放棄重試，
    #   絕對不能再讓安全網把它拉回去強制呼叫工具，否則會無窮迴圈。
    is_after_maxretry = isinstance(last_msg, ToolMessage) and (
        "已達到最大重試次數" in last_msg.content or "系統警示" in last_msg.content
    )

    # ★ 新增：判斷「上一次工具其實已經成功」的狀態。
    #   只要是 ToolMessage，且內容不是系統錯誤、也不是熔斷警示，
    #   就代表已經拿到真正的天氣資料，不該再被安全網拉去重打工具。
    is_after_success = (
        isinstance(last_msg, ToolMessage) and not is_tool_error and not is_after_maxretry
    )

    system_prompt = SystemMessage(
        content="""你是一個負責查詢天氣的助手。
重要規則：
1. 若需要查詢天氣，請【嚴格僅輸出以下 JSON 格式】來呼叫工具（不要包含任何其他說明文字）：
{
  "tool": "get_weather",
  "args": {
    "city": "城市名稱"
  }
}
2. 如果工具回傳包含 "系統錯誤" 或 "失敗"，請【務必再次輸出上述 JSON】進行重試。
3. 如果收到 "系統警示：已達到最大重試次數"，請【不要再輸出 JSON】，改用中文親切告知使用者服務暫時無法使用。"""
    )

    prompt_messages = [system_prompt] + messages
    if is_tool_error:
        prompt_messages.append(
            HumanMessage(content="[系統指令]: 工具執行失敗！請立刻【僅輸出 JSON】重試呼叫 get_weather 工具！")
        )

    # 呼叫自架 LLM，網路/憑證問題不讓整個 graph 崩潰
    try:
        response = llm.invoke(prompt_messages)
        content = (response.content or "").strip()
    except Exception as e:
        content = ""
        print(f"[WARN] LLM 呼叫失敗: {e}")

    # ★ 已經熔斷過（剛執行完 fallback_node），流程必須結束，
    #   不解析、不強制呼叫工具，直接把模型的文字（道歉語）當最終回覆。
    #   即使模型不聽話還是吐出 JSON，也一律忽略，避免無窮迴圈。
    if is_after_maxretry:
        return {"messages": [AIMessage(content=content or "抱歉，天氣查詢服務目前暫時無法使用，請稍後再試。")]}

    # ★ 工具已經成功拿到天氣資料：不管模型這次回覆的文字格式對不對，
    #   都直接結束這輪，避免安全網把「已經成功」的結果又拉回去重打工具。
    #   如果模型有好好用自然語言回覆，就用它的回覆；
    #   萬一它還是吐出奇怪格式，就退而求其次直接用工具原始回傳內容當答案。
    if is_after_success:
        final_text = content if (content and try_parse_tool_json(content) is None) else last_msg.content
        return {"messages": [AIMessage(content=final_text)]}

    args = try_parse_tool_json(content)
    has_tool_call = args is not None

    # 安全網：只有在「一般提問」或「工具真的回傳系統錯誤要重試」這兩種情境下，
    # 若模型沒有照格式輸出合法 JSON，才用關鍵字從歷史裡把城市抓出來，強制補呼叫工具。
    # （已排除 is_after_maxretry 的情境，避免和上面的提前 return 衝突造成無窮迴圈）
    if not has_tool_call:
        city = extract_city_from_text(content) or find_target_city(messages)
        args = {"city": city}

    tool_call_id = f"call_weather_{random.randint(1000, 9999)}"
    ai_msg = AIMessage(
        content="",  # 有 tool_calls 時比照原生 function-call 行為，content 留空避免污染歷史紀錄
        tool_calls=[
            {
                "name": "get_weather",
                "args": args,
                "id": tool_call_id,
            }
        ],
    )

    return {"messages": [ai_msg]}


tool_node_executor = ToolNode(tools)


def fallback_node(state: AgentState):
    """備援節點：當重試次數過多時執行，產生 ToolMessage 結束工具迴圈"""
    last_message = state["messages"][-1]
    tool_call_id = (
        last_message.tool_calls[0]["id"]
        if getattr(last_message, "tool_calls", None)
        else "call_fallback_1"
    )

    error_message = ToolMessage(
        content="系統警示：已達到最大重試次數 (Max Retries Reached)。請停止嘗試，並告知使用者服務暫時無法使用。",
        tool_call_id=tool_call_id,
    )
    return {"messages": [error_message]}


# ================= 4. 定義邊 (Edges & Router) =================

def router(state: AgentState) -> Literal["tools", "fallback", "end"]:
    """路由邏輯"""
    messages = state["messages"]
    last_message = messages[-1]

    if not getattr(last_message, "tool_calls", None):
        return "end"

    # --- 計算連續錯誤次數 ---
    retry_count = 0
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            if "系統錯誤" in msg.content:
                retry_count += 1
            else:
                break
        elif isinstance(msg, HumanMessage):
            break

    print(f"DEBUG: 目前連續重試次數: {retry_count}")

    if retry_count >= 3:
        return "fallback"  # 超過 3 次，走向熔斷備援節點

    return "tools"


# ================= 5. 組裝 Graph =================
workflow = StateGraph(AgentState)

workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {"tools": "tools", "fallback": "fallback", "end": END},
)

workflow.add_edge("tools", "agent")
workflow.add_edge("fallback", "agent")

app = workflow.compile()
print(app.get_graph().draw_ascii())

# ================= 6. 執行測試 =================
if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]:
            break

        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][-1]
                    if getattr(msg, "tool_calls", None):
                        print(f" -> [Agent]: 決定呼叫工具 city={msg.tool_calls[0]['args']} (重試中...)")
                    else:
                        print(f" -> [Agent]: {msg.content}")
                elif key == "tools":
                    tool_msg = value["messages"][-1]
                    print(f" -> [Tools]: {tool_msg.content}")
                elif key == "fallback":
                    print(" -> [Fallback]: 🛑 觸發熔斷機制：停止重試")