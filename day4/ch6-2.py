import random
import json
from typing import Annotated, TypedDict, Union, Literal
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
import random
import json
import os

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
# =========================================

# 1. 定義狀態
class State(TypedDict):
    original_text: str      # 原始文字
    translated_text: str    # 翻譯結果
    critique: str           # 評語
    attempts: int           # 重試次數 (防止無窮迴圈)

# 2. 定義節點 (Node)

def translator_node(state: State):
    """負責翻譯的節點"""
    print(f"\n--- 翻譯嘗試 (第 {state['attempts'] + 1} 次) ---")

    # 建構 Prompt
    prompt = f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋: '{state['original_text']}'"

    # 如果有之前的批評，把它加進去，讓模型知道要改哪裡
    if state['critique']:
        prompt += f"\n\n上一輪的審查意見是: {state['critique']}。請根據意見修正翻譯。"

    # 呼叫 LLM（加上例外處理，避免端點斷線/逾時直接讓程式崩潰）
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        translated = response.content
    except Exception as e:
        print(f"[WARN] 翻譯呼叫失敗: {e}")
        # 失敗時保留原本的翻譯結果（若有），避免狀態被清空；
        # 若是第一次就失敗，退回顯示原文，並讓 attempts 照樣累加，
        # 之後 should_continue 的重試上限一樣能正常擋住無窮迴圈。
        translated = state.get("translated_text") or state["original_text"]

    return {
        "translated_text": translated,
        "attempts": state['attempts'] + 1
    }

def reflector_node(state: State):
    """負責審查的節點 (Critique)"""
    print("--- 審查中 (Reflection) ---")
    print(f"翻譯: {state['translated_text']}")
    prompt = f"""
    你是一個嚴格的翻譯審查員。
    原文: {state['original_text']}
    翻譯: {state['translated_text']}

    請檢查翻譯是否準確且通順。
    - 如果翻譯很完美，請只回覆 "PASS"。
    - 如果需要修改，請給出簡短的具體建議。
    """

    # 呼叫 LLM（同樣加上例外處理）
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        critique = response.content
    except Exception as e:
        print(f"[WARN] 審查呼叫失敗: {e}")
        # 審查端點掛掉時，不要讓程式崩潰；當作「未通過」處理，
        # 讓 should_continue 依 attempts 上限決定要不要繼續重試。
        critique = "REVIEW_FAILED: 審查服務暫時無法使用"

    return {"critique": critique}

# 3. 定義邊 (Edge) - 決策邏輯
def should_continue(state: State) -> Literal["translator", "end"]:
    critique = state['critique'].strip().upper()

    # ★ 修正：原本用 "PASS" in critique 是子字串比對，
    #   如果審查意見裡剛好出現 "PASS" 這幾個字母（例如某個建議句子裡
    #   含有 "passive" 之類的字），會被誤判成「通過」。
    #   改成 startswith，只有模型真的「只回覆 PASS」時才算通過，
    #   跟原本的功能設計（"如果翻譯很完美，請只回覆 PASS"）更一致。
    if critique.startswith("PASS"):
        print("--- 審查通過！ ---")
        return "end"
    elif state['attempts'] >= 3:
        print("--- 達到最大重試次數，強制結束 ---")
        return "end"
    else:
        print(f"--- 審查未通過: {state['critique']} ---")
        print("--- 退回重寫 ---")
        return "translator"

# 4. 組裝 Graph
workflow = StateGraph(State)

workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

workflow.set_entry_point("translator")

workflow.add_edge("translator", "reflector")

workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "translator": "translator",
        "end": END
    }
)

app = workflow.compile()
print(app.get_graph().draw_ascii())
if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]: break
        inputs = {
            "original_text": user_input, # 這種反諷句模型第一次容易翻錯
            "attempts": 0,
            "critique": ""
        }

        result = app.invoke(inputs)
        print("\n========== 最終結果 ==========")
        print(f"原文: {result['original_text']}")
        print(f"最終翻譯: {result['translated_text']}")
        print(f"最終次數: {result['attempts']}")