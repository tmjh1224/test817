import random
import json
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
import random
import json
import os

# ================= 配置區 =================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.7
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

    # 呼叫 LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "translated_text": response.content,
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
    
    # 呼叫 LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"critique": response.content}

# 3. 定義邊 (Edge) - 決策邏輯
def should_continue(state: State) -> Literal["translator", "end"]:
    critique = state['critique'].strip().upper()
    
    if "PASS" in critique:
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