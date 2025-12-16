from dotenv import load_dotenv 
load_dotenv() 

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import operator

# --- 1. Define Tools ---
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

tools = [multiply, add]

# --- 2. Define State ---
class AgentState(TypedDict):
    # 'operator.add' means: append new messages to the existing list
    messages: Annotated[list, operator.add]

# --- 3. Define Nodes ---
# Initialize LLM with tools
llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Pre-built node to run tools
tool_node = ToolNode(tools)

# --- 4. Define Router (Conditional Logic) ---
def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM made a tool call, go to "tools"
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end
    return END

# --- 5. Build Graph ---
builder = StateGraph(AgentState)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.set_entry_point("chatbot")

builder.add_conditional_edges(
    "chatbot",
    router,
    {"tools": "tools", END: END}
)

builder.add_edge("tools", "chatbot") # Loop back to chatbot after tool use

graph = builder.compile()

# --- 6. Run ---
if __name__ == "__main__":
    print("--- Calculating 5 * 23 ---")
    result = graph.invoke({"messages": [HumanMessage(content="What is 7 multiplied by 243?")]})
    for m in result['messages']:
        print(f"{m.type}: {m.content}")
