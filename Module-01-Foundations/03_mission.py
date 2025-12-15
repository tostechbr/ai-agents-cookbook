import os
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Define the State
class State(TypedDict):
    count: int

# 2. Define Nodes
def increment(state: State):
    print(f"Current count: {state['count']}")
    return {"count": state['count'] + 1}

# 3. Build the Graph
builder = StateGraph(State)
builder.add_node("increment", increment)
builder.set_entry_point("increment")
builder.add_edge("increment", END)

# 4. Compile
graph = builder.compile()

# 5. Run
result = graph.invoke({"count": 0})
print(f"Final Result: {result}")
