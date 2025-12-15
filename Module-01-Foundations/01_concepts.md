# Module 01: The Graph Foundation

## Goal
Understand the core building blocks of LangGraph: `State`, `Nodes`, and `Edges`.

## Theory

### 1. The Shift from Chains to Graphs
Standard LangChain applications often rely on "Chains" (sequences of steps). LangGraph introduces the concept of a "StateGraph", which models the application as a state machine. This allows for cyclic execution, persistence, and more complex control flows.

### 2. The State (`TypedDict`)
The "State" is a shared data structure (typically a `TypedDict`) that is passed between all nodes in the graph. Every node receives the current state, performs an operation, and returns an update to that state.

```python
from typing import TypedDict

class AgentState(TypedDict):
    messages: list[str]
    current_step: str
```

### 3. Nodes
Nodes are standard Python functions. They receive the state as input and return a dictionary representing the updates to be applied to the state.

```python
def chatbot_node(state: AgentState):
    # Logic to process state
    return {"messages": ["Hello from the bot!"]}
```

### 4. Edges
Edges define the control flow between nodes.
- **Normal Edge:** Deterministic transition from Node A to Node B.
- **Conditional Edge:** Dynamic transition where the next node is determined by a function (router) based on the current state.
