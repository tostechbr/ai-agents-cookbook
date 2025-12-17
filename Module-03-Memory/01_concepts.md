## Module 03: Memory & Persistence (Time Travel)

### Goal

Understand how to give **real long‑term memory** to agents built with LangGraph, using:

- **Checkpointers** (saving graph state)
- **SQLite (`SqliteSaver`)** to persist that state to disk
- **`thread_id`** to separate conversations/sessions
- **Time travel** concepts (jumping back to a past state)

---

### 1. From “context window” to **memory**

Up to now:

- In **Module 01**, the state only existed **in memory** during execution.
- In **Module 02**, we kept a `messages` history while the graph was running, but nothing was persisted.

This is fine for quick demos, but has issues:

- If the process crashes, everything is lost.
- There are no separate user **sessions**.
- You cannot **go back in time** to a previous state.

LangGraph solves this with the concept of a **Checkpointer**.

---

### 2. Checkpointers: saving graph state

A **checkpointer** is responsible for saving and restoring the internal state of the graph.

Main implementations:

- `InMemorySaver` – simple, great for tutorials, does **not** persist to disk.
- `SqliteSaver` – uses SQLite (a `.db` file), great for development and small projects.
- Others (Postgres, etc.) – for more serious production setups.

Example from the official docs of an in‑memory checkpointer:

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)
```

With `SqliteSaver`, we switch to something persistent:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

builder = StateGraph(int)
builder.add_node("add_one", lambda x: x + 1)
builder.set_entry_point("add_one")
builder.set_finish_point("add_one")

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

result = graph.invoke(3, config)
state_snapshot = graph.get_state(config)
```

This pattern (builder + `compile(checkpointer=...)`) is exactly what we will use in this module’s mission.

---

### 3. `thread_id`: separating sessions and users

When a graph has a checkpointer, it needs to know **which “thread”** (session) it should save state for.

This is done via `config`:

```python
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(input_state, config=config)
```

Important rules:

- `thread_id` is **required** when a checkpointer is configured.
- Each `thread_id` represents an independent timeline.
- If you invoke the same graph multiple times with the same `thread_id`, LangGraph:
  - loads the latest checkpoint for that thread;
  - continues execution from there;
  - writes a new checkpoint at the end.

This is the basis of an assistant’s **long‑term memory**.

---

### 4. Time travel: navigating between checkpoints

With a checkpointer (for example, `SqliteSaver`), you can:

- **List checkpoints** for a thread.
- **Read** the latest saved state.
- **Fetch a specific checkpoint** by `checkpoint_id`.

Example (based on the LangGraph docs):

```python
config = {"configurable": {"thread_id": "user-123"}}

# List checkpoints
checkpoints = list(checkpointer.list(config))

# Get latest checkpoint
latest = checkpointer.get(config)

# Get a specific checkpoint
config_with_id = {
    "configurable": {
        "thread_id": "user-123",
        "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a",
    }
}
tuple_checkpoint = checkpointer.get_tuple(config_with_id)
```

This mechanism allows you to implement features such as:

- **“Undo last action”** (jump back to a previous checkpoint).
- **Review an agent’s decision history**.
- **Visual debugging** in tools like LangGraph Studio.

---

### 5. Example: agent with persistent memory (personal assistant)

Below is an example inspired by the official LangGraph tutorials: an agent that remembers the user’s name across calls, using `SqliteSaver`.

```python
import sqlite3
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def chatbot(state: AgentState):
    """Main node that calls the LLM."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.set_entry_point("chatbot")
builder.add_edge("chatbot", END)

# Configure SQLite checkpointer
conn = sqlite3.connect("assistant_memory.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user-alice"}}

    # 1st call: the agent learns the name
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="My name is Alice. Remember that.")]},
        config=config,
    )

    # 2nd call: the agent should remember
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config=config,
    )

    for m in result2["messages"]:
        print(m.type, ":", m.content)
```

Main idea:

- `thread_id = "user-alice"` identifies Alice’s timeline.
- `SqliteSaver` ensures that the message history is persisted in `assistant_memory.sqlite`.
- On a new run of the script, the agent continues from where it left off.

---

### 6. What you will build in this module

In this module you will:

- Build small graphs with memory (using both `InMemorySaver` and `SqliteSaver`).
- Manipulate `thread_id` to simulate multiple users.
- Explore how to **list** and **inspect** checkpoints.
- Build a **Personal Assistant** that remembers user preferences across sessions.

In the next file (`02_lab.ipynb`), you’ll get guided exercises to practice these concepts. Then, in `03_mission.py`, you’ll apply everything in a more realistic project.
