# Module 02: Tool Use & Reasoning (ReAct)

## Goal
Give the agent "arms" and "eyes" by connecting it to external tools (Search, Calculator, APIs).

## Theory

### 1. The ReAct Pattern
ReAct stands for **Re**asoning + **Act**ing.
Instead of just answering, the model:
1.  **Thinks:** "I need to calculate 25 * 48."
2.  **Acts:** Calls the `multiply(25, 48)` tool.
3.  **Observes:** Gets the result `1200`.
4.  **Answers:** "The answer is 1200."

### 2. Binding Tools (`bind_tools`)
LLMs don't "run" code. They generate text (JSON) that *looks* like a function call.
We use `llm.bind_tools([tool1, tool2])` to tell the model which tools are available.

### 3. The ToolNode
LangGraph provides a pre-built `ToolNode`.
It receives the tool call request from the LLM, executes the actual Python function, and returns the output to the state.

### 4. Conditional Edges (The Router)
This is the brain of the agent.
After the LLM runs, we check:
- **Did it ask for a tool?** -> Go to `tools` node.
- **Did it give a final answer?** -> Go to `END`.

```python
def router(state):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END
```

## Practice: What We Built

In this module's lab and mission, we implemented a simple **Math Agent**.

### 1. The Tool
We defined a simple Python function:
```python
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b
```
And bound it to the LLM using `llm.bind_tools([multiply])`.

### 2. The Graph Structure
We built a `StateGraph` with the following flow:
1.  **Start**: User input ("What is 5 * 23?").
2.  **Chatbot Node**: The LLM analyzes the input. It sees a math problem and decides to call the `multiply` tool.
3.  **Router**: Detects the tool call and directs the flow to the `tools` node.
4.  **Tools Node**: Executes `multiply(5, 23)` and returns `115`.
5.  **Chatbot Node (Again)**: Receives the tool output (`115`). It now has the answer and generates the final response ("The answer is 115.").
6.  **Router**: Detects a final answer (no tool calls) and directs to `END`.

### 3. Key Takeaway
The power of this pattern is that the LLM is not just a text generator anymore; it is an **orchestrator** that can control software (tools) to accomplish tasks.
