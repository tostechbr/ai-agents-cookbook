import os
import sqlite3
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


SYSTEM_PROMPT = (
    "You are a persistent personal assistant. "
    "Your goal is to remember user preferences (name, language, favorite foods, "
    "preferred time, etc.) over time. "
    "Always respond in English, be concise and helpful. "
    "When the user mentions a preference, acknowledge it and use that "
    "information in future interactions."
)


def build_personal_assistant(checkpointer: SqliteSaver) -> StateGraph:
    """Creates the personal assistant graph with persistent memory."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def assistant_node(state: AgentState) -> dict:
        """Main node: receives history + new message and calls the LLM."""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.set_entry_point("assistant")
    builder.add_edge("assistant", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph


def init_checkpointer(db_path: str = "personal_assistant.sqlite") -> SqliteSaver:
    """Initializes a SqliteSaver pointing to the memory file."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def load_env_and_model() -> None:
    """Loads environment variables and ensures the API key exists."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API Key: ")


def main() -> None:
    """
    Simple CLI for the Personal Assistant.

    Special commands:
    - /exit      -> ends the conversation
    - /reset     -> clears all memory for the current thread
    - /history   -> shows how many checkpoints exist for the thread
    - /undo      -> reverts to the previous checkpoint (simple time travel)
    """
    load_env_and_model()

    thread_id = os.environ.get("ASSISTANT_THREAD_ID") or input(
        "Choose a session identifier (thread_id), e.g., 'user-john': "
    )

    checkpointer = init_checkpointer()
    graph = build_personal_assistant(checkpointer)

    # Initial thread configuration
    config = {"configurable": {"thread_id": thread_id}}

    print(
        f"\n✨ Personal Assistant ready! Thread: '{thread_id}'.\n"
        "Type messages normally to chat.\n"
        "Special commands:\n"
        "  /exit    -> exit\n"
        "  /reset   -> clear memory for this thread\n"
        "  /history -> list checkpoint count\n"
        "  /undo    -> revert to previous checkpoint\n"
    )

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            print("Assistant: Goodbye! 👋")
            break

        if user_input == "/reset":
            # Note: In a real app, you might want to delete the thread from DB.
            # For now, we just notify the user.
            print("Assistant: [System] Memory reset is not fully implemented in this demo (requires DB deletion). Start a new thread ID.")
            continue

        if user_input == "/history":
            cfg_list = {"configurable": {"thread_id": thread_id}}
            checkpoints = list(checkpointer.list(cfg_list))
            print(f"Assistant: Found {len(checkpoints)} checkpoints for this thread.")
            if checkpoints:
                last_id = checkpoints[-1].checkpoint.id
                print(f" - Last checkpoint id: {last_id}")
            continue

        if user_input == "/undo":
            cfg_list = {"configurable": {"thread_id": thread_id}}
            checkpoints = list(checkpointer.list(cfg_list))
            if len(checkpoints) < 2:
                print("Assistant: No previous checkpoint to undo to.")
                continue

            # Get the second to last checkpoint
            prev_cp = checkpoints[-2]
            prev_id = prev_cp.checkpoint.id
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": prev_id,
                }
            }
            print(f"Assistant: Reverting to previous checkpoint (id={prev_id}).")
            continue

        # Normal user message -> invoke graph
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        # Update config with the latest checkpoint_id (consistent time travel)
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        latest_cfg = snapshot.config.get("configurable", {})
        checkpoint_id = latest_cfg.get("checkpoint_id")
        if checkpoint_id:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

        # Show only the last assistant message
        last_message = result["messages"][-1]
        print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    main()


