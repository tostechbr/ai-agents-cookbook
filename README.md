# LangGraph Agents Cookbook

A structured, practical guide to mastering AI Agents using LangGraph. This repository is designed as a progressive course, taking you from basic graph concepts to complex, production-ready multi-agent systems.

## Overview

This project adopts a "Course + Cookbook" approach. Each module focuses on a specific architectural pattern or capability, providing both theoretical context and hands-on implementation.

## Repository Structure

The curriculum is organized into sequential modules. Each module contains:

1.  **Concepts (`01_concepts.md`)**: Theoretical explanation of the module's core topics.
2.  **Lab (`02_lab.ipynb`)**: Isolated exercises to practice specific mechanics.
3.  **Mission (`03_mission.py`)**: A complete, functional agent implementation applying the module's concepts.

## Curriculum

| Module | Title | Project | Key Concepts |
| :--- | :--- | :--- | :--- |
| **01** | **Foundations** | *Structured Chatbot* | StateGraph, Nodes, Edges, State Schema |
| **02** | **Tool Use (ReAct)** | *Smart Analyst* | Tool Binding, Conditional Edges, Routing |
| **03** | **Memory** | *Persistent Assistant* | Checkpointers, SQLite, Thread Management |
| **04** | **Human-in-the-Loop** | *Email Drafter* | Interrupts, State Updates, Approval Flows |
| **05** | **Multi-Agent** | *Content Team* | Supervisor Pattern, Hierarchical Graphs |
| **06** | **Production** | *Enterprise RAG* | Configuration, Logging, Docker, Testing |
| **07** | **Full Stack** | *Agent Arena* | FastAPI, Streamlit, API Integration |

## Technical Stack

- **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LLMs:** Agnostic support (OpenAI, Anthropic, Ollama)
- **Validation:** Pydantic
- **Environment:** Python 3.11+

## Getting Started

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure environment variables:
    ```bash
    cp .env.example .env
    # Edit .env with your API keys
    ```
4.  Begin with `Module-01-Foundations`.
