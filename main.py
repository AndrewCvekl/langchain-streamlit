"""Main entry point for the customer support chatbot (terminal).

This runner supports LangGraph human-in-the-loop interrupts.
"""

import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from graph_with_verification import create_agent_with_memory


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 60 + "\n")


def _interrupt_payload(interrupts):
    if not interrupts:
        return None
    first = interrupts[0]
    payload = getattr(first, "value", None)
    if payload is None and isinstance(first, dict):
        payload = first.get("value")
    return payload if payload is not None else first


def main():
    """Run the interactive chatbot in the terminal."""

    load_dotenv()

    print("=" * 60)
    print("🎵 Welcome to the Music Store Customer Support Bot! 🎵")
    print("=" * 60)
    print("\nType 'quit', 'q', or 'exit' to end the conversation.")
    print_separator()

    graph = create_agent_with_memory()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    messages = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"q", "quit", "exit"}:
            print("\nThank you for chatting! Have a great day! 👋")
            break

        messages.append(HumanMessage(content=user_input))

        result = graph.invoke({"messages": messages}, config=config)

        # HITL approval loop
        while isinstance(result, dict) and result.get("__interrupt__"):
            payload = _interrupt_payload(result["__interrupt__"])
            print("\n⏸️ Approval required for a sensitive action.")

            if isinstance(payload, dict) and payload.get("tool_calls"):
                for tc in payload["tool_calls"]:
                    print(f"- Tool: {tc.get('name')} Args: {tc.get('args', {})}")
            else:
                print(payload)

            decision = input("Approve? (y/n): ").strip().lower()
            resume = {"decision": "approve"} if decision in {"y", "yes"} else {"decision": "reject"}
            result = graph.invoke(Command(resume=resume), config=config)

        # Update local message history from graph state
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]

        # Print last assistant message with content
        final = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final = msg
                break

        print("\nBot:", final.content if final else "(no text response)")
        print_separator()


if __name__ == "__main__":
    main()
