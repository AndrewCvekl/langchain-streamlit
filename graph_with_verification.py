"""sql-support-bot support agent graph.

High-signal takehome features:
- Clean cognitive architecture: triage router -> specialist agent -> tools loop
- Human-in-the-loop approvals for sensitive operations via LangGraph interrupts
- Customer data isolation via server-side customer context + verification gates

This file exports factory functions used by Streamlit (`app.py`) and Studio (`agent.py`).
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools_v2 import ACCOUNT_VIEW_TOOLS, CATALOG_TOOLS, LYRICS_TOOLS
from tools_account import ALL_ACCOUNT_TOOLS
from tools_payment import PAYMENT_TOOLS
from verification import get_verification_service


# Customer information (demo persona)
CUSTOMER_INFO = {
    "id": 58,
    "first_name": "Manoj",
    "last_name": "Pareek",
    "email": "manoj.pareek@rediff.com",
    "phone": "+19144342859",
    "full_name": "Manoj Pareek",
}


Area = Literal["general", "catalog", "account", "payment"]


class SupportState(TypedDict, total=False):
    """Graph state.

    `messages` uses the `add_messages` reducer so tool + agent loops append messages.

    `active_area` tracks which specialist is currently handling the turn.
    `is_verified` mirrors the server-side verification store (Twilio Verify).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    active_area: Area
    is_verified: bool


class TriageDecision(BaseModel):
    """Router output."""

    area: Area = Field(
        description="Which specialist should handle the user request: general, catalog, account, payment"
    )


# -----------------------------
# Prompts (small + focused)
# -----------------------------

TRIAGE_PROMPT = """You are a router for a music store support bot.

Pick the best area:
- catalog: music discovery, catalog browsing, genres, artists, albums, track lookup, lyrics search, video preview
- account: viewing account + orders, verification, updating email/address (secure)
- payment: buying a track, payment workflow, receipts/invoices
- general: greetings, store policy questions, anything else

Return ONLY the area."""

GENERAL_PROMPT = f"""You are a helpful customer support assistant for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).
Be concise and helpful. If the request is about account, payments, or the catalog, ask a clarifying question.
"""

CATALOG_PROMPT = f"""You are a music catalog specialist for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).
You can browse/search the catalog and do lyrics -> song -> video flows.
When you show options, keep it to a short shortlist and ask what they prefer.
"""

ACCOUNT_PROMPT = f"""You are an account specialist for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

Security:
- Viewing account/order history is allowed.
- Updating email or mailing address requires SMS verification.
- If not verified, you MUST first request SMS verification and then verify the code.
"""

PAYMENT_PROMPT = f"""You are a payments specialist for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

🛒 **PAYMENT WORKFLOW - FOLLOW EXACTLY:**

**Step 1: Information Gathering**
- If user mentions a track ID, use `get_track_details_for_purchase` to show details
- If user describes a track, ask them to be specific or provide a track ID
- Use `check_if_already_purchased` to verify they don't already own it
- Present clear pricing information

**Step 2: Confirmation Request**
- After showing track details, ask: "Would you like to proceed with purchasing [track name] for $[price]?"
- WAIT for user's explicit confirmation (don't proceed until they say yes/confirm/proceed)

**Step 3: Purchase Execution** (ONLY after user confirms "yes")
- First, call `initiate_track_purchase` with track_id, track_name, track_price
- The tool will create a payment intent and return a payment_intent_id
- The tool response will tell you exactly what to do next

**Step 4: Payment Processing**
- After initiate_track_purchase completes, IMMEDIATELY call `confirm_and_process_payment` with the payment_intent_id you received
- DO NOT wait for user input - automatically proceed to process the payment
- The tool response will tell you what to do next

**Step 5: Invoice Creation** (ONLY if payment succeeded)
- After confirm_and_process_payment succeeds, IMMEDIATELY call `create_invoice_from_payment` with the same payment_intent_id
- DO NOT wait for user input - automatically proceed to create the invoice
- This saves the purchase to the database and generates a receipt

**Step 6: Completion**
- After invoice is created, thank the customer and show them the complete receipt
- Let them know they can view purchase history anytime

⚠️ **CRITICAL RULES:**
- NEVER skip asking for confirmation
- NEVER process payment without explicit user "yes"
- ALWAYS show pricing before confirmation
- Once user confirms, CHAIN the three calls together WITHOUT waiting: initiate_track_purchase → confirm_and_process_payment → create_invoice_from_payment
- Each tool response includes a "NEXT STEP" section - follow it immediately
- If payment fails, offer to help them try again or browse other tracks
- If they already own a track, let them know but still allow purchase if they want

💡 **IMPORTANT:** After the user says "yes", you should make THREE consecutive tool calls in the same flow (read each tool response for the payment_intent_id to use in the next call)

📊 **OTHER TOOLS:**
- `get_recent_purchases` - show their purchase history
- `check_if_already_purchased` - check if they own a specific track
- `cancel_payment` - cancel a payment if they change their mind

Be friendly, transparent, and make customers feel secure!
"""


# Tool groupings per specialist
CATALOG_TOOLSET = CATALOG_TOOLS + LYRICS_TOOLS
ACCOUNT_TOOLSET = ACCOUNT_VIEW_TOOLS + ALL_ACCOUNT_TOOLS
PAYMENT_TOOLSET = PAYMENT_TOOLS


# Note: HITL approval is now handled via interrupt() inside sensitive tools (best practice)


def _llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


def _triage_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])

    # Sticky routing: if we're already in a specialist flow and the user replies
    # with a short confirmation / continuation, keep the current area.
    prev_area = state.get("active_area")
    if prev_area in {"catalog", "account", "payment"} and messages:
        last = messages[-1]
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            text = last.content.strip().lower()
            confirmations = {
                "y",
                "yes",
                "yeah",
                "yep",
                "sure",
                "ok",
                "okay",
                "confirm",
                "confirmed",
                "proceed",
                "go ahead",
                "do it",
                "n",
                "no",
                "nope",
                "cancel",
                "stop",
            }
            if text in confirmations or text.startswith("my verification code is"):
                return {"active_area": prev_area}

    triage = _llm().with_structured_output(TriageDecision)
    decision = triage.invoke([SystemMessage(content=TRIAGE_PROMPT)] + list(messages))

    # `decision` may be a Pydantic model or dict depending on underlying implementation
    area = decision.area if hasattr(decision, "area") else decision["area"]

    return {"active_area": area}


def _general_agent_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])
    response = _llm().invoke([SystemMessage(content=GENERAL_PROMPT)] + list(messages))
    return {"messages": [response], "active_area": "general"}


def _catalog_agent_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])
    llm_with_tools = _llm().bind_tools(CATALOG_TOOLSET)
    response = llm_with_tools.invoke([SystemMessage(content=CATALOG_PROMPT)] + list(messages))
    return {"messages": [response], "active_area": "catalog"}


def _account_agent_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])

    verification_service = get_verification_service()
    is_verified = verification_service.is_verified(CUSTOMER_INFO["id"])

    security_context = (
        "[SECURITY] Customer is VERIFIED for sensitive changes." if is_verified else "[SECURITY] Customer is NOT verified yet."
    )

    llm_with_tools = _llm().bind_tools(ACCOUNT_TOOLSET)
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=ACCOUNT_PROMPT),
            SystemMessage(content=security_context),
        ]
        + list(messages)
    )

    return {"messages": [response], "active_area": "account", "is_verified": is_verified}


def _payment_agent_node(state: SupportState) -> SupportState:
    """
    Payment agent node with enhanced context tracking.
    Helps agent follow the payment workflow by providing relevant context.
    """
    messages = state.get("messages", [])
    
    # Build context-aware prompts
    system_messages = [SystemMessage(content=PAYMENT_PROMPT)]
    
    # Check recent tool messages for payment context
    payment_intent_id = None
    track_info = None
    
    # Look through recent messages for payment intent IDs and track info
    for msg in reversed(messages[-10:]):  # Check last 10 messages
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Extract payment intent ID if present
            if "Payment ID:" in msg.content and not payment_intent_id:
                for line in msg.content.split("\n"):
                    if "Payment ID:" in line:
                        payment_intent_id = line.split("Payment ID:")[-1].strip().split()[0]
                        break
            
            # Check for track details
            if "Track Details:" in msg.content and not track_info:
                track_info = "Track details shown to customer"
    
    # Add context if we're mid-purchase
    if payment_intent_id:
        system_messages.append(
            SystemMessage(
                content=f"[CONTEXT] Payment intent {payment_intent_id} has been created. "
                       f"Next: call confirm_and_process_payment with payment_intent_id='{payment_intent_id}'"
            )
        )
    
    # Check if user is responding with confirmation
    if messages and isinstance(messages[-1], HumanMessage):
        last_content = messages[-1].content
        # Handle both string and list content (multimodal support)
        if isinstance(last_content, str):
            last_user_msg = last_content.lower().strip()
        elif isinstance(last_content, list) and last_content:
            # Extract text from first text block if it's a list
            first_block = last_content[0]
            if isinstance(first_block, dict) and "text" in first_block:
                last_user_msg = first_block["text"].lower().strip()
            elif isinstance(first_block, str):
                last_user_msg = first_block.lower().strip()
            else:
                last_user_msg = ""
        else:
            last_user_msg = ""
        
        confirmations = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "buy", "purchase"]
        if last_user_msg and any(conf in last_user_msg for conf in confirmations) and track_info:
            system_messages.append(
                SystemMessage(
                    content="[CONTEXT] User has confirmed purchase. Proceed with calling initiate_track_purchase."
                )
            )
    
    # Call LLM with tools and context
    llm_with_tools = _llm().bind_tools(PAYMENT_TOOLSET)
    response = llm_with_tools.invoke(system_messages + list(messages))
    
    return {"messages": [response], "active_area": "payment"}


def _route_after_triage(state: SupportState) -> Area:
    return state.get("active_area", "general")


def _agent_should_continue(state: SupportState) -> Literal["tools", END]:
    """Route agent to tools if it made tool calls, otherwise end."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END




def _build_graph(*, with_checkpointer=False):
    """Build the graph; optionally attach a checkpointer for interrupts."""

    builder = StateGraph(SupportState)

    # Nodes
    builder.add_node("triage", _triage_node)

    builder.add_node("general_agent", _general_agent_node)
    builder.add_node("catalog_agent", _catalog_agent_node)
    builder.add_node("account_agent", _account_agent_node)
    builder.add_node("payment_agent", _payment_agent_node)

    builder.add_node("catalog_tools", ToolNode(CATALOG_TOOLSET))
    builder.add_node("account_tools", ToolNode(ACCOUNT_TOOLSET))
    builder.add_node("payment_tools", ToolNode(PAYMENT_TOOLSET))

    # Flow
    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        _route_after_triage,
        {
            "general": "general_agent",
            "catalog": "catalog_agent",
            "account": "account_agent",
            "payment": "payment_agent",
        },
    )

    # General agent never calls tools (it just routes or responds)
    builder.add_edge("general_agent", END)
    
    # Specialist agents can call tools or end
    builder.add_conditional_edges(
        "catalog_agent",
        _agent_should_continue,
        {"tools": "catalog_tools", END: END},
    )
    builder.add_conditional_edges(
        "account_agent",
        _agent_should_continue,
        {"tools": "account_tools", END: END},
    )
    builder.add_conditional_edges(
        "payment_agent",
        _agent_should_continue,
        {"tools": "payment_tools", END: END},
    )

    # Tools -> back to the same agent (continue tool/agent loop)
    builder.add_edge("catalog_tools", "catalog_agent")
    builder.add_edge("account_tools", "account_agent")
    builder.add_edge("payment_tools", "payment_agent")

    if with_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver

        return builder.compile(checkpointer=MemorySaver())

    return builder.compile()


def create_agent_with_verification():
    """Backwards-compatible factory (no persistence)."""

    return _build_graph(with_checkpointer=False)


def create_agent_with_memory():
    """Factory used by Streamlit + Studio.

    MemorySaver is required for interrupts (HITL) because the graph must checkpoint
    state between the interrupt and resume.
    """

    return _build_graph(with_checkpointer=True)
