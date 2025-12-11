"""sql-support-bot support agent graph.

High-signal takehome features:
- Clean cognitive architecture: router -> specialist agent -> tools loop
- Human-in-the-loop approvals for sensitive operations via LangGraph interrupts
- Customer data isolation via server-side customer context + verification gates

This file exports factory functions used by Streamlit (`app.py`) and Studio (`agent.py`).
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict
import logging

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

# Set up logging
logger = logging.getLogger(__name__)

from tools_v2 import ACCOUNT_VIEW_TOOLS, CATALOG_TOOLS, LYRICS_TOOLS
from tools_v2 import search_tracks
from tools_account import ALL_ACCOUNT_TOOLS
from tools_payment import PAYMENT_TOOLS
from verification import get_verification_service
from guardrails import validate_user_input


# Customer information (demo persona)
CUSTOMER_INFO = {
    "id": 58,
    "first_name": "Manoj",
    "last_name": "Pareek",
    "email": "manoj.pareek@rediff.com",
    "phone": "+19144342859",
    "full_name": "Manoj Pareek",
}


Area = Literal["general", "music", "account", "payment"]


class SupportState(TypedDict, total=False):
    """Graph state.

    `messages` uses the `add_messages` reducer so tool + agent loops append messages.

    `active_area` tracks which specialist is currently handling the turn.
    `is_verified` mirrors the server-side verification store (Twilio Verify).
    `approval_status` tracks whether approval gate approved or rejected the operation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    active_area: Area
    is_verified: bool
    approval_status: Literal["approved", "rejected", None]


class RouterDecision(BaseModel):
    """Router output."""

    area: Area = Field(
        description="Which specialist should handle the user request: general, music, account, payment"
    )


# -----------------------------
# Prompts (small + focused)
# -----------------------------

ROUTER_PROMPT = """You are a router for a music store support bot.

Pick the best area:
- music: music discovery, catalog browsing, genres, artists, albums, track lookup, lyrics search, video preview
- account: viewing account + orders, verification, updating email/address (secure)
- payment: buying a track, payment workflow, receipts/invoices
- general: greetings, store policy questions, off-topic requests (will be handled with polite redirection)

IMPORTANT: If the request is clearly unrelated to music stores (sports, movies, cooking, etc.), route to "general" 
where the agent will politely redirect the customer. Don't try to force unrelated topics into music/account/payment areas.

Return ONLY the area."""

GENERAL_PROMPT = f"""You are a helpful customer support assistant for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

🎯 YOUR SCOPE - What You Can Help With:
- Music discovery: finding songs, albums, artists, genres, lyrics search
- Account management: viewing account details, order history, updating information
- Purchases: buying tracks, viewing receipts, purchase history
- Store policies: returns, shipping, general questions about the music store

🚫 OUT OF SCOPE - What You Cannot Help With:
- Topics completely unrelated to music stores (sports scores, movie reviews, cooking recipes, weather, etc.)
- General knowledge questions that aren't about music or the store
- Accessing other customers' data or employee information
- Technical support for devices or software (unless it's about our music store app)

💬 MODERATION - How to Handle Off-Topic Requests:
If a customer asks about something unrelated to the music store, politely but firmly redirect them:

Example responses:
- "I'm here to help with your music store needs! I can help you find music, manage your account, or make purchases. What would you like to do?"
- "I specialize in music store support. I can help you discover new music, manage your account, or make purchases. How can I assist you today?"
- "That's outside my area of expertise, but I'd be happy to help with anything related to our music store - finding music, managing your account, or making purchases!"

Be friendly but clear that you're focused on music store support. Don't be apologetic - just redirect confidently.

🔒 SECURITY RULES (ENFORCED BY SYSTEM - YOU CANNOT BYPASS):
- You can ONLY access this customer's data (Customer ID: {CUSTOMER_INFO['id']})
- The system automatically filters all database queries by customer ID - you don't need to worry about this
- Employee data and other customers' data are completely inaccessible at the system level
- If you try to access unauthorized data, the system will automatically block it

Be concise and helpful. If the request is about account, payments, or music, route appropriately or ask clarifying questions.
"""

MUSIC_PROMPT = f"""You are a music specialist for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

🎵 YOUR EXPERTISE:
- Browse and search the music catalog (tracks, albums, artists, genres)
- Help customers discover new music based on preferences
- Search for songs by lyrics snippets
- Check if songs are available in the catalog
- Show YouTube video previews of tracks
- Provide track details, pricing, and recommendations

💬 MODERATION:
If asked about non-music topics, politely redirect: "I'm a music specialist! I can help you find songs, albums, or artists. What kind of music are you looking for?"

When showing options, keep it to a short shortlist (3-5 items) and ask what they prefer.
"""

ACCOUNT_PROMPT = f"""You are an account specialist for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

👤 YOUR CAPABILITIES:
- View account details and order history
- Show purchase history and spending summaries
- Update email address (requires SMS verification)
- Update mailing address (requires SMS verification)

🔐 SECURITY WORKFLOW:
- Viewing account/order history: Allowed without verification
- Updating email or mailing address: REQUIRES SMS verification first
  → Step 1: Request verification code via `request_phone_verification`
  → Step 2: Customer provides code
  → Step 3: Verify code via `verify_phone_code`
  → Step 4: Once verified, proceed with update

💬 MODERATION:
If asked about non-account topics, redirect: "I specialize in account management. I can help you view your account, order history, or update your information. What would you like to do?"

🔒 SECURITY (ENFORCED BY SYSTEM):
- You can ONLY access this customer's data (Customer ID: {CUSTOMER_INFO['id']})
- All database queries automatically filter by the current customer ID - this is enforced at the system level
- You CANNOT access other customers' information or employee data - the system will block any attempts
- The verification system ensures only the account owner can make changes
"""

PAYMENT_PROMPT = f"""You are a payments specialist for a music store.

The customer is {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

🛒 **PAYMENT WORKFLOW - FOLLOW EXACTLY:**

**Step 1: Information Gathering**
- If user mentions a track ID, use `get_track_details_for_purchase` to show details
- If user mentions a track NAME (e.g., "I'd like to buy Black Country Woman"), use `search_tracks` to find the track ID first
  → After finding the track, use `get_track_details_for_purchase` with the track ID
  → If multiple matches, show the options and ask which one they want
- **CRITICAL:** ALWAYS use `check_if_already_purchased` BEFORE proceeding with purchase
  → If they already own it: Inform them they already own it and DO NOT proceed with purchase. Suggest browsing other tracks instead.
  → If they don't own it: Continue to Step 2
- Present clear pricing information (only if they don't already own it)

**Step 2: Confirmation Request** (ONLY if they don't already own the track)
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
- ALWAYS check if they already own a track BEFORE showing purchase options
- If they already own it: Stop immediately, inform them, and suggest other tracks. DO NOT offer to purchase again.
- NEVER skip asking for confirmation (for tracks they don't own)
- NEVER process payment without explicit user "yes"
- ALWAYS show pricing before confirmation (only for tracks they don't own)
- Once user confirms, CHAIN the three calls together WITHOUT waiting: initiate_track_purchase → confirm_and_process_payment → create_invoice_from_payment
- Each tool response includes a "NEXT STEP" section - follow it immediately
- If payment fails, offer to help them try again or browse other tracks

💡 **IMPORTANT:** After the user says "yes", you should make THREE consecutive tool calls in the same flow (read each tool response for the payment_intent_id to use in the next call)

📊 **OTHER TOOLS:**
- `search_tracks` - search for tracks by name (use this FIRST when user mentions a track name without ID)
- `get_recent_purchases` - show their purchase history
- `check_if_already_purchased` - check if they own a specific track
- `cancel_payment` - cancel a payment if they change their mind

Be friendly, transparent, and make customers feel secure!
"""


# Tool groupings per specialist
MUSIC_TOOLSET = CATALOG_TOOLS + LYRICS_TOOLS
ACCOUNT_TOOLSET = ACCOUNT_VIEW_TOOLS + ALL_ACCOUNT_TOOLS
# Payment tools need search_tracks to look up tracks by name
PAYMENT_TOOLSET = PAYMENT_TOOLS + [search_tracks]


# Sensitive tools that require human approval
SENSITIVE_TOOLS = {
    "initiate_track_purchase",  # Payment operations
    "update_email_address",     # Account changes
    "update_mailing_address",   # Account changes
}


def _llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


def _router_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])

    # Log user input for monitoring (but let LLM handle moderation via system prompts)
    # Code-level validation is kept for critical security (SQL injection, customer data isolation)
    # but topic moderation is handled by system prompts for better contextual understanding
    if messages:
        last = messages[-1]
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            user_input = last.content.strip()
            # Log for monitoring but don't block - let system prompts handle moderation
            is_valid, _ = validate_user_input(user_input)
            if not is_valid:
                logger.info(f"Off-topic query detected (handled by LLM moderation): {user_input[:50]}...")
            # Continue to router - let the LLM handle moderation via system prompts

    # Sticky routing: if we're already in a specialist flow and the user replies
    # with a short confirmation / continuation, keep the current area.
    prev_area = state.get("active_area")
    if prev_area in {"music", "account", "payment"} and messages:
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

    router = _llm().with_structured_output(RouterDecision)
    decision = router.invoke([SystemMessage(content=ROUTER_PROMPT)] + list(messages))

    # `decision` may be a Pydantic model or dict depending on underlying implementation
    area = decision.area if hasattr(decision, "area") else decision["area"]

    return {"active_area": area}


def _general_agent_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])
    response = _llm().invoke([SystemMessage(content=GENERAL_PROMPT)] + list(messages))
    return {"messages": [response], "active_area": "general"}


def _music_agent_node(state: SupportState) -> SupportState:
    messages = state.get("messages", [])
    llm_with_tools = _llm().bind_tools(MUSIC_TOOLSET)
    response = llm_with_tools.invoke([SystemMessage(content=MUSIC_PROMPT)] + list(messages))
    return {"messages": [response], "active_area": "music"}


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


def _route_after_router(state: SupportState) -> Literal["general", "music", "account", "payment", END]:
    area = state.get("active_area", "general")
    # If guardrails blocked the request, route to END
    if area == "__end__":
        return END
    return area


def _agent_should_continue(state: SupportState) -> Literal["approval_gate", "tools", END]:
    """Route agent to approval gate if sensitive tools, tools if safe tools, otherwise end."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        # Check if any tool call is sensitive
        for tool_call in last.tool_calls:
            if tool_call.get("name") in SENSITIVE_TOOLS:
                return "approval_gate"
        return "tools"
    return END


def _approval_gate_node(state: SupportState) -> SupportState:
    """
    Human-in-the-loop approval gate for sensitive operations.
    Interrupts execution and waits for human approval before proceeding.
    
    IMPORTANT: Do NOT wrap interrupt() in try/except - it raises a special
    exception that must propagate to LangGraph runtime to work correctly.
    
    NOTE: When resumed, this node restarts from the beginning per LangGraph docs.
    The interrupt() call will return the resume value.
    """
    logger.info(f"[APPROVAL_GATE] Node called")
    messages = state.get("messages", [])
    logger.info(f"[APPROVAL_GATE] Messages count: {len(messages)}")
    
    if not messages:
        logger.info(f"[APPROVAL_GATE] No messages, returning approved")
        return {"messages": [], "approval_status": "approved"}
    
    last_message = messages[-1]
    logger.info(f"[APPROVAL_GATE] Last message type: {type(last_message)}")
    logger.info(f"[APPROVAL_GATE] Is AIMessage: {isinstance(last_message, AIMessage)}")
    
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        logger.info(f"[APPROVAL_GATE] No tool calls, returning approved")
        return {"messages": [], "approval_status": "approved"}
    
    tool_calls = getattr(last_message, "tool_calls", [])
    logger.info(f"[APPROVAL_GATE] Tool calls found: {len(tool_calls)}")
    
    # Extract sensitive tool calls
    sensitive_calls = [
        tc for tc in tool_calls 
        if tc.get("name") in SENSITIVE_TOOLS
    ]
    
    logger.info(f"[APPROVAL_GATE] Sensitive tool calls: {len(sensitive_calls)}")
    
    if not sensitive_calls:
        # No sensitive calls, pass through
        logger.info(f"[APPROVAL_GATE] No sensitive calls, returning approved")
        return {"messages": [], "approval_status": "approved"}
    
    # Prepare approval request (must be JSON-serializable)
    # Format it nicely for the UI
    approval_request = {
        "action": "approve_sensitive_operation",
        "tool_calls": sensitive_calls,
        "message": f"Approval required for {len(sensitive_calls)} sensitive operation(s)"
    }
    
    # Add specific details for payment operations
    for tc in sensitive_calls:
        if tc.get("name") == "initiate_track_purchase":
            args = tc.get("args", {})
            approval_request["track_id"] = args.get("track_id")
            approval_request["track_name"] = args.get("track_name")
            approval_request["track_price"] = args.get("track_price")
            approval_request["message"] = f"Approve purchase of '{args.get('track_name')}' for ${args.get('track_price')}?"
            break
    
    # Interrupt and wait for approval
    # CRITICAL: Do NOT wrap this in try/except - interrupt() raises a special
    # exception that LangGraph catches to pause execution. Catching it prevents
    # the interrupt from working.
    # NOTE: When resumed, this node restarts from the beginning, and interrupt()
    # returns the resume value (True/False from Command(resume=...))
    logger.info(f"[APPROVAL_GATE] Calling interrupt with request: {approval_request}")
    response = interrupt(approval_request)
    logger.info(f"[APPROVAL_GATE] Interrupt resumed with response: {response} (type: {type(response)})")
    
    # When resumed, the response value is passed back here
    # According to docs, we should accept True/False for simple approval/rejection
    # But also handle dict format for flexibility
    approved = False
    if response is True:
        approved = True
    elif response is False:
        approved = False
    elif isinstance(response, str):
        approved = response.lower() in ("approve", "yes", "y", "true")
    elif isinstance(response, dict):
        decision = response.get("decision", "").lower()
        approved = (
            decision in ("approve", "yes", "y", "true") 
            or response.get("approve") is True
            or response.get("approved") is True
        )
    
    logger.info(f"[APPROVAL_GATE] Approved: {approved}")
    
    if approved:
        # Approved - pass through to tools (tool calls are already in state)
        logger.info(f"[APPROVAL_GATE] Returning approved status")
        return {"messages": [], "approval_status": "approved"}
    else:
        # Rejected - cancel the operation
        # IMPORTANT: We must add ToolMessages for each rejected tool call
        # OpenAI API requires that every tool_call_id has a corresponding ToolMessage
        logger.info(f"[APPROVAL_GATE] Returning rejected status with cancellation message")
        
        # Create ToolMessages for each rejected tool call
        tool_messages = []
        for tc in sensitive_calls:
            tool_id = tc.get("id", "")
            tool_name = tc.get("name", "unknown_tool")
            tool_message = ToolMessage(
                content=f"❌ Operation cancelled: {tool_name} was not approved by the user.",
                tool_call_id=tool_id
            )
            tool_messages.append(tool_message)
            logger.info(f"[APPROVAL_GATE] Created cancellation ToolMessage for tool_call_id: {tool_id}")
        
        # Add a cancellation message from the assistant
        cancellation_message = AIMessage(
            content="❌ Operation cancelled. The sensitive action was not approved. Is there anything else I can help you with?"
        )
        
        # Return both the tool messages (required by OpenAI) and the cancellation message
        return {
            "messages": tool_messages + [cancellation_message],
            "approval_status": "rejected"
        }


def _route_after_approval(state: SupportState) -> Literal["music_tools", "account_tools", "payment_tools", END]:
    """Route after approval gate based on approval status and active area."""
    approval_status = state.get("approval_status")
    area = state.get("active_area", "general")
    logger.info(f"[ROUTE_AFTER_APPROVAL] approval_status={approval_status}, area={area}")
    
    if approval_status == "rejected":
        logger.info(f"[ROUTE_AFTER_APPROVAL] Rejected, routing to END")
        return END
    
    # Approved - route to appropriate tools based on area
    logger.info(f"[ROUTE_AFTER_APPROVAL] Approved, routing to {area}_tools")
    if area == "music":
        return "music_tools"
    elif area == "account":
        return "account_tools"
    elif area == "payment":
        return "payment_tools"
    else:
        logger.warning(f"[ROUTE_AFTER_APPROVAL] Unknown area {area}, routing to END")
        return END


def _build_graph(*, with_checkpointer=False):
    """Build the graph; optionally attach a checkpointer for interrupts."""

    builder = StateGraph(SupportState)

    # Nodes
    builder.add_node("router", _router_node)

    builder.add_node("general_agent", _general_agent_node)
    builder.add_node("music_agent", _music_agent_node)
    builder.add_node("account_agent", _account_agent_node)
    builder.add_node("payment_agent", _payment_agent_node)

    builder.add_node("music_tools", ToolNode(MUSIC_TOOLSET))
    builder.add_node("account_tools", ToolNode(ACCOUNT_TOOLSET))
    builder.add_node("payment_tools", ToolNode(PAYMENT_TOOLSET))
    builder.add_node("approval_gate", _approval_gate_node)

    # Flow
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "general": "general_agent",
            "music": "music_agent",
            "account": "account_agent",
            "payment": "payment_agent",
            END: END,
        },
    )

    # General agent never calls tools (it just routes or responds)
    builder.add_edge("general_agent", END)
    
    # Specialist agents can call tools (via approval gate if sensitive), or end
    builder.add_conditional_edges(
        "music_agent",
        _agent_should_continue,
        {"tools": "music_tools", "approval_gate": "approval_gate", END: END},
    )
    builder.add_conditional_edges(
        "account_agent",
        _agent_should_continue,
        {"tools": "account_tools", "approval_gate": "approval_gate", END: END},
    )
    builder.add_conditional_edges(
        "payment_agent",
        _agent_should_continue,
        {"tools": "payment_tools", "approval_gate": "approval_gate", END: END},
    )
    
    # Approval gate routes: if approved, go to tools; if rejected, end
    builder.add_conditional_edges(
        "approval_gate",
        _route_after_approval,
        {
            "music_tools": "music_tools",
            "account_tools": "account_tools",
            "payment_tools": "payment_tools",
            END: END,
        },
    )

    # Tools -> back to the same agent (continue tool/agent loop)
    builder.add_edge("music_tools", "music_agent")
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
