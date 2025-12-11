"""
Payment Agent using LangGraph.
Handles the complete payment workflow with state management.

Following LangGraph best practices:
- Custom state with proper typing
- Specialized nodes for different stages
- Conditional routing based on payment status
- Clean separation of concerns
"""

from typing import Literal, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from tools_payment import PAYMENT_TOOLS


# =======================
# PAYMENT STATE
# =======================

class PaymentState(TypedDict):
    """
    Custom state for payment workflow.
    Tracks the entire purchase flow from browsing to invoice creation.
    """
    # Messages use the add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Payment workflow tracking
    track_id: Optional[int]
    track_name: Optional[str]
    track_price: Optional[float]
    payment_intent_id: Optional[str]
    payment_status: Optional[str]  # pending, processing, succeeded, failed
    invoice_id: Optional[int]
    
    # Flow control
    awaiting_confirmation: bool  # True when waiting for user to confirm purchase


# System prompt for payment agent
PAYMENT_SYSTEM_PROMPT = """You are a specialized payment assistant for a music store.

Your role is to help customers purchase tracks through a secure, step-by-step process.

🛒 **PAYMENT WORKFLOW:**

When a customer wants to buy a track, follow this exact sequence:

1. **Get Track Details**
   - Use `get_track_details_for_purchase` to show track info and price
   - Use `check_if_already_purchased` to see if they already own it
   - If they already own it, inform them and ask if they still want to purchase again

2. **Confirm Purchase**
   - Clearly show: Track name, artist, price
   - Ask: "Would you like to proceed with purchasing [track name] for $[price]?"
   - Wait for explicit confirmation (yes/proceed/confirm)

3. **Process Payment** (only after confirmation)
   - Use `initiate_track_purchase` to create payment intent
   - Use `confirm_and_process_payment` to process the payment
   - Check if payment succeeded or failed

4. **Create Invoice** (only if payment succeeded)
   - Use `create_invoice_from_payment` to save purchase to database
   - Show complete receipt with invoice number
   - Thank customer and remind them they can view purchase history

5. **Handle Cancellation**
   - If customer says "cancel" or "no", use `cancel_payment` if payment was initiated
   - Confirm cancellation and offer to help with other tracks

🎯 **IMPORTANT RULES:**
- NEVER skip the confirmation step
- NEVER process payment without explicit user confirmation
- ALWAYS check if they already own the track first
- ALWAYS show clear pricing before asking for confirmation
- If payment fails, offer to try again or browse other tracks
- Be friendly, clear, and security-conscious

📊 **ADDITIONAL CAPABILITIES:**
- Show recent purchase history: `get_recent_purchases`
- Check ownership: `check_if_already_purchased`

Be helpful, transparent about pricing, and ensure customers feel secure throughout the purchase process!"""


def create_payment_agent():
    """
    Create a payment agent graph for handling track purchases.
    
    Architecture:
    - Custom payment state with workflow tracking
    - Agent node with payment-specific tools
    - Tool execution node
    - Conditional routing based on payment status
    
    Returns:
        Compiled LangGraph agent
    """
    
    # Initialize LLM with payment tools
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(PAYMENT_TOOLS)
    
    # Tool execution node
    tool_node = ToolNode(PAYMENT_TOOLS)
    
    def payment_agent_node(state: PaymentState) -> PaymentState:
        """
        Main payment agent node - calls LLM with payment tools and context.
        """
        messages = state["messages"]
        
        # Add system message if first interaction
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=PAYMENT_SYSTEM_PROMPT)] + messages
        
        # Add payment context if we're in the middle of a purchase
        context_messages = []
        if state.get("payment_intent_id"):
            context_messages.append(
                SystemMessage(
                    content=f"[PAYMENT CONTEXT] Payment intent {state['payment_intent_id']} "
                           f"created for ${state.get('track_price')} - {state.get('track_name')}"
                )
            )
        
        if state.get("awaiting_confirmation"):
            context_messages.append(
                SystemMessage(
                    content="[SYSTEM] Awaiting user confirmation for purchase. "
                           "Ask user to confirm before proceeding with payment."
                )
            )
        
        # Call LLM with tools
        all_messages = messages + context_messages if context_messages else messages
        response = llm_with_tools.invoke(all_messages)
        
        return {
            "messages": [response]
        }
    
    def should_continue(state: PaymentState) -> Literal["tools", END]:
        """
        Routing logic - continue to tools or end conversation.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If LLM called tools, route to tool execution
        if last_message.tool_calls:
            return "tools"
        
        # Otherwise, we're done
        return END
    
    # Build the graph
    workflow = StateGraph(PaymentState)
    
    # Add nodes
    workflow.add_node("payment_agent", payment_agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "payment_agent")
    workflow.add_conditional_edges(
        "payment_agent",
        should_continue,
        ["tools", END]
    )
    workflow.add_edge("tools", "payment_agent")  # After tools, back to agent
    
    # Compile
    return workflow.compile()


def create_payment_agent_with_memory():
    """
    Create payment agent with checkpointing for conversation persistence.
    This maintains payment state across the conversation.
    
    Returns:
        Compiled LangGraph agent with memory
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    memory = MemorySaver()
    
    # Same graph but with checkpointing
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(PAYMENT_TOOLS)
    tool_node = ToolNode(PAYMENT_TOOLS)
    
    def payment_agent_node(state: PaymentState) -> PaymentState:
        messages = state["messages"]
        
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=PAYMENT_SYSTEM_PROMPT)] + messages
        
        context_messages = []
        if state.get("payment_intent_id"):
            context_messages.append(
                SystemMessage(
                    content=f"[PAYMENT CONTEXT] Payment intent {state['payment_intent_id']} "
                           f"created for ${state.get('track_price')} - {state.get('track_name')}"
                )
            )
        
        if state.get("awaiting_confirmation"):
            context_messages.append(
                SystemMessage(
                    content="[SYSTEM] Awaiting user confirmation for purchase. "
                           "Ask user to confirm before proceeding with payment."
                )
            )
        
        all_messages = messages + context_messages if context_messages else messages
        response = llm_with_tools.invoke(all_messages)
        
        return {"messages": [response]}
    
    def should_continue(state: PaymentState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    workflow = StateGraph(PaymentState)
    workflow.add_node("payment_agent", payment_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "payment_agent")
    workflow.add_conditional_edges("payment_agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "payment_agent")
    
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    # Test the payment agent
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = create_payment_agent()
    
    print("=" * 60)
    print("Testing Payment Agent")
    print("=" * 60)
    
    # Test query - purchase a track
    query = "I'd like to buy the track with ID 1"
    
    print(f"\n💬 User: {query}\n")
    
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
        "track_id": None,
        "track_name": None,
        "track_price": None,
        "payment_intent_id": None,
        "payment_status": None,
        "invoice_id": None,
        "awaiting_confirmation": False
    })
    
    # Get final response
    for msg in result["messages"]:
        if hasattr(msg, 'content') and msg.content and not isinstance(msg, SystemMessage):
            print(f"🤖 Assistant: {msg.content}\n")
    
    print("=" * 60)
    print("Payment agent test completed!")
    print("=" * 60)
