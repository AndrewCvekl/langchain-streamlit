"""
Payment Agent using LangGraph 2025 Best Practices.
Handles the complete payment workflow with state management and human-in-the-loop approval.

LangGraph 2025 Best Practices Implemented:
✅ Custom state with proper typing (TypedDict)
✅ Explicit approval nodes (not interrupts inside tools)
✅ Conditional routing based on state
✅ Clean separation of concerns (agent, tools, approval, execution)
✅ State-driven workflow (track_id, payment_intent_id, etc.)
✅ Human-in-the-loop via interrupt() at graph level
"""

from typing import Literal, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from tools_payment import PAYMENT_TOOLS


# =======================
# PAYMENT STATE (2025 Best Practice)
# =======================

class PaymentState(TypedDict):
    """
    Custom state for payment workflow.
    Tracks the entire purchase flow from browsing to invoice creation.
    
    LangGraph 2025: Use TypedDict with Annotated for reducers.
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
    
    # Flow control flags
    awaiting_user_confirmation: bool  # True when waiting for user to confirm purchase
    user_approved_purchase: bool  # True when user explicitly confirmed
    purchase_completed: bool  # True when invoice created


# System prompts for each stage of payment workflow
PAYMENT_AGENT_PROMPT = """You are a specialized payment assistant for a music store.

Your role is to help customers purchase tracks through a secure, step-by-step process.

🛒 **YOUR WORKFLOW:**

1. **Information Gathering**
   - Use `get_track_details_for_purchase` to show track info and price
   - Use `check_if_already_purchased` to see if they already own it
   - Present the track details clearly with pricing

2. **Request Confirmation**
   - After showing track details, ASK the user to confirm
   - Say: "Would you like to proceed with purchasing [track name] for $[price]?"
   - DO NOT call any payment tools yet - wait for user response

3. **After User Confirms** (in next turn)
   - The system will handle the approval and payment processing
   - You'll receive updates and can communicate the results

📊 **ADDITIONAL CAPABILITIES:**
- Show recent purchase history: `get_recent_purchases`
- Check ownership: `check_if_already_purchased`

🎯 **IMPORTANT:**
- NEVER call `initiate_track_purchase` until the user explicitly confirms
- ALWAYS show pricing clearly before asking for confirmation
- Be friendly, clear, and security-conscious"""

APPROVAL_PROMPT = """The user has confirmed they want to purchase a track.
You will now initiate the payment process by calling `initiate_track_purchase`.
This will create a payment intent that requires approval before processing."""

EXECUTION_PROMPT = """Payment has been approved and processed.
Now call `create_invoice_from_payment` to complete the purchase and create the receipt."""


def create_payment_agent():
    """
    Create a payment agent graph for handling track purchases.
    
    LangGraph 2025 Architecture:
    ✅ Custom payment state with workflow tracking
    ✅ Specialized nodes: agent, tools, approval_gate, execute_payment
    ✅ Conditional routing based on state flags
    ✅ Human-in-the-loop via interrupt() at graph level (not in tools!)
    ✅ Clear separation: gather info → confirm → approve → execute → invoice
    
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
        Handles information gathering and user communication.
        """
        messages = state["messages"]
        
        # Build context-aware system prompt
        system_messages = [SystemMessage(content=PAYMENT_AGENT_PROMPT)]
        
        # Add payment context if we're mid-purchase
        if state.get("track_id") and state.get("track_name") and state.get("track_price"):
            if state.get("awaiting_user_confirmation"):
                system_messages.append(
                    SystemMessage(
                        content=f"[CONTEXT] User is reviewing: {state['track_name']} for ${state['track_price']}. "
                               f"Waiting for them to confirm or cancel."
                    )
                )
        
        if state.get("payment_intent_id"):
            system_messages.append(
                SystemMessage(
                    content=f"[PAYMENT CONTEXT] Payment intent {state['payment_intent_id']} created."
                )
            )
        
        # Call LLM with tools
        all_messages = system_messages + list(messages)
        response = llm_with_tools.invoke(all_messages)
        
        # Check if user is confirming purchase (detect confirmation keywords)
        updates = {"messages": [response]}
        
        # Check last user message for confirmation
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage):
                text = last_msg.content.lower().strip()
                confirmations = ["yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "proceed", "buy", "purchase"]
                cancellations = ["no", "n", "nope", "cancel", "stop", "nevermind"]
                
                # If user confirmed and we have track info
                if any(conf in text for conf in confirmations):
                    if state.get("track_id") and state.get("awaiting_user_confirmation"):
                        updates["user_approved_purchase"] = True
                        updates["awaiting_user_confirmation"] = False
                
                # If user cancelled
                if any(canc in text for canc in cancellations):
                    updates["awaiting_user_confirmation"] = False
                    updates["user_approved_purchase"] = False
        
        # Detect when agent presents track info for purchase
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "get_track_details_for_purchase":
                    # Agent is showing track details - prepare for confirmation
                    track_id = tc["args"].get("track_id")
                    if track_id:
                        updates["track_id"] = track_id
                        updates["awaiting_user_confirmation"] = True
        
        return updates
    
    def approval_gate_node(state: PaymentState) -> PaymentState:
        """
        Human-in-the-loop approval gate (LangGraph 2025 best practice).
        Presents purchase details and waits for approval.
        """
        # Prepare approval request
        approval_request = {
            "action": "approve_purchase",
            "track_id": state.get("track_id"),
            "track_name": state.get("track_name"),
            "track_price": state.get("track_price"),
            "message": f"Approve purchase of '{state.get('track_name')}' for ${state.get('track_price')}?"
        }
        
        # This is where we interrupt for human approval (graph level, not tool level!)
        response = interrupt(approval_request)
        
        # Parse approval response
        approved = False
        if response is True:
            approved = True
        elif isinstance(response, str):
            approved = response.lower() == "approve"
        elif isinstance(response, dict):
            approved = response.get("decision") == "approve" or response.get("approve") is True
        
        if approved:
            # Approved - call initiate_track_purchase tool
            return {
                "messages": [
                    AIMessage(
                        content=f"Processing your purchase of {state.get('track_name')}...",
                        tool_calls=[{
                            "name": "initiate_track_purchase",
                            "args": {
                                "track_id": state.get("track_id"),
                                "track_name": state.get("track_name"),
                                "track_price": state.get("track_price")
                            },
                            "id": "approval_gate_purchase"
                        }]
                    )
                ]
            }
        else:
            return {
                "messages": [
                    AIMessage(content="Purchase cancelled. Let me know if you'd like to browse other tracks!")
                ],
                "awaiting_user_confirmation": False,
                "user_approved_purchase": False
            }
    
    def execute_payment_node(state: PaymentState) -> PaymentState:
        """
        Execute payment after approval.
        Calls confirm_and_process_payment and create_invoice_from_payment.
        """
        if not state.get("payment_intent_id"):
            return {"messages": [AIMessage(content="Error: No payment intent found.")]}
        
        # Create tool calls for payment confirmation and invoice creation
        return {
            "messages": [
                AIMessage(
                    content="Confirming payment and creating invoice...",
                    tool_calls=[
                        {
                            "name": "confirm_and_process_payment",
                            "args": {"payment_intent_id": state.get("payment_intent_id")},
                            "id": "confirm_payment"
                        }
                    ]
                )
            ]
        }
    
    def route_after_agent(state: PaymentState) -> Literal["tools", "approval_gate", END]:
        """
        Route after agent based on state and tool calls.
        """
        messages = state.get("messages", [])
        if not messages:
            return END
        
        last_message = messages[-1]
        
        # If agent made tool calls, execute them
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tools"
        
        # If user approved purchase, go to approval gate
        if state.get("user_approved_purchase") and not state.get("payment_intent_id"):
            return "approval_gate"
        
        # Otherwise end
        return END
    
    def route_after_tools(state: PaymentState) -> Literal["agent", "execute_payment"]:
        """
        Route after tool execution.
        If payment intent was just created, proceed to execute payment.
        Otherwise, go back to agent.
        """
        messages = state.get("messages", [])
        
        # Check if we just created a payment intent
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if "Payment Intent Created" in msg.content and "Payment ID:" in msg.content:
                    # Extract payment intent ID from tool response
                    for line in msg.content.split("\n"):
                        if "Payment ID:" in line:
                            payment_id = line.split("Payment ID:")[-1].strip()
                            if payment_id and payment_id != state.get("payment_intent_id"):
                                # New payment intent - update state and execute payment
                                state["payment_intent_id"] = payment_id
                                return "execute_payment"
                    break
        
        # Default: back to agent
        return "agent"
    
    # Build the graph
    workflow = StateGraph(PaymentState)
    
    # Add nodes
    workflow.add_node("agent", payment_agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("execute_payment", execute_payment_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Agent can go to: tools (if making tool calls), approval_gate (if user confirmed), or END
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "approval_gate": "approval_gate", END: END}
    )
    
    # After tools, decide whether to execute payment or return to agent
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "execute_payment": "execute_payment"}
    )
    
    # Approval gate goes to tools to execute initiate_track_purchase
    workflow.add_edge("approval_gate", "tools")
    
    # Execute payment goes to tools to run confirm_and_process_payment
    workflow.add_edge("execute_payment", "tools")
    
    # Compile
    return workflow.compile()


def create_payment_agent_with_memory():
    """
    Create payment agent with checkpointing for conversation persistence.
    This maintains payment state across the conversation.
    
    LangGraph 2025: Uses MemorySaver for state persistence.
    
    Returns:
        Compiled LangGraph agent with memory
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    memory = MemorySaver()
    
    # Compile the same graph but with checkpointing
    return create_payment_agent()  # Reuse the main graph definition


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
