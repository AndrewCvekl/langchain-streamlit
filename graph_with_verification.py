"""
Enhanced LangGraph implementation with phone verification and account updates.
Follows LangGraph best practices for human-in-the-loop and state management.
"""

from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from tools_v2 import ALL_TOOLS as CATALOG_TOOLS
from tools_account import ALL_ACCOUNT_TOOLS
from tools_payment import PAYMENT_TOOLS
from verification import get_verification_service


# Customer information
CUSTOMER_INFO = {
    "id": 58,
    "first_name": "Manoj",
    "last_name": "Pareek",
    "email": "manoj.pareek@rediff.com",
    "phone": "+19144342859",
    "full_name": "Manoj Pareek"
}


# ============================================
# CUSTOM STATE WITH VERIFICATION TRACKING
# ============================================

class AgentState(TypedDict):
    """
    Custom state that extends MessagesState with verification tracking.
    Following LangGraph best practices for stateful workflows.
    """
    # Messages use the add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Verification tracking (no reducer needed - simple replacement)
    is_verified: bool
    verification_requested: bool
    pending_account_change: str  # Type of change user wants to make


# Enhanced system prompt with verification instructions
SYSTEM_PROMPT = f"""You are a helpful customer support assistant for a music store.

You are currently helping {CUSTOMER_INFO['full_name']} (Customer ID: {CUSTOMER_INFO['id']}).

**Important:** When the user first starts a conversation, greet them warmly by name.

🔒 **SECURITY & VERIFICATION PROTOCOL:**

When a user wants to change SENSITIVE account information (email address, mailing address):

1. Explain that for security, you need to verify their identity via SMS
2. Use the `request_phone_verification` tool to send a code to their phone: {CUSTOMER_INFO['phone']}
3. Wait for them to provide the code they received
4. Use `verify_phone_code` tool to validate the code
5. Once verified, they can update their email or address
6. If already verified in this session, they don't need to verify again for subsequent changes

**NON-SENSITIVE operations** (viewing account info, browsing music, checking orders) do NOT require verification.

You can help with:
1. **Account Information**: View account details, contact information
2. **Secure Account Updates**: Change email or mailing address (requires SMS verification)
3. **Order History**: View past purchases, invoice details, spending summary
4. **Purchased Music**: See what tracks/songs they've bought
5. **Music Catalog**: Search for songs, albums, artists, browse by genre
6. **Pricing**: Check track prices, album details
7. **🎵 Lyrics Search**: Find songs by lyrics snippet - see workflow below
8. **💳 Purchase Tracks**: Buy songs and add them to their collection - see workflow below

🎵 **LYRICS SEARCH WORKFLOW:**
When a customer provides lyrics (e.g., "I heard a song that went 'can't you see'"):
1. Use `search_song_by_lyrics` with the lyrics snippet to find matching songs
2. Present the top matches to the customer
3. Once they confirm which song, use `check_song_in_catalogue` to verify if we have it
4. Tell them if it's in/out of catalogue, then ASK: "Would you like to see a video of this song?"
5. If they say yes, use `search_youtube_video` to get the video
6. If IN CATALOGUE: Offer to help them purchase it (show price and details)
7. If NOT IN CATALOGUE: Say "This song is not in our catalogue yet. Would you like us to add it?" 
   - If yes: Thank them and say "Thanks! We'll pass this feedback to our team."
   - If no: Offer to help find similar music we do have

💳 **TRACK PURCHASE WORKFLOW:**
When a customer wants to buy/purchase a track:
1. Use `get_track_details_for_purchase` to show track info and price
2. Use `check_if_already_purchased` to see if they already own it
3. If already owned, inform them but still offer to purchase again if they want
4. Ask for confirmation: "Would you like to purchase [track name] by [artist] for $[price]?"
5. If confirmed, use `initiate_track_purchase` to create payment intent
6. Then use `confirm_and_process_payment` to process the payment
7. If payment succeeds, use `create_invoice_from_payment` to save to database
8. Show complete receipt with invoice number
9. If they say "cancel" or "no", respect their decision

**IMPORTANT PURCHASE RULES:**
- ALWAYS check if they already own the track first
- ALWAYS show clear pricing before purchase
- NEVER skip the confirmation step
- ALWAYS wait for explicit confirmation (yes/confirm/proceed)
- If payment fails, offer to try again
- After successful purchase, show invoice number and thank them

Be friendly, security-conscious, and helpful. Use {CUSTOMER_INFO['first_name']}'s name naturally in responses.

When handling verification:
- Clearly explain why verification is needed
- Be patient with code entry
- Provide helpful error messages if code is wrong
- Remember verification lasts for the session"""


# Combine all tools
ALL_TOOLS = CATALOG_TOOLS + ALL_ACCOUNT_TOOLS + PAYMENT_TOOLS


def create_agent_with_verification():
    """
    Create a LangGraph agent with verification support.
    
    Architecture:
    - Custom state with verification tracking
    - Agent node with tool calling
    - Tool execution node
    - Conditional routing based on tool calls
    """
    
    # Initialize LLM with all tools
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Tool execution node
    tool_node = ToolNode(ALL_TOOLS)
    
    # Verification service
    verification_service = get_verification_service()
    
    
    def agent_node(state: AgentState) -> AgentState:
        """
        Main agent node - calls LLM with tools and context.
        """
        messages = state["messages"]
        
        # Add system message if first interaction
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        # Check current verification status
        is_verified = verification_service.is_verified(CUSTOMER_INFO['id'])
        
        # Add verification context to help agent
        if is_verified and not state.get("is_verified"):
            # Update state to reflect verification
            context_msg = SystemMessage(
                content=f"[SYSTEM: User is verified. Account changes allowed.]"
            )
            messages = messages + [context_msg]
        
        # Call LLM with tools
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "is_verified": is_verified
        }
    
    
    def should_continue(state: AgentState) -> Literal["tools", END]:
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
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        ["tools", END]
    )
    workflow.add_edge("tools", "agent")  # After tools, back to agent
    
    # Compile
    return workflow.compile()


def create_agent_with_memory():
    """
    Create agent with checkpointing for conversation persistence.
    This maintains verification state across the conversation.
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    memory = MemorySaver()
    
    # Same graph but with checkpointing
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)
    verification_service = get_verification_service()
    
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        is_verified = verification_service.is_verified(CUSTOMER_INFO['id'])
        
        if is_verified and not state.get("is_verified"):
            context_msg = SystemMessage(
                content=f"[SYSTEM: User is verified. Account changes allowed.]"
            )
            messages = messages + [context_msg]
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "is_verified": is_verified}
    
    def should_continue(state: AgentState) -> Literal["tools", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    # Test the verification-enabled agent
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = create_agent_with_verification()
    
    print("Testing agent with verification...\n")
    print("=" * 60)
    print(f"Customer: {CUSTOMER_INFO['full_name']}")
    print(f"Phone: {CUSTOMER_INFO['phone']}")
    print("=" * 60)
    
    # Test query
    query = "I'd like to change my email address to new.email@example.com"
    
    print(f"\n🤔 User: {query}\n")
    
    result = agent.invoke({
        "messages": [HumanMessage(content=query)],
        "is_verified": False,
        "verification_requested": False,
        "pending_account_change": ""
    })
    
    # Get final response
    final_message = result["messages"][-1]
    print(f"🤖 Assistant: {final_message.content}\n")
    print("-" * 60)
