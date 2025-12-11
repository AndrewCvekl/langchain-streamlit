"""
Test script for payment workflow.
Tests the complete purchase flow end-to-end.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph_with_verification import create_agent_with_memory

# Load environment
load_dotenv()

def print_separator(title=""):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}\n")

def test_payment_flow():
    """Test the complete payment flow."""
    
    print_separator("PAYMENT FLOW TEST")
    
    # Create agent
    agent = create_agent_with_memory()
    config = {"configurable": {"thread_id": "test_payment_001"}}
    
    # Test scenario: Buy a track
    test_messages = [
        "I want to buy track ID 1",
        # Agent should show track details and ask for confirmation
        # We'll respond with "yes"
        "yes",
        # Agent should automatically: initiate → confirm → create invoice
    ]
    
    messages = []
    
    for user_msg in test_messages:
        print(f"👤 User: {user_msg}")
        print()
        
        messages.append(HumanMessage(content=user_msg))
        
        # Invoke agent
        result = agent.invoke(
            {"messages": messages},
            config=config
        )
        
        # Update messages with result
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        
        # Print agent response
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                from langchain_core.messages import AIMessage, ToolMessage
                if isinstance(msg, AIMessage):
                    print(f"🤖 Agent: {msg.content}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"\n   🔧 Tool Calls:")
                        for tc in msg.tool_calls:
                            print(f"      - {tc['name']}({tc.get('args', {})})")
                    print()
                    break
                elif isinstance(msg, ToolMessage):
                    # Show tool results
                    print(f"   ⚙️ Tool Result Preview: {msg.content[:200]}...")
                    print()
        
        print_separator()
    
    print("\n✅ Payment flow test completed!")
    print("\nIf you see:")
    print("  1. Track details shown")
    print("  2. Confirmation request")
    print("  3. Payment intent created")
    print("  4. Payment processed")
    print("  5. Invoice created")
    print("\nThen the payment flow is working correctly! 🎉")


if __name__ == "__main__":
    test_payment_flow()
