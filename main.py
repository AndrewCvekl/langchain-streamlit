"""Main entry point for the customer support chatbot."""

import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END

from graph_with_verification import create_agent_with_verification


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 60 + "\n")


async def run_chatbot():
    """
    Run the interactive chatbot in the terminal.
    """
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("🎵 Welcome to the Music Store Customer Support Bot! 🎵")
    print("=" * 60)
    print("\nI can help you with:")
    print("  • Finding music, albums, and artists")
    print("  • Looking up your customer information")
    print("\nType 'quit', 'q', or 'exit' to end the conversation.")
    print_separator()
    
    # Create the graph
    graph = create_agent_with_verification()
    
    # Conversation history
    history = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in {'q', 'quit', 'exit'}:
            print("\nThank you for chatting! Have a great day! 👋")
            break
        
        # Add user message to history
        history.append(HumanMessage(content=user_input))
        
        print("\nBot: ", end="", flush=True)
        
        # Stream the response
        response_content = ""
        try:
            async for output in graph.astream(history):
                if END in output or START in output:
                    continue
                
                # Get the latest message from the output
                for key, value in output.items():
                    if isinstance(value, AIMessage):
                        if value.content:
                            response_content = value.content
        except Exception as e:
            print(f"\n\nError: {str(e)}")
            print("Please try again.")
            history.pop()  # Remove the last user message
            print_separator()
            continue
        
        # Print the final response
        if response_content:
            print(response_content)
            history.append(AIMessage(content=response_content))
        else:
            # If there's no text content, the bot might have made a tool call
            print("I'm processing your request...")
        
        print_separator()


def main():
    """Main function."""
    asyncio.run(run_chatbot())


if __name__ == "__main__":
    main()
