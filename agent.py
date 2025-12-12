"""
Entry point for LangGraph Studio.
Exports the compiled agent graph.
"""

from dotenv import load_dotenv
from graph_with_verification import create_agent_with_verification
from tracing import setup_langsmith_tracing

# Load environment variables and set up tracing
load_dotenv()
setup_langsmith_tracing()

# Create and compile the agent without custom checkpointer
# LangGraph API handles persistence automatically
graph = create_agent_with_verification()

# Export for LangGraph Studio
__all__ = ['graph']
