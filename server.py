"""
FastAPI server for the Music Store Support Bot.
Exposes the LangGraph agent via REST API endpoints.
"""

import sys
import os
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

# IMPORTANT: Set up paths BEFORE importing from sql-support-bot
# This ensures the database module finds chinook.db correctly

# Store our UI directory for serving static files
UI_DIR = Path(__file__).parent.resolve()

# Add sql-support-bot to path and change working directory
# so database.py finds chinook.db with its default path
SQL_BOT_DIR = Path(__file__).parent.parent / "sql-support-bot"
sys.path.insert(0, str(SQL_BOT_DIR))

# Change to sql-support-bot directory so relative paths work
os.chdir(SQL_BOT_DIR)

from dotenv import load_dotenv
load_dotenv(SQL_BOT_DIR / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.types import Command

from graph_with_verification import create_agent_with_memory, CUSTOMER_INFO
from verification import get_verification_service
from tracing import setup_langsmith_tracing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up LangSmith tracing
setup_langsmith_tracing()


# ==================
# Session Management
# ==================

class SessionManager:
    """Manages chat sessions with their own agent instances and state."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "agent": create_agent_with_memory(),
                "thread_id": {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}},
                "messages": [],
                "pending_interrupt": None,
                "verification_store": {},
            }
            logger.info(f"Created new session: {session_id}")
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a session's chat history."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session["messages"] = []
            session["pending_interrupt"] = None
            session["thread_id"] = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
            logger.info(f"Cleared session: {session_id}")


session_manager = SessionManager()


# ==================
# Pydantic Models
# ==================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    tool_calls: Optional[List[Dict]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    pending_interrupt: Optional[Dict] = None
    is_verified: bool = False


# ==================
# Helper Functions
# ==================

def serialize_message(msg: BaseMessage) -> Optional[ChatMessage]:
    """Convert LangChain message to serializable format."""
    if isinstance(msg, HumanMessage):
        return ChatMessage(role="user", content=msg.content)
    elif isinstance(msg, AIMessage):
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
        return ChatMessage(role="assistant", content=msg.content, tool_calls=tool_calls)
    elif isinstance(msg, ToolMessage):
        return None  # Skip tool messages in display
    return None


def extract_interrupt_payload(interrupt):
    """Extract payload from interrupt object."""
    if not interrupt:
        return None
    first = interrupt[0]
    payload = getattr(first, "value", None)
    if payload is None and isinstance(first, dict):
        payload = first.get("value")
    return payload if payload is not None else first


# ==================
# FastAPI App
# ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Starting Music Store Support Bot API")
    yield
    logger.info("👋 Shutting down Music Store Support Bot API")


app = FastAPI(
    title="Music Store Support Bot API",
    description="API for the Music Store customer support chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================
# API Endpoints
# ==================

@app.get("/")
async def root():
    """Serve the main chat interface."""
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/customer")
async def get_customer():
    """Get current customer information."""
    return {
        "id": CUSTOMER_INFO["id"],
        "name": CUSTOMER_INFO["full_name"],
        "email": CUSTOMER_INFO["email"],
        "phone": CUSTOMER_INFO["phone"],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    session = session_manager.get_or_create_session(request.session_id)
    
    # Get verification status
    verification_service = get_verification_service(session["verification_store"])
    is_verified = verification_service.is_verified(CUSTOMER_INFO["id"])
    
    # Add user message
    session["messages"].append(HumanMessage(content=request.message))
    
    # Clear any pending interrupt
    if session["pending_interrupt"]:
        session["pending_interrupt"] = None
    
    try:
        # Invoke agent
        result = session["agent"].invoke(
            {
                "messages": session["messages"],
                "is_verified": is_verified,
                "verification_requested": False,
                "pending_account_change": "",
                "approval_status": None,
            },
            config=session["thread_id"],
        )
        
        # Check for interrupt
        if isinstance(result, dict) and result.get("__interrupt__"):
            session["pending_interrupt"] = result["__interrupt__"]
            payload = extract_interrupt_payload(result["__interrupt__"])
            
            return ChatResponse(
                messages=[serialize_message(m) for m in session["messages"] if serialize_message(m)],
                pending_interrupt=payload if isinstance(payload, dict) else {"message": str(payload)},
                is_verified=is_verified,
            )
        
        # Update messages from result
        if isinstance(result, dict) and "messages" in result:
            new_messages = result["messages"][len(session["messages"]):]
            session["messages"].extend(new_messages)
        
        # Serialize messages for response
        serialized = [serialize_message(m) for m in session["messages"] if serialize_message(m)]
        
        return ChatResponse(
            messages=serialized,
            pending_interrupt=None,
            is_verified=is_verified,
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/approve", response_model=ChatResponse)
async def approve_action(request: ApprovalRequest):
    """Approve or reject a pending interrupt."""
    session = session_manager.get_or_create_session(request.session_id)
    
    if not session["pending_interrupt"]:
        raise HTTPException(status_code=400, detail="No pending approval request")
    
    # Get verification status
    verification_service = get_verification_service(session["verification_store"])
    is_verified = verification_service.is_verified(CUSTOMER_INFO["id"])
    
    try:
        # Resume with approval/rejection
        result = session["agent"].invoke(
            Command(resume=request.approved),
            config=session["thread_id"],
        )
        
        # Clear interrupt
        session["pending_interrupt"] = None
        
        # Check for another interrupt
        if isinstance(result, dict) and result.get("__interrupt__"):
            session["pending_interrupt"] = result["__interrupt__"]
            payload = extract_interrupt_payload(result["__interrupt__"])
            
            # Update messages
            if "messages" in result:
                current_count = len(session["messages"])
                new_messages = result["messages"][current_count:]
                session["messages"].extend(new_messages)
            
            return ChatResponse(
                messages=[serialize_message(m) for m in session["messages"] if serialize_message(m)],
                pending_interrupt=payload if isinstance(payload, dict) else {"message": str(payload)},
                is_verified=is_verified,
            )
        
        # Update messages
        if isinstance(result, dict) and "messages" in result:
            current_count = len(session["messages"])
            new_messages = result["messages"][current_count:]
            session["messages"].extend(new_messages)
        
        # If rejected, create new thread to prevent stuck state
        if not request.approved:
            session["thread_id"] = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
        
        return ChatResponse(
            messages=[serialize_message(m) for m in session["messages"] if serialize_message(m)],
            pending_interrupt=None,
            is_verified=is_verified,
        )
        
    except Exception as e:
        logger.error(f"Error processing approval: {str(e)}")
        session["pending_interrupt"] = None
        session["thread_id"] = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_chat(session_id: str):
    """Clear chat history for a session."""
    session_manager.clear_session(session_id)
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "music-store-support-bot"}


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
