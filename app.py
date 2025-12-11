"""
Streamlit UI for the Music Store Customer Support Bot.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph_with_verification import create_agent_with_memory, CUSTOMER_INFO
from verification import get_verification_service
import os
from dotenv import load_dotenv
from langgraph.types import Command
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up LangSmith tracing
from tracing import setup_langsmith_tracing
setup_langsmith_tracing()

# Page config
st.set_page_config(
    page_title="Music Store Support Bot (Secure)",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = create_agent_with_memory()
    import uuid
    # Use a unique thread_id per session for proper checkpointing
    st.session_state.thread_id = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_greeting = True
else:
    if "show_greeting" not in st.session_state:
        st.session_state.show_greeting = False

# Pending human-in-the-loop approval (LangGraph interrupts)
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

# Initialize persistent verification store
if "verification_store" not in st.session_state:
    st.session_state.verification_store = {}

# Get verification service with persistent store
# This makes verification persist across reruns (until page refresh)
verification_service = get_verification_service(st.session_state.verification_store)
is_verified = verification_service.is_verified(CUSTOMER_INFO['id'])


# Header
st.markdown('<div class="main-header">🔒 Music Store</div>', unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_greeting = True
        st.session_state.pending_interrupt = None  # Clear any pending interrupts
        import uuid
        # Create a new thread_id for the new conversation
        st.session_state.thread_id = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
        st.rerun()


# Main chat interface
chat_container = st.container()

with chat_container:
    
    # Verification code input form (if code was just sent)
    if len(st.session_state.messages) > 0:
        last_msg = st.session_state.messages[-1]
        if isinstance(last_msg, AIMessage) and "verification code sent" in last_msg.content.lower():
            st.info(f"📱 Check your phone ({CUSTOMER_INFO['phone']}) for the verification code!")
            
            with st.form("verification_form", clear_on_submit=True):
                code = st.text_input("Enter 6-digit code:", max_chars=6, placeholder="123456")
                submitted = st.form_submit_button("✅ Verify Code")
                
                if submitted and code:
                    st.session_state.user_input = f"My verification code is {code}"
                    st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Check if the message contains a YouTube video embed signal
                content = message.content
                if "YOUTUBE_VIDEO|" in content:
                    # Parse the video information
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith("YOUTUBE_VIDEO|"):
                            parts = line.split('|')
                            if len(parts) >= 4:
                                video_id = parts[1]
                                video_title = parts[2]
                                channel_title = parts[3]
                                
                                # Display text before the video tag
                                text_before = content.split("YOUTUBE_VIDEO|")[0].strip()
                                if text_before:
                                    st.write(text_before)
                                
                                # Create beautiful header
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, rgba(44,75,75,0.95) 0%, rgba(45,79,79,0.90) 60%, rgba(31,58,58,0.95) 100%); 
                                            padding: 1rem; 
                                            border-radius: 20px 20px 0 0; 
                                            margin-top: 1.5rem;
                                            border: 1px solid rgba(255,255,255,0.12);
                                            border-bottom: none;">
                                    <h3 style="margin: 0; color: white; font-size: 1.2rem;">🎥 {video_title}</h3>
                                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9rem;">
                                        📺 {channel_title}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Use Streamlit's iframe component for full functionality
                                import streamlit.components.v1 as components
                                components.iframe(
                                    f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1&rel=0&modestbranding=1",
                                    height=400,
                                    scrolling=False
                                )
                                
                                # Display text after the video tag if any
                                text_after = content.split(line)[1].strip() if len(content.split(line)) > 1 else ""
                                if text_after:
                                    st.write(text_after)
                                break
                    else:
                        # No video found, display normally
                        st.write(content.replace("YOUTUBE_VIDEO|", "").strip())
                else:
                    st.write(content)
                
                # Show tool calls if any
                if hasattr(message, "tool_calls") and message.tool_calls:
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for tool_call in message.tool_calls:
                            st.markdown(f"""
                            <div class="tool-call">
                                <strong>Tool:</strong> {tool_call['name']}<br>
                                <strong>Args:</strong> {tool_call.get('args', {})}
                            </div>
                            """, unsafe_allow_html=True)
        elif isinstance(message, ToolMessage):
            # Don't display raw tool messages (they're verbose)
            pass

    # Human-in-the-loop approval panel (when graph is interrupted)
    # Display this AFTER chat history so it appears in the natural conversation flow
    if st.session_state.pending_interrupt:
        interrupt_obj = st.session_state.pending_interrupt[0]
        payload = getattr(interrupt_obj, "value", None)
        if payload is None and isinstance(interrupt_obj, dict):
            payload = interrupt_obj.get("value")
        if payload is None:
            payload = interrupt_obj

        # Display as a special message in the chat flow
        with st.chat_message("assistant"):
            st.markdown("⏸️ **Human Approval Required**")
            
            # Check if this is a payment approval (has track details directly in payload)
            if isinstance(payload, dict) and payload.get("track_name"):
                # Payment purchase approval - show it nicely
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(44,75,75,0.95) 0%, rgba(45,79,79,0.90) 55%, rgba(31,58,58,0.95) 100%); 
                            padding: 1.5rem; 
                            border-radius: 20px; 
                            color: white; 
                            margin: 1rem 0;
                            border: 1px solid rgba(255,255,255,0.12);
                            box-shadow: 0 22px 70px rgba(0,0,0,0.45);
                            position: relative;
                            overflow: hidden;">
                    <div style="position: absolute; top: -30px; right: -30px; width: 200px; height: 200px; 
                                background: radial-gradient(circle, rgba(211,182,198,0.12) 0%, rgba(167,196,223,0.10) 50%, transparent 70%);
                                pointer-events: none;"></div>
                    <div style="position: relative; z-index: 1;">
                    <h3 style="margin: 0 0 1rem 0; color: white;">💰 Purchase Approval Required</h3>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                        <strong>Track:</strong> {payload.get('track_name', 'Unknown')}<br>
                        <strong>Track ID:</strong> {payload.get('track_id', 'N/A')}<br>
                        <strong>Price:</strong> ${payload.get('track_price', 0):.2f}
                    </p>
                    <p style="margin: 1rem 0 0 0; font-size: 1rem; opacity: 0.9;">
                        {payload.get('message', 'Approve this purchase?')}
                    </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Generic approval with tool calls
                tool_calls = payload.get("tool_calls", []) if isinstance(payload, dict) else []
                
                if tool_calls:
                    st.markdown("**Operation Details:**")
                    for i, tc in enumerate(tool_calls, 1):
                        tool_name = tc.get('name', 'Unknown')
                        tool_args = tc.get('args', {})
                        
                        # Format based on tool type
                        if tool_name == "initiate_track_purchase":
                            st.info(f"""
                            **💰 Purchase Request #{i}**
                            - **Track:** {tool_args.get('track_name', 'Unknown')}
                            - **Track ID:** {tool_args.get('track_id', 'N/A')}
                            - **Price:** ${tool_args.get('track_price', 0):.2f}
                            """)
                        elif tool_name == "update_email_address":
                            st.info(f"""
                            **📧 Email Update Request #{i}**
                            - **New Email:** {tool_args.get('new_email', 'Unknown')}
                            """)
                        elif tool_name == "update_mailing_address":
                            st.info(f"""
                            **📍 Address Update Request #{i}**
                            - **Street:** {tool_args.get('street_address', 'N/A')}
                            - **City:** {tool_args.get('city', 'N/A')}
                            - **State:** {tool_args.get('state', 'N/A')}
                            - **Postal Code:** {tool_args.get('postal_code', 'N/A')}
                            """)
                        else:
                            st.markdown(
                                f"""
                                <div class="tool-call">
                                    <strong>Tool:</strong> {tool_name}<br>
                                    <strong>Arguments:</strong> {tool_args}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.write("**Action:**", payload.get("message", "Sensitive operation requires approval"))
                    if isinstance(payload, dict):
                        st.json(payload)
                    else:
                        st.write(payload)

            # Approval buttons - displayed inline in the chat
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Approve", type="primary", key="approve_interrupt", use_container_width=True):
                    try:
                        logger.info(f"[APPROVE] Starting approval flow")
                        logger.info(f"[APPROVE] Thread ID: {st.session_state.thread_id}")
                        
                        with st.spinner("Processing approval..."):
                            # Resume the graph with approval
                            # According to docs: Command(resume=True) passes True to interrupt() call
                            logger.info(f"[APPROVE] Invoking graph with Command(resume=True)")
                            result = st.session_state.agent.invoke(
                                Command(resume=True),
                                config=st.session_state.thread_id,
                            )
                            
                            logger.info(f"[APPROVE] Graph result type: {type(result)}")
                            logger.info(f"[APPROVE] Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                            logger.info(f"[APPROVE] Has __interrupt__: {result.get('__interrupt__') if isinstance(result, dict) else 'N/A'}")
                            logger.info(f"[APPROVE] Has messages: {'messages' in result if isinstance(result, dict) else 'N/A'}")
                            if isinstance(result, dict) and "messages" in result:
                                logger.info(f"[APPROVE] Result messages count: {len(result['messages'])}")
                        
                        # Check if there's another interrupt
                        if isinstance(result, dict) and result.get("__interrupt__"):
                            logger.warning(f"[APPROVE] Another interrupt detected: {result['__interrupt__']}")
                            st.session_state.pending_interrupt = result["__interrupt__"]
                            st.warning("Another approval is required.")
                            st.rerun()
                        
                        # Merge new messages from the resumed execution
                        if isinstance(result, dict) and "messages" in result:
                            current_message_count = len(st.session_state.messages)
                            new_messages = result["messages"][current_message_count:]
                            logger.info(f"[APPROVE] Merging {len(new_messages)} new messages")
                            if new_messages:
                                st.session_state.messages.extend(new_messages)
                        
                        # Clear interrupt
                        logger.info(f"[APPROVE] Clearing interrupt")
                        st.session_state.pending_interrupt = None
                        st.rerun()
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        logger.error(f"[APPROVE] Exception: {str(e)}")
                        logger.error(f"[APPROVE] Traceback: {error_trace}")
                        st.error(f"❌ Error processing approval: {str(e)}")
                        st.code(error_trace, language="python")
                        st.session_state.pending_interrupt = None
                        st.rerun()
            with col_b:
                if st.button("❌ Reject", type="secondary", key="reject_interrupt", use_container_width=True):
                    try:
                        logger.info(f"[REJECT] Starting rejection flow")
                        logger.info(f"[REJECT] Thread ID: {st.session_state.thread_id}")
                        logger.info(f"[REJECT] Current messages count: {len(st.session_state.messages)}")
                        
                        with st.spinner("Cancelling..."):
                            # Resume the graph with rejection
                            # According to docs: Command(resume=False) passes False to interrupt() call
                            logger.info(f"[REJECT] Invoking graph with Command(resume=False)")
                            result = st.session_state.agent.invoke(
                                Command(resume=False),
                                config=st.session_state.thread_id,
                            )
                            
                            logger.info(f"[REJECT] Graph result type: {type(result)}")
                            logger.info(f"[REJECT] Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                            logger.info(f"[REJECT] Has __interrupt__: {result.get('__interrupt__') if isinstance(result, dict) else 'N/A'}")
                            logger.info(f"[REJECT] Has messages: {'messages' in result if isinstance(result, dict) else 'N/A'}")
                            if isinstance(result, dict) and "messages" in result:
                                logger.info(f"[REJECT] Result messages count: {len(result['messages'])}")
                        
                        # Check if there's another interrupt
                        if isinstance(result, dict) and result.get("__interrupt__"):
                            logger.warning(f"[REJECT] Another interrupt detected: {result['__interrupt__']}")
                            st.session_state.pending_interrupt = result["__interrupt__"]
                            st.warning("Another approval is required.")
                            st.rerun()
                        
                        # Merge new messages (should include cancellation message)
                        if isinstance(result, dict) and "messages" in result:
                            current_message_count = len(st.session_state.messages)
                            new_messages = result["messages"][current_message_count:]
                            logger.info(f"[REJECT] Merging {len(new_messages)} new messages")
                            if new_messages:
                                st.session_state.messages.extend(new_messages)
                        
                        # Clear interrupt and create new thread_id to start fresh
                        # This prevents checkpoint state from interfering with new messages
                        logger.info(f"[REJECT] Clearing interrupt and creating new thread_id")
                        st.session_state.pending_interrupt = None
                        import uuid
                        old_thread_id = st.session_state.thread_id
                        st.session_state.thread_id = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
                        logger.info(f"[REJECT] New thread_id: {st.session_state.thread_id}")
                        st.rerun()
                    except Exception as e:
                        import traceback
                        error_trace = traceback.format_exc()
                        logger.error(f"[REJECT] Exception occurred: {str(e)}")
                        logger.error(f"[REJECT] Traceback: {error_trace}")
                        st.error(f"Error cancelling: {str(e)}")
                        st.code(error_trace, language="python")
                        st.session_state.pending_interrupt = None
                        # Create new thread_id even on error to prevent stuck state
                        import uuid
                        st.session_state.thread_id = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
                        st.rerun()

# Chat input
user_input = st.chat_input(
    "Ask me anything about your account or music...",
    disabled=bool(st.session_state.pending_interrupt),
)

# Handle example button click from sidebar
if "user_input" in st.session_state and st.session_state.user_input:
    user_input = st.session_state.user_input
    st.session_state.user_input = None

if user_input:
    logger.info(f"[NEW_MESSAGE] User input received: {user_input[:50]}...")
    logger.info(f"[NEW_MESSAGE] Current thread_id: {st.session_state.thread_id}")
    logger.info(f"[NEW_MESSAGE] Current messages count: {len(st.session_state.messages)}")
    logger.info(f"[NEW_MESSAGE] Pending interrupt: {st.session_state.pending_interrupt}")
    
    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Clear any pending interrupt when user sends a new message
    if st.session_state.pending_interrupt:
        logger.warning(f"[NEW_MESSAGE] Clearing pending interrupt before processing")
        st.session_state.pending_interrupt = None
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Ensure we're not in an interrupted state - create new thread if needed
                # This prevents checkpoint issues after rejection
                if st.session_state.pending_interrupt:
                    logger.warning(f"[NEW_MESSAGE] Creating new thread_id due to pending interrupt")
                    import uuid
                    st.session_state.thread_id = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
                    st.session_state.pending_interrupt = None
                
                # Invoke agent with verification state
                # Reset approval_status to None for new messages to avoid stale state
                logger.info(f"[NEW_MESSAGE] Invoking graph with {len(st.session_state.messages)} messages")
                logger.info(f"[NEW_MESSAGE] State: is_verified={is_verified}, approval_status=None")
                
                result = st.session_state.agent.invoke(
                    {
                        "messages": st.session_state.messages,
                        "is_verified": is_verified,
                        "verification_requested": False,
                        "pending_account_change": "",
                        "approval_status": None
                    },
                    config=st.session_state.thread_id
                )
                
                logger.info(f"[NEW_MESSAGE] Graph result type: {type(result)}")
                logger.info(f"[NEW_MESSAGE] Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
                if isinstance(result, dict):
                    logger.info(f"[NEW_MESSAGE] Has __interrupt__: {result.get('__interrupt__')}")
                    logger.info(f"[NEW_MESSAGE] Has messages: {'messages' in result}")
                    if "messages" in result:
                        logger.info(f"[NEW_MESSAGE] Result messages count: {len(result['messages'])}")

                # If graph is interrupted, store interrupt payload and stop.
                if isinstance(result, dict) and result.get("__interrupt__"):
                    logger.info(f"[NEW_MESSAGE] Graph interrupted: {result['__interrupt__']}")
                    st.session_state.pending_interrupt = result["__interrupt__"]
                    st.rerun()
                    # st.rerun() stops execution, no return needed
                
                # Extract all messages after the user input
                if isinstance(result, dict) and "messages" in result:
                    new_messages = result["messages"][len(st.session_state.messages):]
                    logger.info(f"[NEW_MESSAGE] Extracted {len(new_messages)} new messages")
                    
                    # Add new messages to history
                    st.session_state.messages.extend(new_messages)
                else:
                    logger.warning(f"[NEW_MESSAGE] No messages in result!")
                
                # Display the final AI response
                final_message = None
                for msg in reversed(new_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_message = msg
                        break
                
                if final_message:
                    # Check if the message contains a YouTube video embed signal
                    content = final_message.content
                    if "YOUTUBE_VIDEO|" in content:
                        # Parse the video information
                        lines = content.split('\n')
                        for line in lines:
                            if line.startswith("YOUTUBE_VIDEO|"):
                                parts = line.split('|')
                                if len(parts) >= 4:
                                    video_id = parts[1]
                                    video_title = parts[2]
                                    channel_title = parts[3]
                                    
                                    # Display text before the video tag
                                    text_before = content.split("YOUTUBE_VIDEO|")[0].strip()
                                    if text_before:
                                        st.write(text_before)
                                    
                                    # Create beautiful header
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #0A1020 0%, #071526 60%, #031819 100%); 
                                                padding: 1rem; 
                                                border-radius: 15px 15px 0 0; 
                                                margin-top: 1.5rem;
                                                border: 1px solid rgba(255,255,255,0.10);
                                                border-bottom: none;">
                                        <h3 style="margin: 0; color: white; font-size: 1.2rem;">🎥 {video_title}</h3>
                                        <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 0.9rem;">
                                            📺 {channel_title}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Use Streamlit's iframe component for full functionality
                                    import streamlit.components.v1 as components
                                    components.iframe(
                                        f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1&rel=0&modestbranding=1",
                                        height=400,
                                        scrolling=False
                                    )
                                    
                                    # Display text after the video tag if any
                                    text_after = content.split(line)[1].strip() if len(content.split(line)) > 1 else ""
                                    if text_after:
                                        st.write(text_after)
                                    break
                        else:
                            # No video found, display normally
                            st.write(content.replace("YOUTUBE_VIDEO|", "").strip())
                    else:
                        st.write(content)
                    
                    # Show tool calls if any
                    if hasattr(final_message, "tool_calls") and final_message.tool_calls:
                        with st.expander("🔧 Tool Calls", expanded=False):
                            for tool_call in final_message.tool_calls:
                                st.markdown(f"""
                                <div class="tool-call">
                                    <strong>Tool:</strong> {tool_call['name']}<br>
                                    <strong>Args:</strong> {tool_call.get('args', {})}
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning("No response generated. Please try again.")
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"[NEW_MESSAGE] Exception: {str(e)}")
                logger.error(f"[NEW_MESSAGE] Traceback: {error_trace}")
                st.error(f"❌ Error processing message: {str(e)}")
                st.code(error_trace, language="python")
                # Create new thread_id on error to prevent stuck state
                import uuid
                logger.info(f"[NEW_MESSAGE] Creating new thread_id due to error")
                st.session_state.thread_id = {"configurable": {"thread_id": f"thread-{uuid.uuid4().hex[:8]}"}}
                st.session_state.pending_interrupt = None
    
    st.rerun()
