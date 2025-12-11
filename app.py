"""
Streamlit UI for the Music Store Customer Support Bot.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph_with_verification import create_agent_with_memory, CUSTOMER_INFO
from verification import get_verification_service
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Music Store Support Bot (Secure)",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .customer-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .tool-call {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = create_agent_with_memory()
    st.session_state.thread_id = {"configurable": {"thread_id": "1"}}

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_greeting = True
else:
    if "show_greeting" not in st.session_state:
        st.session_state.show_greeting = False

# Initialize persistent verification store
if "verification_store" not in st.session_state:
    st.session_state.verification_store = {}

# Get verification service with persistent store
# This makes verification persist across reruns (until page refresh)
verification_service = get_verification_service(st.session_state.verification_store)
is_verified = verification_service.is_verified(CUSTOMER_INFO['id'])


# Header
st.markdown('<div class="main-header">🔒 Music Store Customer Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered assistant with SMS verification</div>', unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.header("👤 Current Customer")
    st.markdown(f"""
    <div class="customer-info">
        <strong>Name:</strong> {CUSTOMER_INFO['full_name']}<br>
        <strong>Customer ID:</strong> {CUSTOMER_INFO['id']}<br>
        <strong>Email:</strong> {CUSTOMER_INFO['email']}<br>
        <strong>Phone:</strong> {CUSTOMER_INFO['phone']}
    </div>
    """, unsafe_allow_html=True)
    
    # Verification Status
    st.header("🔐 Verification Status")
    if is_verified:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #28a745; color: #155724; margin-bottom: 1rem;">
            ✅ VERIFIED<br>
            <small>You can update account information</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #ffc107; color: #856404; margin-bottom: 1rem;">
            🔒 NOT VERIFIED<br>
            <small>Verification required for account changes</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.header("💡 What Can I Help With?")
    st.markdown("""
    **Browse & View (No Verification):**
    - View account details
    - Check order history
    - Browse music catalog
    - Search songs & artists
    - 🎵 Find songs by lyrics!
    - 🎥 Watch music videos
    
    **Secure Updates (Requires SMS):**
    - 🔒 Change email address
    - 🔒 Update mailing address
    
    *SMS verification required for security*
    """)
    
    st.header("🎯 Example Questions")
    
    st.subheader("📊 Information")
    regular_questions = [
        "Show me my account details",
        "What's my purchase history?",
        "Find Rock music",
    ]
    for question in regular_questions:
        if st.button(question, key=f"reg_{question}"):
            st.session_state.user_input = question
            st.rerun()
    
    st.subheader("🎵 Lyrics Search")
    lyrics_questions = [
        "I heard a song that goes 'can't you see'",
        "Find the song with lyrics 'I will always love you'",
    ]
    for question in lyrics_questions:
        if st.button(question, key=f"lyr_{question}"):
            st.session_state.user_input = question
            st.rerun()
    
    st.subheader("🔒 Secure Changes")
    secure_questions = [
        "Change my email address",
        "Update my mailing address",
    ]
    for question in secure_questions:
        if st.button(question, key=f"sec_{question}"):
            st.session_state.user_input = question
            st.rerun()
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.show_greeting = True
            st.session_state.thread_id = {"configurable": {"thread_id": str(len(st.session_state.messages))}}
            st.rerun()
    
    with col2:
        if st.button("🔓 Clear Verification", type="secondary"):
            verification_service.clear_verification(CUSTOMER_INFO['id'])
            st.success("Verification cleared!")
            st.rerun()
    
    st.divider()
    st.caption("🔒 Secured with Twilio SMS")
    st.caption("Built with LangGraph + LangChain")


# Main chat interface
chat_container = st.container()

with chat_container:
    # Show welcome greeting on first load
    if st.session_state.show_greeting and len(st.session_state.messages) == 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    color: white; 
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="margin: 0 0 1rem 0; color: white;">👋 Welcome, {CUSTOMER_INFO['first_name']}!</h2>
            <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">
                I'm your secure music store assistant. I can help you with:
            </p>
            <ul style="margin: 0.5rem 0 0 1.5rem; font-size: 1rem;">
                <li>Viewing your account and purchase history</li>
                <li>Finding new music - songs, albums, and artists</li>
                <li>🎵 <strong>Finding songs by lyrics</strong> - just tell me what you remember!</li>
                <li>🎥 <strong>Watching music videos</strong> - preview songs before buying</li>
                <li>🔒 <strong>Securely updating</strong> your email or address (requires SMS verification)</li>
                <li>Answering any questions about your orders</li>
            </ul>
            <p style="margin: 1rem 0 0 0; font-size: 0.95rem; opacity: 0.9;">
                🔐 <strong>Security Note:</strong> Account changes require SMS verification to your phone: {CUSTOMER_INFO['phone']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.show_greeting = False
    
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
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            padding: 1rem; 
                                            border-radius: 15px 15px 0 0; 
                                            margin-top: 1.5rem;">
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

# Chat input
user_input = st.chat_input("Ask me anything about your account or music...")

# Handle example button click from sidebar
if "user_input" in st.session_state and st.session_state.user_input:
    user_input = st.session_state.user_input
    st.session_state.user_input = None

if user_input:
    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke agent with verification state
                result = st.session_state.agent.invoke(
                    {
                        "messages": st.session_state.messages,
                        "is_verified": is_verified,
                        "verification_requested": False,
                        "pending_account_change": ""
                    },
                    config=st.session_state.thread_id
                )
                
                # Extract all messages after the user input
                new_messages = result["messages"][len(st.session_state.messages):]
                
                # Add new messages to history
                st.session_state.messages.extend(new_messages)
                
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
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                padding: 1rem; 
                                                border-radius: 15px 15px 0 0; 
                                                margin-top: 1.5rem;">
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
                st.error(f"Error: {str(e)}")
                st.exception(e)
    
    st.rerun()


# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Messages", len([m for m in st.session_state.messages if isinstance(m, HumanMessage)]))
with col2:
    st.metric("Responses", len([m for m in st.session_state.messages if isinstance(m, AIMessage) and m.content]))
with col3:
    status = "✅ Verified" if is_verified else "🔒 Not Verified"
    st.metric("Security", status)
with col4:
    api_key_set = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
    st.metric("API Key", api_key_set)
