import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.title("Chat with Qwen 3.5 (0.8B)")

# Setup Ngrok URL from Secrets or Env
api_url = os.environ.get("NGROK_URL")
if not api_url:
    try:
        api_url = st.secrets["NGROK_URL"]
    except Exception:
        pass

if not api_url:
    st.error("Please set NGROK_URL in .env or Replit Secrets")
    st.stop()

# Initialize OpenAI client to point to the vLLM server via Ngrok
client = OpenAI(
    base_url=f"{api_url.rstrip('/')}/v1",
    api_key="EMPTY" # vLLM doesn't require an API key by default
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS for fixed left-middle navigation
st.markdown("""
    <style>
    html {
        scroll-behavior: smooth;
    }
    .nav-container {
        position: fixed;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        display: flex;
        flex-direction: column;
        gap: 12px;
        z-index: 1000;
    }
    .nav-dot {
        background-color: #ff4b4b;
        width: 28px;
        height: 10px;
        border-radius: 4px;
        transition: all 0.3s ease;
        display: block;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        position: relative;
    }
    .nav-dot:hover {
        background-color: #fff;
        width: 40px;
        cursor: pointer;
    }
    /* Custom tooltip style */
    .nav-dot::after {
        content: attr(data-preview);
        position: absolute;
        left: 55px;
        top: 50%;
        transform: translateY(-50%);
        background-color: rgba(0, 0, 0, 0.85);
        color: white;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 14px;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.2s, visibility 0.2s;
        pointer-events: none;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .nav-dot:hover::after {
        opacity: 1;
        visibility: visible;
    }
    /* High-visibility flash for the targeted prompt */
    @keyframes flash-highlight {
        0% { background-color: #ffff00; transform: scale(1.02); box-shadow: 0 0 20px #ffff00; }
        20% { background-color: #ffff00; }
        100% { background-color: transparent; transform: scale(1); box-shadow: none; }
    }
    
    /* This targets the message container immediately following our anchor div */
    div[id^="msg_"]:target + div {
        animation: flash-highlight 2s ease-out;
        border-radius: 12px;
        padding: 5px;
    }
    
    /* Ensure the anchor has space so the message isn't at the very top edge */
    div[id^="msg_"] {
        scroll-margin-top: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# Build the navigation dots HTML
nav_html = '<div class="nav-container">'
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        # Using data-preview for the custom CSS tooltip and href for scrolling
        preview_text = message["content"][:100].replace('"', '&quot;')
        nav_html += f'<a href="#msg_{i}" class="nav-dot" data-preview="{preview_text}..."></a>'
nav_html += '</div>'

# Render the navigation
st.markdown(nav_html, unsafe_allow_html=True)

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    # Add an HTML anchor before each message for the navigation to work
    st.markdown(f'<div id="msg_{i}"></div>', unsafe_allow_html=True)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            # Stream the response
            for response in client.chat.completions.create(
                model="Qwen/Qwen3.5-0.8B", 
                messages=st.session_state.messages,
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
            st.stop()
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
