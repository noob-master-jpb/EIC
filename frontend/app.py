import streamlit as st
import os
import time
import html
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from dotenv import load_dotenv

# ---------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------
MODEL_PATH = "/home/ankan/projects/EIC/models/gemma-4-E2B-it"
DEVICE_MODE = "gpu"

@st.cache_resource(show_spinner="Loading Model into GPU Memory...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if torch.cuda.is_available():
        import os as _os
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            quantization_config=bnb_config,
            max_memory={0: "7GiB", "cpu": "16GiB"},
            offload_buffers=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16
        )
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ROCm Bridge",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-color: #161616;
    --panel-bg: #1A1A1A;
    --text-main: #E0E0E0;
    --text-muted: #888888;
    --accent-blue: #93C5FD;
    --border-color: #2A2A2A;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-color) !important;
    color: var(--text-main) !important;
}

.stApp, .stApp > header {
    background-color: var(--bg-color) !important;
}

/* Nuke glow */
.stApp, .stApp > div, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > div {
    box-shadow: none !important;
    outline: none !important;
    border: none !important;
}
.stApp::before, .stApp::after, .stApp > div::before, .stApp > div::after,
[data-testid="stAppViewContainer"]::before, [data-testid="stAppViewContainer"]::after,
[data-testid="stMain"]::before, [data-testid="stMain"]::after,
[class*="st-emotion-cache"]::before, [class*="st-emotion-cache"]::after {
    display: none !important;
    content: none !important;
}

/* Hide default elements */
header, footer, .stDeployButton, [data-testid="stToolbar"] {
    display: none !important;
}

/* Force wide layout */
[data-testid="stMainBlockContainer"] {
    max-width: 95% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ------------------------------------------------ */
/* TOPBAR                                           */
/* ------------------------------------------------ */
.topbar-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background-color: #1A1A1A;
    border-bottom: 1px solid #2A2A2A;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 30px;
    z-index: 999999;
}

.topbar-logo {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--accent-blue);
    letter-spacing: 0.5px;
}

.topbar-center {
    display: flex;
    background-color: #252525;
    border-radius: 20px;
    padding: 4px;
    gap: 4px;
}

.topbar-btn {
    text-decoration: none;
    display: inline-block;
    padding: 6px 20px;
    border-radius: 16px;
    font-size: 0.65rem;
    font-weight: 600;
    color: #888;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
    transition: all 0.2s ease;
}
.topbar-btn.active {
    background-color: #B4D5FF;
    color: #121212;
}
.topbar-btn:hover {
    color: #CCC;
}
.topbar-btn.active:hover {
    color: #121212;
}

/* Converter specific CSS */
.editor-header {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-blue);
    margin-bottom: 8px;
    padding: 0 4px;
}
.editor-header span:last-child {
    color: #888;
}

.stTextArea textarea {
    background-color: #121212 !important;
    border: 1px solid #333 !important;
    color: #CCC !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
}

/* Specific styling for the convert button */
.stButton button {
    background-color: #B4D5FF !important;
    color: #121212 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-top: 40px !important;
}
.stButton button:hover {
    background-color: var(--accent-blue) !important;
    color: #121212 !important;
}

.topbar-right {
    display: flex;
    gap: 16px;
    color: #888;
}
.topbar-right svg {
    width: 20px;
    height: 20px;
    cursor: pointer;
}

/* ------------------------------------------------ */
/* VERTICAL TEXTS                                   */
/* ------------------------------------------------ */
.vertical-left, .vertical-right {
    position: fixed;
    top: 50%;
    transform: translateY(-50%) rotate(180deg);
    writing-mode: vertical-rl;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 4px;
    color: #333;
    z-index: 100;
    pointer-events: none;
}
.vertical-left {
    left: 20px;
}
.vertical-right {
    right: 20px;
    transform: translateY(-50%);
}

/* ------------------------------------------------ */
/* MESSAGES                                         */
/* ------------------------------------------------ */
.block-container {
    padding-top: 80px !important;
    padding-bottom: 120px !important;
    max-width: 900px !important;
}

.msg-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-blue);
    margin-bottom: 4px;
    margin-top: 20px;
}
.msg-header .sys-core {
    color: #555;
    margin-left: 8px;
}
.msg-header.user {
    color: #888;
    text-align: right;
    margin-right: 10%;
}

/* Hide avatars completely */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* General Chat Message styling */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

/* Assistant Box */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div {
    background-color: var(--panel-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-left: 2px solid var(--accent-blue) !important;
    border-radius: 4px !important;
    padding: 16px 20px !important;
    color: #CCC !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

/* User Box */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div {
    background-color: transparent !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
    padding: 16px 20px !important;
    color: #CCC !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    max-width: 80% !important;
    margin-left: auto;
}

/* Code Blocks */
pre, code {
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMarkdownContainer"] pre {
    background-color: #0A0A0A !important;
    border: 1px solid #222 !important;
    border-radius: 4px !important;
    padding: 16px !important;
    font-size: 0.8rem !important;
}
.stCodeBlock {
    margin-top: 12px;
}

/* ------------------------------------------------ */
/* INPUT AREA                                       */
/* ------------------------------------------------ */
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background-color: var(--bg-color) !important;
    background: var(--bg-color) !important;
}

[data-testid="stBottomBlockContainer"] {
    background-color: var(--bg-color) !important;
    max-width: 840px !important;
    margin: 0 auto !important;
    padding: 10px 20px 30px 20px !important;
    position: relative !important;
}

/* Bottom stats injected via CSS pseudo-elements */
[data-testid="stBottomBlockContainer"]::before {
    content: "LATENCY: 14MS \\00a0\\00a0\\00a0 COMPUTE: MI300X_NODE_A";
    position: absolute;
    bottom: 5px;
    left: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    color: #444;
    display: block !important;
}

[data-testid="stBottomBlockContainer"]::after {
    content: "ENTER TO SEND / SHIFT+ENTER FOR NEW LINE";
    position: absolute;
    bottom: 5px;
    right: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.5rem;
    color: #444;
    display: block !important;
}

/* Remove default bottom shadow/glow */
[data-testid="stBottom"]::before, [data-testid="stBottom"]::after {
    display: none !important;
}

/* Input Container Wrapper */
.stChatInput,
.stChatInput > div,
.stChatInput form,
.stChatInput form > div,
.stChatInput [data-testid="stChatInputContainer"] {
    background: transparent !important;
    background-color: transparent !important;
}

.stChatInput [data-baseweb="textarea"] {
    background-color: var(--panel-bg) !important;
    background: var(--panel-bg) !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    padding: 24px 16px 12px 16px !important;
    box-shadow: none !important;
}
.stChatInput [data-baseweb="textarea"] > div,
.stChatInput [data-baseweb="textarea"] > div > div {
    background-color: transparent !important;
    background: transparent !important;
}
.stChatInput [data-baseweb="textarea"]:focus-within {
    border-color: #555 !important;
}

/* "COMMAND INPUT" Label */
.stChatInput::before {
    content: "COMMAND INPUT";
    position: absolute;
    top: 12px;
    left: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #666;
    letter-spacing: 1px;
    z-index: 10;
}

/* Paperclip Icon */
.stChatInput::after {
    content: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%23666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>');
    position: absolute;
    right: 56px;
    bottom: 16px;
    pointer-events: none;
    z-index: 10;
}

/* The actual textarea */
.stChatInput textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text-main) !important;
    background: transparent !important;
    border: none !important;
}
.stChatInput textarea::placeholder {
    color: #555 !important;
}
.stChatInput textarea:focus {
    outline: none !important;
}

/* Send Button */
.stChatInput button, .stChatInput [data-testid="stChatInputSubmitButton"] {
    background-color: #B4D5FF !important;
    border-radius: 4px !important;
    padding: 4px !important;
    right: 12px !important;
    bottom: 12px !important;
    height: 32px !important;
    width: 32px !important;
    border: none !important;
    min-height: 32px !important;
}
.stChatInput button:hover {
    background-color: var(--accent-blue) !important;
}
.stChatInput button svg {
    color: #121212 !important;
    fill: #121212 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TOPBAR HTML
# ---------------------------------------------------
page = st.query_params.get("page", "CHAT")
chat_active = "active" if page == "CHAT" else ""
conv_active = "active" if page == "CONVERTER" else ""

st.markdown(f"""
<div class="topbar-wrapper">
    <div class="topbar-logo">ROCm Bridge</div>
    <div class="topbar-center">
        <a href="?page=CHAT" target="_self" class="topbar-btn {chat_active}">CHAT</a>
        <a href="?page=CONVERTER" target="_self" class="topbar-btn {conv_active}">CONVERTER</a>
    </div>
    <div class="topbar-right">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    </div>
</div>
<div class="vertical-left">SYSTEM_MONITOR_ACTIVE</div>
<div class="vertical-right">DATA_STREAM_ENCRYPTED</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# INITIAL STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if page == "CONVERTER":
    st.markdown("<h2 style='text-align: center; color: white; margin-top: 10px; font-weight: 600;'>Kernel Translation Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #aaa; margin-bottom: 40px; font-size: 0.9rem;'>Paste your legacy CUDA kernel code below to instantly transpile it into optimized ROCm compatible syntax using the Bridge API.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='editor-header'><span>CUDA Kernel Input</span><span>.cu</span></div>", unsafe_allow_html=True)
        cuda_code = st.text_area("CUDA Input", label_visibility="collapsed", height=450, placeholder="// Enter CUDA code here...")
    
    with col2:
        st.markdown("<div class='editor-header'><span>ROCm Kernel Output</span><span>.cpp (HIP)</span></div>", unsafe_allow_html=True)
        if "converted_code" not in st.session_state:
            st.session_state.converted_code = "// Output will appear here..."
        st.code(st.session_state.converted_code, language="cpp")

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        if st.button("⚡ Convert", use_container_width=True):
            if not cuda_code.strip():
                st.session_state.converted_code = "// Please provide CUDA code to convert."
            else:
                try:
                    messages = [
                        {"role": "user", "content": f"You are an expert GPU programming engineer specializing in AMD ROCm and HIP. Your task is to PERFECTLY transpile the following CUDA C++ code into AMD ROCm HIP C++ code.\\n\\nRules:\\n1. Replace all CUDA specific API calls with their HIP equivalents (e.g. cudaMalloc -> hipMalloc, cudaMemcpy -> hipMemcpy, threadIdx.x -> hipThreadIdx_x).\\n2. Do NOT provide any markdown formatting or explanations, output raw compilable C++ code ONLY.\\n3. Make sure to include <hip/hip_runtime.h>.\\n\\nCUDA Code:\\n\\n{cuda_code}"}
                    ]
                    with st.spinner("Transpiling CUDA to HIP..."):
                        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        first_layer_device = next(model.parameters()).device
                        inputs = tokenizer(prompt_text, return_tensors="pt").to(first_layer_device)
                        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
                        
                        output_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        st.session_state.converted_code = output_text
                except Exception as exc:
                    st.session_state.converted_code = f"// Conversion Error: {exc}"
            st.rerun()

elif page == "CHAT":
    # ---------------------------------------------------
    # RENDER CHAT
    # ---------------------------------------------------
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.markdown("<div class='msg-header'>ROCm Bridge <span class='sys-core'>SYSTEM CORE V1.0.4</span></div>", unsafe_allow_html=True)
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        else:
            st.markdown("<div class='msg-header user'>Kernel_Dev</div>", unsafe_allow_html=True)
            with st.chat_message("user"):
                st.markdown(msg["content"])

    # ---------------------------------------------------
    # INPUT & CHAT LOGIC
    # ---------------------------------------------------
    prompt = st.chat_input("Input ROCm command or query...")

    if prompt:
        # 1. Display and save user message immediately
        st.markdown("<div class='msg-header user'>Kernel_Dev</div>", unsafe_allow_html=True)
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Display assistant thinking state
        st.markdown("<div class='msg-header'>ROCm Bridge <span class='sys-core'>SYSTEM CORE V1.0.4</span></div>", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            full_response = ""
            try:
                prompt_text = tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
                first_layer_device = next(model.parameters()).device
                inputs = tokenizer(prompt_text, return_tensors="pt").to(first_layer_device)
                
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.7)
                
                thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                for chunk in streamer:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as exc:
                error_msg = f"⚠️ Backend error: `{exc}`"
                placeholder.markdown(error_msg)
                full_response = error_msg
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})