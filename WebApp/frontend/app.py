import streamlit as st
import openai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="CUDA → HIP Transpiler", layout="wide")

# Sidebar - Configuration
st.sidebar.title("🛠️ Settings")
mode = st.sidebar.selectbox("Mode", ["CUDA → HIP Transpiler", "General Chat"])
backend_url = st.sidebar.text_input("Backend URL (e.g., http://localhost:11434)", value=os.getenv("BACKEND_URL", "http://localhost:11434"))
model_name = st.sidebar.text_input("Model Name", value=os.getenv("MODEL_NAME", "gemma-4-transpiler"))

if not backend_url:
    st.sidebar.warning("⚠️ Provide a Backend URL to start.")


# Gemma 4 Reasoning Parser
THINKING_START_TAG = "<|channel>"
THINKING_END_TAG = "<channel|>"
THOUGHT_PREFIX = "thought\n"
TURN_END_TAG = "<turn|>"

def parse_thinking_output(text: str) -> dict:
    if THINKING_END_TAG in text:
        parts = text.split(THINKING_END_TAG, 1)
        thinking_block = parts[0]
        answer = parts[1].strip()
        if THINKING_START_TAG in thinking_block:
            thinking = thinking_block.split(THINKING_START_TAG, 1)[1]
        else:
            thinking = thinking_block
        if thinking.startswith(THOUGHT_PREFIX):
            thinking = thinking[len(THOUGHT_PREFIX):]
        if answer.endswith(TURN_END_TAG):
            answer = answer[:-len(TURN_END_TAG)].rstrip()
        if answer.endswith("<eos>"):
            answer = answer[:-5].rstrip()
        return {"thinking": thinking.strip(), "answer": answer}
    return {"thinking": None, "answer": text.strip()}

# Main UI
if mode == "CUDA → HIP Transpiler":
    st.title("🚀 CUDA → HIP Transpiler")
    st.markdown("Transpile CUDA kernels to performance-aware HIP code.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CUDA Source")
        user_input = st.text_area("Paste CUDA code:", height=400, placeholder="__global__ void kernel(...) { ... }")
    with col2:
        st.subheader("HIP Result")
        output_container = st.empty()
        output_container.code("// HIP code will appear here", language="cpp")
    
    system_prompt = "You are an expert GPU programmer specializing in CUDA to HIP translation."
    user_prompt = f"Transpile the following CUDA code to HIP. Provide your reasoning in a thinking block.\n\n```cuda\n{user_input}\n```"

else:
    st.title("💬 General Chatbot")
    st.markdown("Chat with Gemma 4 (Reasoning enabled).")
    user_input = st.text_area("Message:", height=200, placeholder="Ask me anything...")
    output_container = st.empty()
    
    system_prompt = "You are a helpful and intelligent AI assistant powered by Gemma 4."
    user_prompt = user_input

if st.button("Send / Transpile ✨"):
    if not backend_url:
        st.error("Backend URL is missing!")
    elif not user_input:
        st.error("Please provide some input.")
    else:
        try:
            # Robustly clean the URL
            clean_url = backend_url.strip().rstrip("/").rstrip(".")
            
            client = openai.OpenAI(base_url=f"{clean_url}/v1", api_key="ollama")
            
            # Debug info (hidden by default)
            with st.expander("Debug Connection Details"):
                st.write(f"Target URL: `{clean_url}/v1`")
                st.write(f"Model: `{model_name}`")

            with st.spinner("Processing..."):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.4 if mode == "General Chat" else 0.2,
                )
                
                raw_content = response.choices[0].message.content
                parsed = parse_thinking_output(raw_content)
                
                if parsed["thinking"]:
                    with st.expander("Show Reasoning", expanded=True):
                        st.info(parsed["thinking"])
                
                if mode == "CUDA → HIP Transpiler":
                    hip_code = parsed["answer"]
                    code_blocks = re.findall(r"```(?:cpp|roc|cuda|c|hip)?\n(.*?)```", hip_code, re.DOTALL)
                    if code_blocks:
                        hip_code = code_blocks[0].strip()
                    output_container.code(hip_code, language="cpp")
                else:
                    st.markdown(parsed["answer"])
                
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
            st.info("Check if Ollama is running and the URL is correct.")

# Footer
st.markdown("---")
st.caption("Built for AMD Hackathon | Powered by Gemma 4 ")
