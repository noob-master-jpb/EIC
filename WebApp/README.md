# CUDA → HIP Transpiler Frontend

This is a Streamlit-based frontend for transpiling CUDA code to HIP using a Gemma 4-based model.

## Features
- **CUDA to HIP Translation**: Specialized mode for GPU kernel conversion.
- **General Chat**: Interact with the model for general queries.
- **Reasoning Display**: Automatically parses and displays the model's "thinking" process.

## Setup & Running

1. **Install Requirements**:
   ```bash
   pip install streamlit openai python-dotenv
   ```

2. **Configuration**:
   Create a `.env` file in this directory or provide the values in the sidebar:
   - `BACKEND_URL`: The URL where your Ollama/vLLM server is running (e.g., `http://localhost:11434`).
   - `MODEL_NAME`: The name of the model to use (e.g., `gemma-4-transpiler`).

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Integration
The frontend expects an OpenAI-compatible API endpoint provided by the backend. It targets `{BACKEND_URL}/v1`.
