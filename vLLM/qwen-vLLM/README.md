# Qwen 3.5 (0.8B) Chat - Local Backend with Streamlit Frontend

> This project sets up a high-performance, low-latency chat interface using the **Qwen 3.5 (0.8B)** model. It uses **vLLM** for an OpenAI-compatible local backend and **Streamlit** for a modern, interactive frontend that can be hosted on Replit.

## 🚀 Architecture
- **Backend:** Locally hosted **vLLM** server running on an NVIDIA GPU (optimized for 8GB VRAM).
- **Frontend:** **Streamlit** app with custom CSS navigation for chat history.
- **Connectivity:** **Ngrok** tunnel to securely expose the local backend to the public internet (for Replit connectivity).

## 📁 Project Structure
```text
Arya-Files/qwen/
├── app.py              # Streamlit frontend (Replit/Local)
├── .env                # Environment variables (HF_TOKEN, NGROK_AUTHTOKEN, NGROK_URL)
├── start_backend.sh    # Script to launch the vLLM server
├── start_ngrok.sh      # Script to start the Ngrok tunnel
├── download_model.sh   # Script to download Qwen3.5-0.8B from Hugging Face
└── README.md           # This file
```

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU 
- [Ngrok account](https://dashboard.ngrok.com/)

### 2. Configuration
Create a `.env` file in the `Arya-Files/qwen/` directory:
```env
HF_TOKEN=your_huggingface_token
NGROK_AUTHTOKEN=your_ngrok_authtoken
NGROK_URL=your_ngrok_public_url (update this after starting ngrok)
```

### 3. Execution (Local Machine)

**Step A: Download the Model**
```bash
./download_model.sh
```

**Step B: Start the Backend Server**
```bash
./start_backend.sh
```

**Step C: Start the Tunnel**
```bash
./start_ngrok.sh
```
*Copy the `Forwarding` URL (e.g., `https://xxxx.ngrok-free.dev`) and update your `NGROK_URL` in `.env` or Replit Secrets.*

### 4. Frontend (Streamlit)
**On Replit:**
- Copy `app.py` and `.env` (or set Secrets).
- Deploy/Run.

**Locally:**
```bash
streamlit run app.py
```
