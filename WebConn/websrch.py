import argparse
import asyncio
import json
import torch
import warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from transformers import AutoModelForCausalLM, AutoTokenizer
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../qwen3.5-0.8b-local")
    parser.add_argument("--question", type=str, default="what is the latest market cap of nividia in may 8 2026")
    parser.add_argument("--mcp_cmd", type=str, default="python")
    parser.add_argument("--mcp_args", type=str, default="server.py")
    return parser.parse_args()

def fallback_search_bs4(query):
    print("-> Attempting fallback search using BeautifulSoup (DuckDuckGo HTML)...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for a in soup.find_all('a', class_='result__snippet'):
            results.append(a.text)
            
        if not results:
            print("-> No snippets found in fallback search. Extracting visible text...")
            text = soup.get_text(separator=' ', strip=True)
            return text[:3000]
            
        return "\n".join(results)
    except Exception as e:
        print(f"-> Fallback search failed: {e}")
        return ""

async def search_and_answer(args):
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")

    print("\n[1] Formulating search query with Qwen...")
    query_prompt = [
        {"role": "system", "content": "You are a web search query generator. Output ONLY the short search query to use on a search engine. Do not answer the question. Do not include quotes."},
        {"role": "user", "content": "what is the capital of france?"},
        {"role": "assistant", "content": "capital of france"},
        {"role": "user", "content": "who won the super bowl in 2024"},
        {"role": "assistant", "content": "super bowl 2024 winner"},
        {"role": "user", "content": args.question}
    ]
    query_text = tokenizer.apply_chat_template(query_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([query_text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
    search_query = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"-> Search query: '{search_query}'")

    print(f"\n[2] Querying MCP browser server with '{search_query}'...")
    server_params = StdioServerParameters(command=args.mcp_cmd, args=[args.mcp_args])
    
    context_text = ""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("-> Connected to MCP browser server. Executing search_web...")
            await session.call_tool("search_web", {"query": search_query})
            
            print("-> Waiting 3 seconds for page to load...")
            await asyncio.sleep(3) 
            
            print("-> Extracting text from page...")
            result = await session.call_tool("extract_text", {})
            
            content_json = result.content[0].text
            try:
                parsed = json.loads(content_json)
                context_text = parsed.get("content", content_json)
            except Exception as e:
                context_text = content_json
            print(f"-> Extracted {len(context_text)} characters.")
            
            if len(context_text) < 500 or "Please complete the following challenge" in context_text or "captcha" in context_text.lower():
                print("-> MCP browser results look incomplete or blocked.")
                fallback_context = fallback_search_bs4(search_query)
                if fallback_context:
                    context_text = fallback_context
                    print(f"-> Fallback search succeeded. Extracted {len(context_text)} characters.")
            
            print(f"--- Context snippet ---\n{context_text[:1000]}\n-----------------------")
                
    print("\n[3] Generating final answer with Qwen...")
    answer_prompt = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided web search context to answer the user's question. If the context does not contain the answer, say you don't know."},
        {"role": "user", "content": f"Context from web search:\n{context_text[:3000]}\n\nQuestion: {args.question}"}
    ]
    answer_text = tokenizer.apply_chat_template(answer_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([answer_text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    final_answer = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(final_answer)

if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(search_and_answer(args))
    except RuntimeError as e:
        if str(e) == "Event loop is closed":
            pass
        else:
            raise
