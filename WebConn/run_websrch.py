import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Parent script to run the Web Search Qwen Service")
    parser.add_argument("--question", type=str, default="what is the latest market cap of nividia in may 8 2026", help="The question to ask")
    parser.add_argument("--model_path", type=str, default="../qwen3.5-0.8b-local", help="Path to the local Qwen model")
    args = parser.parse_args()
    
    cmd = [
        "python", "websrch.py",
        "--model_path", args.model_path,
        "--question", args.question
    ]
    
    print(f"Starting parent script. Delegating to websrch.py...")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run the websrch.py script and stream output to stdout
    result = subprocess.run(cmd)
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
