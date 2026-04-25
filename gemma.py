# Gemma 4 Reasoning Parser Utilities
# Ported from vLLM/Transformers native implementations for Gemma 4

import re

# Thinking delimiter tokens as they appear in decoded text.
# Gemma4 uses <|channel> (start) and <channel|> (end) as thinking delimiters.
THINKING_START_TAG = "<|channel>"
THINKING_END_TAG = "<channel|>"
THOUGHT_PREFIX = "thought\n"
TURN_END_TAG = "<turn|>"

def _strip_thought_label(text: str) -> str:
    """Strip the 'thought\\n' label from the start of text."""
    if text.startswith(THOUGHT_PREFIX):
        return text[len(THOUGHT_PREFIX):]
    return text

def _clean_answer(text: str) -> str:
    """Clean trailing sentinel tokens from the answer text."""
    text = text.strip()
    # Strip trailing <turn|> (Gemma4 turn-end marker)
    if text.endswith(TURN_END_TAG):
        text = text[:-len(TURN_END_TAG)].rstrip()
    # Strip trailing <eos> if present
    if text.endswith("<eos>"):
        text = text[:-5].rstrip()
    return text

def parse_thinking_output(text: str) -> dict:
    """
    Parse decoded Gemma4 model output to extract thinking and answer.
    
    Returns:
        A dict with keys:
            - "thinking": The chain-of-thought text, or None if not found.
            - "answer": The final answer text.
    """
    if THINKING_END_TAG in text:
        parts = text.split(THINKING_END_TAG, 1)
        thinking_block = parts[0]
        answer = _clean_answer(parts[1])

        # Extract thinking content: strip the start tag if present
        if THINKING_START_TAG in thinking_block:
            thinking = thinking_block.split(THINKING_START_TAG, 1)[1]
        else:
            thinking = thinking_block

        # Strip the "thought\n" role label
        thinking = _strip_thought_label(thinking.strip())
        thinking = thinking.strip()

        return {"thinking": thinking, "answer": answer}

    # No thinking delimiters found.
    answer = _strip_thought_label(text)
    answer = _clean_answer(answer)
    return {"thinking": None, "answer": answer}

class Gemma4ReasoningParser:
    """
    A class-based implementation of the Gemma 4 reasoning parser.
    Matches the naming requested by the user.
    """
    def __init__(self):
        pass

    def parse(self, text: str) -> dict:
        return parse_thinking_output(text)
