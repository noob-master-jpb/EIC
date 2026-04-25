import gemma

test_input = """<|channel>thought
This is a test of the reasoning process.
I am thinking about Fibonacci.
<channel|>The Fibonacci sequence starts with 0 and 1."""

print("Testing Gemma4ReasoningParser...")
parser = gemma.Gemma4ReasoningParser()
result = parser.parse(test_input)

print(f"Thinking: {result['thinking']}")
print(f"Answer: {result['answer']}")

assert result['thinking'] == "This is a test of the reasoning process.\nI am thinking about Fibonacci."
assert result['answer'] == "The Fibonacci sequence starts with 0 and 1."
print("Test passed!")
