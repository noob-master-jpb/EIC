import json

with open('d:/EIC/prompts.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print('Current count:', len(d['prompts']))
for i, e in enumerate(d['prompts']):
    print(f"  [{i+1:02d}] {e['label']}")
