import pandas as pd
import json

# Since it's a JSONL file, we use lines=True
df = pd.read_json('dataset/problems.jsonl', lines=True)

# Print the head of the DataFrame
# print(df.columns)
from pprint import pprint
pprint(df['type', 'schema_version', 'task_id', 'date', 'prompt', 'metadata',
       'group', 'context_files', 'test_files', 'source_references',
       'build_command', 'test_command', 'benchmark_command', 'timing_mode',
       'min_cuda_toolkit', 'compute_capability', 'requires_datacenter_gpu',
       'timeout_seconds', 'baseline_solution'][0], width = 800, compact=True)