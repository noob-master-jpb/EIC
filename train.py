import torch
from transformers import AutoTokenizer, AutoProcessor
from data import *


MODEL_ID = "./models/gemma-4-E2B-it"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

train_dataset = process_dataset("./Datasets/oss-ins-75k.parquet","problem","solution",processor=processor,num_proc=1)

