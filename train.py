import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from data import *


MODEL_ID = "./models/gemma-4-E2B-it"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

train_dataset = process_dataset("./Datasets/oss-ins-75k.parquet","problem","solution",processor=processor,num_proc=1)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


training_args = SFTConfig(
    output_dir="./gemma-4-finetuned",
    max_seq_length=1024,
    per_device_train_batch_size=4,      
    gradient_accumulation_steps=4,       
    learning_rate=2e-4,
    bf16=True,
    packing=True,
    assistant_only_loss=True,            
    processing_class=processor,          
    logging_steps=10,
    save_steps=100                      
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
)

print("Starting training...")
trainer.train()


trainer.save_model("./gemma-4-finetuned-final")