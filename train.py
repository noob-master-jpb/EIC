import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from data import process_dataset

MODEL_ID = "google/gemma-4-E2B-it"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 4096,
    load_in_4bit = True, # Unsloth 4-bit is still faster than full bf16
    use_gradient_checkpointing = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, 
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ], 
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",    
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4"
)

train_dataset = process_dataset(
    path = "./Datasets/oss-ins-75k.parquet",
    user_prompt = "problem",
    agent_response = "solution",
    processor = tokenizer, 
    num_proc = 1 
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text", 
    max_seq_length = 4096,
    dataset_num_proc = 4, # Unsloth/TRL tokenization processes
    packing = True, 
    args = SFTConfig(
        output_dir = "./gemma-4-finetuned",
        per_device_train_batch_size = 64, # Even larger for 2B on MI300X
        gradient_accumulation_steps = 1,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 1,
        save_steps = 500,
        optim = "adamw_8bit", # 8-bit is usually faster due to less memory IO
        weight_decay = 0.01,
        seed = 3407,
        dataloader_num_workers = 16, # Faster data loading
        dataloader_pin_memory = True,
    ),
)


trainer.train()

model.save_pretrained("./gemma-4-finetuned-final")
tokenizer.save_pretrained("./gemma-4-finetuned-final")