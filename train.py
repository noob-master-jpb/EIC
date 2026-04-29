import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from data import process_dataset, combine_dataset, split_dataset

REPO="google"
MODEL = "gemma-4-31B"
MODEL_ID = f"{REPO}/{MODEL}"
OUTPUT_DIR = f"./{MODEL}-finetuned"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit = False, 
    use_gradient_checkpointing = "unsloth",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 1, 
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ], 
    lora_alpha = 4,
    lora_dropout = 0,
    bias = "none",    
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4"
)

dataset = combine_dataset(paths=["./Datasets/cass_part1.parquet", "./Datasets/cass_part2.parquet"],
                          output_mapping=["query","output"],)

train,val = split_dataset(dataset, split=0.1, seed=3407)
train_dataset = process_dataset(
    df = train,
    user_prompt = "query", 
    agent_response = "output",
    processor = tokenizer, 
    num_proc = 1 
)

eval_dataset = process_dataset(
    df = val,
    user_prompt = "query", 
    agent_response = "output",
    processor = tokenizer, 
    num_proc = 1
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text", 
    max_seq_length = 2048,
    packing = True, 
    args = SFTConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 16, 
        gradient_accumulation_steps = 6,
        learning_rate = 2e-4,
        bf16 = True,
        num_train_epochs = .2, 
        logging_steps = 1,
        save_steps = 500,
        optim = "adamw_8bit", 
        weight_decay = 0.01,
        seed = 3407,
        dataloader_num_workers = 16, 
        dataloader_pin_memory = True,
    ),
)

try:
    print("Starting training. Press Ctrl+C to interrupt and save the model early.")
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving the model at the current state...")

print("Saving model and tokenizer to ./gemma-4-finetuned-final")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
print("Save complete!")