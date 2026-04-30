import torch
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer
from data import process_dataset, combine_dataset, split_dataset
# import multiprocessing
REPO="unsloth"
MODEL = "gemma-4-E2B-it"
MODEL_ID = f"{REPO}/{MODEL}"
OUTPUT_DIR = f"./{MODEL}-finetuned"

# torch._inductor.config.dce = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.triton.cudagraphs = False
# torch._inductor.config.compile_threads = multiprocessing.cpu_count()
# # Skips the exhaustive search for the fastest kernels
# torch._inductor.config.max_autotune = False
# torch._inductor.config.max_autotune_gemm = False

# # Skips autotuning for simple element-wise operations (like Add/Mul)
# torch._inductor.config.triton.autotune_pointwise = False
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = 2048,
    load_in_4bit = False, 
    use_gradient_checkpointing = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16 , 
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ], 
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",    
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4"
)

dataset = combine_dataset(
    paths=["./Datasets/cass_part1.parquet", "./Datasets/cass_part2.parquet"],
    column_mapping={
        "./Datasets/cass_part1.parquet": ["problem", "answer"],
        "./Datasets/cass_part2.parquet": ["problem", "answer"],
    },
    output_mapping=["query", "output"],
)

if os.path.exists("./processed_train"):
    train_dataset = load_from_disk("./processed_train")
    eval_dataset = load_from_disk("./processed_eval")
else:
    train, val = split_dataset(dataset, split=0.99, random_seed=3407)
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

    train_dataset.save_to_disk("./processed_train")
    eval_dataset.save_to_disk("./processed_eval")

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
        dataset_num_proc = 4,
        eval_strategy = "steps",
        eval_steps = 5,
        per_device_train_batch_size = 45, 
        gradient_accumulation_steps = 1,
        learning_rate = 1e-4,
        max_grad_norm = 1.0,
        warmup_steps = 32,
        bf16 = True,
        num_train_epochs = .5, 
        logging_steps = 1,
        save_steps = 500,
        optim = "adamw_8bit", 
        weight_decay = 0.01,
        seed = 3407,
        dataloader_num_workers = 16, 
        dataloader_pin_memory = True,
        # torch_compile = True,
        # torch_compile_backend = "inductor",
        # torch_compile_mode = "default",
    ),
)

try:
    print("Starting training. Press Ctrl+C to interrupt and save the model early.")
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving the model at the current state...")

print(f"Saving model and tokenizer to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Save complete!")