from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    dtype="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./my_lora"
)

merged_model = model.merge_and_unload()

merged_model.save_pretrained("./merged_model")