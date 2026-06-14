"""Download Mistral-7B-Instruct-v0.3 with 4-bit quantization."""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
print(f"Downloading {model_name}...")

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Download complete! {n_params:.1f}B params loaded.")
print("Mistral-7B ready.")
