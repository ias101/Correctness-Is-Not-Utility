from transformers import AutoTokenizer, AutoModelForCausalLM
print("Downloading Qwen2.5-7B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
print("Tokenizer downloaded")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", torch_dtype="auto", trust_remote_code=True)
print("Model downloaded successfully")
