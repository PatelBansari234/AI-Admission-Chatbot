import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/Qwen2.5-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading Qwen2.5-3B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful college admission assistant."
    },
    {
        "role": "user",
        "content": "Hello, what documents are normally required for college admission?"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

print("Generating answer...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\nQwen Answer:")
print(response)