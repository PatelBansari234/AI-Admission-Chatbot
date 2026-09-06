"""Side-by-side check: base Qwen vs base+LoRA adapter, on trained AND
held-out questions - run this after any retrain before flipping
USE_FINE_TUNED on, to catch overfitting/fact drift before it reaches students.

    python -m backend.finetune.compare

Loads both variants in the same process rather than through
backend.llm.qwen's cached singleton, since that only holds one model.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.config import FINE_TUNED_ADAPTER_DIR, LLM_PATH
from backend.rag.prompt import build_messages
from backend.rag.retriever import retrieve

QUESTIONS = [
    # in the training set - style should match closely
    "How many seats are there for BA Economics?",
    # NOT in the training set - checks generalization vs memorization
    "What is the eligibility for BCA?",
    "What programmes does the Computer Science department offer?",
    "What is the seat count for the MSc Big Data Analytics programme?",
    # off-topic - must still refuse correctly
    "What is the capital of France?",
]


def _generate(tokenizer, model, messages, max_new_tokens=250) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.float16, device_map="auto")
    base_model.eval()

    print("Generating with the base model...")
    base_answers = {}
    for q in QUESTIONS:
        messages = build_messages(q, retrieve(q), [])
        base_answers[q] = _generate(tokenizer, base_model, messages)

    print("Attaching the LoRA adapter...")
    ft_model = PeftModel.from_pretrained(base_model, str(FINE_TUNED_ADAPTER_DIR)).merge_and_unload()
    ft_model.eval()

    print("Generating with the fine-tuned model...\n")
    for q in QUESTIONS:
        messages = build_messages(q, retrieve(q), [])
        ft_answer = _generate(tokenizer, ft_model, messages)
        print("=" * 70)
        print("Q:", q)
        print("-- base --\n" + base_answers[q])
        print("-- fine-tuned --\n" + ft_answer)


if __name__ == "__main__":
    main()
