"""QLoRA fine-tune of Qwen2.5-3B-Instruct on prepare_dataset.py's output.

    python -m backend.finetune.prepare_dataset   # build finetune_dataset.jsonl first
    python -m backend.finetune.train

Trains a LoRA adapter (not the base weights), so it stays a few MB and loads
on top of the base model on demand - see USE_FINE_TUNED in backend/config.py
and backend/llm/qwen.py. This is a demonstration-scale run: a few dozen
logged Q&A pairs, a handful of epochs, meant to prove the pipeline works
end-to-end. Real deployment should accumulate much more real student traffic
before a fine-tune is worth serving in place of the base model - see the
README's "Why RAG and not fine-tuning" section for why facts stay with RAG
regardless.

Only the base model needs ~6GB VRAM at fp16 for inference; this instead
4-bit-quantizes the base weights (QLoRA) and trains small LoRA adapter
matrices on top, so it fits in the same 8GB card. The inference server must
not be running at the same time - there isn't VRAM for both.
"""

import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from backend.config import FINE_TUNED_ADAPTER_DIR, LLM_PATH, PROCESSED_DIR

DATASET_PATH = PROCESSED_DIR / "finetune_dataset.jsonl"
MAX_LENGTH = 1536


def load_examples() -> list[dict]:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"{DATASET_PATH} not found - run `python -m backend.finetune.prepare_dataset` first"
        )
    examples = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(examples) < 5:
        raise SystemExit(
            f"Only {len(examples)} training examples - log more real Q&A first (see README)."
        )
    return examples


def tokenize_example(tokenizer, example: dict) -> dict:
    user_content = f"CONTEXT:\n{example['context']}\n\nSTUDENT QUESTION: {example['question']}"
    prompt_messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + example["answer"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH
    )
    input_ids = full["input_ids"]
    labels = list(input_ids)
    # Loss only on the answer tokens - the model shouldn't be trained to
    # predict its own system prompt or the retrieved context back.
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def main() -> None:
    examples = load_examples()
    print(f"Loaded {len(examples)} training examples")

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(examples).map(
        lambda ex: tokenize_example(tokenizer, ex),
        remove_columns=["system", "context", "question", "answer"],
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=str(FINE_TUNED_ADAPTER_DIR / "_checkpoints"),
        num_train_epochs=6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    FINE_TUNED_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINE_TUNED_ADAPTER_DIR))
    tokenizer.save_pretrained(str(FINE_TUNED_ADAPTER_DIR))
    print(f"Saved LoRA adapter to {FINE_TUNED_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
