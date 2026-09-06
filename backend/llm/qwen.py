"""Local Qwen2.5-3B-Instruct wrapper. Loaded once, reused for every request."""

import threading
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from backend.config import FINE_TUNED_ADAPTER_DIR, LLM_DEVICE, LLM_PATH, MAX_NEW_TOKENS, USE_FINE_TUNED

# generate() is not thread-safe and we hold a single copy of the weights.
_generation_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=LLM_DEVICE,
    )
    if USE_FINE_TUNED:
        if not FINE_TUNED_ADAPTER_DIR.exists():
            raise RuntimeError(
                f"USE_FINE_TUNED is set but no adapter found at {FINE_TUNED_ADAPTER_DIR} - "
                "run backend/finetune/train.py first, or unset USE_FINE_TUNED."
            )
        from peft import PeftModel

        # Trained on a 4-bit base (QLoRA) but the small LoRA matrices apply
        # cleanly on top of this fp16 copy too - same approach used for
        # serving, just full precision instead of the memory-saving one
        # training needed.
        model = PeftModel.from_pretrained(model, str(FINE_TUNED_ADAPTER_DIR))
        model = model.merge_and_unload()  # fold the adapter in, one plain model to serve
    model.eval()
    return tokenizer, model


def warmup() -> None:
    _load()


def _prepare(messages: list[dict]):
    tokenizer, model = _load()
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    return tokenizer, model, inputs


def generate(messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    tokenizer, model, inputs = _prepare(messages)
    with _generation_lock, torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # factual answers: greedy, no creative drift
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def stream(messages: list[dict], max_new_tokens: int = MAX_NEW_TOKENS):
    """Yield the answer token-by-token so the UI can render it as it arrives."""
    tokenizer, model, inputs = _prepare(messages)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        streamer=streamer,
    )

    def _run():
        with _generation_lock, torch.inference_mode():
            model.generate(**kwargs)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for piece in streamer:
        yield piece
    thread.join()
