"""Modular Translation Service using IndicTrans2 for 22 Scheduled Indian Languages.

Handles:
1. Feature flag control (INDIC_TRANSLATION_ENABLED).
2. Lazy-loaded IndicTrans2 models for Indic->En and En->Indic.
3. Strict preservation of numbers, fees, percentages, emails, phone numbers,
   course codes (BCA, BA), and official SXCA links via placeholder masking.
4. Zero-overhead passthrough for English queries.
5. Graceful fallback on model failure or OOM without hallucinated facts.
"""

import logging
import os
import threading
import time
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.translation.config import (
    SUPPORTED_LANGUAGES,
    TRANSLATION_BEAM_SIZE,
    TRANSLATION_DEVICE,
    TRANSLATION_ENABLED,
    TRANSLATION_MAX_INPUT_LEN,
    TRANSLATION_MAX_OUTPUT_LEN,
)
from backend.translation.detector import detector
from backend.translation.indic_processor import IndicProcessor

logger = logging.getLogger(__name__)

# Model names on HuggingFace Hub (downloaded locally in HF cache)
INDIC_EN_MODEL_NAME = "prajdabre/rotary-indictrans2-indic-en-dist-200M"
EN_INDIC_MODEL_NAME = "prajdabre/rotary-indictrans2-en-indic-dist-200M"


class TranslationService:
    """Singleton service for AI4Bharat IndicTrans2 translation."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.enabled = TRANSLATION_ENABLED
        self.processor = IndicProcessor(inference=True)

        self._device = self._resolve_device()
        self._ie_tokenizer = None
        self._ie_model = None
        self._ei_tokenizer = None
        self._ei_model = None
        self._load_lock = threading.Lock()

        self._initialized = True
        logger.info(
            "TranslationService initialized. Enabled=%s, Device=%s",
            self.enabled,
            self._device,
        )

    def _resolve_device(self) -> str:
        pref = TRANSLATION_DEVICE.lower()
        if pref in ("cuda", "gpu") and torch.cuda.is_available():
            return "cuda"
        if pref == "cpu":
            return "cpu"

        # Auto: check GPU memory
        if torch.cuda.is_available():
            try:
                free_bytes = torch.cuda.mem_get_info()[0]
                # If at least 3.0 GB VRAM free, use CUDA, else CPU
                if free_bytes > 3.0 * (1024**3):
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def _ensure_indic_en_loaded(self):
        """Lazy loads the Indic -> English translation model."""
        if self._ie_model is not None:
            return

        with self._load_lock:
            if self._ie_model is not None:
                return

            logger.info("Loading Indic -> English model (%s) onto %s...", INDIC_EN_MODEL_NAME, self._device)
            t0 = time.time()
            self._ie_tokenizer = AutoTokenizer.from_pretrained(
                INDIC_EN_MODEL_NAME, trust_remote_code=True
            )
            # Use float16 if cuda, float32 if cpu
            dt = torch.float16 if self._device == "cuda" else torch.float32
            self._ie_model = AutoModelForSeq2SeqLM.from_pretrained(
                INDIC_EN_MODEL_NAME, dtype=dt, trust_remote_code=True
            ).to(self._device)
            self._ie_model.eval()
            logger.info("Loaded Indic -> English model in %.2fs", time.time() - t0)

    def _ensure_en_indic_loaded(self):
        """Lazy loads the English -> Indic translation model."""
        if self._ei_model is not None:
            return

        with self._load_lock:
            if self._ei_model is not None:
                return

            logger.info("Loading English -> Indic model (%s) onto %s...", EN_INDIC_MODEL_NAME, self._device)
            t0 = time.time()
            self._ei_tokenizer = AutoTokenizer.from_pretrained(
                EN_INDIC_MODEL_NAME, trust_remote_code=True
            )
            dt = torch.float16 if self._device == "cuda" else torch.float32
            self._ei_model = AutoModelForSeq2SeqLM.from_pretrained(
                EN_INDIC_MODEL_NAME, dtype=dt, trust_remote_code=True
            ).to(self._device)
            self._ei_model.eval()
            logger.info("Loaded English -> Indic model in %.2fs", time.time() - t0)

    def translate_to_english(
        self, text: str, src_lang: str
    ) -> Tuple[str, bool]:
        """Translates student's native Indic query to English.

        Returns: (translated_english_text, is_translated)
        If src_lang is English or translation is disabled, returns original text.
        """
        if not self.enabled or not text or src_lang == "eng_Latn":
            return text, False

        if src_lang not in SUPPORTED_LANGUAGES:
            logger.warning("Unsupported source language: %s", src_lang)
            return text, False

        try:
            self._ensure_indic_en_loaded()

            # Preprocess and placeholder-mask numbers/emails/dates/urls
            batch = self.processor.preprocess_batch(
                [text], src_lang=src_lang, tgt_lang="eng_Latn"
            )

            inputs = self._ie_tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=TRANSLATION_MAX_INPUT_LEN,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._ie_model.generate(
                    **inputs,
                    num_beams=max(1, TRANSLATION_BEAM_SIZE),
                    max_new_tokens=TRANSLATION_MAX_OUTPUT_LEN,
                    use_cache=False,
                )

            decoded = self._ie_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            postprocessed = self.processor.postprocess_batch(decoded, lang="eng_Latn")

            translated = postprocessed[0].strip() if postprocessed else text
            if not translated:
                return text, False

            logger.info("Translated [%s -> en]: '%s' -> '%s'", src_lang, text, translated)
            return translated, True

        except Exception as exc:
            logger.error("Failed to translate to English: %s", exc, exc_info=True)
            return text, False

    def translate_from_english(
        self, text: str, tgt_lang: str
    ) -> Tuple[str, bool]:
        """Translates English grounded admission response back to the student's native language.

        Returns: (translated_native_text, is_translated)
        If tgt_lang is English or translation is disabled, returns original text.
        """
        if not self.enabled or not text or tgt_lang == "eng_Latn":
            return text, False

        if tgt_lang not in SUPPORTED_LANGUAGES:
            logger.warning("Unsupported target language: %s", tgt_lang)
            return text, False

        tgt_info = SUPPORTED_LANGUAGES.get(tgt_lang)
        if tgt_info and detector.detect_script(text) == tgt_info.script:
            # Already in target script (e.g. native template or Gujarati fast-path)
            return text, False

        try:
            self._ensure_en_indic_loaded()

            # Preprocess and protect numbers, course names, fees, emails, urls
            batch = self.processor.preprocess_batch(
                [text], src_lang="eng_Latn", tgt_lang=tgt_lang
            )

            inputs = self._ei_tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=TRANSLATION_MAX_INPUT_LEN,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._ei_model.generate(
                    **inputs,
                    num_beams=max(1, TRANSLATION_BEAM_SIZE),
                    repetition_penalty=1.3,
                    length_penalty=1.0,
                    early_stopping=True,
                    max_new_tokens=TRANSLATION_MAX_OUTPUT_LEN,
                    use_cache=False,
                )

            decoded = self._ei_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            postprocessed = self.processor.postprocess_batch(decoded, lang=tgt_lang)

            translated = postprocessed[0].strip() if postprocessed else text
            if not translated:
                return text, False

            logger.info("Translated [en -> %s]: '%s' -> '%s'", tgt_lang, text[:50], translated[:50])
            return translated, True

        except Exception as exc:
            logger.error("Failed to translate from English: %s", exc, exc_info=True)
            return text, False


# Global singleton instance
translation_service = TranslationService()
