"""Dual LLM backends for local_dev aligned with live SN33 validators."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator
from typing import Literal
from typing import Optional

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib

BackendName = Literal["chutes", "openrouter", "openai", "live_validator"]

_llm_override: ContextVar[Optional[LlmLib]] = ContextVar("local_dev_llm_override", default=None)


def _ensure_dotenv_loaded() -> None:
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    except Exception:
        pass


def make_backend(name: BackendName | None = None) -> LlmLib:
    _ensure_dotenv_loaded()
    name = (name or os.environ.get("LLM_TYPE_OVERRIDE") or "openrouter").strip().lower()

    if name in ("live_validator", "validator"):
        from local_dev.live_validator_llm import LiveMatchedValidatorLlm

        return LiveMatchedValidatorLlm()
    if name == "chutes":
        from conversationgenome.llm.llm_chutes import LlmChutes

        return LlmChutes()
    if name == "openrouter":
        from conversationgenome.llm.llm_openrouter import LlmOpenRouter

        return LlmOpenRouter()
    if name == "openai":
        from conversationgenome.llm.llm_openai import LlmOpenAI

        return LlmOpenAI()
    raise ValueError(f"Unsupported backend: {name}")


def validator_llm() -> LlmLib:
    backend = os.environ.get("VALIDATOR_LLM_BACKEND", "live_validator")
    return make_backend(backend)  # type: ignore[arg-type]


def miner_llm() -> LlmLib:
    # Miner always OpenRouter with MINER/OPENROUTER_MODEL
    prev = os.environ.get("OPENROUTER_MODEL")
    miner_model = os.environ.get("MINER_OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL") or "google/gemini-2.5-flash"
    os.environ["OPENROUTER_MODEL"] = miner_model
    try:
        return make_backend("openrouter")
    finally:
        if prev is None:
            os.environ["OPENROUTER_MODEL"] = miner_model
        else:
            os.environ["OPENROUTER_MODEL"] = prev


def embedding_llm() -> LlmLib:
    # Same embedding space as validator (OpenRouter openai/text-embedding-3-small)
    return validator_llm()


@contextmanager
def override_get_llm_backend(instance: LlmLib) -> Iterator[None]:
    """Force conversationgenome.llm.llm_factory.get_llm_backend() to return instance."""
    import conversationgenome.llm.llm_factory as factory

    token = _llm_override.set(instance)
    original = factory.get_llm_backend

    def _patched(llm_type_override=None):
        current = _llm_override.get()
        return current if current is not None else original(llm_type_override)

    factory.get_llm_backend = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        factory.get_llm_backend = original  # type: ignore[assignment]
        _llm_override.reset(token)


@contextmanager
def use_miner_backend_env() -> Iterator[LlmLib]:
    """Activate miner OpenRouter model for task.mine()/get_llm_backend()."""
    backend = os.environ.get("MINER_LLM_BACKEND", "openrouter")
    miner_model = os.environ.get("MINER_OPENROUTER_MODEL") or os.environ.get("OPENROUTER_MODEL") or "google/gemini-2.5-flash"
    prev_override = os.environ.get("LLM_TYPE_OVERRIDE")
    prev_model = os.environ.get("OPENROUTER_MODEL")
    prev_locked = c.get("system", "llm_overrides_locked", False)

    os.environ["LLM_TYPE_OVERRIDE"] = backend
    os.environ["OPENROUTER_MODEL"] = miner_model
    c.set("system", "llm_overrides_locked", False)
    llm = make_backend("openrouter")
    with override_get_llm_backend(llm):
        try:
            yield llm
        finally:
            if prev_override is None:
                os.environ.pop("LLM_TYPE_OVERRIDE", None)
            else:
                os.environ["LLM_TYPE_OVERRIDE"] = prev_override
            if prev_model is None:
                os.environ.pop("OPENROUTER_MODEL", None)
            else:
                os.environ["OPENROUTER_MODEL"] = prev_model
            c.set("system", "llm_overrides_locked", prev_locked)


def describe_models() -> dict:
    _ensure_dotenv_loaded()
    return {
        "production_validator_mainnet": {
            "chat_gt_default": "gpt-5.2",
            "conversation_quality": "gpt-5-mini",
            "skill_coverage_chain": "gpt-5.6-luna",
            "embeddings": "text-embedding-3-small",
            "note": "Mainnet validators lock overrides to OpenAI defaults",
        },
        "local_dev_validator": {
            "backend": os.environ.get("VALIDATOR_LLM_BACKEND", "live_validator"),
            "gt_model": os.environ.get("VALIDATOR_OPENROUTER_MODEL", "openai/gpt-5.2"),
            "quality_model": os.environ.get("VALIDATOR_QUALITY_MODEL", "openai/gpt-5-mini"),
            "skill_model": os.environ.get("VALIDATOR_SKILL_MODEL", "openai/gpt-5.6-luna"),
            "embeddings": os.environ.get("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        },
        "local_dev_miner": {
            "backend": os.environ.get("MINER_LLM_BACKEND", "openrouter"),
            "model": os.environ.get("MINER_OPENROUTER_MODEL", os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")),
        },
    }
