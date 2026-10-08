"""OpenRouter validator LLM that mirrors live LlmOpenAI model routing.

Live mainnet (LlmOpenAI):
  - default chat / GT tagging: gpt-5.2
  - quality gate: gpt-5-mini
  - skill coverage chain + judge: gpt-5.6-luna
  - embeddings: text-embedding-3-small

OpenRouter IDs use the openai/ prefix.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from conversationgenome.ConfigLib import c
from conversationgenome.llm.LlmLib import LlmLib
from conversationgenome.llm.llm_openrouter import LlmOpenRouter


def _env(key: str, default: str) -> str:
    return (os.environ.get(key) or c.get("env", key) or default).strip()


class LiveMatchedValidatorLlm(LlmOpenRouter):
    """Validator-side OpenRouter client with live-matching model routing."""

    def __init__(self):
        # Temporarily pin OPENROUTER_MODEL so parent picks validator default
        prev = os.environ.get("OPENROUTER_MODEL")
        os.environ["OPENROUTER_MODEL"] = _env("VALIDATOR_OPENROUTER_MODEL", "openai/gpt-5.2")
        try:
            super().__init__()
        finally:
            if prev is None:
                # leave miner model for later; parent already read validator model
                pass
            else:
                os.environ["OPENROUTER_MODEL"] = prev

        self.model = _env("VALIDATOR_OPENROUTER_MODEL", "openai/gpt-5.2")
        self.embedding_model = _env("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.quality_model = _env("VALIDATOR_QUALITY_MODEL", "openai/gpt-5-mini")
        self.skill_model = _env("VALIDATOR_SKILL_MODEL", "openai/gpt-5.6-luna")
        self.max_tokens = int(os.environ.get("VALIDATOR_MAX_TOKENS", os.environ.get("OPENROUTER_MAX_TOKENS", "2048")))

    @contextmanager
    def _model(self, model_name: str):
        prev = self.model
        self.model = model_name
        try:
            yield
        finally:
            self.model = prev

    def validate_conversation_quality(self, conversation):
        with self._model(self.quality_model):
            return LlmLib.validate_conversation_quality(self, conversation)

    def skill_request_to_section_map(self, seed):
        with self._model(self.skill_model):
            return LlmLib.skill_request_to_section_map(self, seed)

    def skill_request_to_skill(self, seed, section_map):
        with self._model(self.skill_model):
            return LlmLib.skill_request_to_skill(self, seed, section_map)

    def skill_request_to_skill_bundle(self, seed, section_map):
        with self._model(self.skill_model):
            return LlmLib.skill_request_to_skill_bundle(self, seed, section_map)

    def skill_to_tdd_plan(self, skill_markdown, section_map):
        with self._model(self.skill_model):
            return LlmLib.skill_to_tdd_plan(self, skill_markdown, section_map)

    def skill_to_section_tests(self, skill_markdown, tdd_plan, section_map):
        with self._model(self.skill_model):
            return LlmLib.skill_to_section_tests(self, skill_markdown, tdd_plan, section_map)

    def judge_section_tests(self, skill_markdown, section_map, section_tests):
        with self._model(self.skill_model):
            return LlmLib.judge_section_tests(self, skill_markdown, section_map, section_tests)
