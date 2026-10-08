"""Task routing for the hybrid tag pipeline."""

from __future__ import annotations

from typing import Literal

Route = Literal["chat", "text", "skill", "skip"]

_CHAT = {"conversation_tagging", "survey_tagging"}
_TEXT = {"named_entities_extraction", "webpage_metadata_generation"}
_SKILL = {"skill_coverage_evaluation", "skill_generation"}


def classify_task(task_type: str) -> Route:
    if task_type in _SKILL:
        return "skill"
    if task_type in _CHAT:
        return "chat"
    if task_type in _TEXT:
        return "text"
    return "skip"
