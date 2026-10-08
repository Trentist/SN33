"""Per-task mining policies tuned to SN33 scoring.

GroundTruthTagSimilarityScoringMechanism cares about:
- at least 3 tags (else zero score)
- at most 20 tags scored
- top-3 unique tag cosine means weigh 55%
- penalties for too few tags / no GT overlap / junk tags
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagPolicy:
    min_tags: int = 5
    target_tags: int = 10
    max_tags: int = 18
    max_tag_tokens: int = 5
    drop_singleton_chars: bool = True
    expand_variants: bool = True


TASK_POLICIES: dict[str, TagPolicy] = {
    "conversation_tagging": TagPolicy(
        min_tags=10, target_tags=12, max_tags=14, max_tag_tokens=5, expand_variants=True
    ),
    "webpage_metadata_generation": TagPolicy(
        min_tags=8, target_tags=12, max_tags=16, max_tag_tokens=5, expand_variants=True
    ),
    "survey_tagging": TagPolicy(
        min_tags=3, target_tags=6, max_tags=10, max_tag_tokens=4, expand_variants=False
    ),
    "named_entities_extraction": TagPolicy(
        min_tags=10, target_tags=14, max_tags=18, max_tag_tokens=6, expand_variants=True
    ),
    "skill_generation": TagPolicy(
        min_tags=5, target_tags=10, max_tags=15, max_tag_tokens=5, expand_variants=False
    ),
}

DEFAULT_TAG_POLICY = TagPolicy()


def get_tag_policy(task_type: str) -> TagPolicy:
    return TASK_POLICIES.get(task_type, DEFAULT_TAG_POLICY)
