"""Skill-coverage mining policy tuned to SkillCoverageScoringMechanism.

The validator scores a hidden trio of sections:
- section coverage 60% (mean of the best 2 judged-correct tests)
- skill coverage 40% (skill vs the mean of those tests)
- near-duplicate penalty at cosine 0.95
- fewer than 5 tests in the scored subset multiplies by 0.5
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillPolicy:
    tests_per_section: int = 2
    max_tests_per_section: int = 6  # intake before the local pair selector
    min_total_tests: int = 5
    max_skill_chars: int = 6000
    target_skill_chars: int = 6000
    max_test_name_chars: int = 200
    max_test_field_chars: int = 1000
    # Validator penalizes at 0.95. Reject a pair locally a notch under that.
    hard_duplicate_cosine: float = 0.94
    warn_duplicate_cosine: float = 0.91
    near_duplicate_cosine: float = 0.94
    embed_min_remaining_s: float = 1.2


DEFAULT_SKILL_POLICY = SkillPolicy()


def get_skill_policy(task_type: str = "skill_coverage_evaluation") -> SkillPolicy:
    if task_type == "skill_coverage_evaluation":
        return DEFAULT_SKILL_POLICY
    return DEFAULT_SKILL_POLICY
