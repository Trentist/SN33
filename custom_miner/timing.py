"""Per-request mining time budgets aligned to SN33 task timeouts."""

from __future__ import annotations

import time
from dataclasses import dataclass


# Leave headroom for axon serialization / network RTT before validator timeout.
_DEFAULT_BUDGETS = {
    "skill_coverage_evaluation": 22.0,  # task timeout 24s
    "conversation_tagging": 10.5,  # task timeout 12s
    "webpage_metadata_generation": 10.5,
    "named_entities_extraction": 10.5,
    "survey_tagging": 10.5,
    "skill_generation": 10.5,
}
_DEFAULT_BUDGET = 10.5

# Minimum remaining seconds required to attempt embed-aware selection.
EMBED_MIN_REMAINING = 2.0


@dataclass
class TimeBudget:
    task_type: str
    started: float
    limit_s: float

    @classmethod
    def for_task(cls, task_type: str, timeout: float | None = None) -> "TimeBudget":
        if task_type == "skill_coverage_evaluation" and timeout is not None and timeout > 0:
            # The skill model often needs ~21s. 88% of 24s (21.1s) cancels that
            # response. Stay 1s under the validator timeout for the return trip.
            limit = max(4.0, float(timeout) - 1.0)
        elif timeout is not None and timeout > 0:
            # Use 88% of declared task timeout as soft budget
            limit = max(4.0, float(timeout) * 0.88)
        else:
            limit = _DEFAULT_BUDGETS.get(task_type, _DEFAULT_BUDGET)
        return cls(task_type=task_type, started=time.perf_counter(), limit_s=limit)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def remaining(self) -> float:
        return self.limit_s - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def allow_embed(self) -> bool:
        return self.remaining() >= EMBED_MIN_REMAINING
