"""Production miner: one pipeline per task, several tasks at once.

Validators send a single task per request, and two or three of those requests
often arrive together. Each mine runs on a worker thread under a small
in-flight cap (``MINER_MAX_INFLIGHT``, default 3) so one skill-coverage call
does not freeze the axon while a conversation window is waiting.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import bittensor as bt

from conversationgenome.miner.MinerLib import MinerLib
from conversationgenome.task.Task import Task
from custom_miner import dispatch
from custom_miner.prompt_overrides import install_custom_prompts
from custom_miner.prompt_overrides import is_installed
from custom_miner.timing import TimeBudget


def max_inflight() -> int:
    raw = (os.environ.get("MINER_MAX_INFLIGHT", "3") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(1, min(n, 8))


class CompetitiveMinerLib(MinerLib):
    verbose = False

    def __init__(self):
        super().__init__()
        self._slots = asyncio.Semaphore(max_inflight())
        self._inflight = 0
        if not is_installed():
            install_custom_prompts()
            bt.logging.info("CompetitiveMinerLib: custom prompt overlay installed")

    async def do_mining(self, task: Task) -> dict[str, Any]:
        task_type = getattr(task, "type", "") or ""
        pipeline = dispatch.pipeline_for(task_type)
        budget = TimeBudget.for_task(task_type, getattr(task, "timeout", None))
        bt.logging.info(
            f"CompetitiveMiner: queued {task_type} pipeline={pipeline} "
            f"budget={budget.limit_s:.1f}s cap={max_inflight()}"
        )

        async with self._slots:
            self._inflight += 1
            started = time.perf_counter()
            try:
                if budget.remaining() < 0.5:
                    bt.logging.warning(
                        f"CompetitiveMiner: dropped {task_type} after waiting "
                        f"{budget.elapsed():.2f}s in the inflight queue"
                    )
                    return dispatch.empty_result(task_type)
                bt.logging.info(
                    f"CompetitiveMiner: mining {task_type} pipeline={pipeline} "
                    f"inflight={self._inflight} remaining={budget.remaining():.2f}s"
                )
                try:
                    result = await asyncio.to_thread(dispatch.mine_task, task, budget)
                except Exception as exc:
                    bt.logging.error(f"CompetitiveMiner: {task_type} pipeline failed: {exc}")
                    result = dispatch.empty_result(task_type)
            finally:
                self._inflight -= 1

        if not isinstance(result, dict):
            result = dispatch.empty_result(task_type)

        elapsed = time.perf_counter() - started
        tag_count = 0
        n_tests = 0
        if task_type == "skill_coverage_evaluation":
            n_tests = sum(len(v) for v in (result.get("section_tests") or {}).values() if isinstance(v, list))
            detail = f"sections={len(result.get('section_tests') or {})} tests={n_tests}"
        else:
            tag_count = len(result.get("tags") or [])
            detail = f"tags={tag_count}"
        over = " OVER_BUDGET" if budget.elapsed() > budget.limit_s else ""
        bt.logging.info(
            f"CompetitiveMiner: finished {task_type} pipeline={pipeline} "
            f"in {elapsed:.2f}s (budget={budget.limit_s:.1f}s {detail}){over}"
        )

        try:
            from conversationgenome.extensions.Extensions import Extensions

            Extensions().execute(
                "Metrics",
                "incStat",
                {
                    "metric_name": f"mine_{task_type}",
                    "elapsed": budget.elapsed(),
                    "tag_count": tag_count,
                    "test_count": n_tests,
                    "budget_s": budget.limit_s,
                    "over_budget": budget.elapsed() > budget.limit_s,
                    "pipeline": pipeline,
                },
            )
        except Exception:
            pass

        return result
