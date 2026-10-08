"""Routing and concurrent mining for the production miner."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import custom_miner.dispatch as dispatch
from custom_miner.miner_lib import CompetitiveMinerLib
from custom_miner.timing import TimeBudget


def _task(task_type: str, timeout: float = 12.0):
    return SimpleNamespace(type=task_type, timeout=timeout)


def test_each_task_type_has_a_pipeline():
    assert dispatch.pipeline_for("conversation_tagging") == "tag"
    assert dispatch.pipeline_for("survey_tagging") == "tag"
    assert dispatch.pipeline_for("webpage_metadata_generation") == "webpage"
    assert dispatch.pipeline_for("named_entities_extraction") == "ner"
    assert dispatch.pipeline_for("skill_coverage_evaluation") == "skill_coverage"
    assert dispatch.pipeline_for("skill_generation") == "skill_generation"


def test_mine_task_calls_the_pipeline_for_that_type(monkeypatch):
    seen: list[str] = []

    def _skill(task, budget):
        seen.append("skill_coverage")
        return {"skill": "s", "tdd_plan": "t", "section_tests": {"a": [{}]}}

    def _web(task, budget):
        seen.append("webpage")
        return {"tags": ["page", "meta", "title"], "vectors": None}

    def _ner(task, budget):
        seen.append("ner")
        return {"tags": ["Ada", "Lovelace", "London"], "vectors": None}

    def _tags(task, llm_tags, budget=None):
        seen.append(f"tag:{task.type}")
        return ["chat", "topic"]

    monkeypatch.setattr(dispatch, "mine_skill_coverage", _skill)
    monkeypatch.setattr(dispatch, "mine_webpage", _web)
    monkeypatch.setattr(dispatch, "mine_named_entities", _ner)
    monkeypatch.setattr(dispatch, "select_tags_for_task", _tags)
    monkeypatch.setattr(dispatch, "parallel_mine_task", lambda task: {"tags": ["raw"], "vectors": None})
    monkeypatch.setattr(
        dispatch,
        "_mine_skill_generation",
        lambda task, budget: seen.append("skill_generation") or {"tags": ["skill"], "vectors": None},
    )

    budget = TimeBudget.for_task("conversation_tagging", 12)
    assert dispatch.mine_task(_task("conversation_tagging"), budget)["tags"] == ["chat", "topic"]
    assert dispatch.mine_task(_task("webpage_metadata_generation"), budget)["tags"][0] == "page"
    assert dispatch.mine_task(_task("named_entities_extraction"), budget)["tags"][0] == "Ada"
    skill = dispatch.mine_task(_task("skill_coverage_evaluation", 24), budget)
    assert skill["section_tests"]["a"]
    assert dispatch.mine_task(_task("skill_generation"), budget)["tags"] == ["skill"]
    assert seen == ["tag:conversation_tagging", "webpage", "ner", "skill_coverage", "skill_generation"]


@pytest.mark.asyncio
async def test_three_tasks_mine_at_the_same_time(monkeypatch):
    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def slow(task, budget):
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.35)
        with lock:
            state["current"] -= 1
        return {"tags": [task.type], "vectors": None}

    monkeypatch.setattr(dispatch, "mine_task", slow)
    miner = CompetitiveMinerLib()
    tasks = [
        _task("conversation_tagging"),
        _task("webpage_metadata_generation"),
        _task("named_entities_extraction"),
    ]
    started = time.perf_counter()
    results = await asyncio.gather(*(miner.do_mining(task) for task in tasks))
    elapsed = time.perf_counter() - started

    assert state["max"] == 3
    assert elapsed < 0.9
    assert [r["tags"][0] for r in results] == [
        "conversation_tagging",
        "webpage_metadata_generation",
        "named_entities_extraction",
    ]
