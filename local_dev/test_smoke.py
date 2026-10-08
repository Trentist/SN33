import pytest

from custom_miner.postprocess import clean_tag_list
from custom_miner.postprocess import postprocess_result
from custom_miner.policies import get_tag_policy
from custom_miner.skill_pipeline import normalize_skill_bundle
from custom_miner.skill_pipeline.policies import get_skill_policy
from custom_miner.timing import TimeBudget
from local_dev.fixtures import ready_conversation_bundle
from local_dev.fixtures import ready_skill_coverage_bundle
from local_dev.harness import run_conversation_sim
from local_dev.harness import run_skill_coverage_sim
from tests.mocks.DummyData import DummyData


def test_clean_tag_list_dedupes_and_normalizes():
    policy = get_tag_policy("conversation_tagging")
    tags = clean_tag_list(["Wedding Planning", "wedding planning", "  Ferrari ", "x", ""], policy)
    assert tags == ["wedding planning", "ferrari"]


def test_postprocess_skill_coverage_normalizes():
    raw = {"skill": "s", "tdd_plan": "t", "section_tests": {"a": []}}
    out = postprocess_result("skill_coverage_evaluation", raw)
    assert "skill" in out and "section_tests" in out
    assert out["tdd_plan"]


def test_normalize_does_not_invent_tests():
    sections = DummyData.section_map()
    out = normalize_skill_bundle(
        {"skill": "# Slug skill\n\nUse `slugify`.\n", "tdd_plan": "plan", "section_tests": {}, "seed": "slugify text"},
        sections,
        policy=get_skill_policy(),
    )
    assert out["section_tests"] == {}
    assert "{'ok'" not in out["skill"]
    assert out["tdd_plan"]


def test_normalize_locks_skill_to_assertions():
    sections = DummyData.section_map()
    raw = {
        "skill": "this skill is discarded",
        "tdd_plan": "plan",
        "seed": "Skill for slugifying arbitrary text.",
        "section_tests": {
            "s1": [
                {"name": "a", "description": "lowercase", "assertion": "slugify('A B') == 'a-b'"},
                {"name": "a", "description": "empty", "assertion": "slugify('') == ''"},
                {"name": "vague", "description": "bad", "assertion": "the function behaves as expected"},
            ],
            "s2": [
                {"name": "c", "description": "accent", "assertion": 'slugify("Café") == "cafe"'},
            ],
        },
    }
    out = normalize_skill_bundle(raw, sections)
    assert [t["assertion"] for t in out["section_tests"]["s1"]] == [
        "slugify('A B') == 'a-b'",
        "slugify('') == ''",
    ]
    assert len({t["name"] for t in out["section_tests"]["s1"]}) == 2
    assert "Basic ASCII transformation" in out["section_tests"]["s1"][0]["description"]
    assert "slugify('A B') == 'a-b'" in out["skill"]
    assert 'slugify("Café") == "cafe"' in out["skill"]
    assert "this skill is discarded" in out["skill"]
    assert "s2" in out["section_tests"]


def test_ready_bundle_has_windows_and_metadata():
    bundle = ready_conversation_bundle()
    assert bundle.input.metadata is not None
    assert len(bundle.input.metadata.tags) >= 5
    tasks = bundle.to_mining_tasks(number_of_tasks_per_bundle=2)
    assert len(tasks) >= 1
    assert tasks[0].type == "conversation_tagging"


def test_ready_skill_coverage_bundle():
    bundle = ready_skill_coverage_bundle()
    assert bundle.type == "skill_coverage_evaluation"
    assert bundle.input.data.section_map
    assert bundle.input.metadata.section_vectors
    tasks = bundle.to_mining_tasks(number_of_tasks_per_bundle=1)
    assert len(tasks) == 1
    assert tasks[0].type == "skill_coverage_evaluation"
    assert tasks[0].input.data.seed
    assert tasks[0].input.data.section_map


@pytest.mark.asyncio
async def test_mock_sim_returns_scores():
    results = await run_conversation_sim(miner="custom", mode="mock", window_limit=1)
    assert len(results) == 1
    assert results[0].tags
    assert results[0].final_miner_score >= 0.0


@pytest.mark.asyncio
async def test_mock_skill_coverage_sim_returns_scores():
    results = await run_skill_coverage_sim(miner="custom", mode="mock", window_limit=1)
    assert len(results) == 1
    assert results[0].task_type == "skill_coverage_evaluation"
    assert results[0].tags  # section ids
    assert results[0].extra.get("n_tests", 0) >= 4
    assert results[0].final_miner_score > 0.0
    assert results[0].section_coverage is not None
    assert results[0].skill_coverage is not None


def test_time_budgets_match_timeouts():
    tag = TimeBudget.for_task("conversation_tagging", 12)
    skill = TimeBudget.for_task("skill_coverage_evaluation", 24)
    assert tag.limit_s <= 12
    assert tag.limit_s >= 8
    assert skill.limit_s <= 24
    assert skill.limit_s >= 18
    assert tag.allow_embed()
    # Exhausted budget skips embed
    tag.started -= 100
    assert not tag.allow_embed()


def test_parallel_merge_prefers_enrichment_order():
    from custom_miner.parallel_mine import _merge_tag_groups

    out = _merge_tag_groups([["a", "b"], ["c", "a"]])
    # enrichment group (last) is merged first
    assert out["tags"][0] == "c"
    assert "a" in out["tags"] and "b" in out["tags"]
