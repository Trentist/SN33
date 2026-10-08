from custom_miner.skill_pipeline.normalize import _is_concrete
from custom_miner.skill_pipeline.normalize import stretch_description
from custom_miner.skill_pipeline.normalize import normalize_skill_bundle
from custom_miner.skill_pipeline.policies import get_skill_policy
from custom_miner.skill_pipeline.proxy_score import hidden_subsets
from custom_miner.skill_pipeline.proxy_score import score_subset
from custom_miner.skill_pipeline.section_parse import parse_section
from custom_miner.skill_pipeline.select import mmr_indices
from custom_miner.skill_pipeline.select import embed_text
from custom_miner.skill_pipeline.select import select_best_pair
from tests.mocks.DummyData import DummyData


def test_mmr_drops_near_duplicate_and_keeps_diverse():
    section = [1.0, 0.0]
    vectors = [
        [1.0, 0.0],
        [0.99, 0.05],
        [0.6, 0.8],
    ]
    kept = mmr_indices(section, vectors, limit=6, dup=0.93)
    assert kept[0] == 0
    assert 1 not in kept
    assert 2 in kept


def test_embed_text_matches_validator_shape():
    text = embed_text({"description": "Lowercase input", "assertion": "slugify('A') == 'a'"})
    assert text == "Lowercase input Assertion: slugify('A') == 'a'"
    assert "\n" not in text


def test_two_tests_share_the_full_section_sentence_and_differ_by_case():
    sections = [
        {
            "section_id": "s1",
            "title": "Events of Default and Enforcement Remedies",
            "description": "Define defaults, notice and cure periods, acceleration, foreclosure procedures, deficiency rights, and application of proceeds.",
        }
    ]
    out = normalize_skill_bundle(
        {
            "seed": "deed",
            "tdd_plan": "plan",
            "section_tests": {
                "s1": [
                    {"name": "cure", "description": "Cure before acceleration.", "assertion": "accelerate(notice_sent=False) == False"},
                    {"name": "proceeds", "description": "Apply sale proceeds.", "assertion": "apply_proceeds(proceeds=100, costs=20, debt=70) == 10"},
                ],
            },
        },
        sections,
    )
    first, second = (test["description"] for test in out["section_tests"]["s1"])
    assert first.startswith("Events of Default and Enforcement Remedies:")
    assert "notice and cure periods" in first
    assert "application of proceeds" in first
    assert "notice and cure periods" in second
    assert "application of proceeds" in second
    assert "Cure before acceleration" in first
    assert "Apply sale proceeds" in second
    assert first != second
    assert "Define defaults" in out["skill"]
    assert "Assertion: accelerate(notice_sent=False) == False" in out["skill"]


def test_stretch_keeps_the_unique_tail_inside_the_field_cap():
    stretched = stretch_description(
        "x" * 1000,
        "apply_proceeds(100) == 10",
        "notice and cure, application of proceeds",
        half=1,
        limit=1000,
        title="Events of Default",
    )
    assert len(stretched) <= 1000
    assert "Only the Events of Default check 2" in stretched
    assert "apply_proceeds(100) == 10" in stretched


def test_prepare_keeps_two_and_drops_vague():
    sections = DummyData.section_map()
    tests = [
        {"name": f"t{i}", "description": f"case {i}", "assertion": f"slugify('{i}') == '{i}'"}
        for i in range(8)
    ]
    tests.append({"name": "vague", "description": "nope", "assertion": "handles edge cases correctly"})
    out = normalize_skill_bundle(
        {"seed": "slugify", "tdd_plan": "plan", "section_tests": {"s1": tests}},
        sections,
    )
    policy = get_skill_policy()
    assert policy.tests_per_section == 2
    assert len(out["section_tests"]["s1"]) == 2
    assert all("edge cases" not in t["assertion"] for t in out["section_tests"]["s1"])
    assert len(out["skill"]) <= policy.max_skill_chars


def test_traceability_keeps_assertions_when_skill_is_truncated():
    policy = get_skill_policy()
    out = normalize_skill_bundle(
        {
            "skill": "preamble " + ("word " * 5000),
            "tdd_plan": "plan",
            "section_tests": {
                "s1": [
                    {"name": "a", "description": "lowercase", "assertion": "slugify('A B') == 'a-b'"},
                ],
            },
        },
        DummyData.section_map(),
        policy=policy,
    )
    assert len(out["skill"]) <= policy.max_skill_chars
    assert "slugify('A B') == 'a-b'" in out["skill"]


def test_concrete_assertion_keeps_include_and_omit_checks():
    assert _is_concrete("draft_ad(monthly_rent=1850) includes the text '$1,850/month'")
    assert _is_concrete("draft_ad(max_occupancy=2, code_citation=None) omits any occupancy limit text")
    assert not _is_concrete("the advertisement includes the expected value")


def test_concrete_assertion_accepts_equality_without_quotes():
    assert _is_concrete("accelerate(notice_sent=False) == False")
    assert _is_concrete("given a monetary default, acceleration == false until the cure period expires")
    assert not _is_concrete("conflicts are handled correctly")


def test_concrete_assertion_accepts_command_outcome_and_rejects_vague():
    assert _is_concrete(
        "after resolving all conflicted files and staging them with git add, "
        "git rebase --continue resumes the paused interactive rebase"
    )
    assert not _is_concrete("conflicts are handled correctly")
    assert not _is_concrete("properly resolves conflicts")


def test_select_best_pair_skips_near_duplicate():
    section = [1.0, 0.0]
    vectors = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.7, 0.7],
    ]
    kept = select_best_pair(section, vectors, skill=section, hard_dup=0.94)
    assert kept == [0, 2]


def test_hidden_subset_counts():
    assert len(hidden_subsets(["a", "b", "c", "d"])) == 4
    assert len(hidden_subsets(["a", "b", "c", "d", "e"])) == 10
    assert len(hidden_subsets(["a", "b", "c", "d", "e", "f"])) == 20
    assert len(hidden_subsets(["a", "b", "c", "d", "e", "f", "g"])) == 35
    assert hidden_subsets(["a", "b"]) == [("a", "b")]


def test_duplicate_pair_halves_subset_score():
    sections = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
    duplicated = {
        "a": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "b": [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
        "c": [[0.0, 0.0, 1.0], [0.0, 0.1, 0.9]],
    }
    skill = [1.0, 1.0, 1.0]
    raw = score_subset(("a", "b", "c"), sections, duplicated, skill, apply_penalties=False)
    penalized = score_subset(("a", "b", "c"), sections, duplicated, skill)
    assert penalized == raw * 0.5


def test_section_parser_keeps_supplied_title():
    brief = parse_section({
        "section_id": "rebase_conflicts",
        "title": "Conflict Resolution",
        "description": "Explain how interactive rebase handles conflicts and how to continue or abort. Use `git rebase --abort`.",
    })
    assert brief.section_id == "rebase_conflicts"
    assert brief.title == "Conflict Resolution"
    assert "abort" in [word.lower() for word in brief.boundaries]
    assert any("abort" in command for command in brief.commands)
