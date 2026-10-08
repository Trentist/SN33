"""Named-entity pipeline: stock prompts, aliases, and centroid portfolio."""

import numpy as np

from custom_miner.ner_pipeline.aliases import alias_forms
from custom_miner.ner_pipeline.aliases import extract_spans
from custom_miner.ner_pipeline.aliases import grounded_aliases
from custom_miner.ner_pipeline.gate import clean_tag
from custom_miner.ner_pipeline.parse import parse_ner_task
from custom_miner.ner_pipeline.pipeline import mine_named_entities
from custom_miner.ner_pipeline.pipeline import ner_chat_model
from custom_miner.ner_pipeline.pipeline import use_ner_chat_model
from custom_miner.ner_pipeline.portfolio import select_portfolio
from custom_miner.ner_pipeline.stock import parse_tag_response
from custom_miner.ner_pipeline.stock import transcript_prompt
from custom_miner.timing import TimeBudget


class _Data:
    def __init__(self, window):
        self.window = window


class _Input:
    def __init__(self, window):
        self.data = _Data(window)


class _Task:
    type = "named_entities_extraction"

    def __init__(self, window):
        self.input = _Input(window)


def test_parse_keeps_transcript_separate_from_snippets():
    transcript, snippets = parse_ner_task(
        _Task(
            [
                (0, "Mayor Eric Adams proposed an amendment to Local Law 55."),
                (1, "Downtown Grant\nThe grant funds the city waterfront."),
            ]
        )
    )
    assert transcript.startswith("Mayor Eric Adams")
    assert snippets == ["Downtown Grant\nThe grant funds the city waterfront."]


def test_parse_unpacks_raw_json_and_prefers_selected_snippets():
    raw = (
        '{"transcript_text": "Mayor Adams discussed the Downtown Grant in detail.", '
        '"enrichment": {"search_results": {"adams": ['
        '{"title": "Unused Hit", "snippet": "should not win when a line was selected"}]}}}'
    )
    transcript, snippets = parse_ner_task(
        _Task([(0, raw), (1, "Chosen Title\nChosen snippet about Local Law 55")])
    )
    assert transcript.startswith("Mayor Adams")
    assert len(transcript) <= 1000
    assert snippets == ["Chosen Title\nChosen snippet about Local Law 55"]


def test_clean_tag_strips_role_and_header_edges():
    assert clean_tag("Member Eddie Lin") == "eddie lin"
    assert clean_tag("Seattle City Council Tuesday") == "seattle city council"
    assert clean_tag("Councilmember Eddie Lin") == "councilmember eddie lin"
    assert clean_tag("Columbia Street") is None


def test_clean_tag_matches_validator_safe_form():
    assert clean_tag("Mayor Eric Adams") == "mayor eric adams"
    assert clean_tag("Local Law 55") == "local law 55"
    assert clean_tag("speaker 1") is None
    assert clean_tag("ab") is None
    assert clean_tag("mayor") is None
    assert clean_tag("1. Downtown Grant") == "downtown grant"


def test_ner_chat_model_defaults_to_gemini_flash_and_restores(monkeypatch):
    monkeypatch.delenv("NER_OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("VALIDATOR_OPENROUTER_MODEL", "openai/gpt-5.2")
    assert ner_chat_model() == "anthropic/claude-haiku-5.5"
    monkeypatch.setenv("NER_OPENROUTER_MODEL", "openai/gpt-5.2")
    assert ner_chat_model() == "openai/gpt-5.2"
    monkeypatch.setenv("NER_OPENROUTER_MODEL", "anthropic/claude-haiku-5.5")

    class _Llm:
        model = "inception/mercury-2"
        reasoning_effort = "high"

    llm = _Llm()
    with use_ner_chat_model(llm):
        assert llm.model == "anthropic/claude-haiku-5.5"
        assert llm.reasoning_effort is None
    assert llm.model == "inception/mercury-2"
    assert llm.reasoning_effort == "high"


def test_stock_transcript_prompt_is_the_validator_template():
    prompt = transcript_prompt("HELLO-TRANSCRIPT-MARKER")
    assert "HELLO-TRANSCRIPT-MARKER" in prompt
    assert "Mayor Example" in prompt
    assert "eddie lin" not in prompt
    assert "SN33" not in prompt


def test_parse_tag_response_reads_a_comma_list():
    raw = "Here you go:\nMayor Adams, Local Law 55, Downtown Grant"
    assert parse_tag_response(raw) == ["mayor adams", "local law 55", "downtown grant"]


def test_alias_forms_restore_collapsed_names():
    assert "eric adams" in alias_forms("mayor eric adams")
    assert "adams" in alias_forms("mayor eric adams")
    assert "new york" in alias_forms("new york city")
    assert "cosby" in alias_forms("bill cosby")
    assert alias_forms("adams") == []


def test_spans_catch_bills_and_council_names():
    text = "Councilmember Eddie Lin sponsored CB 121254 and Council Bill 121242."
    spans = extract_spans(text)
    assert "eddie lin" in spans or "councilmember eddie lin" in spans
    assert "cb 121254" in spans
    assert "council bill 121242" in spans


def test_spans_and_grounded_aliases_read_the_document():
    text = "Mayor Eric Adams proposed an amendment to Local Law 55 for the Downtown Grant."
    spans = extract_spans(text)
    assert "mayor eric adams" in spans
    assert "local law 55" in spans
    aliases = grounded_aliases(text, ["mayor eric adams", "local law 55", "downtown grant"])
    assert "eric adams" in aliases
    assert "adams" in aliases
    assert "mayor eric adams" not in aliases


def _vec(*coords: float) -> np.ndarray:
    return np.asarray(coords, dtype=np.float32)


def _formula(tags: list[str], true_gt: list[str], vectors: dict[str, np.ndarray]) -> float:
    """Same weights as NoPenaltyGroundTruthTagSimilarityScoringMechanism."""
    centroid = np.mean(np.stack([vectors[tag] for tag in true_gt], axis=0), axis=0)

    def cosine(tag: str) -> float:
        vec = vectors[tag]
        return float(np.dot(centroid, vec) / (np.linalg.norm(centroid) * np.linalg.norm(vec)))

    scores = [cosine(tag) for tag in tags]
    unique_scores = [cosine(tag) for tag in tags if tag not in set(true_gt)]
    top = sorted(unique_scores, reverse=True)[:3]
    while len(top) < 3:
        top.append(0.0)
    return float(
        0.55 * np.mean(top)
        + 0.10 * np.median(scores)
        + 0.25 * np.mean(scores)
        + 0.10 * np.max(scores)
    )


def test_portfolio_keeps_aliases_and_drops_off_topic():
    gt = ["mayor eric adams", "local law 55", "downtown grant", "new york city"]
    uniques = [
        "eric adams",
        "adams",
        "new york",
        "nyc council",
        "waterfront grant",
        "municipal budget",
        "football league",
        "recipe blog",
    ]
    vectors = {tag: _vec(1.0, 0.08, 0.02) for tag in gt + uniques[:6]}
    vectors["eric adams"] = _vec(0.96, 0.12, 0.0)
    vectors["adams"] = _vec(0.9, 0.2, 0.0)
    vectors["new york"] = _vec(0.94, 0.05, 0.1)
    vectors["football league"] = _vec(0.0, 1.0, 0.0)
    vectors["recipe blog"] = _vec(0.02, 0.1, 1.0)
    # Close enough to pass a loose 0.45 floor, far enough to pull the mean down.
    vectors["city hall rumor"] = _vec(0.45, 0.85, 0.15)
    uniques = uniques + ["city hall rumor"]

    vectors["eric adams downtown grant"] = _vec(1.0, 0.08, 0.02)
    selected = select_portfolio(
        gt,
        uniques,
        vectors,
        bridges=["eric adams downtown grant"],
    )
    assert 6 <= len(selected) <= 10
    assert "eric adams downtown grant" in selected
    assert "mayor eric adams" in selected
    assert any(tag in selected for tag in ("eric adams", "new york", "nyc council"))
    assert "football league" not in selected
    assert "recipe blog" not in selected
    assert "city hall rumor" not in selected

    exact_only = list(gt)
    padded = list(gt) + ["football league", "recipe blog", "city hall rumor"]
    assert _formula(selected, gt, vectors) > _formula(exact_only, gt, vectors)
    assert _formula(selected, gt, vectors) > _formula(padded, gt, vectors)


class _FakeLlm:
    def __init__(self):
        self.prompts: list[str] = []

    def basic_prompt(self, prompt, response_format="text"):
        self.prompts.append(prompt)
        if "<transcript_to_analyze>" in prompt:
            return "Mayor Eric Adams, Local Law 55, Downtown Grant, New York City"
        if "<enrichment_content>" in prompt:
            return "Downtown Grant, New York City Council, Waterfront Project"
        if "<sets_to_analyze>" in prompt:
            return "Mayor Eric Adams, Local Law 55, Downtown Grant, New York City"
        if "CORE ENTITIES" in prompt:
            return "Eric Adams, Adams, New York, NYC Council, Waterfront Project, Municipal Budget"
        return ""

    def get_vector_embeddings(self, tag):
        topical = (
            "adams",
            "law",
            "grant",
            "york",
            "nyc",
            "waterfront",
            "municipal",
            "council",
            "downtown",
        )
        on_topic = 1.0 if any(word in tag for word in topical) else 0.0
        return [on_topic, 0.05 if on_topic else 1.0, 0.0]


def test_mine_named_entities_follows_stock_prompts(monkeypatch):
    llm = _FakeLlm()
    monkeypatch.setattr("custom_miner.ner_pipeline.pipeline.get_llm_backend", lambda: llm)
    task = _Task(
        [
            (0, "HELLO-TRANSCRIPT-MARKER Mayor Eric Adams proposed Local Law 55 in New York City."),
            (1, "Downtown Grant\nThe grant rebuilds the waterfront."),
        ]
    )
    result = mine_named_entities(task, TimeBudget.for_task("named_entities_extraction"))
    blob = "\n".join(llm.prompts)
    assert "HELLO-TRANSCRIPT-MARKER" in blob
    assert "Mayor Example" in blob
    assert "eddie lin" not in blob
    tags = result["tags"]
    assert result["vectors"] is None
    assert 6 <= len(tags) <= 10
    assert "mayor eric adams" in tags
    assert "eric adams" in tags or "adams" in tags
    assert all(tag == tag.lower() for tag in tags)
    assert all(ch.isalnum() or ch == " " for tag in tags for ch in tag)
