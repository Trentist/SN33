"""Webpage metadata pipeline: stock prompts, cleaning, and centroid portfolio."""

import numpy as np

from custom_miner.timing import TimeBudget
from custom_miner.webpage_pipeline.anchors import phrases_in_text
from custom_miner.webpage_pipeline.anchors import variant_tags
from custom_miner.webpage_pipeline.gate import clean_tag
from custom_miner.webpage_pipeline.parse import is_stub_page
from custom_miner.webpage_pipeline.portfolio import too_close
from custom_miner.webpage_pipeline.parse import parse_webpage_task
from custom_miner.webpage_pipeline.pipeline import mine_webpage
from custom_miner.webpage_pipeline.pipeline import use_webpage_chat_model
from custom_miner.webpage_pipeline.pipeline import webpage_chat_model
from custom_miner.webpage_pipeline.portfolio import select_portfolio
from custom_miner.webpage_pipeline.stock import parse_tag_response
from custom_miner.webpage_pipeline.stock import website_prompt


class _Data:
    def __init__(self, window):
        self.window = window


class _Input:
    def __init__(self, window, categories=None):
        self.data = _Data(window)
        self.input_categories = categories


class _Task:
    type = "webpage_metadata_generation"

    def __init__(self, window, categories=None):
        self.input = _Input(window, categories)


def test_stub_page_and_snippet_phrases():
    assert is_stub_page(
        "This domain is for use in documentation examples without needing permission."
    )
    found = phrases_in_text("Patterns such as retry and circuit breaker keep microservices up.")
    assert "circuit breaker" in found
    assert "retry" in found
    assert "microservices" in found
    assert "retry pattern" in variant_tags(found)
    assert "circuit breaker pattern" in variant_tags(found)
    assert too_close("circuit breaker", "circuit breaker pattern") is True
    assert too_close("retry", "retry pattern") is True
    assert too_close("resilience patterns", "resiliency patterns") is True
    assert too_close("vulnerability management", "vulnerability scanning") is False
    assert clean_tag("examples") is None
    assert clean_tag("api") is None


def test_webpage_chat_model_is_gemini_flash():
    assert webpage_chat_model() == "google/gemini-2.5-flash"
    llm = type("L", (), {"model": "inception/mercury-2"})()
    with use_webpage_chat_model(llm):
        assert llm.model == "google/gemini-2.5-flash"
    assert llm.model == "inception/mercury-2"


def test_parse_keeps_page_separate_from_snippets():
    page, snippets, coding = parse_webpage_task(
        _Task(
            [
                (0, "Solar farms in Nevada store daylight as electricity."),
                (1, "Battery storage\nUtilities add grid batteries"),
            ]
        )
    )
    assert coding is False
    assert page.startswith("Solar farms")
    assert snippets == ["Battery storage\nUtilities add grid batteries"]


def test_parse_unpacks_raw_json_and_coding_flag():
    raw = (
        '{"website_markdown": "Python asyncio tutorial for web servers", '
        '"enrichment": {"search_results": {"asyncio": ['
        '{"title": "Async IO", "snippet": "event loop and tasks"}]}}}'
    )
    page, snippets, coding = parse_webpage_task(_Task([(0, raw)], categories=["coding"]))
    assert coding is True
    assert "asyncio" in page
    assert len(page) <= 1000
    assert snippets == ["Async IO\nevent loop and tasks"]


def test_clean_tag_matches_validator_safe_form():
    assert clean_tag("AI-Driven Drug Discovery") == "ai driven drug discovery"
    assert clean_tag("contact us") is None
    assert clean_tag("https://example.com/story") is None
    assert clean_tag("ab") is None
    assert clean_tag("1. solar power") == "solar power"


def test_stock_website_prompt_is_the_validator_template():
    prompt = website_prompt("HELLO-PAGE-MARKER", coding=False)
    assert "HELLO-PAGE-MARKER" in prompt
    assert "artificial intelligence, machine learning" in prompt
    assert "circuit breaker" not in prompt
    assert "galen growth" not in prompt


def test_parse_tag_response_reads_a_comma_list():
    raw = "Here you go:\nsolar power, renewable energy, photovoltaic panels"
    assert parse_tag_response(raw) == ["solar power", "renewable energy", "photovoltaic panels"]


def test_portfolio_keeps_centroid_uniques_and_drops_off_topic():
    def vec(x, y):
        return np.asarray([x, y], dtype=np.float32)

    gt = ["solar power", "renewable energy", "photovoltaic panels", "energy storage"]
    uniques = [
        "clean energy",
        "solar farms",
        "electricity generation",
        "power grid",
        "energy transition",
        "sustainable power",
        "football league",
        "recipe blog",
    ]
    vectors = {tag: vec(1.0, 0.05) for tag in gt + uniques[:6]}
    vectors["football league"] = vec(0.0, 1.0)
    vectors["recipe blog"] = vec(0.05, 1.0)

    selected = select_portfolio(gt, uniques, vectors)
    assert 8 <= len(selected) <= 10
    assert selected[0] == "solar power"
    assert "clean energy" in selected
    assert "football league" not in selected
    assert "recipe blog" not in selected


def test_portfolio_keeps_core_variant_and_drops_peripheral():
    def vec():
        return np.asarray([1.0, 0.1], dtype=np.float32)

    gt = [
        "circuit breaker",
        "prometheus",
        "microservices",
        "retry pattern",
        "retry",
        "vulnerability management",
        "threat detection",
        "security monitoring",
    ]
    uniques = [
        "retry",
        "security posture",
        "audit trail",
        "incident handling",
        "least privilege principle",
    ]
    vectors = {tag: vec() for tag in gt + uniques}
    selected = select_portfolio(gt, uniques, vectors)
    assert "retry pattern" in selected
    assert "retry" in selected
    assert "security posture" in selected
    assert "circuit breaker" not in selected
    assert "prometheus" not in selected
    assert "microservices" not in selected


class _FakeLlm:
    def __init__(self):
        self.prompts: list[str] = []

    def basic_prompt(self, prompt, response_format="text"):
        self.prompts.append(prompt)
        if "<webpage_content>" in prompt:
            return "solar power, renewable energy, photovoltaic panels, energy storage"
        if "<enrichment_content>" in prompt:
            return "battery storage, grid modernization, clean energy policy"
        if "<sets_to_analyze>" in prompt:
            return "solar power, renewable energy, battery storage, clean energy policy"
        if "CHOSEN TAGS" in prompt:
            return (
                "clean electricity, solar generation, renewable power, grid storage, "
                "energy systems, daylight capture"
            )
        if "CORE TAGS" in prompt:
            return (
                "clean energy, solar farms, electricity generation, power grid, "
                "energy transition, sustainable power"
            )
        return ""

    def get_vector_embeddings(self, tag):
        topical = (
            "solar",
            "energy",
            "renewable",
            "power",
            "grid",
            "battery",
            "photovoltaic",
            "clean",
            "electricity",
        )
        on_topic = 1.0 if any(word in tag for word in topical) else 0.0
        return [on_topic, 0.0 if on_topic else 1.0]


def test_mine_webpage_follows_stock_prompts(monkeypatch):
    llm = _FakeLlm()
    monkeypatch.setattr("custom_miner.webpage_pipeline.pipeline.get_llm_backend", lambda: llm)
    task = _Task(
        [
            (0, "HELLO-PAGE-MARKER solar farms store daylight as electricity."),
            (1, "Battery storage\nUtilities add grid batteries"),
        ]
    )
    result = mine_webpage(task, TimeBudget.for_task("webpage_metadata_generation"))
    blob = "\n".join(llm.prompts)
    assert "HELLO-PAGE-MARKER" in blob
    assert "artificial intelligence, machine learning" in blob
    assert "circuit breaker" not in blob
    assert "galen growth" not in blob
    assert "CHOSEN TAGS" in blob
    tags = result["tags"]
    assert result["vectors"] is None
    assert 8 <= len(tags) <= 10
    assert "solar power" in tags
    assert all(tag == tag.lower() for tag in tags)
    assert all(" " == ch or ch.isalnum() for tag in tags for ch in tag)
