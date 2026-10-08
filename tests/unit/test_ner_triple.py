"""Exact-aware three-tag score."""

import numpy as np

from custom_miner.ner_pipeline.triple import prepare_document
from custom_miner.ner_pipeline.triple import select_triple
from custom_miner.ner_pipeline.triple import validator_score


def test_three_equal_uniques_score_themselves():
    score = validator_score([0.70, 0.70, 0.70], [False, False, False])
    assert abs(score - 0.70) < 1e-9


def test_exact_match_zero_pads_the_unique_term():
    score = validator_score([0.80, 0.80, 0.90], [False, False, True])
    top3 = (0.80 + 0.80 + 0.0) / 3.0
    mean = (0.80 + 0.80 + 0.90) / 3.0
    expected = 0.55 * top3 + 0.25 * mean + 0.10 * 0.80 + 0.10 * 0.90
    assert abs(score - expected) < 1e-9


def test_one_match_in_many_simulations_is_not_a_reject():
    council = np.asarray([1.0, 0.0], dtype=np.float32)
    vectors = {
        "seattle city council": council,
        "always exact": council.copy(),
        "bridge a": np.asarray([0.72, 0.05], dtype=np.float32),
        "bridge b": np.asarray([0.70, 0.02], dtype=np.float32),
        "bridge c": np.asarray([0.69, 0.04], dtype=np.float32),
    }
    simulations = [{"seattle city council", "always exact"} for _ in range(8)]
    simulations[0] = set(simulations[0]) | {"bridge a"}
    choice = select_triple(
        ["always exact", "bridge a", "bridge b", "bridge c"],
        vectors,
        simulations,
    )
    assert choice is not None
    assert "always exact" not in choice.tags
    assert "bridge a" in choice.tags


def test_city_state_place_beats_a_repeated_surname():
    text = (
        "Seattle Council Committee Report\n"
        "Council Chamber, City Hall\n"
        "Seattle, WA 98104\n"
        "Joy Hollingsworth, Council President\n"
        "Eddie Lin, Member\n"
        "Council President Hollingsworth presiding.\n"
    )
    _anchors, _groups, bridges = prepare_document(text, [])
    assert bridges
    assert all(tag.startswith("seattle ") for tag in bridges)
    assert all("hollingsworth" not in tag for tag in bridges)
