#!/usr/bin/env python3
"""Score the three-tag selector against stored named-entity ground truth.

Reports, per bundle, the selector's simulated score, its score against the
stored centroid, and the best triple in the same candidate pool. A leave-one-
bundle-out shift is fit on the other bundles only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

from custom_miner.ner_pipeline.portfolio import embed_tags
from custom_miner.ner_pipeline.triple import best_actual_triple
from custom_miner.ner_pipeline.triple import candidate_pool
from custom_miner.ner_pipeline.triple import document_seed
from custom_miner.ner_pipeline.triple import load_stored_ground_truth
from custom_miner.ner_pipeline.triple import prepare_document
from custom_miner.ner_pipeline.triple import score_against
from custom_miner.ner_pipeline.triple import select_triple
from custom_miner.ner_pipeline.triple import simulate_ground_truths
from custom_miner.ner_pipeline.triple import simulation_centroids
from local_dev.backends import make_backend

PREP = _ROOT / "local_dev" / "cache" / "miner_problems" / "prepared_bundles"


def _lines(bundle: dict) -> tuple[str, list[str]]:
    lines = ((bundle.get("input") or {}).get("data") or {}).get("lines") or []
    texts = []
    for line in lines:
        if isinstance(line, (list, tuple)) and len(line) >= 2:
            texts.append(str(line[1] or "").strip())
        elif isinstance(line, str):
            texts.append(line.strip())
    texts = [text for text in texts if text]
    if not texts:
        return "", []
    return texts[0][:1000], [text[:1000] for text in texts[1:6]]


def _anchors(transcript_anchors: list[str], groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in transcript_anchors + [tag for group in groups for tag in group]:
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _mean_centroid(packed: list[tuple[set[str], np.ndarray]]) -> np.ndarray | None:
    if not packed:
        return None
    return np.mean(np.stack([centroid for _, centroid in packed], axis=0), axis=0).astype(np.float32)


def main() -> None:
    paths = sorted(PREP.glob("*named_entities_extraction*.json"))
    if not paths:
        raise SystemExit(f"no prepared NER bundles in {PREP}")
    llm = make_backend("openrouter")
    docs = []
    for path in paths:
        bundle = json.loads(path.read_text())
        transcript, snippets = _lines(bundle)
        true_centroid, true_tags = load_stored_ground_truth(bundle)
        transcript_anchors, groups, bridges = prepare_document(transcript, snippets)
        anchors = _anchors(transcript_anchors, groups)
        pool = candidate_pool(anchors, bridges)
        vectors = embed_tags(llm, anchors + pool)
        seed = document_seed(transcript, snippets)
        simulations = simulate_ground_truths(transcript_anchors, groups, seed=seed)
        packed = simulation_centroids(simulations, vectors)
        sim_mean = _mean_centroid(packed)
        docs.append(
            {
                "name": path.name,
                "anchors": anchors,
                "bridges": bridges,
                "pool": pool,
                "vectors": vectors,
                "simulations": simulations,
                "true_centroid": true_centroid,
                "true_tags": true_tags,
                "sim_mean": sim_mean,
            }
        )

    print(f"{'bundle':<42} {'sim':>6} {'actual':>7} {'oracle':>7} {'loo':>7}  picked")
    sim_vals = []
    actual_vals = []
    oracle_vals = []
    loo_vals = []
    for index, doc in enumerate(docs):
        choice = select_triple(doc["pool"], doc["vectors"], doc["simulations"])
        actual = None
        if choice is not None and doc["true_centroid"] is not None:
            actual = score_against(choice.tags, doc["vectors"], doc["true_centroid"], doc["true_tags"])
        oracle = None
        if doc["true_centroid"] is not None:
            found = best_actual_triple(doc["pool"], doc["vectors"], doc["true_centroid"], doc["true_tags"])
            if found is not None:
                oracle = found
        shift = None
        others = []
        for other_index, other in enumerate(docs):
            if other_index == index or other["sim_mean"] is None or other["true_centroid"] is None:
                continue
            others.append(other["true_centroid"] - other["sim_mean"])
        if others:
            shift = np.mean(np.stack(others, axis=0), axis=0).astype(np.float32)
        loo_choice = select_triple(
            doc["pool"],
            doc["vectors"],
            doc["simulations"],
            centroid_shift=shift,
        )
        loo_actual = None
        if loo_choice is not None and doc["true_centroid"] is not None:
            loo_actual = score_against(
                loo_choice.tags, doc["vectors"], doc["true_centroid"], doc["true_tags"]
            )
        if choice is not None:
            sim_vals.append(choice.expected)
        if actual is not None:
            actual_vals.append(actual)
        if oracle is not None:
            oracle_vals.append(oracle[1])
        if loo_actual is not None:
            loo_vals.append(loo_actual)
        picked = choice.tags if choice is not None else []
        oracle_tags = oracle[0] if oracle is not None else []
        print(
            f"{doc['name'][:42]:<42} "
            f"{(choice.expected if choice else float('nan')):6.3f} "
            f"{(actual if actual is not None else float('nan')):7.3f} "
            f"{(oracle[1] if oracle else float('nan')):7.3f} "
            f"{(loo_actual if loo_actual is not None else float('nan')):7.3f}  "
            f"{picked}"
        )
        print(f"  bridges={doc['bridges']}")
        print(f"  oracle={oracle_tags}")
        if loo_choice is not None:
            print(
                f"  loo={loo_choice.tags} sim={loo_choice.expected:.3f} "
                f"q25={loo_choice.lower:.3f}"
            )
        if choice is not None:
            print(f"  q25={choice.lower:.3f} objective={choice.objective:.3f} pool={len(doc['pool'])}")

    def _avg(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    sim_mean = _avg(sim_vals)
    actual_mean = _avg(actual_vals)
    oracle_mean = _avg(oracle_vals)
    loo_mean = _avg(loo_vals)
    print(
        f"MEAN sim={sim_mean:.3f} actual={actual_mean:.3f} "
        f"oracle={oracle_mean:.3f} loo_actual={loo_mean:.3f}"
    )
    if actual_mean + 0.02 >= oracle_mean and actual_mean >= sim_mean:
        print(
            "DIAGNOSIS pool ceiling: the selector matches the best triple in the pool. "
            "The stored score is higher than the simulated score, so the centroid estimate is pessimistic."
        )
    elif sim_mean - actual_mean > 0.08 and oracle_mean - actual_mean > 0.05:
        print("DIAGNOSIS centroid: simulated score is ahead of the stored ground truth, and a better triple exists.")
    elif oracle_mean < 0.60 and abs(sim_mean - actual_mean) <= 0.08:
        print("DIAGNOSIS candidates: simulated and stored scores agree, and the pool itself stays under 0.60.")
    elif oracle_mean - actual_mean > 0.05:
        print("DIAGNOSIS ranking: the pool contains a stronger triple than the selector picked.")
    else:
        print("DIAGNOSIS mixed: selector and pool are close.")


if __name__ == "__main__":
    main()
