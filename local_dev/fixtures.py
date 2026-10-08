"""Local development fixtures with ground-truth tags that match the sample dialogue."""

from __future__ import annotations

from conversationgenome.api.models.conversation_metadata import ConversationMetadata
from conversationgenome.task_bundle.TaskBundle import TaskBundle
from tests.mocks.DummyData import DummyData


# Sensible GT for DummyData.lines() (wedding / babysitting / cars / movie)
CONVERSATION_GT_TAGS = [
    "wedding planning",
    "babysitting",
    "parenthood",
    "luxury cars",
    "mercedes",
    "ferrari",
    "movies",
    "titanic",
    "family",
    "side hustle",
    "marriage",
    "children",
]


def _pseudo_vec(text: str, dim: int = 8) -> list[float]:
    seed = sum(ord(c) for c in (text or "").lower()) or 1
    vec = [((seed * (i + 3)) % 97) / 97.0 for i in range(dim)]
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _pseudo_vectors(tags: list[str], dim: int = 8) -> dict:
    """Deterministic fake embeddings so offline cosine scoring works without OpenAI."""
    return {tag: {"vectors": _pseudo_vec(tag, dim)} for tag in tags}


def ready_conversation_bundle() -> TaskBundle:
    bundle = DummyData.setup_conversation_tagging_task_bundle()
    tags = list(CONVERSATION_GT_TAGS)
    bundle.input.metadata = ConversationMetadata(
        participantProfiles=DummyData.participants(),
        tags=tags,
        vectors=_pseudo_vectors(tags),
    )
    return bundle


def ready_skill_coverage_bundle() -> TaskBundle:
    """Ready skill_coverage_evaluation bundle with section map + dim-matched vectors."""
    from conversationgenome.api.models.skill_coverage import SkillCoverageMetadata

    bundle = DummyData.setup_skill_coverage_evaluation_task_bundle()
    # DummyData.section_vectors are dim-3; offline pseudo embeddings use dim-8.
    # Keep both sides aligned so SkillCoverageScoringMechanism cosine works offline.
    section_vectors = {
        getattr(s, "section_id"): _pseudo_vec(getattr(s, "section_id"))
        for s in (bundle.input.data.section_map or [])
    }
    bundle.input.metadata = SkillCoverageMetadata(
        sections=list(bundle.input.data.section_map or []),
        section_vectors=section_vectors,
    )
    return bundle


def attach_pseudo_vectors(miner_result: dict) -> dict:
    tags = miner_result.get("tags") or []
    return {"tags": tags, "vectors": _pseudo_vectors(tags), **{k: v for k, v in miner_result.items() if k not in ("tags", "vectors")}}


def attach_pseudo_skill_scoring(miner_result: dict, section_vectors: dict[str, list[float]] | None = None) -> dict:
    """Offline stand-in for format_results(): skill_vector + judged_correct test_vectors."""
    result = dict(miner_result or {})
    skill = result.get("skill") or ""
    section_tests = result.get("section_tests") or {}
    if not isinstance(section_tests, dict):
        section_tests = {}

    section_vectors = section_vectors or {}
    skill_l = skill.lower()
    dim = 8
    test_vectors: dict[str, list] = {}

    for section_id, tests in section_tests.items():
        if not isinstance(tests, list):
            continue
        # Always use dim-matched pseudo GT derived from section_id (+ optional seed from real vectors)
        provided = section_vectors.get(section_id)
        if provided and len(provided) == dim:
            gt = list(provided)
            norm = sum(v * v for v in gt) ** 0.5 or 1.0
            gt = [v / norm for v in gt]
        else:
            gt = _pseudo_vec(section_id, dim)
        out = []
        for idx, test in enumerate(tests):
            if not isinstance(test, dict):
                continue
            description = test.get("description") or ""
            assertion = test.get("assertion") or ""
            # Unique dominant direction per test (avoids near_duplicate ≥0.95 offline),
            # with a light section GT pull so section_coverage stays meaningful.
            uniq = [0.0] * dim
            uniq[idx % dim] = 1.0
            uniq[(idx * 3 + hash(assertion) % dim) % dim] += 0.7
            noise = _pseudo_vec(f"uniq:{section_id}:{test.get('name')}:{assertion}", dim)
            blended = [(0.25 * a + 0.55 * u + 0.20 * n) for a, u, n in zip(gt, uniq, noise)]
            norm = sum(v * v for v in blended) ** 0.5 or 1.0
            vector = [v / norm for v in blended]
            # Heuristic judge: non-empty concrete assertion; bonus if tokens appear in skill
            tokens = [t.strip("\"'`.,") for t in assertion.replace("(", " ").replace(")", " ").split() if len(t.strip("\"'`.,")) > 3]
            hit = any(tok.lower() in skill_l for tok in tokens[:8])
            judged = bool(assertion.strip()) and (hit or len(assertion) >= 12)
            out.append(
                {
                    "name": test.get("name"),
                    "description": description,
                    "assertion": assertion,
                    "vector": vector,
                    "judged_correct": judged,
                }
            )
        test_vectors[section_id] = out

    result["skill_vector"] = _pseudo_vec(skill, dim) if skill.strip() else None
    # Blend skill vector toward mean of judged-correct tests so skill_coverage isn't random offline
    judged_vecs = [
        t["vector"]
        for tests in test_vectors.values()
        for t in tests
        if t.get("judged_correct") and t.get("vector")
    ]
    if result["skill_vector"] and judged_vecs:
        mean = [sum(col) / len(judged_vecs) for col in zip(*judged_vecs)]
        blended = [(0.4 * a + 0.6 * b) for a, b in zip(result["skill_vector"], mean)]
        norm = sum(v * v for v in blended) ** 0.5 or 1.0
        result["skill_vector"] = [v / norm for v in blended]
    result["test_vectors"] = test_vectors
    return result
