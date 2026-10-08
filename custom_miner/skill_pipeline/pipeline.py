"""Skill-coverage mining pipeline for SN33.

Flow:
  seed + the validator's section map
    → one skill_bundle call (skill and exactly two tests written together)
    → JSON repair on truncation
    → lint (concrete assertions, unique names, identifiers present in the skill)
    → one text-embedding-3-small batch
    → keep the best pair per section and score every hidden 3-section trio

Tag tasks do not use this path. A second generation candidate is not started
until this single call has a reliable latency margin.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import bittensor as bt

from custom_miner.prompt_overrides import install_custom_prompts
from custom_miner.prompt_overrides import is_installed
from custom_miner.skill_pipeline.json_repair import repair_skill_bundle_json
from custom_miner.skill_pipeline.normalize import ensure_traceability
from custom_miner.skill_pipeline.normalize import limit_tests
from custom_miner.skill_pipeline.normalize import prepare_section_tests
from custom_miner.skill_pipeline.normalize import section_parts
from custom_miner.skill_pipeline.policies import get_skill_policy
from custom_miner.skill_pipeline.proxy_score import score_candidate
from custom_miner.skill_pipeline.section_parse import format_section_brief
from custom_miner.skill_pipeline.section_parse import parse_section_map
from custom_miner.skill_pipeline.select import embed_text
from custom_miner.skill_pipeline.select import section_target_text
from custom_miner.skill_pipeline.select import select_best_pair
from custom_miner.timing import TimeBudget


def _section_map_from_task(task) -> list:
    data = getattr(getattr(task, "input", None), "data", None)
    if data is None:
        return []
    return list(getattr(data, "section_map", None) or [])


def _seed_from_task(task) -> str:
    data = getattr(getattr(task, "input", None), "data", None)
    if data is None:
        return ""
    seed = getattr(data, "seed", None)
    if isinstance(seed, str) and seed.strip():
        return seed.strip()
    lines = getattr(data, "lines", None) or []
    if lines:
        raw = lines[0][1] if isinstance(lines[0], (list, tuple)) else lines[0]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and parsed.get("seed"):
                    return str(parsed["seed"])
            except Exception:
                return raw
    return ""


def _format_section_map(section_map: list) -> str:
    """Validator section ids, plus terms pulled from the supplied descriptions."""
    return "\n".join(format_section_brief(brief) for brief in parse_section_map(section_map))


def _usable(raw: dict[str, Any] | None) -> bool:
    if not raw:
        return False
    tests = raw.get("section_tests") or {}
    if not isinstance(tests, dict):
        return False
    return any(isinstance(v, list) and v for v in tests.values())


def _skill_model() -> str:
    return os.environ.get("SKILL_OPENROUTER_MODEL", "anthropic/claude-haiku-5.5").strip() or "anthropic/claude-haiku-5.5"


def _skill_reasoning() -> str:
    """Pick the effort that actually stays inside the 24s task timeout.

    Gemini rejects a disabled flag and then thinks unless effort is minimal.
    Claude Haiku thinks for several thousand tokens unless effort is none.
    """
    effort = (os.environ.get("SKILL_REASONING_EFFORT", "off") or "off").strip().lower()
    model = _skill_model().lower()
    if effort in ("none", "off", "null", ""):
        if "gemini" in model:
            return "minimal"
        if "claude" in model or "anthropic" in model or "haiku" in model:
            return "none"
        return ""
    return effort


def _raw_prompt_bundle(
    llml,
    seed: str,
    section_map: list,
    *,
    max_tokens: int,
    model: str,
    reasoning: str,
) -> dict[str, Any] | None:
    """Direct prompt + repair. Model and reasoning apply only to this call."""
    from conversationgenome.llm.prompt_manager import prompt_manager

    prompt = prompt_manager.skill_request_to_skill_bundle_prompt(seed, _format_section_map(section_map))
    prev_tokens = getattr(llml, "max_tokens", None)
    prev_model = getattr(llml, "model", None)
    prev_effort = getattr(llml, "reasoning_effort", None)
    try:
        if hasattr(llml, "max_tokens"):
            llml.max_tokens = max_tokens
        if model and hasattr(llml, "model"):
            llml.model = model
        if hasattr(llml, "reasoning_effort"):
            llml.reasoning_effort = reasoning or None
        content = llml.basic_prompt(prompt, response_format="json")
    except Exception as e:
        bt.logging.warning(f"SkillCoverage: raw prompt failed: {e}")
        return None
    finally:
        if prev_tokens is not None and hasattr(llml, "max_tokens"):
            try:
                llml.max_tokens = prev_tokens
            except Exception:
                pass
        if prev_model is not None and hasattr(llml, "model"):
            try:
                llml.model = prev_model
            except Exception:
                pass
        if hasattr(llml, "reasoning_effort"):
            try:
                llml.reasoning_effort = prev_effort
            except Exception:
                pass

    if not isinstance(content, str):
        return None
    repaired = repair_skill_bundle_json(content)
    if repaired:
        n_tests = sum(len(v) for v in (repaired.get("section_tests") or {}).values() if isinstance(v, list))
        bt.logging.info(f"SkillCoverage: parsed bundle tests={n_tests}")
    return repaired


def _generate_bundle(
    llml,
    seed: str,
    section_map: list,
    budget: TimeBudget,
    max_tokens: int,
    *,
    model: str,
    reasoning: str,
) -> dict[str, Any]:
    """One prompt call. Repair that same text. Retry only when it yielded no tests."""
    raw: dict[str, Any] = {"skill": "", "tdd_plan": "", "section_tests": {}, "seed": seed}
    # A measured generation is about 21s. Wait long enough to receive it, and
    # run the embedding batch only when time is actually left afterward.
    wait_s = max(1.0, min(budget.remaining() - 0.4, 22.6))
    pool = ThreadPoolExecutor(max_workers=1)
    closed = False
    try:
        fut = pool.submit(
            _raw_prompt_bundle,
            llml,
            seed,
            section_map,
            max_tokens=max_tokens,
            model=model,
            reasoning=reasoning,
        )
        try:
            parsed = fut.result(timeout=wait_s)
        except Exception as e:
            # shutdown(wait=True) blocks until the HTTP call ends, so a 16s
            # timeout was still holding the miner for the full model latency.
            bt.logging.error(f"SkillCoverage: skill_bundle failed/timed out: {e}")
            pool.shutdown(wait=False, cancel_futures=True)
            closed = True
            parsed = None
        else:
            pool.shutdown(wait=True)
            closed = True
        if parsed:
            parsed["seed"] = seed
            if _usable(parsed):
                return parsed
            raw = parsed
    finally:
        if not closed:
            pool.shutdown(wait=False, cancel_futures=True)

    if not _usable(raw) and budget.remaining() > 8.0:
        bt.logging.info("SkillCoverage: no tests from first call, shorter retry")
        retry_wait = max(1.0, min(budget.remaining() - 1.5, 8.0))
        pool = ThreadPoolExecutor(max_workers=1)
        closed = False
        try:
            fut = pool.submit(
                _raw_prompt_bundle,
                llml,
                seed,
                section_map,
                max_tokens=min(max_tokens, 4096),
                model=model,
                reasoning=reasoning,
            )
            try:
                repaired = fut.result(timeout=retry_wait)
            except Exception as e:
                bt.logging.warning(f"SkillCoverage: retry failed: {e}")
                pool.shutdown(wait=False, cancel_futures=True)
                closed = True
                repaired = None
            else:
                pool.shutdown(wait=True)
                closed = True
            if repaired and _usable(repaired):
                repaired["seed"] = seed
                return repaired
        finally:
            if not closed:
                pool.shutdown(wait=False, cancel_futures=True)
    raw["seed"] = seed
    return raw


def _log_proxy(report: dict[str, Any], policy) -> None:
    bt.logging.info(
        "SkillCoverage: proxy "
        f"mean={float(report.get('mean') or 0):.3f} "
        f"min={float(report.get('minimum') or 0):.3f} "
        f"p10={float(report.get('p10') or 0):.3f} "
        f"selection={float(report.get('selection') or 0):.3f} "
        f"max_pair={float(report.get('max_pairwise') or 0):.3f} "
        f"weakest={report.get('weakest_section') or '-'}"
    )
    max_pair = float(report.get("max_pairwise") or 0)
    if max_pair >= policy.hard_duplicate_cosine:
        bt.logging.warning(
            f"SkillCoverage: max test cosine {max_pair:.3f} is at the local duplicate reject line"
        )
    elif max_pair >= policy.warn_duplicate_cosine:
        bt.logging.info(
            f"SkillCoverage: max test cosine {max_pair:.3f} is near the duplicate margin"
        )


def embed_select_tests(
    section_tests: dict[str, list[dict[str, str]]],
    section_map: list | None,
    budget: TimeBudget | None,
    *,
    policy=None,
    skill: str = "",
) -> dict[str, list[dict[str, str]]]:
    """Keep two tests per section using the validator's embedding strings.

    One batch embeds every section title+description, every test
    description+assertion, and the skill. Every 3-section trio is scored
    locally. Skips the API when the budget is tight or the call fails, and
    then keeps the first two linted tests.
    """
    policy = policy or get_skill_policy()
    if not section_tests:
        return section_tests
    fallback = limit_tests(section_tests, policy.tests_per_section)
    if budget is not None and budget.remaining() < policy.embed_min_remaining_s:
        bt.logging.info(
            f"SkillCoverage: skip embed select, remaining={budget.remaining():.2f}s"
        )
        return fallback

    by_id = {}
    for section in list(section_map or []):
        sid, title, desc = section_parts(section)
        by_id[sid] = (title, desc)

    provisional = ensure_traceability(skill, "", section_map, section_tests, policy=policy)
    texts: list[str] = []
    slots: list[tuple[str, str, int]] = []
    if provisional.strip():
        texts.append(provisional)
        slots.append(("skill", "", -1))
    for sid, tests in section_tests.items():
        title, desc = by_id.get(sid, (sid, sid))
        texts.append(section_target_text(title, desc))
        slots.append(("section", sid, -1))
        for idx, test in enumerate(tests):
            texts.append(embed_text(test))
            slots.append(("test", sid, idx))

    try:
        from conversationgenome.llm.llm_factory import get_llm_backend

        vectors = get_llm_backend().get_vector_embeddings_batch(texts)
    except Exception as e:
        bt.logging.warning(f"SkillCoverage: embed select failed: {e}")
        return fallback

    if not vectors or len(vectors) != len(texts):
        bt.logging.warning("SkillCoverage: embed select returned a short vector list")
        return fallback

    skill_vec: list[float] | None = None
    section_vecs: dict[str, list[float] | None] = {}
    test_vecs: dict[tuple[str, int], list[float] | None] = {}
    for slot, vector in zip(slots, vectors):
        kind, sid, idx = slot
        if kind == "skill":
            skill_vec = vector
        elif kind == "section":
            section_vecs[sid] = vector
        else:
            test_vecs[(sid, idx)] = vector

    selected: dict[str, list[dict[str, str]]] = {}
    selected_vecs: dict[str, list[list[float]]] = {}
    dropped = 0
    for sid, tests in section_tests.items():
        vecs = [test_vecs.get((sid, i)) for i in range(len(tests))]
        if any(v is None or len(v) == 0 for v in vecs):
            chosen_idx = list(range(min(policy.tests_per_section, len(tests))))
        else:
            chosen_idx = select_best_pair(
                section_vecs.get(sid),
                vecs,
                skill_vec,
                hard_dup=policy.hard_duplicate_cosine,
            )
        chosen = [tests[i] for i in chosen_idx]
        if not chosen:
            chosen = tests[:1]
            chosen_idx = [0] if tests else []
        dropped += len(tests) - len(chosen)
        selected[sid] = chosen
        selected_vecs[sid] = [vecs[i] for i in chosen_idx if i < len(vecs) and vecs[i]]

    if dropped:
        bt.logging.info(f"SkillCoverage: embed select kept 2 tests and dropped {dropped}")
    report = score_candidate(list(selected), section_vecs, selected_vecs, skill_vec)
    _log_proxy(report, policy)
    return selected


def _sections_short_of_two(
    section_map: list,
    section_tests: dict[str, list[dict[str, str]]],
) -> list[tuple[str, str, str, list[dict[str, str]]]]:
    short = []
    for section in section_map:
        sid, title, desc = section_parts(section)
        have = list(section_tests.get(sid) or [])
        if len(have) < 2:
            short.append((sid, title, desc, have))
    return short


def _fill_prompt(seed: str, short: list[tuple[str, str, str, list[dict[str, str]]]]) -> str:
    blocks = []
    for sid, title, desc, have in short:
        need = 2 - len(have)
        lines = [f"- [{sid}] {title}: {desc}", f"  write {need} new test(s)"]
        for test in have:
            lines.append(f"  already asserted: {test['assertion']}")
        blocks.append("\n".join(lines))
    return (
        "Return one JSON object and nothing else.\n"
        "A section with fewer than 2 tests scores as empty. "
        "Write only the new tests for the sections below. "
        "Each new assertion must check a different outcome from the assertion already listed. "
        "Use f(input) == exact_output, f(input) raises ExactError, or a command that produces an exact state. "
        "No 'because'. No vague claim.\n"
        "skill_addendum is markdown. For each section, one heading with the exact title, "
        "the section description, and the new assertion copied exactly.\n"
        '{"section_tests":{"<section_id>":[{"name":"string","description":"string","assertion":"string"}]},'
        '"skill_addendum":"markdown"}\n\n'
        f"<skill_request>\n{seed}\n</skill_request>\n\n"
        "<sections>\n" + "\n".join(blocks) + "\n</sections>"
    )


def _fill_missing_tests(
    llml,
    seed: str,
    section_map: list,
    raw: dict[str, Any],
    prepared: dict[str, list[dict[str, str]]],
    budget: TimeBudget,
    *,
    model: str,
    reasoning: str,
    policy,
) -> dict[str, list[dict[str, str]]]:
    """One short call for sections that arrived with fewer than two tests."""
    short = _sections_short_of_two(section_map, prepared)
    if not short or budget.remaining() < 7.0:
        return prepared
    bt.logging.info(
        f"SkillCoverage: filling {len(short)} sections that have fewer than 2 tests"
    )
    wait_s = max(1.0, min(budget.remaining() - 1.2, 8.0))
    pool = ThreadPoolExecutor(max_workers=1)
    closed = False
    filled = None
    try:
        fut = pool.submit(
            _raw_prompt_text,
            llml,
            _fill_prompt(seed, short),
            max_tokens=2500,
            model=model,
            reasoning=reasoning,
        )
        try:
            filled_text = fut.result(timeout=wait_s)
        except Exception as e:
            bt.logging.warning(f"SkillCoverage: gap fill timed out: {e}")
            pool.shutdown(wait=False, cancel_futures=True)
            closed = True
            filled_text = None
        else:
            pool.shutdown(wait=True)
            closed = True
        if isinstance(filled_text, str):
            filled = repair_skill_bundle_json(filled_text)
    finally:
        if not closed:
            pool.shutdown(wait=False, cancel_futures=True)
    if not isinstance(filled, dict):
        return prepared
    merged_tests = dict(raw.get("section_tests") or {})
    for sid, rows in (filled.get("section_tests") or {}).items():
        if isinstance(sid, str) and isinstance(rows, list):
            merged_tests[sid] = list(merged_tests.get(sid) or []) + rows
    raw["section_tests"] = merged_tests
    addendum = filled.get("skill_addendum") or filled.get("skill") or ""
    if isinstance(addendum, str) and addendum.strip():
        raw["skill"] = ((raw.get("skill") or "") + "\n\n" + addendum.strip()).strip()
    return prepare_section_tests(raw, section_map, policy=policy)


def _raw_prompt_text(llml, prompt: str, *, max_tokens: int, model: str, reasoning: str) -> str | None:
    prev_tokens = getattr(llml, "max_tokens", None)
    prev_model = getattr(llml, "model", None)
    prev_effort = getattr(llml, "reasoning_effort", None)
    try:
        if hasattr(llml, "max_tokens"):
            llml.max_tokens = max_tokens
        if model and hasattr(llml, "model"):
            llml.model = model
        if hasattr(llml, "reasoning_effort"):
            llml.reasoning_effort = reasoning or None
        content = llml.basic_prompt(prompt, response_format="json")
    except Exception as e:
        bt.logging.warning(f"SkillCoverage: gap prompt failed: {e}")
        return None
    finally:
        if prev_tokens is not None and hasattr(llml, "max_tokens"):
            try:
                llml.max_tokens = prev_tokens
            except Exception:
                pass
        if prev_model is not None and hasattr(llml, "model"):
            try:
                llml.model = prev_model
            except Exception:
                pass
        if hasattr(llml, "reasoning_effort"):
            try:
                llml.reasoning_effort = prev_effort
            except Exception:
                pass
    return content if isinstance(content, str) else None


def mine_skill_coverage(task, budget: TimeBudget | None = None) -> dict[str, Any]:
    """Run the contract-first skill-coverage pipeline within the task budget."""
    from conversationgenome.llm.llm_factory import get_llm_backend

    if not is_installed():
        install_custom_prompts()

    seed = _seed_from_task(task)
    section_map = _section_map_from_task(task)
    policy = get_skill_policy(getattr(task, "type", "skill_coverage_evaluation"))
    budget = budget or TimeBudget.for_task("skill_coverage_evaluation", getattr(task, "timeout", 24))
    skill_max_tokens = int(os.environ.get("SKILL_OPENROUTER_MAX_TOKENS", "8192"))
    skill_model = _skill_model()
    skill_reasoning = _skill_reasoning()
    bt.logging.info(
        f"SkillCoverage: model={skill_model} reasoning={skill_reasoning or 'off'} max_tokens={skill_max_tokens}"
    )

    if not seed or not section_map:
        bt.logging.warning(
            f"SkillCoverage: missing seed/section_map "
            f"(seed={bool(seed)}, sections={len(section_map)})"
        )
        return {
            "skill": "",
            "tdd_plan": "",
            "section_tests": {},
        }

    llml = get_llm_backend()
    raw = _generate_bundle(
        llml,
        seed,
        section_map,
        budget,
        skill_max_tokens,
        model=skill_model,
        reasoning=skill_reasoning,
    )
    raw["seed"] = seed
    prepared = prepare_section_tests(raw, section_map, policy=policy)
    prepared = _fill_missing_tests(
        llml,
        seed,
        section_map,
        raw,
        prepared,
        budget,
        model=skill_model,
        reasoning=skill_reasoning,
        policy=policy,
    )
    skill_in = raw.get("skill") if isinstance(raw.get("skill"), str) else ""
    selected = embed_select_tests(
        prepared,
        section_map,
        budget,
        policy=policy,
        skill=skill_in,
    )
    tdd_plan = (raw.get("tdd_plan") or "").strip() or (
        "Two concrete checks per section, both written into the skill."
    )
    result = {
        "skill": ensure_traceability(skill_in, seed, section_map, selected, policy=policy),
        "tdd_plan": tdd_plan[:240],
        "section_tests": selected,
    }
    n_sections = len(result["section_tests"])
    n_tests = sum(len(v) for v in result["section_tests"].values())
    bt.logging.info(
        f"SkillCoverage: bundle sections={n_sections} tests={n_tests} "
        f"skill_chars={len(result['skill'])} remaining={budget.remaining():.2f}s"
    )
    return result


def pipeline_debug(task) -> dict[str, Any]:
    """Inspect inputs without calling the LLM."""
    return {
        "seed": _seed_from_task(task)[:200],
        "section_ids": [
            section_parts(s)[0] for s in _section_map_from_task(task)
        ],
        "policy": get_skill_policy().__dict__,
    }
