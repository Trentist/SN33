# Local SN33 Miner Development (live-matched)

Mirrors mainnet validator behavior over OpenRouter:

1. Quality gate (`openai/gpt-5-mini`)
2. Ground-truth tags (`openai/gpt-5.2`)
3. Embeddings (`openai/text-embedding-3-small`)
4. Miner window tags (`google/gemini-2.5-flash` by default)
5. Score with `GroundTruthTagSimilarityScoringMechanism`

Skill coverage path uses `openai/gpt-5.6-luna` for section map + judge, and the
competitive `custom_miner/skill_pipeline` on the miner side.

## Commands

```bash
source .env   # or rely on python-dotenv

python -m local_dev.run_sim --models

# Default: LIVE testnet problem from ReadyAI API (netuid 138)
python -m local_dev.run_sim --mode validator --source api --windows 1

# Filter to a task type (retries reserves until match)
python -m local_dev.run_sim --source api --task conversation_tagging
python -m local_dev.run_sim --source api --task skill_coverage_evaluation

# Offline fixture (DummyData)
python -m local_dev.run_sim --source fixture --task conversation_tagging
python -m local_dev.run_sim --mode validator --source fixture --task skill_coverage_evaluation

# Offline mock skill coverage (no API key) — normalize + pseudo judge/score
python -m local_dev.run_sim --mode mock --task skill_coverage_evaluation
python -m local_dev.run_sim --mode mock --task skill_coverage_evaluation --compare

# Cache ~20 live testnet problems, then replay from disk
python -m local_dev.fetch_problems --count 20
python -m local_dev.fetch_problems --list
python -m local_dev.run_sim --source cache --cache-index 0

# Build miner-facing payloads (masked CgSynapse.cgp_input) from cached bundles
python -m local_dev.build_miner_problems --tasks-per-bundle 2
python -m local_dev.build_miner_problems --list
# → local_dev/cache/miner_problems/tasks/*.json

# Per-problem eyeball view: raw / gt / miner / metrics / compare
python -m local_dev.problem_view
# → local_dev/cache/problems/<stem>/{raw,gt,miner,metrics,compare}.json
# Every score run updates those files after EACH problem (+ index.json):
python local_dev/score_prepared_bundles.py

# Smoke tests (tags + skill coverage)
pytest local_dev/test_smoke.py -q

# Auto monitor: WandB scores + pm2 miner logs → local_dev/cache/monitor/uid244/
python -m local_dev.monitor --once --json
# loop (also started via pm2 as sn33-uid244-monitor):
python -m local_dev.monitor --interval 120
```

Uses `testnet_readyai_api_data.json` / `READYAI_API_KEY` (testnet only). Mainnet needs a validator-issued `readyai_api_data.json`.

There is no hard public cap for testnet reserves in practice — we pulled 20 in ~5–7s with 0 errors. 20 is a good local set.

## Models

### Live mainnet validators

| Role | Model |
|------|--------|
| GT / default chat | `gpt-5.2` |
| Quality gate | `gpt-5-mini` |
| Skill coverage + judge | `gpt-5.6-luna` |
| Embeddings | `text-embedding-3-small` |

### This local_dev setup (OpenRouter)

| Role | Model |
|------|--------|
| Validator GT | `openai/gpt-5.2` |
| Quality gate | `openai/gpt-5-mini` |
| Skill coverage | `openai/gpt-5.6-luna` |
| Embeddings | `openai/text-embedding-3-small` |
| Miner | `google/gemini-2.5-flash` |

Change miner with `MINER_OPENROUTER_MODEL=...` in `.env`.
