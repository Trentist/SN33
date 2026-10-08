# Competitive SN33 Miner

Drop-in miner improvements used when `CUSTOM_MINER=1` (default).

## What it changes

1. **Robust hybrid tag pipeline** (`tag_pipeline/`) — embed-aware exact/unique mix aligned to SN33 cosine scoring
2. **Skill coverage pipeline** (`skill_pipeline/`) — score-aligned skill + TDD tests for `skill_coverage_evaluation`
3. **Parallel LLM mine** (`parallel_mine.py`) — window + enrichment LLM calls run concurrently
4. **Time budgets** — tag tasks target ≤10.5s (timeout 12s); skill ≤22s (timeout 24s); skips embed select if budget nearly exhausted
5. **Prompt overlay** — score-aligned Jinja prompts under `prompts/`
6. **Deterministic tag merge** — skips stock LLM combine that collapses paraphrases
7. **Metrics hook** — optional `Extensions.Metrics` call after each mine

Default miner model: `google/gemini-2.5-flash` via `MINER_OPENROUTER_MODEL`.

## Tag pipeline selection

```bash
export TAG_PIPELINE=hybrid      # default: FAISS/clf/alias hybrid
export TAG_PIPELINE=wiki        # WikiGround: segment → Wikidata link → KG expand
```

WikiGround (`tag_pipeline_wiki/`) does **not** use historical GT FAISS / concept-clf /
hand alias tables. It segments dialogue, mines mentions, links to Wikidata, expands
from entity descriptions (+ optional 1-hop claims), evidence-gates, then selects 10–14 tags.
Wikidata search responses cache under `tag_pipeline_wiki/data/wd_cache/`.

```bash
export TAG_PIPELINE=wiki
export WIKI_FETCH_CLAIMS=0   # default: description expansion only (fast, avoids 429s)
export WIKI_MENTION_LIMIT=28
export WIKI_LINK_WORKERS=4
```

## Hybrid tag pipeline (v2 — embedding-aware)

```
INPUT → classifier (chat/text/skill)
          ├─ explicit anchors (window+enrichment, junk-stripped)
          ├─ tag-bank lexical kNN
          ├─ LLM candidates
          └─ validate-safe paraphrases
                ↓
          batch embed (text-embedding-3-small)
                ↓
          neighborhood = mean(evidenced anchors)
                ↓
          mix: 3–4 evidenced anchors + 6–8 high-cosine uniques
                ↓
          8–12 validate-safe tags
```

Targets live WandB band (~0.75–0.90) via the same objective validators use:
`0.55 * top3_unique_cosine + 0.25 * mean + …`

Rebuild the tag bank after fetching new prepared problems:

```bash
python -m custom_miner.rebuild_tag_bank
```

## Skill coverage pipeline

One generation call, then a local score of every hidden 3-section trio (`SkillCoverageScoringMechanism`, section 60% + skill 40%):

```
seed + validator section map
    → one skill_bundle call (skill and exactly 2 tests written together)
    → JSON repair, retry only if that call produced no tests
    → lint (concrete assertion, unique names, identifiers present in the skill)
    → one text-embedding-3-small batch
    → keep the best pair and score every hidden 3-section trio
    → {skill, tdd_plan, section_tests}
```

`skill_generation` remains tag-like (stock mine + tag postprocess).

## Files

| Path | Role |
|------|------|
| `miner_lib.py` | `CompetitiveMinerLib` |
| `tag_pipeline/` | Hybrid tag selector |
| `skill_pipeline/` | Skill-coverage miner |
| `tag_pipeline/data/tag_bank.json` | Seeded GT/alias bank |
| `prompt_overrides.py` | ChoiceLoader overlay |
| `postprocess.py` | Tag cleanup + skill normalize hook |
| `policies.py` | Per-task tag min/target/max |
| `prompts/*.j2` | Competitive prompts (tags + skill bundle) |

## Save live problems (mainnet / testnet)

When the axon is running, every incoming validator request is saved (default on):

```bash
export SAVE_MINER_PROBLEMS=1
# optional override:
# export MINER_PROBLEMS_DIR=./local_dev/cache/mainnet_problems
```

Files land in `local_dev/cache/mainnet_problems/tasks/*.json` (or testnet dir when `NETWORK=test`).

```bash
# List / stats
python -m local_dev.list_saved_problems
python -m local_dev.list_saved_problems --stats

# Replay one through CompetitiveMinerLib
python -m local_dev.replay_saved_problem --latest
```

### Run miner on mainnet (finney / netuid 33)

```bash
export NETWORK=finney          # entrypoint maps to netuid 33
export TYPE=miner
export CUSTOM_MINER=1
export SAVE_MINER_PROBLEMS=1
export MINER_OPENROUTER_MODEL=google/gemini-2.5-flash
export OPENROUTER_MODEL=google/gemini-2.5-flash
# wallet + axon IP/PORT via entrypoint / docker env
./entrypoint.sh
# or:
python -m neurons.miner --netuid 33 --wallet.name <cold> --wallet.hotkey <hot> \
  --subtensor.network finney --axon.port <port> --axon.ip <public_ip>
```

## Disable

```bash
export CUSTOM_MINER=0
```
