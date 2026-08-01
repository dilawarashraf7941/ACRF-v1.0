# Adaptive Critic Routing Framework (ACRF) v1.0

A research framework for studying **policy-guided, adaptive critic
routing** in multi-agent LLM pipelines, and for evaluating candidate
routing policies entirely **offline**, from logged execution data,
before any such policy is trusted with live traffic.

ACRF pairs a deterministic LangGraph execution pipeline with a
pluggable policy-scoring layer (a heuristic scorer and a LinUCB
contextual bandit) that sits *alongside* — and never overrides — the
pipeline's existing, fixed critic-routing rule. A companion offline
evaluation suite (replay, bootstrap statistics, ablation studies, and
learning-curve analysis) makes it possible to answer "would this
adaptive policy have helped?" without ever deploying it. This
repository contains the full implementation, the experimental campaign
that evaluates it, and the accompanying manuscript.

**Headline finding, reported without hedging:** in the evaluated
setting, adaptive routing did not outperform the deterministic
baseline. Cold-start LinUCB was statistically indistinguishable from
the heuristic baseline at every tested exploration coefficient, and
sequential (in-place) learning only diverged from cold-start behavior,
and only for the worse, at the higher end of the tested exploration
range. The framework's contribution is the methodology that produced
and validated this negative result reproducibly — see
[`docs/`](docs/) for the full manuscript.

## Project overview

| | |
|---|---|
| **Status** | Research framework, feature-complete; manuscript in preparation (not yet submitted) |
| **Language** | Python 3.12 |
| **Core stack** | LangGraph (orchestration), FastAPI (API layer), Pydantic v2 (data models), NumPy/SciPy (LinUCB, statistics), LiteLLM (provider-agnostic LLM access, not yet wired into any node), ChromaDB (vector memory, reserved) |
| **Package manager** | [uv](https://docs.astral.sh/uv/) |

**What is, and isn't, implemented.** The graph topology, context
representation, policy layer, reward function, experience recording,
and the entire offline evaluation pipeline are fully implemented and
tested. The *generative* stages of the pipeline — planning, worker
generation, and critic evaluation — are, by design, deterministic
placeholders rather than real LLM calls (Methodology §3.1 explains why:
the study is about the routing *decision*, not about generation
quality). No node currently calls an LLM; `LiteLLM` and `ChromaDB` are
declared dependencies for a future integration, not yet exercised by
any code path.

## Architecture overview

ACRF is a `StateGraph` over one shared `AgentState`, with a fixed node
sequence:

```
START → planner → worker → error_feature_extractor → policy_engine → router
                                                                        │
                              ┌─────────────────────────────────────────┤
                              ▼                                         ▼
                     worker | critic | self_correction | safety | evaluation → END
```

- **`planner` / `worker` / `error_feature_extractor`** produce a task
  decomposition, an initial response, and a structured error/complexity
  profile (all deterministic placeholders at present).
- **`policy_engine`** builds a numeric `ContextVector` from the current
  state and asks the active policy — `HeuristicPolicy` (a fixed linear
  scorer) or `LinUCBPolicy` (a disjoint contextual bandit) — to score
  and rank the candidate critics. This is recorded for research purposes
  only; it does **not** decide routing.
- **`router`** independently decides which critic(s) actually run, via
  a simple, fixed rule (`task_type == "code" → CodeCritic`, else
  `LogicCritic`). This separation — a live routing rule the policy layer
  never touches — is what makes it safe to study adaptive policies
  without risking the pipeline's live behavior.
- **`critic` / `self_correction` / `safety`** execute the selected
  critics, apply a correction policy if warranted, and are eligible to
  loop back to `worker`.
- **`evaluation`** records the completed run as an `ExperienceRecord`,
  computes its `RewardSignal`, and terminates the graph.

Every recorded `ExperienceRecord` is consumed **only** by the offline
evaluation suite (`app/evaluation/`), never by the live graph — replay,
bootstrap evaluation, ablation studies, and learning-curve analysis all
read exclusively from this log and never execute the graph or call an
LLM.

Full architectural description, the context-feature specification, the
LinUCB update equations, and the reward function: [`docs/Methodology.md`](docs/Methodology.md).

## Repository structure

```
adaptive-critic-routing/
├── app/
│   ├── graph/            # StateGraph wiring, node implementations (nodes.py), routing (edges.py)
│   ├── state/             # Shared AgentState schema
│   ├── context/            # ContextEncoder: AgentState -> numeric ContextVector, normalization
│   ├── policy/               # HeuristicPolicy, LinUCBPolicy behind a common interface
│   ├── policy_engine/         # Policy registry / selection process used by policy_engine node
│   ├── correction_policy/      # CorrectionDecisionEngine (self-correction rule set)
│   ├── error_features/          # Error-feature extraction
│   ├── critics/                  # Critic result models + aggregation (placeholder evaluators)
│   ├── experience/                 # ExperienceRecord, ExperienceRecorder, repository
│   ├── reward/                      # RewardCalculator (weighted + quality-only strategies)
│   ├── metrics/                      # ExecutionMetrics summary
│   ├── evaluation/                    # Offline evaluation suite (see below)
│   │   ├── offline/                      # ReplayEngine, OfflineEvaluator, Benchmark
│   │   ├── experiments/                   # Bootstrap-based ExperimentRunner
│   │   ├── statistics/                     # Paired-test Analyzer, confidence intervals, effect size
│   │   ├── ablation/                        # AblationRunner (7 configurable ablations)
│   │   └── learning_analysis/                # LearningAnalyzer (regret, convergence, learning rate)
│   ├── api/, config/                          # FastAPI app scaffold, pydantic-settings config
│   └── agents/, models/, memory/, prompts/,     # Reserved for future implementation
│       router/, utils/                           # (empty packages; see note below)
├── docs/                  # The manuscript — one chapter per file; see docs/README.md
├── experiments/
│   ├── campaign_v1/         # Earlier pilot campaign, superseded, not referenced by the manuscript
│   └── campaign_v2/          # THE campaign reported in the manuscript — see experiments/campaign_v2/README.md
├── tests/                       # pytest suite (63 files)
├── datasets/                       # Reserved for locally cached datasets (empty)
├── main.py                            # FastAPI entrypoint (GET /health only)
├── pyproject.toml                        # Dependencies, tool config (uv, ruff, pytest)
├── LICENSE                                  # MIT
└── .env.example                                # Environment variable template
```

`app/policies/` also exists alongside `app/policy/` — an earlier,
data-contract-only design iteration (`AdaptivePolicy` model) that
predates the `app/policy/` + `app/policy_engine/` implementation this
manuscript describes. It is still exercised by one test file but is not
part of the live pipeline; treat `app/policy/`/`app/policy_engine/` as
authoritative.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repository>
cd adaptive-critic-routing
uv sync                    # creates .venv, installs runtime + dev dependencies
cp .env.example .env       # only needed to run the FastAPI app; not required for tests or experiments
```

> **Note:** if your active Python is not 3.12.x, `uv sync` will fetch
> and use an appropriate interpreter itself (uv manages this); you do
> not need to separately install Python 3.12 first.

## Quick start

```bash
# Run the test suite
uv run pytest

# Lint
uv run ruff check .

# Start the (minimal) API — exposes GET /health only; no LLM-backed
# endpoints exist yet
uv run uvicorn main:app --reload
```

Expected test outcome: 1078 passed, 1 failed. The one failure
(`test_pipeline_integration.py::test_compiled_graph_runs_implemented_nodes_then_stops_at_unimplemented_routing`)
requires `langgraph`, which is only installable under the pinned Python
3.12 environment (`requires-python = ">=3.12,<3.13"` in
`pyproject.toml`); it is not a code defect.

## Reproducing the experiments

Every number and figure in the manuscript comes from one script:

```bash
uv run python experiments/campaign_v2/run_campaign.py
```

This is fully seeded and deterministic (`DATA_SEED`, `CAMPAIGN_SEED` in
the script) — re-running it regenerates
`experiments/campaign_v2/results/` and `experiments/campaign_v2/figures/`
identically. It touches no LLM, no external service, and no file
outside `experiments/campaign_v2/`. Runtime is under a minute on a
typical laptop; NumPy, SciPy, and Matplotlib (installed via `uv sync`)
are the only non-trivial dependencies it exercises.

See [`experiments/campaign_v2/README.md`](experiments/campaign_v2/README.md)
for what each output file is, and **do not** read anything under
`experiments/campaign_v2/archive/` as current — that directory holds
pre-revalidation artifacts kept only for audit purposes (see its own
README).

## Manuscript structure

The paper lives in [`docs/`](docs/) as one Markdown file per chapter —
see [`docs/README.md`](docs/README.md) for the full reading order,
chapter-to-file map, and a list of five still-open placeholders
(two algorithm listings, one architecture figure, one weight table,
one learning-curve figure) that remain before this is submission-ready.

| Chapter | File |
|---|---|
| Abstract | [`docs/Abstract.md`](docs/Abstract.md) |
| 1. Introduction | [`docs/Introduction.md`](docs/Introduction.md) |
| 2. Related Work | [`docs/Related_Work.md`](docs/Related_Work.md) |
| 3. Methodology | [`docs/Methodology.md`](docs/Methodology.md) |
| 4. Experimental Setup | [`docs/Experimental_Setup.md`](docs/Experimental_Setup.md) |
| 5. Results | [`docs/Results.md`](docs/Results.md) |
| 6. Discussion | [`docs/Discussion.md`](docs/Discussion.md) |
| 7. Threats to Validity | [`docs/Threats_to_Validity.md`](docs/Threats_to_Validity.md) |
| 8. Conclusion | [`docs/Conclusion.md`](docs/Conclusion.md) |
| References | [`docs/References.md`](docs/References.md) |

## Citation

The manuscript in `docs/` is not yet published or submitted; please do
not cite it as a formal publication. If you build on this codebase,
please cite the repository itself:

```bibtex
@software{acrf2026,
  title  = {Adaptive Critic Routing Framework (ACRF)},
  author = {{ACRF Contributors}},
  year   = {2026},
  version = {1.0},
  url    = {<repository URL>}
}
```

Every external work this project itself relies on or discusses is
listed, in IEEE format, in [`docs/References.md`](docs/References.md).

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

ACRF is built directly on top of
[LangGraph](https://github.com/langchain-ai/langgraph),
[FastAPI](https://github.com/fastapi/fastapi),
[Pydantic](https://github.com/pydantic/pydantic),
[NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and
[Matplotlib](https://matplotlib.org/) — the LinUCB implementation,
statistical-comparison procedure, and every figure in the manuscript
depend directly on the latter three. [LiteLLM](https://github.com/BerriAI/litellm)
and [ChromaDB](https://github.com/chroma-core/chroma) are integrated as
dependencies for a future generation/memory layer not yet exercised by
this research.
