# Michelin Taiwan: Starred-vs-Not "Explain" Classifier + Predictor & Static Dashboard — Plan

## Context

Build a Taiwan-only Michelin analysis centered on a binary **"Starred" vs "not Starred"**
classifier, delivered as a fully static GitHub Pages **map site** with **in-browser ONNX
inference** (the analytical panels live in an external write-up, not on the site).
Latency is the steadfast priority; trading some absolute accuracy for low latency is explicitly
acceptable.

**Class definition (central — read this first):**
- **Class 1 = Starred** — a restaurant that holds 1, 2, or 3 Michelin Stars.
- **Class 0 = not Starred**, drawn from two sub-populations:
  - **Ordinary non-Guide restaurants** — a large sample from a general source (Google Places /
    Yelp / OpenStreetMap, etc.); the broad, *easy* negatives.
  - **Recognized-but-not-starred** — every **Bib Gourmand** and **Selected Restaurants** entry; a
    **hard, high-quality negative set**. These are good enough to be *in* the Guide yet did not
    earn a star, so they force the model to learn what separates *starred* from *merely
    recognized* — not just *starred* from *random*.

**Map display — two groups, scored and labeled differently (read with the class definition):**
The map shows two distinct groups, distinguished in the data and in the UI. This refines (does not
undo) the out-of-fold (OOF) scoring of the modeling rows; it adds a genuinely out-of-sample display
pool alongside them.
- **In-sample group** — the ~10k bounded modeling rows (all Taipei Starred + all Bib/Selected + the
  bounded ordinary non-Guide pool). Every one was used to train/evaluate.
- **Out-of-sample group** — ordinary non-Guide Taipei restaurants *outside* the ~10k, never used in
  training or evaluation. By construction this group is **entirely unlabeled ordinary restaurants**:
  all Guide rows (Starred and Bib/Selected) already sit in the modeling sample, so there are **no
  out-of-sample positives**.

The two groups serve different purposes:
- **In-sample = the evaluation surface (known labels).** Every reported metric — 2×2 confusion
  matrix, PR-AUC, calibration, per-class precision/recall, the Bib/Selected slice — comes from
  **out-of-fold (OOF) scores on this group only**.
- **Out-of-sample = a demonstration surface only.** Live ONNX inference produces a genuinely
  out-of-sample P(Starred), but there is no ground truth, so these predictions are **never
  evaluated and never feed any metric**. The UI presents them as "the model's judgment on an unseen
  restaurant," not as validated.

**Scoring rule per group (explicit):**
- **In-sample row →** displayed score is its **OOF probability** (scored by the fold that did not
  train on it). The final all-data model's in-sample score for these rows is never displayed as
  honest.
- **Out-of-sample row →** displayed score is the **live ONNX inference output directly**; it is
  honestly out-of-sample because the model never saw the row. No OOF needed.

Two distinct pipelines:
- **Explanation pipeline (primary):** train the Starred-vs-not classifier on the **same feature set
  available for any restaurant anywhere** — price level, cuisine, average rating, number of reviews,
  city, neighborhood density, and optionally text features derived from reviews. The model learns to
  separate the classes; its purpose is **explanatory, not predictive**.
- **Prediction pipeline (separate):** a lightweight **GBM or small neural network**, exported and
  served via **ONNX** for the interactive in-browser scoring.

- **Feature-importance interpretation (canonical caveat):** importances describe
  **over-representation / correlation, NOT causation**. *"Price is the top feature" does not mean
  inspectors reward expense — it means expensive restaurants are over-represented among starred
  restaurants relative to the not-starred population.* This caveat travels with every
  importance/explanation surface so results are never misread as causal claims about inspector
  behavior.
- **Current state:** the repo's `eda.py` is wired only to a *single-year 2021 global* Kaggle
  snapshot (`ngshiheng/michelin-guide-restaurants-2021`). The labeled Starred(1)/not-Starred(0)
  Taipei dataset the task requires **does not exist in the repo today** — so data sourcing is the
  first and highest-risk build step.
- **Product vision (clarified with the human):** the not-starred class must combine a *broad,
  unbiased* ordinary sample with the *hard* Bib/Selected negatives — so the classifier learns what
  genuinely separates starred restaurants both from the general population and from their
  closest-but-unstarred peers.
- **Intended outcome:** the GitHub Pages site is **just the interactive Taipei map** — click *or
  search* a restaurant → show its basic info; if the model deems it **similar to a Starred
  restaurant**, additionally surface the **top 5 features** that explain the resemblance. The
  analytical panels (binary **2×2** confusion matrix, global feature importance with the
  correlation-not-causation caveat, overall model-characteristic metrics) are **not on the site** —
  they move to an **external write-up** (e.g. a Medium article) that consumes the Phase 7 result
  JSON.
- **Goal & scope (decided):** this is a **portfolio / showcase** build — not a product or paper.
  The binding limit is data, not engineering (~tens of starred Taipei rows vs a much larger
  not-starred class, and the same-feature-set constraint strips Michelin-internal signals), so the
  honest ceiling is modest and the project's *value is its rigor and candor*. The decision points
  below are therefore **resolved for a scoped v1** (existing data only · Taipei-only · bounded
  sample); the broader ambitions are **parked as future work**, not deleted. Headline framing stays
  "how I avoided fooling myself, and exactly how strong/weak the signal is" — never a crystal ball.

## Resolved with the human (not open)

- **Labeled universe:** class 1 = **all Taipei Starred restaurants** (1/2/3 Stars); class 0 =
  **not Starred**, combining (a) a **large sample of ordinary non-Guide Taipei restaurants** from a
  general source and (b) **all Taipei Bib Gourmand + Selected Restaurants** as a hard, high-quality
  negative set. Sample breadth/source surfaced as decision points below (tied to the class-0 source
  choice), not hard-coded.
- **Confusion matrix shape:** **binary 2×2** — predicted {Starred, not} × actual {Starred, not}.
  (Matrix shape is independent of the threshold value.)
- **Core feature constraint:** the feature set is restricted to **signals available for any
  restaurant anywhere** (geo/cuisine/price/ratings/review-volume + optional review text). This is
  what makes all three sub-populations (starred, ordinary, Bib/Selected) comparable. Michelin-
  internal fields (the guide's Description/Facilities text) exist only for listed rows; using them
  would trivially separate the classes and destroy the explanatory value. "Not in any guide" is a
  valid feature value (e.g., prior-award = none). **Beyond a shared schema, every class must share
  feature *provenance*: each field is sourced from the same provider for starred, Bib/Selected, and
  ordinary rows, so a column's presence/missingness can never proxy the class (see decision #8).**

## Resolved decisions (v1 — portfolio scope)

Each previously-open decision is now resolved for a buildable v1. The options + tradeoffs are
kept for context; the **v1:** line states the choice. All are easy to flip later.

**1. Label definition** — how class membership is operationalized.
- **Starred vs not** (binary; the not-class folds in ordinary non-Guide *and* Bib/Selected as hard
  negatives). · In-Guide vs not. · Star tier (1/2/3). · Multi-class star level.
- Tradeoff: Starred-vs-not is the sharpest explanatory target — folding **Bib Gourmand + Selected
  Restaurants** into the negatives asks "what makes a *star*, not merely a *recognized*
  restaurant." A broader "in-Guide vs not" target would have more positives but a muddier question.
  Interacts with the labeled universe and the confusion matrix's "actual" axis.
- **v1:** binary **class 1 = Starred (1/2/3) / class 0 = not Starred** (ordinary non-Guide sample +
  Bib Gourmand + Selected as hard negatives). A **hard-negative slice** (performance on Bib/Selected
  specifically) is reported as a separate honest eval, not a separate model.

**2. Train/test split strategy** — the task is a cross-sectional Starred-vs-not-Starred separation.
- Stratified holdout (class-balanced train/test). · Stratified k-fold CV. · Grouped split by
  city/neighborhood to probe geographic generalization.
- Tradeoff: stratified k-fold is data-efficient and standard for cross-sectional classification;
  grouped splits test geographic transfer but shrink each fold. (The temporal rolling-origin
  split of the old next-year framing is parked — see Parked.)
- **v1:** **stratified holdout** (class-balanced), with **stratified k-fold CV** for metric
  stability given the modest class-1 count.

**3. Model family** — split across the two pipelines (explanation vs prediction).
- **Explanation pipeline:** an **interpretable classifier** (logistic regression / a single tree
  or GBDT read via global feature importance) — the readout is importances, with the
  correlation-not-causation caveat.
- **Prediction pipeline:** a **lightweight GBM or small neural network**, exported and served via
  **ONNX**; drives ONNX size and in-browser latency.
- Tradeoff (latency-first): logistic/linear = smallest ONNX, fastest, most interpretable; GBDT =
  strong tabular accuracy, fast tree-eval, size grows with #trees/depth; small MLP = moderate,
  needs scaling.
- **v1:** **logistic regression for the explanation readout**, and a **lightweight GBM (or small
  NN) → ONNX** for prediction — smallest/fastest ONNX, interpretable, latency-first; the GBM is
  added only after the LR skeleton + baselines pass.

**4. Decision threshold** — mapping P(Starred) → a "similar to a Starred restaurant" flag for the
dashboard.
- Fixed probability cutoff. · Quantile-based. · Calibrated probabilities (Platt/isotonic) then a
  cutoff.
- Tradeoff: a fixed cutoff is stable/interpretable but uncalibrated; calibration makes the
  probability mean a true likelihood at a small build-time cost.
- **v1:** **calibrated probabilities (Platt/isotonic) → a single "similar-to-Starred" cutoff**; when
  a clicked restaurant clears it, the dashboard surfaces the top-5 explaining features. (Conformal
  ranked shortlist / wide prediction sets are parked.)

**5. Class-0 source** — supplies the non-Guide restaurant sample and its features.
- **Google Places/Maps** (rich: rating, review count, price band — but **not redistributable** in
  a public static site → licensing blocker for baking data in). · **Yelp** (rich ratings/reviews;
  ToS restricts redistribution). · **OpenStreetMap** (free, **redistributable** ODbL; sparse:
  name/cuisine/geo, weak ratings). · Open government business / food-hygiene registry
  (authoritative coverage; weak quality signal).
- Tradeoff: signal richness vs (a) whether it can legally be **baked into a public static site**,
  (b) payload weight on the latency budget. Google Places/Yelp are rich but **build-time-only** —
  derived features may be served, raw licensed records may not.
- **v1:** **OSM (ODbL) as the bakeable class-0 base**; Google Places / Yelp used **build-time
  only** for richer rating/review/price features where licensing permits. Mass + live enrichment
  and serving raw licensed records are parked (network/licensing/static-site tradeoffs analyzed
  separately).

**6. Not-starred composition & breadth** — downstream of #5; the representativeness-vs-latency knob
for the *ordinary* sub-population only (the Bib Gourmand + Selected hard negatives are a fixed,
fully-included set, not a sampling choice).
- Maximal: every ordinary non-Guide row in the chosen base dataset. · Bounded: ordinary non-Guide
  restaurants with some online presence/reviews. · Curated ordinary sample.
- Tradeoff: maximal = most representative ordinary negatives but heaviest payload, worst imbalance
  (tens of starred rows vs tens of thousands), and a map needing WebGL clustering; narrower samples
  cut latency/payload but make the ordinary negatives less representative.
- **v1:** **all Taipei Bib Gourmand + Selected (hard negatives) + a bounded ordinary non-Guide
  Taipei sample**; the maximal ~40k ordinary pool is parked (payload / imbalance / clustering
  cost).

**7. Two-group map (in-sample evaluated vs out-of-sample demo).** The map shows the in-sample
modeling rows *and* a genuinely out-of-sample ordinary display pool (see Context). In-sample rows
are displayed with OOF scores and are the only evaluation surface; out-of-sample rows are a
live-inference demonstration surface, excluded from every metric.
- Tradeoff: a larger out-of-sample pool is a richer demo but adds payload/latency, the same
  pressure as the in-sample bound.
- **v1:** record a **bounded size for the out-of-sample display pool** (latency/payload budget
  governs it, the same constraint as the in-sample bound); the maximal pool stays parked.

**8. Feature-provenance parity (across all classes *and* the out-of-sample pool).** Every row —
Starred, Bib/Selected, ordinary in-sample, and out-of-sample — **must** be enriched through the
**identical feature pipeline / provider**. This matters most **between class 1 and class 0**:
features like `average rating` / `number of reviews` exist only on consumer platforms, so if the
guide rows (Starred + Bib/Selected) and the ordinary rows are enriched from different sources, the
*presence/missingness pattern of those fields becomes a near-perfect class label* and the model
separates classes on a data-source artifact, not restaurant characteristics. (Same risk, lower
stakes, for the out-of-sample pool: a provenance mismatch makes its live scores distribution-shift
artifacts, not judgments.)
- **v1:** enrich **all sub-populations through the same provider/pipeline** (match every guide row
  to the same rating/review/price source used for ordinary rows) with the **same feature schema
  *and* provenance**. A provenance mismatch — or class-correlated field missingness — is a build
  error, not a degraded-but-acceptable case.

**Methodological flag: geography / coverage confounder.** Taiwan guide geography expanded over
time (2018 Taipei → 2020 +Taichung → 2022 +Tainan/Kaohsiung → 2025 +New Taipei/Hsinchu). Mixing
cities makes "Starred" partly a function of *which cities the Guide covers* rather than restaurant
characteristics. Handling: restrict to a single consistently-covered city, or add a
"city-covered" indicator.
- **v1 resolution:** model **only Taipei** (consistently covered since 2018) — removes the
  confounder cleanly at the cost of a smaller sample. (The cross-year temporal aspect is parked.)

**Methodological flag: feature importance ≠ causation.** Every importance/explanation surface
(the decisions here, the Phase 2 EDA, Phase 5, and the external write-up that presents the
importance/metrics) must state that importances describe **over-representation / correlation, not
causation**. Canonical example:
*"price is the top feature" does not mean inspectors reward expense — it means expensive
restaurants are over-represented among starred restaurants relative to the not-starred
population.*

### Parked (future work — not built in v1)

Maximal ~40k ordinary non-Guide sample; mass + live external enrichment (OSM/Places/Yelp) and live
arbitrary-restaurant lookup; serving raw licensed (Google/Yelp) records; `guide.michelin.com/tw`
+ Wayback scraping; at-scale guide↔non-guide identity resolution; WebGL/clustered map. **Also
parked from the old next-year framing:** the temporal year-over-year **transition** dataset,
**rolling-origin / expanding-window** split, **conformal** ranked shortlist / wide prediction sets,
and the **3-band** likelihood output (superseded by the binary 2×2 confusion matrix). Documenting
*why* these are deferred is part of the deliverable, not a gap.

---

## Phase Overview

Strict order. No phase begins until the prior phase's Validation passes.

> **All validation in this plan is *soft*.** Every "Validation gate" cell below, every `## Validation`
> / `**Validation:**` block in the phases, and the `tests/*.py` files named throughout are **soft
> tests**: lightweight checks that enforce **functional correctness** and confirm the phase's
> **acceptance criteria** are met — just enough to gate progression to the next phase. They are
> **not** the real test suite. A **separate agent will author the REAL tests** (edge cases,
> statistical rigor, adversarial leakage/provenance checks, property-based and integration tests);
> the test files here are placeholders/stubs for that agent to flesh out. Do not treat a green soft
> check as production-grade coverage.

| Phase | Scope | Frontend? | Validation gate |
|---|---|---|---|
| 0 | Project scaffolding (repo, package, config, deps) | — | `uv sync` works; package imports; config validates |
| 1 | Data sourcing + labeled Starred(1)/not-Starred(0) dataset + out-of-sample pool | — | Dataset shape/dtype/NaN + idempotency + Starred-count reconciliation + `group` tag tests green |
| 2 | Same-feature-set matrix (leakage-safe) | — | Transform fit-on-train purity + leakage tests green |
| 3 | Labeled classification dataset + stratified split | — | Label assembly + split-boundary (no leakage, class balance) tests green |
| 4 | Model skeleton + baselines + validation protocol A–F | — | Reproducibility, init-sanity, baseline, row-independence, leakage tests green |
| 5 | Model selection, calibration, threshold, explanation importances (G) | — | Calibrated similar-to-Starred cutoff + 2×2 confusion matrix + importances (caveat) on held-out set |
| 6 | ONNX export + parity | — | ONNX↔Python parity within tolerance on a holdout batch |
| 7 | Build artifacts: map data (`docs/data/`) + analysis JSON (`reports/`, for write-up) | — | Map-served files + write-up JSON written + schema-checked |
| 8 | Static map site (Leaflet + onnxruntime-web), **map only** | yes | Two groups distinct; click/search → in-sample OOF score vs out-of-sample live score + (if similar) top-5; inference < 10 ms; no panels; fully static |
| 9 | End-to-end verification + deploy | yes | In-sample scores = OOF (metrics exclude out-of-sample); out-of-sample live score matches Python; site serves with no runtime backend |

---

# Project Scaffolding

Everything below must exist before Phase 1 begins. Nothing here is created yet — this is the
layout an execution session builds directly. The tree spans the full pipeline: data sourcing →
features → modeling → ONNX export → static dashboard.

## Repository structure

```
project-michelin/
├── pyproject.toml              # deps + package metadata (uv); add deps via `uv add`
├── uv.lock
├── README.md
├── CLAUDE.md
├── michelin-plan.md            # this plan
├── .gitignore                  # excludes data/ and cached inputs (no data committed)
│
├── config/
│   ├── pipeline.yaml           # the resolved v1 decisions live here: years, seeds, label def,
│   │                           #   split strategy, model family, thresholds, enrichment
│   │                           #   source(s), pool breadth, coverage-expansion handling
│   └── features.yaml           # feature list, encoders, scaling/imputation spec
│
├── data/                       # gitignored — never committed; raw zone is immutable
│   ├── raw/                    # append-only ingestion zone
│   │   ├── guide/              # per-year guide snapshots (Path A CSV/JSON; Path B HTML cache)
│   │   └── enrichment/         # external base list + enrichment snapshots (point-in-time)
│   ├── interim/                # resolved identities, partial panels, DLQ for bad rows
│   └── processed/              # final labeled dataset + feature matrix
│
├── src/
│   └── michelin/
│       ├── __init__.py
│       ├── config.py           # load + validate YAML into typed dataclasses/Pydantic
│       ├── pipeline.py         # orchestrates ingest → features → train → export end-to-end
│       ├── data/
│       │   ├── sources.py      # Path A loaders (michelin-my-maps / kagglehub, Wikipedia)
│       │   ├── scrape.py       # (PARKED v1) Path B fallback: guide.michelin.com/tw + Wayback
│       │   ├── enrichment.py   # v1: uniform enrichment for ALL rows (guide + ordinary), one provider; OSM base; Places/Yelp build-time
│       │   ├── identity.py     # v1: minimal matching within guide + sample (at-scale PARKED)
│       │   └── panel.py        # idempotent labeled (restaurant → class) dataset; MERGE/UPSERT; DLQ
│       ├── features/
│       │   ├── build.py        # assemble same-feature-set matrix (starred + Bib/Selected + ordinary)
│       │   ├── transforms.py   # fit-on-train scalers/encoders/imputers (pure, leakage-safe)
│       │   └── temporal.py     # cross-sectional feats: neighborhood density, tenure; label-derived (star-neighbor) computed in-fold
│       ├── model/
│       │   ├── dataset.py      # labeled-set assembly + stratified split strategy
│       │   ├── train.py        # deterministic training per config; calibration
│       │   ├── thresholds.py   # calibrated P(Starred) → similar-to-Starred cutoff
│       │   ├── conformal.py    # (PARKED v1) conformal ranked shortlist + prediction sets (MAPIE/crepes)
│       │   └── evaluate.py     # baselines + binary 2×2 confusion matrix + precision@k
│       └── export/
│           ├── to_onnx.py      # skl2onnx / onnxmltools export + quantization
│           ├── parity.py       # ONNX↔Python parity check (tolerance gate before ship)
│           └── artifacts.py    # write the browser-facing assets into docs/data/
│
├── eda.py                      # exploratory analysis (stars vs Bib, importance EDA)
├── notebooks/                  # optional Jupyter exploration
│
├── docs/                       # GitHub Pages root — the static site is the MAP ONLY
│   ├── index.html              # map shell (markup only; no panels; styling supplied separately)
│   ├── css/                    # styles (supplied later via Claude Design)
│   ├── js/
│   │   ├── main.js             # app bootstrap + wiring
│   │   ├── inference.js        # in-sample → OOF score; out-of-sample → live ONNX P(Starred) + top-5
│   │   └── map.js              # v1: Leaflet only (small bounded sample, no clustering); two groups distinct + click/search
│   ├── vendor/                 # pinned onnxruntime-web WASM + map lib (CDN is the alternative)
│   └── data/                   # the ONLY browser-served data (written by export/artifacts.py):
│                               #   v1 map needs only: model.onnx, feature_table.json.gz
│                               #   (per-row `group` tag; OOF score on in_sample rows only),
│                               #   points.geojson, explanations.json (per-restaurant top-5)
│
├── reports/                    # analysis JSON for the EXTERNAL write-up (not browser-served):
│                               #   metrics.json, confusion_matrix.json, feature_importance.json
│
└── tests/
    ├── test_panel.py           # shape/dtype/NaN, idempotency, Starred-count reconciliation
    ├── test_identity.py        # matching spot checks
    ├── test_leakage.py         # no Michelin-internal fields; class 1/0 share schema; presence/provenance ⊥ class
    ├── test_transforms.py      # fit-on-train purity (no leakage from val/test)
    └── test_onnx_parity.py     # ONNX vs Python predictions within tolerance
```

## Core modules/packages

- `src/michelin/` is the importable package; each subpackage owns one pipeline stage —
  `data/` (source → resolve identity → idempotent labeled dataset), `features/` (leakage-safe
  matrix), `model/` (labeled dataset → train → threshold → evaluate), `export/` (ONNX + parity +
  site assets). `pipeline.py` wires them; `config.py` is the typed boundary for the YAML.
- `data/` is the local-only data lake: `raw/` is immutable and append-only, `interim/` holds
  resolved/partial artifacts and the dead-letter queue, `processed/` holds the final labeled
  dataset and feature matrix. The whole folder is gitignored.
- `docs/` is the GitHub Pages root and the deployable surface. `js/` separates concerns
  (bootstrap / inference / map / panels); `docs/data/` is the single location the browser
  reads, written by `export/artifacts.py` so build output and served data never diverge.
- `tests/` mirrors the pipeline's risk points: dataset integrity & idempotency, identity
  resolution, feature-schema leakage, transform purity, and ONNX↔Python parity.

## Shared types/interfaces

Defined in `config.py` and the stage modules; referenced across phases:
- `Award` — enum of guide values: `"1 Star"`, `"2 Stars"`, `"3 Stars"`, `"Bib Gourmand"`,
  `"Selected Restaurants"`.
- Labeled row — `restaurant_id → class` (1 = Starred / 0 = not Starred) plus geo/cuisine/price/
  ratings and resolved identity. `Award` is retained to derive the label (Star → class 1; Bib
  Gourmand / Selected / none → class 0) and to mark the Bib/Selected hard-negative eval slice.
- `group` — `in_sample` | `out_of_sample`, tagged on every restaurant record. `in_sample` rows are
  the labeled modeling set (displayed with OOF scores; the only evaluation surface); `out_of_sample`
  rows are the unlabeled display pool (scored live; excluded from all metrics).
- Class + similar flag — the binary label space `{Starred, not}` for the 2×2 confusion matrix,
  plus a `similar-to-Starred` flag from the calibrated cutoff that drives the dashboard top-5
  (independent of the threshold value).
- Typed config dataclasses/Pydantic models loaded from `config/pipeline.yaml` and
  `config/features.yaml` (no magic numbers in modules).

## Configuration

- `config/` holds every switchable choice, including the six (now-resolved) v1 decisions, so they
  are set in one place rather than hard-coded in modules.
- `config/pipeline.yaml` — one global seed, label definition (Starred vs not), split strategy,
  model family (explanation + ONNX prediction), similar-to-Starred threshold, class-0 source(s),
  not-starred composition/breadth, geography handling (Taipei-only).
- `config/features.yaml` — feature list, encoders, scaling/imputation spec.

## Environment setup

This project uses `uv` for dependency management with Python 3.14.

```bash
# Install dependencies
uv sync

# Run a script
uv run python eda.py

# Add a dependency
uv add <package>
```

## Dependencies

- Core: `kagglehub`, `pandas`/`polars`, `numpy`, `scikit-learn`, `lightgbm`/`xgboost`,
  `pydantic`/`pyyaml` (typed config), `pytest`.
- Calibration: scikit-learn (Platt/isotonic). Uncertainty: `MAPIE` or `crepes` (conformal —
  parked v1).
- Export: `skl2onnx` (sklearn/linear), `onnxmltools` (LightGBM/XGBoost), `onnxruntime`
  (Python-side parity).
- Browser (vendored into `docs/vendor/`, CDN is the alternative): `onnxruntime-web`, Leaflet.
- Add every dependency via `uv add` so `pyproject.toml`/`uv.lock` stay authoritative.

## Initial file creation

0.1. Initialize the package layout under `src/michelin/` with the subpackages and `__init__.py`
files shown in the tree.
0.2. Create `config/pipeline.yaml` and `config/features.yaml` and author `config.py` to load +
validate them into typed dataclasses/Pydantic.
0.3. Create `.gitignore` excluding `data/` and cached inputs (no data committed).
0.4. Create empty stage modules (`data/`, `features/`, `model/`, `export/`) and the `tests/`
files as stubs.
0.5. Create the `docs/` shell (`index.html`, `js/`, `css/`, `vendor/`, `data/`) as empty
placeholders to be filled in Phases 7–8.

**Validation:** `uv sync` succeeds; `import michelin` works; `config.py` loads both YAML files
and raises on a missing/invalid key.

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

---

# Phase 1: Data Sourcing & Labeled-Dataset Construction

## Objective

Build the Michelin Taiwan labeled **Starred(1) / not-Starred(0)** dataset restricted to **Taipei**
using **Path A (existing/published) only** (Path B scraping is parked). Class 1 = Taipei **Starred**
restaurants (1/2/3 Stars; locate existing published data first; scraping is the fallback). Class 0 =
**not Starred** = a bounded sample of ordinary non-Guide Taipei restaurants from a general source
**plus** all Taipei **Bib Gourmand + Selected Restaurants** (the hard, high-quality negatives).
Produce an idempotent labeled dataset whose class-1 (Starred) counts reconcile against published
starred totals.

**Prerequisites:** Project Scaffolding complete (`config.py` loads YAML; package imports).

## Deliverables

```
src/michelin/data/
  sources.py        # Path A loaders (michelin-my-maps / kagglehub, Wikipedia)
  enrichment.py     # uniform enrichment for ALL rows (guide + ordinary), one provider; OSM base; Places/Yelp build-time
  identity.py       # v1: minimal matching within guide + sample
  panel.py          # idempotent labeled (restaurant → class) dataset; MERGE/UPSERT; DLQ
data/raw/guide/       # guide snapshots, Star + Bib + Selected (Path A CSV/JSON)
data/raw/enrichment/  # ordinary non-Guide sample snapshots
data/interim/         # resolved identities, partial datasets, DLQ for bad rows
data/processed/       # final labeled dataset
tests/test_panel.py
tests/test_identity.py
```

## Implementation Steps

1.1. **Path A — existing / published (primary; Guide rows).** In `sources.py`:
   - Load `ngshiheng/michelin-my-maps` ("Michelin Guide Awards Historical Database,
     2019–present"; also mirrored on Kaggle) — includes the **Award** type (Star / Bib Gourmand /
     Selected) plus geo/price/cuisine for every Guide row. **Starred → class 1; Bib Gourmand +
     Selected → class 0 hard negatives.** Verify whether it is a true per-year panel or
     current-state with a first-seen year.
   - Load Wikipedia "List of Michelin-starred restaurants in Taiwan" + news archives — cross-checks
     the Taipei starred lists.
1.2. **Class-0 ordinary sample + uniform enrichment** (`enrichment.py`): pull a bounded sample of
   ordinary non-Guide Taipei restaurants from a general source — OSM (ODbL, bakeable) as the base,
   with Google Places / Yelp used build-time-only for richer rating/review/price features where
   licensing permits. **Enrich every row — Starred, Bib/Selected, and ordinary — through the
   *identical* provider/pipeline** (match the guide rows from 1.1 to the same rating/review/price
   source used for ordinary rows), so no field's presence or missingness can proxy the class
   (decision #8). Together with the Bib/Selected hard negatives from 1.1, the ordinary rows form
   class 0.
   - **Out-of-sample display pool:** also produce a **bounded** pool of ordinary non-Guide Taipei
     restaurants *outside* the modeling sample, built with the **same feature schema *and* the same
     provenance** as the modeling rows (decision #8). This pool is never used in training or
     evaluation — it is the live-inference demonstration surface (decision #7).
1.3. **Filter to Taipei** (geography-confounder resolution) before dataset assembly.
1.4. **Identity resolution** (`identity.py`) across the guide rows and the ordinary sample via name
   + address + geo fuzzy matching (dedupe, alias table) so a Guide restaurant (starred or
   Bib/Selected) is not also duplicated as an ordinary class-0 row. v1 scope = minimal matching
   within the guide + the ordinary sample; at-scale resolution is parked.
1.5. **Labeled-dataset construction** (`panel.py`): build `restaurant_id → class` (1 = Starred /
   0 = not Starred — ordinary sample + Bib/Selected) with the shared feature columns. **Tag every
   restaurant record with a `group` field (`in_sample` | `out_of_sample`)** — the in-sample group
   is the labeled modeling set; the out-of-sample group is the unlabeled display pool from 1.2.
   - **Idempotent** ingestion (MERGE/UPSERT, never blind INSERT); raw ingestion zone is
     append-only; schema-validation failures routed to a DLQ rather than crashing the batch.
1.6. **Point-in-time note (leakage guard):** rating/review counts are current-state snapshots. The
   cross-sectional Starred-vs-not framing avoids the next-year-prediction leakage, but record the
   snapshot date and document that features are point-in-time-as-collected, not historical.
1.7. **Reconcile** class-1 (Starred) counts against published starred totals as a data-quality gate
   (e.g., Taipei 2023 ≈ 44 starred / 321 establishments; 2024 ≈ 49 / 343; 2025 ≈ 53 / 419); the
   Bib/Selected hard-negative count reconciles against the remaining establishments.

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- `tests/test_panel.py`: pytest asserts on labeled-dataset shape/dtype/NaN; idempotency (re-run
  yields identical rows); class-1 (Starred) counts reconcile to published starred totals; class
  balance and the Bib/Selected hard-negative count recorded.
- `tests/test_identity.py`: identity-resolution spot checks (no Guide row — starred or Bib/Selected
  — is duplicated as an ordinary class-0 row).

---

# Phase 2: Shared-Feature-Set Feature Engineering

## Objective

Assemble the **same-feature-set** matrix from the labeled dataset — only signals available for any
restaurant anywhere, identical across class 1 and class 0 — with **leakage-safe** transforms
(fit-on-train, transform val/test).

**Prerequisites:** Phase 1 labeled dataset in `data/processed/`.

## Deliverables

```
src/michelin/features/
  build.py       # assemble same-feature-set matrix from the labeled dataset
  transforms.py  # fit-on-train scalers/encoders/imputers (pure, leakage-safe)
  temporal.py    # label-agnostic feats (density, tenure); label-derived (star-neighbor) materialized per-fold
tests/test_transforms.py
tests/test_leakage.py
```

## Implementation Steps

2.1. **Shared feature set** in `build.py`: price level, cuisine, average rating, number of reviews,
   city/region, neighborhood (restaurant) density, and optionally text features derived from
   reviews — every signal must exist identically for class 1 and class 0. Michelin-internal fields
   (the guide's Description/Facilities text) are excluded — they exist only for Guide rows, would
   trivially separate the classes, and destroy the explanatory value.
2.2. **Derived features** (`temporal.py`): neighborhood density (label-agnostic restaurant count
   nearby) and other-guide award flag (e.g. Gault&Millau; "none" if unlisted elsewhere — **never
   the Michelin award itself**, which would leak the Starred label and isn't available for ordinary
   restaurants), tenure/newness — computed cross-sectionally per restaurant. **`star-neighbor
   density` (count of *starred* neighbors) is label-derived: it MUST be computed inside each CV fold
   from train-fold labels only (leave-one-out neighbor stats), never once over the full dataset —
   otherwise a held-out row's own/neighbors' star labels leak into its features. If fold-safe
   computation isn't in place, ship only the label-agnostic density and drop the starred variant.**
2.3. **Leakage controls** (`transforms.py`): fit scalers/encoders/imputers — **and any label-derived
   feature (e.g. `star-neighbor density`)** — on **train only inside each CV fold**, transform/score
   val/test; validate shapes/dtypes/NaN at ingestion before modeling. The global Phase 2 build emits
   only **label-agnostic** columns; anything that touches the label is materialized **per-fold**, not
   once over the full dataset.
2.4. **EDA — Starred vs not-Starred separation:** descriptive split of starred restaurants against
   both the ordinary sample and the Bib/Selected hard negatives — "what is over-represented among
   starred restaurants," and (the sharper question) "what separates a star from a merely-recognized
   Bib/Selected restaurant." **Caveat:** any importance / over-representation here is **correlation,
   not causation** — e.g. *"price is the top feature" does not mean inspectors reward expense;
   expensive restaurants are simply over-represented among starred restaurants relative to the
   not-starred population.* (The model-importance side is produced in Phase 5 from the trained
   model.)

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- `tests/test_transforms.py`: fit-on-train purity (no leakage from val/test).
- `tests/test_leakage.py`: Michelin-internal fields are absent from the feature matrix; class 1 and
  class 0 share an identical feature schema; **and each feature's presence/missingness and source
  provenance are statistically independent of class** (no column whose null-pattern or provider
  proxies the label — catches the cross-class enrichment artifact of decision #8).

---

# Phase 3: Labeled Classification Dataset & Stratified Split

## Objective

Assemble the labeled classification dataset and the **stratified** train/test split
(class-balanced, with stratified k-fold CV for stability). Target = binary **class 1 = Starred /
class 0 = not Starred** (ordinary sample + Bib/Selected hard negatives).

**Prerequisites:** Phase 2 feature matrix.

## Deliverables

```
src/michelin/model/
  dataset.py     # labeled-set assembly + stratified split strategy
```

## Implementation Steps

3.1. **Label assembly** (`dataset.py`): construct `restaurant_id → label` where label = "is Starred"
   (class 1) vs "not Starred" (class 0 = ordinary sample + Bib/Selected), joined to the Phase 2
   feature matrix.
3.2. **Split strategy:** stratified holdout (class-balanced train/test), with stratified k-fold CV
   for metric stability given the modest class-1 count. (The temporal rolling-origin split is
   parked: the task is cross-sectional, not next-year.)
3.3. **Hard-negative eval slice:** define a Bib/Selected-only slice (how often the
   recognized-but-not-starred negatives are misread as starred-similar) as a reporting view over
   the same dataset — not a separate model.

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- Label assembly + split-boundary tests green: assert no restaurant appears in both train and test;
  assert stratification preserves the class balance; assert positive label = Starred restaurant and
  Bib/Selected rows carry class 0.

---

# Phase 4: Model Skeleton, Baselines & Validation Protocol (A–F)

## Objective

Build the modeling skeleton clean and bracket it with baselines, following the ordered checklist
adapted from Karpathy's "Recipe for Training Neural Networks" to this tabular, heavily-imbalanced,
latency-first setting. Run the stages in order; do not advance until a stage's check passes. Every
check that can be automated becomes a test under `tests/`.

**Prerequisites:** Phase 3 labeled dataset + split.

## Deliverables

```
src/michelin/model/
  train.py       # deterministic training per config (skeleton; calibration added Phase 5)
  evaluate.py    # baselines + precision@k (confusion matrix added Phase 5)
```

## Implementation Steps

4.1. **A. Reproducibility & restraint — build the skeleton clean.**
   - One seed from `config/pipeline.yaml` seeds numpy / random / the model lib; the full
     train+eval run twice yields identical metrics.
   - Strip fanciness for the first pass: no class-weighting, calibration, resampling
     (SMOTE/over-/under-sampling), feature selection, or ensembling. These are regularizers added
     only after the skeleton is trusted — the tabular analog of "turn off data augmentation."
4.2. **B. A metric you can trust.**
   - Evaluate over the *entire* held-out set per the split — never per-batch losses, no
     smoothing; report log-loss / PR-AUC at full precision.
   - Track human-checkable metrics beside loss: starred-class precision/recall, the binary 2×2
     confusion matrix, and error rate on the Bib/Selected hard-negative slice.
   - Baselines to beat/match: **majority-class / base-rate** (predict the prevailing class — the
     trivial guess the model must beat) and reconciliation of predicted-positive counts vs the
     known starred count.
4.3. **C. Initialization sanity — no hockey-stick.**
   - Verify loss @ init = base-rate cross-entropy: with positive rate `p`,
     `-(p·log p + (1-p)·log(1-p))`. A zero-/randomly-initialized model must measure this.
   - Init the model's prior to `p`: logistic → intercept = `logit(p)`; GBDT → `base_score`
     (XGBoost) / `init_score` (LightGBM) = `p`. Absorbs the heavy imbalance at step 0.
4.4. **D. Baselines — bracket the model (floor & ceiling).**
   - **Input-independent floor:** zero/constant the feature matrix; the model must collapse to
     predicting `p`, and the real model (features in) must beat it on PR-AUC — proof the features
     carry signal.
   - **Overfit-a-handful ceiling:** take ~a dozen rows, drop regularization / raise capacity
     (deep unbounded trees, or enough features for linear separability), and drive train loss to
     ~0 with predictions matching labels exactly. If it can't, stop — there is a pipeline bug.
   - **Capacity sanity:** nudge capacity up on the full train set and confirm train loss drops as
     expected.
4.5. **E. Visualize the source of truth — right before `model.predict(X)`.**
   - Dump the exact feature row entering the model and decode it back to human-readable form
     (restaurant name, region, raw feature values) — the single source of truth for catching
     preprocessing / encoding / leakage bugs.
   - Prediction dynamics: track predicted probabilities for a fixed watch-list (e.g. a 3-star
     mainstay, a Bib, a never-listed eatery) across boosting rounds / training; watching them move
     reveals instability and mis-scaled inputs.
4.6. **F. Dependency & code-correctness checks — tabular analog of the backprop trick.**
   - **Row-independence:** a row's prediction must depend only on its own features — shuffle all
     *other* rows and assert the target row's score is unchanged (catches cross-row leakage from a
     bad group-by or transform fit). For differentiable models, the gradient form (non-zero grad
     only on row `i`) applies.
   - **Feature-build independence (not just predict-time):** the row-independence check above tests
     the *model* on a fixed feature vector, so it cannot catch leakage introduced when features are
     *built*. Add a build-stage check: recompute a held-out fold's label-derived features (e.g.
     `star-neighbor density`) and assert they do **not** change when the held-out rows' own labels
     are altered — i.e. they were derived from train-fold labels only.
   - **Feature-schema dependency:** a prediction must not move when a Michelin-internal field is
     introduced or a class-only column is added — the model may only use the shared feature set
     (ties to the leakage guard in Phase 2; becomes `tests/test_leakage.py`).
   - **Generalize a special case:** write the explicit per-restaurant feature computation as a loop
     first, verify by hand on a few rows, then vectorize the dataset build and assert identical
     output.

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- Deterministic re-train reproduces metrics (stage A).
- Loss @ init equals base-rate cross-entropy (stage C).
- Input-independent floor collapses to `p`; real model beats it on PR-AUC; overfit-a-handful
  reaches ~0 train loss (stage D).
- Row-independence, feature-build independence (label-derived features recomputed per fold), and
  feature-schema dependency tests green (stage F).

---

# Phase 5: Model Selection, Calibration, Thresholds & Conformal Sets (G)

## Objective

Pick the models (stage G) — the interpretable **explanation** classifier and the lightweight
**prediction** model (GBM/NN → ONNX) — then turn calibrated probabilities into the
similar-to-Starred flag, and compute the binary **2×2** confusion matrix on the held-out set.
Imbalance is handled in stages: `logit(p)` init bias first (Phase 4); class-weighting /
calibration added here as regularizers.

**Prerequisites:** Phase 4 skeleton + baselines pass.

## Deliverables

```
src/michelin/model/
  train.py        # explanation classifier + prediction model; + calibration (Platt/isotonic)
  thresholds.py   # calibrated P(Starred) → similar-to-Starred cutoff
  conformal.py    # (PARKED v1) conformal ranked shortlist + prediction sets (MAPIE/crepes, LOO)
  evaluate.py     # + binary 2×2 confusion matrix
```

## Implementation Steps

5.1. **G. Overfit stage — pick the models, change one thing at a time.**
   - **Don't be a hero:** explanation = **logistic regression** (read via importances); prediction
     = a **lightweight GBM (or small NN)** before anything exotic — also the smallest/fastest ONNX,
     aligning with the latency-first priority and decision #3.
   - **Safe defaults first:** stock library hyperparameters (the tabular analog of "Adam @ 3e-4");
     do not tune before the skeleton and baselines pass.
   - **Complexify one signal at a time:** add feature groups (geo → ratings → density → …) one by
     one, confirming each expected lift before adding the next.
   - **Distrust schedule defaults:** for the GBM, treat `learning_rate × n_estimators` and early
     stopping explicitly on the stratified split; keep settings constant and tune learning-rate /
     decay dead last.
5.2. **Calibration** (`train.py`): calibrated probabilities (Platt/isotonic).
5.3. **Threshold** (`thresholds.py`): calibrated P(Starred) → a single **similar-to-Starred** cutoff
   that drives the dashboard's top-5 trigger.
5.4. **Conformal (PARKED v1)** (`conformal.py`): conformal ranked shortlist + wide prediction sets
   (MAPIE / `crepes`, leave-one-out) — retained as a stub; the v1 dashboard uses the calibrated
   similar-to-Starred cutoff instead.
5.5. **Evaluate** (`evaluate.py`): compute the binary **2×2** confusion matrix on the held-out set
   per the split; report the Bib/Selected hard-negative slice (starred-vs-recognized) as a separate
   honest eval.
5.6. **Feature importance (with the canonical caveat):** extract global importance from the trained
   explanation classifier. State explicitly that importances are **over-representation /
   correlation, NOT causation** — *"price is the top feature" does not mean inspectors reward
   expense; expensive restaurants are over-represented among starred restaurants relative to the
   not-starred population.* Also derive the **per-restaurant top-5** contributing features for the
   dashboard.

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- Calibrated similar-to-Starred cutoff produced; binary 2×2 confusion matrix computed on the
  held-out set.
- Model beats the majority-class / base-rate baseline; predicted-positive counts reconcile vs the
  known starred count.
- Feature importance (with the correlation-not-causation caveat) and per-restaurant top-5 features
  produced.

---

# Phase 6: ONNX Export & Parity

## Objective

Export the trained **prediction** model to a quantized ONNX artifact and verify ONNX↔Python parity
within tolerance before shipping. (The explanation classifier stays Python-side; only the
prediction model is served.)

**Prerequisites:** Phase 5 trained + calibrated prediction model.

## Deliverables

```
src/michelin/export/
  to_onnx.py     # skl2onnx / onnxmltools export + quantization
  parity.py      # ONNX↔Python parity check (tolerance gate before ship)
tests/test_onnx_parity.py
```

## Implementation Steps

6.1. **Export** (`to_onnx.py`): train in Python → ONNX (`skl2onnx` for sklearn/linear;
   `onnxmltools` for LightGBM/XGBoost; standard ONNX export for a small NN). Minimize opset/feature
   dimension; quantize for latency.
6.2. **Parity** (`parity.py`): verify ONNX↔Python parity on a holdout batch within tolerance.

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- `tests/test_onnx_parity.py`: ONNX vs Python predictions within tolerance on a holdout batch.

---

# Phase 7: Build Artifact Generation (map data + write-up JSON)

## Objective

Write the map's browser-facing static assets into `docs/data/` (so build output and served data
never diverge) **and** the analysis JSON into `reports/` for the external write-up. The site is
**map-only**; the metrics / confusion-matrix / feature-importance results are not served — they feed
the write-up. All heavy compute is precomputed here at build time.

**Prerequisites:** Phase 6 ONNX artifact + parity pass.

## Deliverables

```
src/michelin/export/
  artifacts.py   # write map assets into docs/data/ + analysis JSON into reports/
docs/data/       # map only: model.onnx, feature_table.json.gz, points.geojson, explanations.json
reports/         # for the external write-up: metrics.json, confusion_matrix.json,
                 #   feature_importance.json
```

## Implementation Steps

7.1. **`artifacts.py`** writes the **map-served** assets into `docs/data/`:
   - `model.onnx` (quantized prediction model).
   - `feature_table.json.gz` — a **compact precomputed feature table** for **both groups**, each row
     tagged with its `group` (`in_sample` | `out_of_sample`). **In-sample** rows carry the
     precomputed **OOF score**; **out-of-sample** rows carry **no precomputed score** (scored live
     in the browser). Columnar/Arrow or gzip/brotli JSON.
   - `points.geojson` — GeoJSON points from lat/long (present in the data), each carrying its
     `group` tag so the map can render the two groups distinctly.
   - `explanations.json` — per-restaurant **top-5** contributing features (for the map click), for
     rows in either group.
7.2. **`artifacts.py`** writes the **analysis JSON** into `reports/` for the external write-up (not
   served to the browser):
   - `metrics.json` — summary metrics describing behavior (e.g., ROC-AUC / PR-AUC, calibration,
     per-class precision/recall, Bib/Selected hard-negative error rate, base rates, sample size),
     **computed from in-sample OOF rows only**; the schema records that out-of-sample rows are
     excluded from all metrics.
   - `feature_importance.json` — global importance from the trained explanation classifier,
     **carrying the correlation-not-causation caveat as a field** (so the write-up cannot show
     importance without it).
   - `confusion_matrix.json` — binary 2×2, **computed from in-sample OOF rows only** (schema records
     the out-of-sample exclusion).
7.3. **Latency mitigation:** optionally bake per-restaurant scores at build so the browser can
   fall back to a lookup if a device is slow (live inference retained for the interactive
   requirement).

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- Map-served files present in `docs/data/` and schema-checked (`model.onnx`, `feature_table.json.gz`
  with per-row `group` tag and OOF score on in-sample rows only, `points.geojson`,
  `explanations.json`); `docs/data/` contains **no panel/analysis JSON**.
- Write-up files present in `reports/` and schema-checked (`metrics.json`, `confusion_matrix.json`
  with the out-of-sample exclusion recorded, `feature_importance.json` with the caveat field).
- `confusion_matrix.json` and `metrics.json` match the Python evaluation from Phase 5 (in-sample
  OOF rows only; out-of-sample rows excluded).

---

# Phase 8: Static Map Site (Leaflet + onnxruntime-web), map only

## Objective

Build the fully static GitHub Pages site: the **interactive Taipei map only** (no panels), with
**in-browser ONNX inference**. The map **visibly distinguishes the two groups** (in-sample vs
out-of-sample) and states each group's meaning. The user **clicks a marker or searches by name** to
select a restaurant; show its basic info, then — **by group** — show its score: an **in-sample** row
shows its precomputed **held-out (out-of-fold) estimate**; an **out-of-sample** row is **scored
live** and shown as a model judgment on an unseen restaurant. If the score clears the
similar-to-Starred cutoff, also surface the **top 5 features** explaining the resemblance to a
Starred restaurant. The analytical panels live in the external write-up, not here. Runtime is a tiny
matmul / tree-eval; all heavy compute is precomputed. **Latency MUST be controlled to < 10 ms** for
the per-restaurant inference step (feature-vector lookup → ONNX `predict` → band); this is a hard
budget, not a target — if it cannot be met, fall back to the precomputed baked score. (Visual design
supplied separately — structure only here.)

**Prerequisites:** Phase 7 artifacts in `docs/data/`.

## Deliverables

```
docs/
  index.html     # map shell (markup only; no panels; styling supplied separately)
  js/
    main.js      # app bootstrap + wiring
    inference.js # onnxruntime-web load + per-restaurant predict → P(Starred) + top-5
    map.js       # v1: Leaflet only (small bounded sample, no clustering) + click/search
  vendor/        # pinned onnxruntime-web WASM + map lib (CDN is the alternative)
```

## Implementation Steps

8.1. **`inference.js`** — load `onnxruntime-web` (WASM + SIMD/threads) and `model.onnx`. Per click,
   **branch on the row's `group`**:
   - **`in_sample`** → display the **precomputed OOF score** from `feature_table.json.gz`, labeled
     **"held-out (out-of-fold) estimate"**.
   - **`out_of_sample`** → look up the precomputed feature vector → **run live ONNX inference** →
     P(Starred), labeled **"live estimate on an unseen restaurant (no ground truth)"**.
   If the score clears the similar-to-Starred cutoff, read the top-5 features from
   `explanations.json` and render them — for an out-of-sample row, make clear the explanation
   describes **the model's reasoning, not a verified outcome**. **The inference step (lookup → ONNX
   `predict` → band) MUST stay < 10 ms**; instrument it (`performance.now()`) and fall back to the
   baked per-restaurant score whenever a call exceeds the budget or the device is slow.
8.2. **`map.js`** — **v1 uses Leaflet** (small bounded sample, no clustering). Render
   `points.geojson`, **distinguishing the two groups** by marker treatment + a legend that states
   each group's meaning (in-sample = held-out OOF estimate with known label; out-of-sample = live,
   unvalidated model judgment). Provide a **search box (find a restaurant by name)** alongside
   marker click; both select a restaurant → show basic info + per-group score + (if similar) top-5
   explaining features. (MapLibre GL / deck.gl with clustering is parked for the maximal sample.)
8.3. **`main.js`** — app bootstrap + wiring across map and inference. (No `panels.js`: the confusion
   matrix, feature importance, and metrics are presented in the **external write-up** from the
   `reports/` JSON, not rendered on the site.)

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- Serve `docs/` locally; the map renders the two groups distinctly; **click or search** selects a
  restaurant → basic info + per-group score (in-sample OOF estimate vs out-of-sample live judgment)
  + (if similar) top-5 features. **No panels on the site.**
- **Latency gate:** the measured per-restaurant inference step (lookup → ONNX `predict` → band) is
  **< 10 ms**; assert the baked-score fallback triggers when the budget is exceeded.
- Confirm the build is fully static with no backend/network calls at runtime.

---

# Phase 9: End-to-End Verification & Deploy

## Objective

Verify the full pipeline end-to-end and confirm the static site is deployable to GitHub Pages
with no runtime backend.

**Prerequisites:** Phases 1–8 complete.

## Deliverables

- Passing end-to-end test run across data, leakage, model, inference, and site.
- `docs/` served as the GitHub Pages root.

## Implementation Steps

9.1. **Data:** pytest asserts on labeled-dataset shape/dtype/NaN; idempotency (re-run yields
   identical rows); identity-resolution spot checks; class-1 (Starred) counts reconcile to published
   starred totals.
9.2. **Leakage:** test asserting Michelin-internal fields are absent and class 1 / class 0 share an
   identical feature schema.
9.3. **Model:** deterministic re-train reproduces metrics; evaluate on the chosen stratified split;
   ONNX↔Python parity within tolerance on a holdout batch. Verify **in-sample displayed scores equal
   the Python `cross_val_predict` OOF output**, and that `metrics.json` / `confusion_matrix.json`
   reconcile to the OOF **in-sample** evaluation with **out-of-sample rows excluded**.
9.4. **Inference:** load `model.onnx` + feature table in a headless browser. For an **in-sample**
   row, assert the displayed score equals its precomputed OOF score. For a known **out-of-sample**
   restaurant, assert its **live in-browser P(Starred)** matches Python inference from the same
   exported model on the same feature vector; assert the similar-to-Starred flag matches.
9.5. **Site:** serve `docs/` locally; verify the map renders the two groups distinctly and that
   **both click and search** select a restaurant → basic info + per-group score + (if similar) top-5
   features (the site is **map-only — no panels**); confirm the build is fully static with no
   backend/network calls at runtime.

## Validation

> _Soft test — functional correctness + acceptance criteria only; the REAL suite is authored by a
> separate agent (see Phase Overview)._

- All of the above pass; in-sample displayed scores equal the OOF output and metrics reconcile with
  out-of-sample rows excluded; a known out-of-sample restaurant's live in-browser P(Starred) /
  similar-to-Starred flag matches the Python prediction; the site is confirmed fully static (no
  backend/network calls at runtime).
