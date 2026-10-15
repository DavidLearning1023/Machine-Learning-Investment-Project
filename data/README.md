# Data

## Provenance

The study panel **cannot be redistributed**, so it is not in this repository.
What that panel actually contains is worth stating precisely, because the answer
shapes how every number in `outputs/` should be read:

| Layer | Real or simulated? |
| --- | --- |
| Returns (`ret_next_week`, `rf_weekly`) | **Simulated.** Not observed market data. |
| All 17 predictors | **Simulated.** Constructed signals, not historical factor exposures. |
| Universe of 80 assets | **Real.** A real Hong Kong large-cap equity universe. |
| Industry buckets | **Real** classification, relabelled here. |
| Weekly calendar | **Real** week dates, Fridays, 2021-07-09 → 2026-06-26. |

In the study panel the assets carried exchange tickers, company names and
industry classifications. None of that is redistributed here: throughout this
repository tickers and company names appear as `asset_01`…`asset_80` and
`Asset 01`…`Asset 80`, and the four industry buckets appear as
`Sector_A`…`Sector_D`.

**Why the returns count as simulated.** Three properties rule out observed
weekly equity returns: excess kurtosis of 0.21 (observed weekly returns run
3–6), mean lag-1 autocorrelation of −0.03 (observed returns cluster in
volatility), and a cross-sectional dispersion that barely moves over five years
(0.035–0.058, with no stress periods). The 0.35 average correlation *between*
assets is a deliberately injected common factor, not an estimated market beta.
The practical consequence: no performance figure in this repository is a market
result or a live record, and none of the numbers carry investment meaning.

## Reproducing the pipeline

This folder contains a generator that produces a panel with the identical
schema — same splits, same predictor structure, same four industry buckets with
the same sizes (50 / 13 / 11 / 6), and the same weekly calendar start:

```
python data/make_synthetic_data.py      # writes data/panel.csv  (~6.3 MB)
```

`data/panel.csv` is git-ignored. When present, `src/pipeline.py` picks it up
automatically; otherwise point the pipeline at any file with the schema below
via `--data` or the `MLIP_DATA` environment variable.

> **The generator reproduces the *pipeline*, not the study's conclusions.**
> The tables, figures and `outputs/portfolio_weights.csv` in this repository are
> the study run on the original panel. Running the generator and the pipeline
> reproduces the code path, the assertions and the structure of every artefact —
> but its predictive structure is deliberately different, so it selects a
> different portfolio rule and reports different ratios. The honest summary is:
> the method reproduces, the numbers do not.

## Schema

One row = one asset × one forecast week. 260 weeks × 80 assets = 20,800 rows.

| Column | Role | Description |
| --- | --- | --- |
| `week` | identifier | Forecast week, 1–260 |
| `forecast_date` | identifier | Weekly date (Friday); a real calendar week, not an observation |
| `set` | split | `training` (weeks 1–78), `validation` (79–104), `private_evaluation` (105–260) |
| `asset_id` | identifier | Asset id 1–80; stable across weeks; not a model input |
| `ticker` | identifier | Anonymised label `asset_01`…`asset_80`; not a model input |
| `company_name` | identifier | Anonymised label `Asset 01`…`Asset 80`; not a model input |
| `official_sector` | identifier | Anonymised industry bucket `Sector_A`…`Sector_D`; used for portfolio summaries only |
| `ret_next_week` | outcome | Next-week total simple return; blank in the evaluation window |
| `rf_weekly` | adjustment | Known weekly risk-free return; subtracted from the total return |
| 17 predictors | inputs | 6 market-level + 10 asset-level + 1 sector-level simulated signals |

### Predictors

| Group | Columns | Behaviour |
| --- | --- | --- |
| Market-level (6) | `market_trend`, `market_risk`, `market_sentiment`, `liquidity_state`, `rate_pressure`, `policy_uncertainty` | One value per week, identical across the 80 assets — no cross-sectional ordering power on their own |
| Asset-level (10) | `momentum_short`, `momentum_medium`, `reversal_signal`, `risk_signal`, `turnover_signal`, `size_signal`, `value_signal`, `quality_signal`, `beta_signal`, `illiquidity_signal` | Vary across assets within every week |
| Sector-level (1) | `sector_trend` | Shared by the assets of the same industry bucket within a week |

### Target

```
y = ret_next_week - rf_weekly        # weekly excess return, decimal
```

The outcome is already aligned to its forecast row (no shift). `y` is only
defined for the training and validation splits; the 156-week evaluation window
carries predictors but withheld outcomes.

## Splits and discipline

- The split is **by date, never shuffled**: each week contributes all 80 assets
  to exactly one split, so a model never sees part of a week's market state and
  is evaluated on the rest.
- The pipeline fits its scaler on training rows only; every hyper-parameter
  (LASSO penalty, forest depth/leaf, portfolio rule K) is selected on the
  validation weeks 79–104; the evaluation window is used exactly once, to
  produce the final frozen weights.

## Using your own panel

Any CSV with the schema above works. `src/pipeline.py` asserts: 20,800 rows,
80 assets per week, 260 weeks, the six market predictors constant within each
week, and outcomes present in exactly the first 104 weeks.
