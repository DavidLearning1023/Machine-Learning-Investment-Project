# Machine-Learning Investment Project

**Weekly cross-sectional return prediction and long-only portfolio construction on an 80-asset panel — one pipeline from data integrity checks to frozen out-of-sample weights.**

**David Hui** · October 2026 · MIT

> **Simulated returns, real methodology.** The panel is a real Hong Kong
> large-cap universe carrying **simulated** returns and signals, which is why it
> cannot be redistributed. Asset identifiers and industry buckets are anonymised
> throughout this repository. No figure below is a market result or a
> live-performance record — the deliverable is the method and its diagnostics.
> See [`data/README.md`](data/README.md) for the full provenance, including the
> evidence that distinguishes simulated returns from observed ones.

---

## What it does

A weekly panel of 80 assets × 260 forecast dates is split **by date, never
shuffled** into 78 training weeks, 26 validation weeks and a 156-week evaluation
window whose outcomes are withheld. On it the pipeline:

1. verifies data integrity and sets a zero-excess-return prediction benchmark;
2. fits OLS and a validation-tuned LASSO (100-point log penalty path);
3. fits a random forest selected on a 6-point complexity ladder;
4. measures importance by drop-to-zero OOS R² on the frozen forest;
5. scores 33 long-only top-K rules on a **three-window robust composite**
   (Sharpe / information ratio / drawdown) — a rule must look good in the full
   validation block *and* in each of its two halves;
6. re-scores the whole rule family per window to check stability;
7. refits the winner once on weeks 1–104, freezes it, and writes 156 weekly
   weight vectors for the evaluation window.

## Key results

**Prediction (validation weeks 79–104, MSE in squared decimal returns)**

| Model | Training MSE | Validation MSE | Validation OOS R² |
| --- | ---: | ---: | ---: |
| Zero forecast | 0.0031939 | 0.0030909 | 0.0000 |
| OLS (17 predictors) | 0.0031130 | 0.0031545 | −0.0206 |
| LASSO (α = 0.002801) | 0.0031552 | 0.0030640 | **+0.0087** |
| Random forest (depth 2, leaf 60) | 0.0029415 | 0.0032194 | −0.0416 |

Only LASSO beats the zero forecast out of sample. The forest has the lowest
training error and the worst validation error — a textbook overfit, and a
reminder that OOS R² is a fit diagnostic, not a P&L.

**Portfolios (weekly, 26 validation weeks)**

| Model / rule | Weekly long-short spread | Long-only Sharpe | IR vs equal weight | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| OLS / top 16 | +0.0220 | 4.45 | 7.87 | −0.0477 |
| LASSO / top 16 | +0.0209 | 4.41 | 10.31 | −0.0478 |
| Random forest / top 16 | +0.0244 | 4.90 | 9.87 | −0.0426 |
| **Random forest / top 14 — selected** | — | **5.15** | **10.19** | **−0.0402** |
| 80-stock equal weight | — | 1.78 | 0.00 | −0.0777 |

**The two rankings disagree.** LASSO wins the prediction table and loses the
portfolio table: a top-K rule uses only the cross-sectional *order* of forecasts
within a week, so scaling every forecast by the same constant would leave the
holdings unchanged. What separates them is model form, not forecast dispersion.

The selected rule scored 17.13 on the three-window composite against 15.60 for
the runner-up, and its weakest window score (15.09) was the highest worst-window
score of any candidate — the win does not rest on one lucky stretch of weeks.

## Figures

| | |
| --- | --- |
| ![Training correlation matrix](outputs/figures/fig1.png) | **Figure 1.** Training correlation matrix of the 17 predictors (training rows only); cells with \|r\| ≥ 0.25 are labelled. |
| ![LASSO penalty path](outputs/figures/fig2.png) | **Figure 2.** Validation MSE against the LASSO penalty (left axis) and the number of retained signals (right axis). The selected point sits in a flat region, not a knife edge. |
| ![Drop-to-zero importance](outputs/figures/fig3.png) | **Figure 3.** Five largest signed drop-to-zero importance scores of the frozen forest. |
| ![Response profile](outputs/figures/fig4.png) | **Figure 4.** Average response of the most important predictor across its training 10th–90th percentiles — piecewise constant, the signature of tree thresholds. |

## Repository layout

```
src/pipeline.py              7-stage pipeline (the whole study, end to end)
data/make_synthetic_data.py  generator for a structurally identical panel
data/README.md               schema, splits and data provenance
outputs/results.json         every number in the tables, machine-readable
outputs/tables/              one CSV per result table
outputs/figures/             the four figures above
outputs/portfolio_weights.csv  156 weekly weight vectors (evaluation window)
docs/report.md               full technical report
```

## Data

The study panel **cannot be redistributed**, so this repository ships a
generator instead: `make_synthetic_data.py` writes a panel with the identical
schema, splits, weekly calendar, industry-bucket sizes, within-week-constant
market signals and a weak true predictive structure. See
[`data/README.md`](data/README.md) for the full schema and provenance.

> **What you will and will not reproduce.** The tables, figures and
> `outputs/portfolio_weights.csv` committed here are the **study run on the
> original panel**. Running the generator plus the pipeline reproduces the code
> path, the assertions and the shape of every artefact — but not the numbers:
> the generator's predictive structure is deliberately different, so it selects
> a different portfolio rule and reports different ratios. Treat the run as a
> check that the method executes end to end, not as a check on these figures.

## Quickstart

```bash
pip install -r requirements.txt

# reproduce the pipeline on a generated panel
python data/make_synthetic_data.py
python src/pipeline.py

# or run it on a panel of your own
python src/pipeline.py --data /path/to/panel.csv
# MLIP_DATA=/path/to/panel.csv python src/pipeline.py
```

Running the pipeline **overwrites `outputs/`** with the results for whichever
panel it was given. Clone first if you want to keep the committed study run.

The pipeline writes `outputs/` (tables, figures, `results.json`,
`portfolio_weights.csv`) and a `run_log.txt` with the full trace. It is
deterministic: seed 2026 for the forest, deterministic solvers elsewhere.
Verified on Python 3.13.14 with numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1
and matplotlib 3.11.2.

## Limitations

- **Simulated returns.** The panel's returns and signals are simulated on a real
  universe, so no inference about real markets can be drawn from these numbers.
- **26 validation weeks.** The portfolio rule was chosen on 26 weekly
  observations, so the annualised ratios are optimistic.
- **No costs.** Transaction costs, borrow costs and liquidity constraints are
  ignored; the rules rebalance weekly.
- **Sector concentration.** The 10% single-name cap imposes no sector cap.
  Across the 156 evaluation weeks the largest sector allocation averages 62.2%,
  exceeds 80% in 17 weeks and reaches 100% in 2.
- **Single metrics disagree.** The Sharpe-maximising rule is random forest /
  top 10, the IR-maximising one is random forest / top 20, and the shallowest
  drawdown is OLS / top 11. Selection is therefore made on a composite and the
  disagreement is reported rather than hidden.

## License

MIT — © 2026 David Hui. See [LICENSE](LICENSE).
