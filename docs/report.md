# Machine-Learning Investment Project — Technical Report

*Weekly cross-sectional return prediction and long-only portfolio construction
on an 80-asset panel.*

**David Hui** · October 2026

> **Simulated returns, real methodology.** The panel is a real Hong Kong
> large-cap universe carrying **simulated** returns and signals, which is why it
> cannot be redistributed; asset identifiers and industry buckets are anonymised
> throughout. The figures below describe a research exercise — not market
> results and not a performance record. Provenance and the evidence that the
> returns are simulated are in [`data/README.md`](../data/README.md).

---

## 1. Overview

The study asks two questions that are often conflated:

1. **Can 17 simulated signals predict next-week excess returns out of sample?**
   They can — but barely, and only after shrinkage.
2. **Does the model that predicts best also build the best portfolio?**  It
   does not, and the reversal is sharp enough to be the main finding.

A weekly panel of 80 assets × 260 forecast dates is split by date (never
shuffled) into 78 training weeks, 26 validation weeks and a 156-week evaluation
window whose outcomes are withheld. Seven stages run end to end:

| Stage | Purpose |
| --- | --- |
| 0 | Data integrity checks |
| 1 | Target definition and the zero-forecast benchmark |
| 2 | Linear models: OLS and a validation-tuned LASSO |
| 3 | One nonlinear model: random forest, chosen on a complexity ladder |
| 4 | Importance analysis by drop-to-zero OOS R² |
| 5 | Portfolio construction: 33 long-only top-K rules scored on a three-window composite |
| 6 | Per-window re-scoring of the whole rule family |
| 7 | Refit once on weeks 1–104, freeze, write 156 weekly weight vectors |

Every hyper-parameter is selected on the 26 validation weeks. The evaluation
window is used exactly once, to produce the final weights.

## 2. Data and the zero-forecast benchmark

**Table 1. Sample counts by split**

| Split | Weeks | Stocks | Rows | Outcomes |
| --- | ---: | ---: | ---: | --- |
| Training | 78 | 80 | 6,240 | visible |
| Validation | 26 | 80 | 2,080 | visible |
| Evaluation | 156 | 80 | 12,480 | withheld |

The target is `y = ret_next_week − rf_weekly`, in decimal weekly excess returns,
already aligned to its forecast row. Over the 6,240 training rows the mean is
**−0.0010226791** and the sample standard deviation (n − 1) is **0.0565097717**.
The mean is negative, so the intercept matters and the zero forecast is not the
sample mean. Only the 17 predictors are used as inputs; asset ids, labels and
sector labels are identifiers, not features.

![Training correlation matrix](../outputs/figures/fig1.png)

**Figure 1.** Training correlation matrix of the 17 predictors (6,240 training
rows only); cells with \|r\| ≥ 0.25 are labelled.

The strongest absolute off-diagonal pair is `momentum_short` ↔
`momentum_medium` at **+0.6399936121**; the next is `size_signal` ↔
`illiquidity_signal` at −0.4421681987. Apart from those two pairs every
\|correlation\| is below 0.17, so the 17 predictors are close to orthogonal —
correlation alone does not justify deleting one.

**Why the calendar order and whole weeks are kept.** Six of the 17 predictors
are identical across the 80 stocks within a week, and `rf_weekly` is a single
weekly number. A random stock-week split would place some stocks of one week in
training and others in validation, letting the model see that week's market
state. Each week is therefore kept intact in calendar order, so the validation
block imitates an investor who only knows the past at each forecast date.

**Benchmark.** The zero forecast predicts zero excess return for every row. Its
validation MSE is **0.0030908984** (training MSE 0.0031938884; sum of squared
validation targets 6.4290686681). For every model below,
OOS R² = 1 − Σ(y − ŷ)² / Σy², so a negative value means worse than predicting
zero.

## 3. Linear models: OLS and LASSO

Both models are pooled regressions with an unpenalised intercept, fitted on all
6,240 training rows after standardising the 17 predictors with training means
and standard deviations only. OLS uses `LinearRegression`; LASSO uses `Lasso`
with a cyclic-coordinate solver, tolerance 1e-10, `max_iter` 200,000, minimising
SSE/(2n) + α·Σ\|βj\|. No shuffled cross-validation is used anywhere.

**Table 2. Prediction results (MSE in squared decimal returns; OOS R² as a decimal)**

| Model | Training MSE | Validation MSE | Validation OOS R² |
| --- | ---: | ---: | ---: |
| Zero forecast | 0.0031938884 | 0.0030908984 | 0.0000000000 |
| OLS | 0.0031130152 | 0.0031544950 | −0.0205754475 |
| LASSO (α = 0.0028011752) | 0.0031551909 | 0.0030640089 | +0.0086995595 |
| Random forest | 0.0029414926 | 0.0032193579 | −0.0415605689 |

![LASSO penalty path](../outputs/figures/fig2.png)

**Figure 2.** Validation MSE against the LASSO penalty on a log scale (left
axis). The grey dashed line is the zero forecast; the green step curve
(right axis) is the number of retained signals.

α_max = max\|X′(y − ȳ)\|/n = **0.0064710781**, computed from training data only.
Across 100 log-spaced penalties from α_max·10⁻⁴ to α_max, the lowest validation
MSE is at **α = 0.0028011752**, retaining **4 of 17 slopes**
(validation MSE 0.0030640089, OOS R² +0.0086995595). The selected point is
interior, not on a grid boundary.

**The curve is flat, and that matters.** Between α = 0.001 and 0.003 the
validation MSE moves only between 0.0030640089 and 0.0031035289 (a 1.29% range)
while the retained set shrinks from 8 to 4 signals. The ranking of penalties is
not knife-edge, and the exact penalty should not be read as economically
meaningful. One caution: the LASSO intercept equals the training mean, so at
very large penalties the model predicts that mean rather than zero.

**Table 3. Standardised coefficients (intercept reported separately)**

| Predictor | OLS | Selected LASSO |
| --- | ---: | ---: |
| *intercept* | −0.0010226791 | −0.0010226791 |
| market_trend | +0.0017780617 | 0 |
| market_risk | +0.0019947431 | +0.0000150212 |
| market_sentiment | −0.0016787078 | 0 |
| liquidity_state | −0.0005016862 | 0 |
| rate_pressure | −0.0027584277 | −0.0001411235 |
| policy_uncertainty | +0.0014924866 | 0 |
| momentum_short | −0.0008702420 | 0 |
| momentum_medium | +0.0014350886 | 0 |
| reversal_signal | +0.0006500409 | 0 |
| risk_signal | +0.0007796242 | 0 |
| turnover_signal | −0.0003453226 | 0 |
| size_signal | −0.0009922023 | 0 |
| value_signal | +0.0030151058 | +0.0004521775 |
| quality_signal | +0.0064684204 | +0.0036680868 |
| beta_signal | −0.0001335541 | 0 |
| illiquidity_signal | +0.0003153755 | 0 |
| sector_trend | −0.0002469338 | 0 |

LASSO retains `market_risk`, `rate_pressure`, `value_signal` and
`quality_signal`. Selection helps, but only slightly: OLS on all 17 predictors
has a negative validation OOS R² (−0.0205754475) while LASSO recovers
+0.0086995595. The full regression overfits 6,240 rows; shrinkage removes
enough noise to move the sign out of sample. Because the penalty was chosen on
these same 26 weeks, this improvement is not an untouched-test result.

## 4. A nonlinear model: random forest

The forest averages 300 trees, each on a bootstrap sample of training rows with
a random subset of inputs (`max_features = sqrt(17) = 4`), squared-error splits,
`min_samples_split = 2`, seed 2026. Complexity is varied over
`max_depth ∈ {2, 4, 8}` × `min_samples_leaf ∈ {30, 60}` — the ladder starts at
depth 2 because a tree needs two splits to express an interaction, which is
exactly the nonlinearity of interest.

**Table 4. Forest settings and validation MSE**

| Max depth | Min. leaf | Training MSE | Validation MSE | Validation OOS R² |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 30 | 0.0029399988 | 0.0032231428 | −0.0427850999 |
| **2** | **60** | **0.0029414926** | **0.0032193579** | **−0.0415605689** |
| 4 | 30 | 0.0025479311 | 0.0033543030 | −0.0852194460 |
| 4 | 60 | 0.0025951406 | 0.0033251511 | −0.0757879172 |
| 8 | 30 | 0.0020571795 | 0.0035064431 | −0.1344414129 |
| 8 | 60 | 0.0021962617 | 0.0033997844 | −0.0999340458 |

Minimum validation MSE selects depth 2 / leaf 60 (selected row above). Two
independent checks confirm the implementation: on the 2,080 validation rows the
forest prediction equals the mean of its 300 tree predictions, and refitting on
a positive affine map of the inputs changes no prediction (largest absolute
difference 8.7e-19). The second check covers affine rescalings only — a
non-linear monotone map can move rows across a split threshold.

**Every step up the complexity ladder lowers training error and raises
validation error**, and no setting beats the zero forecast. One structural fact
explains why: the shallow forest still places **86.2%** of its built-in split
importance on the six market predictors that are identical across the 80 stocks
within a week. Those variables cannot rank stocks within a week on their own;
a tree can only use them to choose which *other* variable does the splitting,
so the importance they carry does not translate into cross-sectional order.

## 5. Importance analysis

The selected forest is fitted once on weeks 1–78 and never refitted. For one
standardised validation predictor at a time, its whole column is set to zero —
equal to that predictor's training mean under the same scaler — and the OOS R²
is recomputed on the same 2,080 rows. Importance = original OOS R² − replaced
OOS R². Unlike the common training-fit variant, this uses validation OOS R².
Importance is predictive, not causal, and correlated signals can share credit.

Baseline OOS R² = −0.0415605689.

![Drop-to-zero importance](../outputs/figures/fig3.png)

**Figure 3.** Five largest signed drop-to-zero importance scores.

**Table 5. Five largest drop-to-zero scores**

| Rank | Predictor | Importance |
| ---: | --- | ---: |
| 1 | policy_uncertainty | +0.0086911000 |
| 2 | liquidity_state | +0.0076031920 |
| 3 | quality_signal | +0.0026298300 |
| 4 | value_signal | +0.0002336646 |
| 5 | sector_trend | +0.0001793685 |

Eight predictors carry a negative score, and they are informative: setting
`market_trend` to zero *raises* the OOS R² by 0.0274132449, so out of sample the
forest's reliance on it is a training-sample artefact. The next most negative
are `rate_pressure` (−0.010325), `market_sentiment` (−0.003303) and
`market_risk` (−0.002156).

![Response profile](../outputs/figures/fig4.png)

**Figure 4.** Average response to the most important predictor between its
training 10th and 90th percentiles. Each validation row keeps its own values for
all other inputs; predictions are averaged over the 2,080 rows.

The most important predictor is `policy_uncertainty`. Over the evaluated range
(−1.2264495953 to +1.2761109476 standardised) the mean prediction moves from
−0.0032013596 to +0.0002515380, a change of **+0.0034528976** in decimal weekly
excess return. The curve is piecewise constant with a non-monotonic segment and
a sharp rise near +1 standard deviation — the signature of tree thresholds
rather than a smooth effect. It describes this fitted model, not a causal
effect; and the forest's baseline OOS R² is negative, so interpreting its
reliance does not make it a successful forecaster.

**Comparison with LASSO.** LASSO retained 4 signals; of these **2** appear in
the forest's top five (`quality_signal`, `value_signal`). `market_risk` and
`rate_pressure` are outside the forest's top five, while `policy_uncertainty`,
`liquidity_state` and `sector_trend` are relied on by the forest but not
retained by LASSO. The overlap is partial and structural: a penalised linear
model uses a predictor only through its slope, whereas a tree can partition
stocks with it and fit a different level in each leaf. Refitting the forest
under seeds 0, 42, 123 and 9999 returns the same top three predictors
(`liquidity_state`, `policy_uncertainty`, `quality_signal`) every time, so the
ranking is not a seed artefact.

## 6. Portfolio construction

Each validation week the 80 stocks are ranked by predicted excess return
descending, ties broken by ascending asset id. Long-short equally weights the
top 16 and shorts the bottom 16; long-only equally weights the top K and holds
zero elsewhere. Sharpe = √52 × mean(r_p − r_f)/sd(r_p − r_f) and information
ratio = √52 × mean(r_p − r_EW)/sd(r_p − r_EW), both with sample standard
deviations. Maximum drawdown compounds total returns from wealth 1 and is
reported as a nonpositive number. Transaction costs are ignored.

**Table 6. Validation portfolios (weeks 79–104; decimals)**

| Model / rule | Weekly long-short spread | Long-only Sharpe | IR vs equal weight | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| OLS / top 16 | +0.0220083 | 4.4530 | 7.8654 | −0.0477 |
| LASSO / top 16 | +0.0208958 | 4.4086 | 10.3092 | −0.0478 |
| Random forest / top 16 | +0.0244138 | 4.8950 | 9.8742 | −0.0426 |
| 80-stock equal weight | — | 1.7803 | 0.0000 | −0.0777 |

**Does the best prediction model produce the best portfolio? No.** The best
validation prediction is LASSO (OOS R² +0.0087), yet the largest long-short
spread and the best long-only Sharpe both belong to the random forest, whose
OOS R² is negative (−0.0416). OOS R² measures the numerical fit of the *level*
of returns; a top-K rule uses only the cross-sectional *order* of forecasts
within a week, and scaling every forecast of a week by the same positive
constant would leave the holdings unchanged. The two models therefore differ in
the ranking itself, not in how dispersed their forecasts are: LASSO keeps 4 of
the 17 slopes and those slopes are not a common multiple of the OLS slopes,
while the forest partitions stocks on thresholds and interactions. Model form,
not shrinkage of the forecast scale, is what moves the top-K rule.

**Selecting K.** Thirty-three long-only top-K rules (K ∈ {10, 11, 12, 13, 14,
15, 16, 20, 24, 32, 40} × 3 models; a K below 10 would breach the 10% cap) are
scored with the metric weights Sharpe 8 / IR 6 / drawdown 6. To avoid picking
the K that looks best on one stretch of weeks, the weighted percentile score is
computed separately on the full validation block and on each of its two halves,
and candidates are ranked on the **mean** of the three window scores.

**Table 7. Eight highest-ranked candidates by three-window composite**

| Model / rule | Sharpe | IR | Max drawdown | Composite |
| --- | ---: | ---: | ---: | ---: |
| **Random forest / top 14** | **5.1522** | **10.1914** | **−0.0402** | **17.131** |
| Random forest / top 15 | 4.8799 | 10.3675 | −0.0452 | 15.596 |
| Random forest / top 16 | 4.8950 | 9.8742 | −0.0426 | 14.263 |
| Random forest / top 11 | 5.1993 | 9.4462 | −0.0434 | 14.222 |
| Random forest / top 10 | 5.4809 | 9.3871 | −0.0484 | 14.121 |
| OLS / top 10 | 5.1739 | 7.5019 | −0.0416 | 14.040 |
| Random forest / top 20 | 4.6504 | 11.2416 | −0.0436 | 13.939 |
| Random forest / top 13 | 4.9822 | 8.8525 | −0.0424 | 13.697 |

The winner, random forest / top 14, has composite 17.131 against 15.596 for the
runner-up, and its weakest window score (15.091) is the highest worst-window
score of any candidate — the selection does not rest on one lucky window.

**What the stability checks do not show.** Single metrics disagree: the highest
full-block Sharpe is random forest / top 10 (5.4809), the highest information
ratio is random forest / top 20 (11.2416), and the shallowest drawdown is OLS /
top 11 (−0.0395). The Sharpe-maximising K also moves between windows
(random forest / top 10 on the full block, / top 11 in the first half, LASSO /
top 13 in the second). Selection is therefore made on a composite of all three
metrics, and the disagreement is reported rather than hidden.

**Sector concentration.** Under the final rule the largest single-sector
allocation during validation is 92.86% (`Sector_A`, week 103), where
13 of the 14 holdings share the same official sector. The 10% single-name cap
imposes no sector cap.

## 7. Final weights for the evaluation window

The scaler and the selected forest are refit once on all 8,320 rows of weeks
1–104 and then frozen. For each of the 156 evaluation weeks, only that week's
17 predictors are used to score the 80 stocks, rank them and assign 1/14 to the
top 14. The model is never refitted in the evaluation window, no withheld
outcome is used anywhere, and forecasts change only because the inputs change.

The weight file holds 156 rows × 81 columns, keeps the weeks in date order, uses
eight decimal places, holds exactly 14 non-zero entries per week at 1/14 =
7.142857%, and every row sums to 1 within rounding. Files for the full
156-week evaluation window are written to `outputs/portfolio_weights.csv`.

**Material risk: concentration in common sector shocks.** Across the 156
evaluation weeks the largest sector allocation averages **62.2%**, exceeds 80%
in **17** weeks and reaches **100%** in **2**. A sector shock in one of those
weeks would be amplified relative to the 80-stock equal-weight benchmark. This
is a portfolio-level risk rather than a deliberate sector bet — the crowding may
come from sector-level signal, from sector co-movement, or from the sample
structure — and it is disclosed rather than attributed to a single cause. A
second risk is that the rule was chosen on 26 validation weeks, so the selection
is optimistic and the annualised figures do not represent a year of independent
evidence.

## 8. Reproducibility

| | |
| --- | --- |
| Main script | `src/pipeline.py` (all seven stages, every table and figure) |
| Environment | Python 3.13.14, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, matplotlib 3.11.2 |
| Reproducibility | Deterministic: seed 2026 for the forest; OLS, cyclic-coordinate LASSO and the weekly sorts are deterministic |
| Reproduction | `python data/make_synthetic_data.py && python src/pipeline.py` |
| Machine-readable results | `outputs/results.json`, one CSV per table in `outputs/tables/` |

**What "reproduction" does and does not mean here.** The study panel cannot be
redistributed, so the shipped generator writes a panel with the identical schema,
splits, weekly calendar and industry-bucket sizes but a deliberately different
predictive structure. Re-running therefore reproduces the code path, the
assertions and the shape of every artefact, but selects a different portfolio
rule and reports different ratios. The numbers in this report and in
`outputs/` are the study run on the original panel; a generated run verifies
that the method executes end to end, not that these figures recur.

Two independent checks are run inside the pipeline:

1. The zero-excess-return validation MSE is recomputed by exact summation
   (`math.fsum`) rather than by a pairwise mean, and matches to 1e-15.
2. The frozen forest's validation prediction equals the mean of its 300 tree
   predictions, and is unchanged under a positive affine remap of the inputs.

Every "best / most / only" statement in this report is read from the tables the
pipeline writes to `outputs/tables/`, not asserted by hand.

## 9. Limitations

- **Simulated returns.** The panel's returns and signals are simulated on a real
  universe, so no inference about real markets can be drawn.
- **26 validation weeks.** Rule selection is optimistic; annualised ratios are
  not a year of independent evidence.
- **No costs.** Transaction costs, borrow costs and liquidity constraints are
  ignored; all rules rebalance weekly.
- **Sector concentration**, disclosed above, is not controlled for.
- **The evaluation window is unreported.** Its outcomes are withheld, so the
  final weights are produced under the same discipline as a live deployment and
  their realised performance cannot be shown here.
