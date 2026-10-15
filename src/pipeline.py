"""Machine-Learning Investment Project -- end-to-end research pipeline.

A compact quantitative-investment study on an 80-asset panel observed over 260
weekly forecast dates. The panel's returns and signals are simulated on a real
universe and cannot be redistributed, so this repository ships a generator for
a structurally identical panel instead (see data/README.md). The pipeline runs
seven stages:

  1. Data integrity checks and the zero-forecast benchmark
  2. Linear models: OLS and a validation-tuned LASSO path
  3. A random forest selected on a small complexity ladder
  4. Drop-to-zero importance analysis of the fitted forest
  5. Portfolio construction: long-only top-K rules scored on a
     three-window robust composite (Sharpe / IR / drawdown)
  6. Robustness re-scoring of the whole rule family per window
  7. Final weights: refit on weeks 1-104, frozen, then applied to the
     156-week evaluation window whose labels are withheld

Discipline enforced by assertions: never shuffle; the scaler is fit on
training rows only; every hyper-parameter is selected on weeks 79-104;
the evaluation split is never fitted on.

Usage:
    python src/pipeline.py --data /path/to/panel.csv
    # or place the panel at data/panel.csv and run without arguments
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

def _locate_data(cli_path=None) -> Path:
    """Locate the panel CSV: --data argument, then MLIP_DATA, then data/panel.csv."""
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    env = os.environ.get("MLIP_DATA")
    if env:
        return Path(env).expanduser().resolve()
    root = Path(__file__).resolve().parent.parent
    cand = root / "data" / "panel.csv"
    if cand.exists():
        return cand
    raise SystemExit("Panel CSV not found: pass --data /path/to/panel.csv or set MLIP_DATA.")


DATA: Path = Path("data") / "panel.csv"   # reassigned in main() before use
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TAB, FIG = OUT / "tables", OUT / "figures"

PREDICTORS = [
    "market_trend", "market_risk", "market_sentiment", "liquidity_state",
    "rate_pressure", "policy_uncertainty",
    "momentum_short", "momentum_medium", "reversal_signal", "risk_signal",
    "turnover_signal", "size_signal", "value_signal", "quality_signal",
    "beta_signal", "illiquidity_signal", "sector_trend",
]
MARKET_PREDICTORS = PREDICTORS[:6]
# sklearn's max_features="sqrt" resolves to int(sqrt(n_features)); derived here
# so the logged/tabled value cannot drift from what the estimator actually uses.
N_FEATURES = int(np.sqrt(len(PREDICTORS)))

SEED = 2026
LASSO_PATH_POINTS = 100
RF_GRID = [dict(max_depth=d, min_samples_leaf=l) for d in (2, 4, 8) for l in (30, 60)]
RF_FIXED = dict(n_estimators=300, max_features="sqrt", bootstrap=True,
                criterion="squared_error", random_state=SEED, n_jobs=1,
                min_samples_split=2, oob_score=False, ccp_alpha=0.0)
TOPK_GRID = [10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 40]
CAP, TOP_N, WEEKS_PER_YEAR = 0.10, 16, 52
METRIC_WEIGHTS = {"sharpe": 8.0, "ir": 6.0, "mdd": 6.0}

LOG: list[str] = []


def log(msg: str = "") -> None:
    LOG.append(str(msg))
    print(msg)


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def oos_r2(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(1.0 - np.sum((a - b) ** 2) / np.sum(a ** 2))


def max_drawdown(returns) -> float:
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns))])
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def ann_ratio(x) -> float:
    x = np.asarray(x, dtype=float)
    sd = float(x.std(ddof=1))
    return 0.0 if sd < 1e-12 else float(np.sqrt(WEEKS_PER_YEAR) * x.mean() / sd)


# --------------------------------------------------------------------------- #
# STAGE 0 - data loading
# --------------------------------------------------------------------------- #
def load_data():
    log("=" * 78); log("STAGE 0 - DATA LOAD AND INTEGRITY CHECKS"); log("=" * 78)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["y"] = df["ret_next_week"] - df["rf_weekly"]
    assert len(df) == 20800
    for name in ["training", "validation", "private_evaluation"]:
        sub = df[df["set"] == name]
        ok = sub["ret_next_week"].isna().all() if name == "private_evaluation" else sub["ret_next_week"].notna().all()
        assert ok, name
        log(f"  {name:18s} weeks={sub['week'].nunique():3d} stocks={sub['asset_id'].nunique():3d} rows={len(sub):6d}")
    assert df["week"].nunique() == 260
    assert df.groupby("week")["asset_id"].nunique().eq(80).all()
    const = [c for c in PREDICTORS if df.groupby("week")[c].nunique().eq(1).all()]
    assert all(c in const for c in MARKET_PREDICTORS), const
    log(f"  predictors constant within a week (no cross-sectional power): {len(const)}")

    # ---- anonymise identifiers ONCE, at load time ------------------------------
    # Every identifier is replaced here so that no downstream table, log line or
    # JSON blob can ever carry a real ticker, company name or industry label.
    # The mapping is derived from the panel itself (buckets sorted, labelled
    # A, B, C, ...), so it works for any panel and is stable across runs.
    secs = sorted({str(s) for s in df["official_sector"].dropna().unique()})
    smap = {s: f"Sector_{chr(ord('A') + i)}" for i, s in enumerate(secs)}
    df["official_sector"] = df["official_sector"].astype(str).map(smap)
    df["ticker"] = df["asset_id"].map(lambda a: f"asset_{int(a):02d}")
    df["company_name"] = df["asset_id"].map(lambda a: f"Asset {int(a):02d}")
    log(f"  anonymised {len(secs)} industry buckets -> {', '.join(smap.values())}; "
        f"tickers/company names -> asset_NN / Asset NN")
    return df


# --------------------------------------------------------------------------- #
# STAGE 1 - data and prediction benchmark
# --------------------------------------------------------------------------- #
def stage1_benchmark(df):
    log(""); log("=" * 78); log("STAGE 1 - DATA AND PREDICTION BENCHMARK"); log("=" * 78)
    counts = [dict(split=n, stocks=df.loc[df["set"] == n, "asset_id"].nunique(),
                   weeks=df.loc[df["set"] == n, "week"].nunique(),
                   rows=int((df["set"] == n).sum()))
              for n in ["training", "validation", "private_evaluation"]]
    tr, va = df[df["set"] == "training"], df[df["set"] == "validation"]
    y_tr, y_va = tr["y"].to_numpy(), va["y"].to_numpy()
    corr = tr[PREDICTORS].corr()
    corr.to_csv(TAB / "training_correlation_matrix.csv", float_format="%.6f")
    up = corr.where(~np.tril(np.ones(corr.shape), k=0).astype(bool))
    pairs = up.stack().sort_values(key=lambda s: s.abs(), ascending=False)
    top_pair = (pairs.index[0][0], pairs.index[0][1], float(pairs.iloc[0]))
    zero_tr, zero_va = mse(y_tr, 0 * y_tr), mse(y_va, 0 * y_va)
    # summation-robustness cross-check of the zero benchmark: the same
    # quantity (mean of y^2 on the validation rows) computed by exact summation
    # (math.fsum) rather than by numpy's pairwise mean, so a summation artefact
    # in either direction would show up as a mismatch
    alt_va = math.fsum(float(v) * float(v) for v in y_va) / len(y_va)
    assert abs(alt_va - zero_va) < 1e-15, (alt_va, zero_va)
    log(f"  zero-MSE cross-check (exact summation): {alt_va:.12f} vs {zero_va:.12f}"
        f"  |diff|={abs(alt_va - zero_va):.2e}")
    log(f"  train mean={y_tr.mean():.10f}  sd={y_tr.std(ddof=1):.10f}")
    log(f"  strongest pair: {top_pair[0]} <-> {top_pair[1]} = {top_pair[2]:+.10f}")
    log(f"  zero-forecast: train MSE={zero_tr:.10f}  val MSE={zero_va:.10f}  sum(y^2)={np.sum(y_va**2):.10f}")
    return dict(counts=counts, train_mean=float(y_tr.mean()), train_sd=float(y_tr.std(ddof=1)),
                strongest_pair=[top_pair[0], top_pair[1]], strongest_corr=top_pair[2],
                runner_up=[pairs.index[1][0], pairs.index[1][1], float(pairs.iloc[1])],
                zero_mse_train=zero_tr, zero_mse_val=zero_va, sum_y2_val=float(np.sum(y_va ** 2)),
                zero_mse_val_fsum=float(alt_va),
                corr_csv="training_correlation_matrix.csv")


# --------------------------------------------------------------------------- #
# STAGES 2 AND 3 - linear and nonlinear prediction models
# --------------------------------------------------------------------------- #
def stage2_linear(df):
    log(""); log("=" * 78); log("STAGE 2 - LINEAR MODELS (OLS / LASSO)"); log("=" * 78)
    tr, va = df[df["set"] == "training"], df[df["set"] == "validation"]
    X_tr, y_tr = tr[PREDICTORS].to_numpy(), tr["y"].to_numpy()
    X_va, y_va = va[PREDICTORS].to_numpy(), va["y"].to_numpy()
    scaler = StandardScaler().fit(X_tr)                 # training rows only
    Z_tr, Z_va = scaler.transform(X_tr), scaler.transform(X_va)
    assert scaler.n_samples_seen_ == 6240 and int(scaler.mean_.size) == 17

    ols = LinearRegression().fit(Z_tr, y_tr)
    log(f"  OLS    train MSE={mse(y_tr, ols.predict(Z_tr)):.10f}  val MSE={mse(y_va, ols.predict(Z_va)):.10f}"
        f"  OOS R2={oos_r2(y_va, ols.predict(Z_va)):+.10f}")

    alpha_max = float(np.max(np.abs(Z_tr.T @ (y_tr - y_tr.mean()))) / len(y_tr))
    grid = alpha_max * np.logspace(-4, 0, LASSO_PATH_POINTS)
    rows, fits = [], []
    for a in grid:
        m = Lasso(alpha=float(a), fit_intercept=True, max_iter=200000, tol=1e-10,
                  selection="cyclic").fit(Z_tr, y_tr)
        # first-order optimality check for the penalised least-squares solution
        grad = Z_tr.T @ (y_tr - m.predict(Z_tr)) / len(y_tr)
        act = m.coef_ != 0
        err = np.where(act, np.abs(grad - a * np.sign(m.coef_)), np.maximum(np.abs(grad) - a, 0.0))
        kkt = float(max(err.max(), abs(np.mean(y_tr - m.predict(Z_tr)))))
        assert kkt < 1e-8, "KKT check failed"
        rows.append(dict(alpha=float(a), train_mse=mse(y_tr, m.predict(Z_tr)),
                         val_mse=mse(y_va, m.predict(Z_va)), oos_r2=oos_r2(y_va, m.predict(Z_va)),
                         nonzero=int(act.sum()), max_kkt_error=kkt))
        fits.append(m)
    g = pd.DataFrame(rows)
    sel = int(np.argmin(g["val_mse"].to_numpy()))
    assert 0 < sel < len(grid) - 1, "selected penalty sits on the grid boundary"
    lasso = fits[sel]
    g["selected"] = np.arange(len(grid)) == sel
    g.to_csv(TAB / "lasso_path.csv", index=False, float_format="%.12g")
    log(f"  alpha_max={alpha_max:.12g}  {LASSO_PATH_POINTS}-point log path")
    for _, r in g.iterrows():
        flag = "  <== selected" if r["selected"] else ""
        log(f"    alpha={r['alpha']:.10g}  valMSE={r['val_mse']:.10f}  R2={r['oos_r2']:+.8f}  nz={int(r['nonzero'])}{flag}")
    log(f"  LASSO  train MSE={mse(y_tr, lasso.predict(Z_tr)):.10f}  val MSE={mse(y_va, lasso.predict(Z_va)):.10f}"
        f"  OOS R2={oos_r2(y_va, lasso.predict(Z_va)):+.10f}  nz={int((lasso.coef_ != 0).sum())}")

    coef = pd.DataFrame({
        "predictor": ["intercept"] + PREDICTORS,
        "ols_std": np.r_[ols.intercept_, ols.coef_],
        "lasso_sel": np.r_[lasso.intercept_, lasso.coef_],
    })
    coef.to_csv(TAB / "linear_coefficients.csv", index=False, float_format="%.10g")

    # robustness remark: how flat is the validation-MSE curve, and how does the
    # retained set move? The window is expressed as a fraction of alpha_max
    # (not as absolute alphas) so it stays meaningful on any panel, and a window
    # with fewer than two grid points is reported rather than silently yielding
    # a NaN that would propagate into results.json.
    win_lo, win_hi = 0.15 * alpha_max, 0.46 * alpha_max
    span = g.loc[(g["alpha"] >= win_lo) & (g["alpha"] <= win_hi)]
    if len(span) >= 2:
        span_lo, span_hi = float(span["val_mse"].min()), float(span["val_mse"].max())
        span_pct = 100.0 * (span_hi - span_lo) / span_lo
        nz_lo, nz_hi = int(span["nonzero"].min()), int(span["nonzero"].max())
        log(f"  over alpha in [{win_lo:.6g},{win_hi:.6g}] ({len(span)} grid points):"
            f" valMSE {span_lo:.10f}..{span_hi:.10f}  (range {span_pct:.3f}%),"
            f" nonzero {nz_lo}..{nz_hi}")
    else:
        span_lo = span_hi = span_pct = nz_lo = nz_hi = None
        log(f"  alpha window [{win_lo:.6g},{win_hi:.6g}] captured only {len(span)} grid point(s);"
            f" flatness remark skipped")
    span_out = dict(lo=span_lo, hi=span_hi, pct=span_pct, nz_lo=nz_lo, nz_hi=nz_hi,
                    n_points=int(len(span)), alpha_lo=win_lo, alpha_hi=win_hi)
    return dict(alpha_max=alpha_max, selected_alpha=float(lasso.alpha),
                path=rows, Z_tr=Z_tr, Z_va=Z_va, y_tr=y_tr, y_va=y_va,
                p_ols_tr=ols.predict(Z_tr), p_ols_va=ols.predict(Z_va),
                p_las_tr=lasso.predict(Z_tr), p_las_va=lasso.predict(Z_va),
                nonzero=int((lasso.coef_ != 0).sum()),
                retained=[p for p, c in zip(PREDICTORS, lasso.coef_) if c != 0],
                val_curve_span=span_out)


def stage3_forest(lin):
    log(""); log("=" * 78); log("STAGE 3 - NONLINEAR MODEL (RANDOM FOREST)"); log("=" * 78)
    Z_tr, Z_va, y_tr, y_va = lin["Z_tr"], lin["Z_va"], lin["y_tr"], lin["y_va"]
    rows, models = [], []
    for spec in RF_GRID:
        m = RandomForestRegressor(**RF_FIXED, **spec).fit(Z_tr, y_tr)
        vp = m.predict(Z_va)
        rows.append(dict(**spec, n_estimators=RF_FIXED["n_estimators"], max_features=N_FEATURES,
                         train_mse=mse(y_tr, m.predict(Z_tr)), val_mse=mse(y_va, vp),
                         oos_r2=oos_r2(y_va, vp)))
        models.append(m)
        log(f"    depth={spec['max_depth']:<2} leaf={spec['min_samples_leaf']:<3} "
            f"trainMSE={mse(y_tr, m.predict(Z_tr)):.10f}  valMSE={mse(y_va, vp):.10f}  R2={oos_r2(y_va, vp):+.8f}")
    cand = pd.DataFrame(rows)
    # pre-declared tie-break: shallower trees, then larger leaves
    sel = int(min(range(len(cand)), key=lambda i: (cand.at[i, "val_mse"], cand.at[i, "max_depth"],
                                                   -cand.at[i, "min_samples_leaf"])))
    cand["selected"] = np.arange(len(cand)) == sel
    cand.to_csv(TAB / "forest_grid.csv", index=False, float_format="%.12g")
    model = models[sel]
    vp = model.predict(Z_va)
    # independent mean-of-trees check
    avg = np.mean(np.stack([t.predict(Z_va) for t in model.estimators_]), axis=0)
    assert np.allclose(avg, vp, rtol=0, atol=1e-14)
    # independent invariance check: refitting on a positive affine map of the
    # inputs leaves the fitted partition - and therefore every prediction -
    # unchanged in exact arithmetic; under floating point a near-tied split can
    # still flip, so the check asserts the effect stays numerically negligible.
    # (A non-linear monotone map does NOT have this property, so the claim is
    # deliberately restricted to positive affine rescalings.)
    sel_spec = {k: int(v) for k, v in cand.iloc[sel][["max_depth", "min_samples_leaf"]].to_dict().items()}
    aff = RandomForestRegressor(**RF_FIXED, **sel_spec).fit(3.0 * Z_tr + 7.0, y_tr)
    aff_max_diff = float(np.max(np.abs(aff.predict(3.0 * Z_va + 7.0) - vp)))
    # Reported as a diagnostic rather than a tight invariant: in exact arithmetic
    # a positive affine remap leaves the partition -- and therefore every
    # prediction -- unchanged, but under floating point a near-tied split gain
    # can flip and move a row across a threshold. The meaningful quantity is
    # therefore the ratio to the prediction scale, which is logged alongside.
    aff_scale = float(np.abs(vp).max())
    aff_ratio = aff_max_diff / max(aff_scale, 1e-30)
    assert aff_max_diff < 1e-3, aff_max_diff
    log(f"  affine-invariance diagnostic: max |pred(3Z+7) - pred(Z)| = {aff_max_diff:.2e}"
        f"  (prediction scale {aff_scale:.2e}, ratio {aff_ratio:.2e})")
    log(f"  SELECTED: max_depth={cand.at[sel,'max_depth']} min_samples_leaf={cand.at[sel,'min_samples_leaf']}"
        f"  n_estimators={RF_FIXED['n_estimators']} max_features={N_FEATURES} seed={SEED}")
    log(f"  train MSE={mse(y_tr, model.predict(Z_tr)):.10f}  val MSE={mse(y_va, vp):.10f}"
        f"  OOS R2={oos_r2(y_va, vp):+.10f}")
    fi = pd.DataFrame({"predictor": PREDICTORS, "builtin_importance": model.feature_importances_})
    fi["market_type"] = fi["predictor"].isin(MARKET_PREDICTORS)
    fi = fi.sort_values("builtin_importance", ascending=False).reset_index(drop=True)
    fi.to_csv(TAB / "forest_builtin_importance.csv", index=False)
    share = float(fi.loc[fi["market_type"], "builtin_importance"].sum())
    log(f"  built-in importance mass on the six within-week-constant market predictors = {share:.4f}")
    return dict(model=model, spec=dict(cand.iloc[sel][["max_depth", "min_samples_leaf"]].to_dict()),
                val_mse=float(cand.at[sel, "val_mse"]), oos_r2=float(cand.at[sel, "oos_r2"]),
                train_mse=mse(y_tr, model.predict(Z_tr)), grid=rows, p_va=vp,
                market_share=share,
                affine_max_diff=aff_max_diff, affine_scale=aff_scale, affine_ratio=aff_ratio,
                builtin=fi.to_dict("records"))


def stage4_importance(lin, nonlin):
    log(""); log("=" * 78); log("STAGE 4 - IMPORTANCE ANALYSIS (DROP-TO-ZERO)"); log("=" * 78)
    Z_tr, Z_va, y_va = lin["Z_tr"], lin["Z_va"], lin["y_va"]
    model = nonlin["model"]
    base = oos_r2(y_va, model.predict(Z_va))
    rows = []
    for j, name in enumerate(PREDICTORS):
        Z2 = Z_va.copy(); Z2[:, j] = 0.0
        rows.append(dict(predictor=name, importance=base - oos_r2(y_va, model.predict(Z2)),
                         builtin_importance=dict((r["predictor"], r["builtin_importance"])
                                                 for r in nonlin["builtin"])[name]))
    imp = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", np.arange(1, len(PREDICTORS) + 1))
    imp["market_type"] = imp["predictor"].isin(MARKET_PREDICTORS)
    imp.to_csv(TAB / "drop_to_zero_importance.csv", index=False, float_format="%.10g")
    log(f"  baseline OOS R2 = {base:.10f}")
    for _, r in imp.head(5).iterrows():
        log(f"    {r['predictor']:22s} {r['importance']:+.10f}")
    neg = imp.loc[imp["importance"] < 0, "predictor"].tolist()
    log(f"  negative importance: {len(neg)} predictors, most negative {imp.iloc[-1]['predictor']} "
        f"{imp.iloc[-1]['importance']:+.10f}")

    # ---- LASSO-vs-forest overlap, computed (not asserted by hand) ----
    top5 = imp.head(5)["predictor"].tolist()
    lasso_retained = list(lin["retained"])
    overlap = [p for p in top5 if p in lasso_retained]
    only_lasso = [p for p in lasso_retained if p not in top5]
    only_forest = [p for p in top5 if p not in lasso_retained]
    log(f"  forest top-5        : {top5}")
    log(f"  LASSO retained      : {lasso_retained}")
    log(f"  OVERLAP ({len(overlap)})          : {overlap}")
    log(f"  LASSO-only ({len(only_lasso)})     : {only_lasso}")
    log(f"  forest-only ({len(only_forest)})    : {only_forest}")
    # The overlap size is a RESULT of the comparison, not an invariant of the
    # pipeline: a panel on which LASSO and the forest agree less would still be a
    # valid study. It is logged, never asserted.

    # response profile of the most important predictor
    most = imp.iloc[0]["predictor"]; j = PREDICTORS.index(most)
    lo, hi = np.quantile(Z_tr[:, j], [0.10, 0.90], method="linear")
    gridv = np.linspace(lo, hi, 101)
    prof = []
    for v in gridv:
        Z = Z_va.copy(); Z[:, j] = v
        prof.append(float(model.predict(Z).mean()))
    prof = np.array(prof)
    pd.DataFrame({"standardized_value": gridv, "mean_predicted_excess_return": prof}).to_csv(
        TAB / "response_profile.csv", index=False, float_format="%.12g")
    log(f"  most important = {most}  p10={lo:.8f} p90={hi:.8f}  profile {prof.min():.8f}..{prof.max():.8f}")
    log(f"  endpoint change = {prof[-1]-prof[0]:+.10f}")
    # seed-sensitivity check for the importance ranking
    spec_int = {k: int(v) for k, v in nonlin["spec"].items()}
    sets = []
    for sd in [0, 42, 123, 9999]:
        mm = RandomForestRegressor(**{**RF_FIXED, "random_state": sd}, **spec_int).fit(Z_tr, lin["y_tr"])
        b = oos_r2(y_va, mm.predict(Z_va))
        per = []
        for k2, n in enumerate(PREDICTORS):
            Z2 = Z_va.copy(); Z2[:, k2] = 0.0
            per.append((n, b - oos_r2(y_va, mm.predict(Z2))))
        per.sort(key=lambda t: -t[1])
        sets.append([n for n, _ in per[:4]])
    log(f"  top-4 predictors under seeds 0/42/123/9999: {sets}")
    return dict(base_oos_r2=base, top5=imp.head(5).to_dict("records"), negative=neg,
                most_important=most, p10=float(lo), p90=float(hi),
                response_x=gridv.tolist(), response_y=prof.tolist(),
                profile_min=float(prof.min()), profile_max=float(prof.max()),
                profile_start=float(prof[0]), profile_end=float(prof[-1]),
                profile_delta=float(prof[-1] - prof[0]),
                seed_top4=sets, full=imp.to_dict("records"),
                overlap=overlap, lasso_retained=lasso_retained,
                lasso_only=only_lasso, forest_only=only_forest)


# --------------------------------------------------------------------------- #
# STAGE 5 - portfolio construction
# --------------------------------------------------------------------------- #
def _weekly(va, pred, k):
    vv = va.copy(); vv["pred"] = pred
    out = []
    for _, g in vv.groupby("week", sort=True):
        o = g.sort_values(["pred", "asset_id"], ascending=[False, True])
        out.append(float(o.head(k)["ret_next_week"].mean()))
    return np.array(out)


def _spread16(va, pred):
    vv = va.copy(); vv["pred"] = pred
    s = []
    for _, g in vv.groupby("week", sort=True):
        o = g.sort_values(["pred", "asset_id"], ascending=[False, True])
        s.append(float(o.head(TOP_N)["ret_next_week"].mean() - o.tail(TOP_N)["ret_next_week"].mean()))
    return np.array(s)


def stage5_portfolio(df, lin, nonlin):
    log(""); log("=" * 78); log("STAGE 5 - PORTFOLIO CONSTRUCTION"); log("=" * 78)
    va = df[df["set"] == "validation"].copy()
    rf_s = np.array([float(g["rf_weekly"].iloc[0]) for _, g in va.groupby("week", sort=True)])
    ew = np.array([float(g["ret_next_week"].mean()) for _, g in va.groupby("week", sort=True)])
    models = {"OLS": lin["p_ols_va"], "LASSO": lin["p_las_va"], "Random forest": nonlin["p_va"]}

    def met(lo):
        # Every mean/sd pair published below sits on the SAME basis, so each
        # ratio can be recomputed from the columns shipped beside it:
        #   sharpe = sqrt(52) * mean_excess / sd_excess
        #   ir     = sqrt(52) * mean_active / sd_active   (active vs equal weight)
        #   mdd    = max drawdown of the gross weekly return series
        exc, act = lo - rf_s, lo - ew
        return dict(mean=float(lo.mean()), sd=float(lo.std(ddof=1)),
                    mean_excess=float(exc.mean()), sd_excess=float(exc.std(ddof=1)),
                    mean_active=float(act.mean()), sd_active=float(act.std(ddof=1)),
                    sharpe=ann_ratio(exc), ir=ann_ratio(act), mdd=max_drawdown(lo))

    table3 = []
    for name, pred in models.items():
        sp = _spread16(va, pred)
        m = met(_weekly(va, pred, TOP_N))
        t = float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))))
        table3.append(dict(model=name, rule=f"top{TOP_N}", mean_ls_spread=float(sp.mean()), spread_t=t, **m))
        log(f"  {name:14s} top16  spread={sp.mean():+.8f} (t={t:.2f})  Sharpe={m['sharpe']:+.6f}"
            f"  IR={m['ir']:+.6f}  MDD={m['mdd']:+.8f}")
    m_ew = met(ew)
    table3.append(dict(model="80-stock equal weight", rule="equal weight", mean_ls_spread=None, spread_t=None, **m_ew))
    log(f"  {'Equal weight':14s}        Sharpe={m_ew['sharpe']:+.6f}  IR={m_ew['ir']:+.6f}  MDD={m_ew['mdd']:+.8f}")

    cand = []
    for name, pred in models.items():
        for k in TOPK_GRID:
            m = met(_weekly(va, pred, k))
            cand.append(dict(model=name, rule=f"top{k}", k=k, **m))
    cd = pd.DataFrame(cand)

    # ---- three-window robust composite -------------------------------------
    # The candidate set is scored separately on the full validation block and on
    # each of its two halves; each window contributes one weighted
    # metric percentile score, and the rule is ranked on the AVERAGE.
    windows = {"full": va,
               "first half (79-91)": va[va["week"] <= 91],
               "second half (92-104)": va[va["week"] >= 92]}
    per_window = {}
    for label, sub in windows.items():
        rf_w = np.array([float(g["rf_weekly"].iloc[0]) for _, g in sub.groupby("week", sort=True)])
        ew_w = np.array([float(g["ret_next_week"].mean()) for _, g in sub.groupby("week", sort=True)])

        def met_w(lo):
            return dict(sharpe=ann_ratio(lo - rf_w), ir=ann_ratio(lo - ew_w), mdd=max_drawdown(lo))

        tmp = []
        for name, pred in models.items():
            # slice the model's validation predictions down to this window's rows
            sub_full = va.copy(); sub_full["pred"] = pred
            pred_w = sub_full.loc[sub.index, "pred"].to_numpy()
            for k in TOPK_GRID:
                tmp.append(dict(model=name, k=k, **met_w(_weekly(sub, pred_w, k))))
        td = pd.DataFrame(tmp)
        for c in METRIC_WEIGHTS:
            td[c + "_pct"] = td[c].rank(pct=True)
        td["score_w"] = sum(METRIC_WEIGHTS[c] * td[c + "_pct"] for c in METRIC_WEIGHTS)
        per_window[label] = td.set_index(["model", "k"])["score_w"]
        cd[f"score[{label}]"] = [float(per_window[label].loc[(r["model"], r["k"])]) for _, r in cd.iterrows()]
        # also record the within-window Sharpe rank for the stability story
        sr = td.set_index(["model", "k"])["sharpe"].rank(ascending=False)
        cd[f"sharpe_rank[{label}]"] = [int(sr.loc[(r["model"], r["k"])]) for _, r in cd.iterrows()]

    cd["score"] = cd[[c for c in cd.columns if c.startswith("score[")]].mean(axis=1)
    cd["score_min"] = cd[[c for c in cd.columns if c.startswith("score[")]].min(axis=1)
    cd = cd.sort_values(["score", "score_min"], ascending=False).reset_index(drop=True)
    cd.to_csv(TAB / "portfolio_rule_candidates.csv", index=False, float_format="%.10g")

    log("  candidate long-only rules (K >= 10):")
    log("  [three-window robust composite = mean of the weighted")
    log("   percentile score on the full block and on each validation half]")
    for _, r in cd.head(12).iterrows():
        log(f"    {r['model']:14s} {r['rule']:6s} Sharpe={r['sharpe']:+.6f} IR={r['ir']:+.6f}"
            f" MDD={r['mdd']:+.8f}  composite={r['score']:.3f} (min {r['score_min']:.3f})")
    best = cd.iloc[0]
    final = dict(model=best["model"], rule=best["rule"], k=int(best["k"]), kind="topk",
                 sharpe=float(best["sharpe"]), ir=float(best["ir"]), mdd=float(best["mdd"]),
                 composite=float(best["score"]), composite_min=float(best["score_min"]),
                 runner_up=f"{cd.iloc[1]['model']} / top {int(cd.iloc[1]['k'])}",
                 runner_up_composite=float(cd.iloc[1]["score"]))
    log(f"  FINAL RULE -> {final['model']} / {final['rule']}  Sharpe={final['sharpe']:+.6f}"
        f" IR={final['ir']:+.6f} MDD={final['mdd']:+.8f} composite={final['composite']:.3f}"
        f" (runner-up {final['runner_up']} {final['runner_up_composite']:.3f})")

    # largest validation-week sector weight for every rule in Table 3 and for the final rule
    def max_sector(pred, k):
        vv = va.copy(); vv["pred"] = pred
        best = None
        for w, g in vv.groupby("week", sort=True):
            o = g.sort_values(["pred", "asset_id"], ascending=[False, True]).head(k)
            c = o["official_sector"].value_counts()
            share = int(c.iloc[0]) / k
            if best is None or share > best[0]:
                best = (share, str(c.index[0]), int(w), str(g["forecast_date"].iloc[0]), int(c.iloc[0]))
        return best
    secs = {n: max_sector(p, TOP_N) for n, p in models.items()}
    final_sec = max_sector(models[final["model"]], final["k"])
    for n, s in secs.items():
        log(f"  {n:14s} top16 largest validation sector: {s[0]*100:.2f}% {s[1]} week {s[2]} ({s[3]})")
    log(f"  FINAL RULE largest validation sector: {final_sec[0]*100:.2f}% {final_sec[1]} "
        f"week {final_sec[2]} ({final_sec[3]}), {final_sec[4]}/{final['k']} names")
    return dict(table3=table3, candidates=cd.to_dict("records"), final=final, final_sector=final_sec,
                model_sectors=secs)


def stage6_robustness(df, lin, nonlin):
    log(""); log("=" * 78); log("STAGE 6 - ROBUSTNESS OF THE RULE FAMILY"); log("=" * 78)
    pred_all = {"OLS": lin["p_ols_va"], "LASSO": lin["p_las_va"], "Random forest": nonlin["p_va"]}
    out = []
    va = df[df["set"] == "validation"].copy()
    halves = {"full 79-104": va, "first 79-91": va[va["week"] <= 91], "second 92-104": va[va["week"] >= 92]}
    for label, sub in halves.items():
        rf_s = np.array([float(g["rf_weekly"].iloc[0]) for _, g in sub.groupby("week", sort=True)])
        ew = np.array([float(g["ret_next_week"].mean()) for _, g in sub.groupby("week", sort=True)])
        for name, pred in pred_all.items():
            vv = va.copy(); vv["pred"] = pred
            pred_sub = vv.loc[sub.index, "pred"].to_numpy()
            # scan the SAME K grid as the candidate set, so a window's "best
            # rule" is the best of the whole family and not of a sub-grid
            for k in TOPK_GRID:
                lo = _weekly(sub, pred_sub, k)
                out.append(dict(window=label, model=name, rule=f"top{k}", k=k,
                                sharpe=ann_ratio(lo - rf_s), mdd=max_drawdown(lo)))
    rob = pd.DataFrame(out)
    rob.to_csv(TAB / "portfolio_rule_subperiods.csv", index=False, float_format="%.8g")
    for label in halves:
        s = rob[rob["window"] == label]
        top = s.sort_values("sharpe", ascending=False).iloc[0]
        log(f"  {label:14s} best Sharpe -> {top['model']} {top['rule']} ({top['sharpe']:+.6f}); "
            f"best MDD -> {s.sort_values('mdd', ascending=False).iloc[0]['model']} "
            f"{s.sort_values('mdd', ascending=False).iloc[0]['rule']} "
            f"({s.sort_values('mdd', ascending=False).iloc[0]['mdd']:+.8f})")
    return rob.to_dict("records")


def stage7_final_weights(df, lin, nonlin, final):
    log(""); log("=" * 78); log("STAGE 7 - FINAL WEIGHTS (REFIT ON WEEKS 1-104, THEN FIXED)"); log("=" * 78)
    dev = df[df["week"] <= 104]
    assert len(dev) == 8320 and dev["set"].isin(["training", "validation"]).all()
    scaler = StandardScaler().fit(dev[PREDICTORS].to_numpy())
    Z_dev, y_dev = scaler.transform(dev[PREDICTORS].to_numpy()), dev["y"].to_numpy()
    name = final["model"]
    if name == "OLS":
        model = LinearRegression().fit(Z_dev, y_dev)
    elif name == "LASSO":
        model = Lasso(alpha=lin["selected_alpha"], fit_intercept=True, max_iter=200000,
                      tol=1e-10, selection="cyclic").fit(Z_dev, y_dev)
    else:
        model = RandomForestRegressor(**RF_FIXED,
                                      **{k: int(v) for k, v in nonlin["spec"].items()}).fit(Z_dev, y_dev)
    log(f"  refit {name} on {len(dev)} rows (Weeks 1-104); scaler n_samples_seen={scaler.n_samples_seen_}")

    private = df[df["set"] == "private_evaluation"].copy()
    private["pred"] = model.predict(scaler.transform(private[PREDICTORS].to_numpy()))
    assert np.isfinite(private["pred"]).all()
    groups = list(private.groupby("week", sort=True))
    chunks = [model.predict(scaler.transform(g[PREDICTORS].to_numpy())) for _, g in groups]
    # Batching by week must reproduce the whole-frame prediction row for row.
    # The comparison is index-aligned rather than positional, so it still holds
    # on a panel whose rows are not sorted by week.
    full = dict(zip(private.index, private["pred"].to_numpy()))
    ref = np.array([full[i] for (_, g), c in zip(groups, chunks) for i in g.index])
    assert np.allclose(np.concatenate(chunks), ref, rtol=0, atol=1e-14)

    # anonymised, stable column names in asset_id order (1..80 -> asset_01..asset_80)
    asset_ids = sorted(int(a) for a in df["asset_id"].unique())
    columns = [f"asset_{a:02d}" for a in asset_ids]
    id2col = {a: f"asset_{a:02d}" for a in asset_ids}
    dmap = private.groupby("week")["forecast_date"].first().to_dict()
    weeks = sorted(int(w) for w in private["week"].unique())
    dates = [str(dmap[w]) for w in weeks]
    assert dates == sorted(dates), "forecast dates must be ascending"

    rows, diag = [], []
    for w in weeks:
        g = private[private["week"] == w].sort_values(["pred", "asset_id"], ascending=[False, True]).head(final["k"])
        wts = {c: 0.0 for c in columns}
        for a in g["asset_id"]:
            wts[id2col[int(a)]] = 1.0 / final["k"]
        # constraints are checked on the exact values; the file stores 8 decimals
        assert abs(sum(wts.values()) - 1.0) < 1e-12 and max(wts.values()) <= CAP + 1e-12
        stored = {c: f"{v:.8f}" for c, v in wts.items()}
        rows.append(dict(forecast_date=dmap[w], **stored))
        sec = g["official_sector"].value_counts()
        diag.append(dict(week=w, n_holdings=len(g),
                         max_sector_weight=float(sec.iloc[0] / final["k"]), top_sector=str(sec.index[0])))
    out = pd.DataFrame(rows, columns=["forecast_date"] + columns)
    out.to_csv(OUT / "portfolio_weights.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(TAB / "weight_diagnostics.csv", index=False)
    log(f"  weights {out.shape[0]}x{out.shape[1]}; evaluation-window sector weight mean={dg['max_sector_weight'].mean():.4f}"
        f" max={dg['max_sector_weight'].max():.4f} weeks>=0.8={int((dg['max_sector_weight']>=0.8).sum())}"
        f" weeks=1.0={int((dg['max_sector_weight']>=1.0-1e-9).sum())}")
    return dict(shape=list(out.shape), diagnostics=dg.to_dict("records"),
                sector_mean=float(dg["max_sector_weight"].mean()),
                sector_max=float(dg["max_sector_weight"].max()),
                weeks_ge80=int((dg["max_sector_weight"] >= 0.8).sum()),
                weeks_full=int((dg["max_sector_weight"] >= 1.0 - 1e-9).sum()))


# --------------------------------------------------------------------------- #
FIG_RC = {"font.size": 8, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
          "xtick.labelsize": 7, "ytick.labelsize": 7}


def draw_fig2(p, zero_mse, out_path, figsize=(6.9, 3.35)):
    """Validation MSE and the number of retained signals against the LASSO penalty.

    The retained-signal count is drawn as a step curve on a secondary right-hand
    axis (the usual glmnet convention): near the elbow the path points sit so
    close together on the log axis that per-point text labels overlap, so the
    count is encoded as a curve instead.
    """
    plt.rcParams.update(FIG_RC)
    fig, ax = plt.subplots(figsize=figsize)
    x = p["alpha"].to_numpy(dtype=float)
    ax.semilogx(x, p["val_mse"].to_numpy(dtype=float), marker="o", ms=2.7, lw=1.5,
                color="#12507e", label="LASSO validation MSE (left axis)", zorder=4)
    ax.axhline(zero_mse, ls="--", lw=1.0, color="#8a8a8a", label="zero forecast", zorder=2)
    s = p.loc[p["selected"]].iloc[0]
    ax.scatter([s["alpha"]], [s["val_mse"]], s=105, facecolors="none", edgecolors="#b3202f",
               lw=1.7, zorder=6)
    ax.annotate(f"selected: alpha = {s['alpha']:.4g}, {s['nonzero']:.0f} signals",
                xy=(s["alpha"], s["val_mse"]), xycoords="data",
                xytext=(0.985, 0.035), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=7.2, color="#b3202f",
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#b3202f", lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#b3202f", lw=1.0, shrinkA=2, shrinkB=4))
    ax.set_xlabel("LASSO penalty alpha (log scale; standardized predictors)")
    ax.set_ylabel("Validation MSE (squared decimal returns)")
    ax.grid(alpha=0.18)
    ax.margins(x=0.03, y=0.20)

    ax2 = ax.twinx()
    nz = p["nonzero"].to_numpy(dtype=float)
    ax2.step(x, nz, where="post", lw=1.5, ls=(0, (4, 2.2)),
             color="#2f8a72", label="retained signals (right axis)")
    # direct label for the starting plateau: the top tick is 16, not 17
    ax2.text(x[0], nz[0], f" {int(nz[0])} signals", color="#2f8a72", fontsize=7.0,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))
    ax2.set_ylabel("Number of retained signals", color="#2f8a72")
    ax2.set_ylim(-1.2, 24.0)
    ax2.set_yticks([0, 4, 8, 12, 16])
    ax2.tick_params(axis="y", colors="#2f8a72")
    # keep the step curve behind the MSE curve
    ax2.set_zorder(1)
    ax.set_zorder(3)
    ax.patch.set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.1, loc="lower left", framealpha=0.96)
    fig.tight_layout()
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def build_figures(lin, q4res):
    plt.rcParams.update(FIG_RC)
    corr = pd.read_csv(TAB / "training_correlation_matrix.csv", index_col=0)
    labels = list(corr.columns)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(17), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(17), labels)
    for i in range(17):
        for j in range(17):
            if i != j and abs(corr.values[i, j]) >= 0.25:
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7.5, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson correlation (training)")
    # (no in-image title: the report caption carries the figure label)
    fig.tight_layout(); fig.savefig(FIG / "fig1.png", dpi=230); plt.close(fig)

    p = pd.read_csv(TAB / "lasso_path.csv")
    draw_fig2(p, float(lin["y_va"].dot(lin["y_va"]) / len(lin["y_va"])), FIG / "fig2.png")

    imp = pd.read_csv(TAB / "drop_to_zero_importance.csv")
    top5 = imp.head(5)
    fig, ax = plt.subplots(figsize=(5.9, 3.3))
    cols = ["#12507e" if v >= 0 else "#b3202f" for v in top5["importance"]]
    ax.barh(top5["predictor"][::-1], top5["importance"][::-1], color=cols[::-1])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("Original OOS R2 minus OOS R2 after the predictor is set to zero")
    # (no in-image title)
    xmax = max(top5["importance"].max() * 1.42, 1e-4)
    ax.set_xlim(min(0.0, top5["importance"].min() * 1.42), xmax)
    for i, v in enumerate(top5["importance"][::-1]):
        ax.text(v + xmax * 0.02, i, f"{v:+.6f}", va="center", fontsize=7.4)
    ax.grid(axis="x", alpha=0.18); fig.tight_layout(); fig.savefig(FIG / "fig3.png", dpi=230); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.9, 3.5))
    ax.plot(q4res["response_x"], q4res["response_y"], color="#12507e", lw=1.8)
    ax.axvline(0, color="#8a8a8a", ls=":", lw=1, label="training mean (standardized 0)")
    ax.set_xlabel(f"{q4res['most_important']} (standardized; training 10th-90th percentile)")
    ax.set_ylabel("Mean predicted weekly excess return (decimal)")
    # (no in-image title)
    ax.grid(alpha=0.18); ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(FIG / "fig4.png", dpi=230); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Machine-Learning Investment Project pipeline")
    ap.add_argument("--data", default=None, help="path to the panel CSV "
                    "(default: data/panel.csv, or the MLIP_DATA environment variable)")
    args = ap.parse_args()
    global DATA
    DATA = _locate_data(args.data)
    log(f"data file: {DATA.name}")
    for d in (OUT, TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    df = load_data()
    q1res = stage1_benchmark(df)
    lin = stage2_linear(df)
    nonlin = stage3_forest(lin)
    q4res = stage4_importance(lin, nonlin)
    q5res = stage5_portfolio(df, lin, nonlin)
    rob = stage6_robustness(df, lin, nonlin)
    fw = stage7_final_weights(df, lin, nonlin, q5res["final"])
    build_figures(lin, q4res)

    table1 = [
        dict(model="Zero forecast", train_mse=q1res["zero_mse_train"], val_mse=q1res["zero_mse_val"], oos_r2=0.0),
        dict(model="OLS", train_mse=mse(lin["y_tr"], lin["p_ols_tr"]),
             val_mse=mse(lin["y_va"], lin["p_ols_va"]), oos_r2=oos_r2(lin["y_va"], lin["p_ols_va"])),
        dict(model=f"LASSO (alpha={lin['selected_alpha']:.6g})", train_mse=mse(lin["y_tr"], lin["p_las_tr"]),
             val_mse=mse(lin["y_va"], lin["p_las_va"]), oos_r2=oos_r2(lin["y_va"], lin["p_las_va"])),
        dict(model="Random forest", train_mse=nonlin["train_mse"], val_mse=nonlin["val_mse"],
             oos_r2=nonlin["oos_r2"]),
    ]
    pd.DataFrame(table1).to_csv(TAB / "prediction_results.csv", index=False, float_format="%.12g")
    pd.DataFrame(q5res["table3"]).to_csv(TAB / "validation_portfolios.csv", index=False, float_format="%.12g")

    results = dict(
        environment=dict(python=sys.version.split()[0], platform=platform.platform(),
                         numpy=np.__version__, pandas=pd.__version__,
                         scikit_learn=sklearn.__version__, matplotlib=matplotlib.__version__,
                         seed=SEED),
        data=q1res,
        linear=dict(alpha_max=lin["alpha_max"], selected_alpha=lin["selected_alpha"], nonzero=lin["nonzero"],
                retained=lin["retained"], path=lin["path"], val_curve_span=lin["val_curve_span"],
                ols=dict(train_mse=mse(lin["y_tr"], lin["p_ols_tr"]), val_mse=mse(lin["y_va"], lin["p_ols_va"]),
                         oos_r2=oos_r2(lin["y_va"], lin["p_ols_va"])),
                lasso=dict(train_mse=mse(lin["y_tr"], lin["p_las_tr"]), val_mse=mse(lin["y_va"], lin["p_las_va"]),
                           oos_r2=oos_r2(lin["y_va"], lin["p_las_va"]))),
        forest=dict(spec=nonlin["spec"], grid=nonlin["grid"], val_mse=nonlin["val_mse"], oos_r2=nonlin["oos_r2"],
                train_mse=nonlin["train_mse"], market_share=nonlin["market_share"],
                affine_max_diff=nonlin["affine_max_diff"], affine_scale=nonlin["affine_scale"],
                affine_ratio=nonlin["affine_ratio"],
                builtin=nonlin["builtin"], seed=SEED),
        importance=q4res,
        portfolio=dict(table3=q5res["table3"], candidates=q5res["candidates"], final=q5res["final"],
                final_sector=q5res["final_sector"], model_sectors=q5res["model_sectors"],
                robustness=rob),
        table1=table1,
        final_weights=dict(shape=fw["shape"], diagnostics=fw["diagnostics"],
                           sector_mean=fw["sector_mean"], sector_max=fw["sector_max"],
                           weeks_ge80=fw["weeks_ge80"], weeks_full=fw["weeks_full"]),
    )
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    # the closing lines must be logged BEFORE run_log.txt is written, otherwise
    # the committed log is not a faithful copy of what this run produced
    log(""); log("DONE -> outputs/")
    (OUT / "run_log.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
