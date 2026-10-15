# -*- coding: utf-8 -*-
"""Generate a synthetic panel with the same schema as the study dataset.

The original study panel cannot be redistributed (see data/README.md), so this
script generates a structurally identical simulated panel:

  * 260 weekly forecast dates x 80 stocks = 20,800 rows
  * splits: weeks 1-78 training, 79-104 validation, 105-260 evaluation
    (the evaluation window carries predictors but blank outcomes)
  * 17 simulated predictors: 6 market-level signals that are constant within
    a week, 10 stock-level signals, and 1 sector-level signal shared by the
    stocks of the same sector
  * the same four industry buckets with the same sizes (50 / 13 / 11 / 6) and
    the same weekly calendar start as the study panel, so the structural
    diagnostics in outputs/ are directly comparable
  * a weak but genuine predictive structure, so that the pipeline produces
    meaningful model selection and portfolio results rather than pure noise

Note that the predictive structure here is *not* a copy of the study panel's:
running on this panel reproduces the pipeline, not the study's conclusions.

Run:  python data/make_synthetic_data.py   (writes data/panel.csv)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_WEEKS, N_STOCKS = 260, 80
TRAIN_END, VAL_END = 78, 104          # weeks 1-78 / 79-104 / 105-260
START_DATE = pd.Timestamp("2021-07-09")

MARKET_PREDICTORS = ["market_trend", "market_risk", "market_sentiment",
                     "liquidity_state", "rate_pressure", "policy_uncertainty"]
STOCK_PREDICTORS = ["momentum_short", "momentum_medium", "reversal_signal",
                    "risk_signal", "turnover_signal", "size_signal",
                    "value_signal", "quality_signal", "beta_signal",
                    "illiquidity_signal"]
# Same four buckets, same sizes as the study panel; labels are anonymised.
SECTOR_SIZES = [("Sector_A", 50), ("Sector_B", 13), ("Sector_C", 11),
                ("Sector_D", 6)]

# predictive structure: which stock-level signals actually drive the outcome,
# and how strongly (deliberately modest, so selection still matters)
BETAS = {"quality_signal": 0.0040, "value_signal": 0.0030,
         "momentum_medium": 0.0025, "reversal_signal": -0.0020,
         "sector_trend": 0.0025}
NOISE_SD = 0.050

OUT = Path(__file__).resolve().parent / "panel.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)
    weeks = np.arange(1, N_WEEKS + 1)
    assets = np.arange(1, N_STOCKS + 1)
    SECTORS = [name for name, _ in SECTOR_SIZES]
    # contiguous grouping reproduces the study panel's unequal bucket sizes
    sector_of, nxt = {}, 1
    for name, size in SECTOR_SIZES:
        for a in range(nxt, nxt + size):
            sector_of[a] = name
        nxt += size
    assert len(sector_of) == N_STOCKS
    sec_index = {s: i for i, s in enumerate(SECTORS)}

    # ---- market-level signals: one value per week, constant across stocks ----
    market = {}
    for name in MARKET_PREDICTORS:
        rho = 0.55
        e = rng.standard_normal(N_WEEKS)
        x = np.empty(N_WEEKS)
        x[0] = e[0]
        for t in range(1, N_WEEKS):                     # AR(1) in calendar order
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
        market[name] = x

    # ---- stock-level signals: genuine week x stock noise ---------------------
    stock = {name: rng.standard_normal((N_WEEKS, N_STOCKS)) for name in STOCK_PREDICTORS}

    # ---- sector-level signal: shared within a sector, week by week -----------
    sector_trend = rng.standard_normal((N_WEEKS, len(SECTORS)))

    # ---- outcome: weak true structure + idiosyncratic noise -----------------
    excess = np.empty((N_WEEKS, N_STOCKS))
    for j, a in enumerate(assets):
        s = sec_index[sector_of[a]]
        trend = sector_trend[:, s]
        excess[:, j] = (BETAS["quality_signal"] * stock["quality_signal"][:, j]
                        + BETAS["value_signal"] * stock["value_signal"][:, j]
                        + BETAS["momentum_medium"] * stock["momentum_medium"][:, j]
                        + BETAS["reversal_signal"] * stock["reversal_signal"][:, j]
                        + BETAS["sector_trend"] * trend
                        + NOISE_SD * rng.standard_normal(N_WEEKS))

    rf_weekly = 0.0002 + 0.00005 * np.sin(np.arange(N_WEEKS) / 26.0)
    ret_next_week = excess + rf_weekly[:, None]

    # ---- long-form panel ------------------------------------------------------
    rows = []
    dates = (START_DATE + pd.to_timedelta((weeks - 1) * 7, unit="D")).strftime("%Y-%m-%d")
    for t, w in enumerate(weeks):
        split = ("training" if w <= TRAIN_END
                 else "validation" if w <= VAL_END
                 else "private_evaluation")
        for j, a in enumerate(assets):
            rec = {
                "week": int(w),
                "forecast_date": dates[t],
                "set": split,
                "asset_id": int(a),
                "ticker": f"asset_{a:02d}",
                "company_name": f"Asset {a:02d}",
                "official_sector": sector_of[a],
                "ret_next_week": np.nan if split == "private_evaluation" else float(ret_next_week[t, j]),
                "rf_weekly": float(rf_weekly[t]),
            }
            for name in MARKET_PREDICTORS:
                rec[name] = float(market[name][t])
            for name in STOCK_PREDICTORS:
                rec[name] = float(stock[name][t, j])
            rec["sector_trend"] = float(sector_trend[t, sec_index[sector_of[a]]])
            rows.append(rec)

    df = pd.DataFrame(rows)
    assert len(df) == N_WEEKS * N_STOCKS == 20800
    assert df.groupby("week")["asset_id"].nunique().eq(N_STOCKS).all()
    assert df.loc[df["set"] == "private_evaluation", "ret_next_week"].isna().all()
    assert df.loc[df["set"] != "private_evaluation", "ret_next_week"].notna().all()
    assert (df.groupby("official_sector")["asset_id"].nunique()
            .to_dict() == {name: size for name, size in SECTOR_SIZES})
    for name in MARKET_PREDICTORS:
        assert df.groupby("week")[name].nunique().eq(1).all()

    df.to_csv(OUT, index=False)
    print(f"written: {OUT}  ({len(df)} rows, "
          f"{df['week'].nunique()} weeks x {df['asset_id'].nunique()} stocks)")


if __name__ == "__main__":
    main()
