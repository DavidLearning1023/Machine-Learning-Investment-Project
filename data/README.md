# Data | 数据

## Provenance | 数据溯源

**中文** 研究面板**不可再分发**，因此不随本仓库提供。为了让读者准确理解 `outputs/` 里的每一个数字，下面把面板的构成逐层拆开说明：哪些是模拟的，哪些是真实的。

| 层级 | 真实还是模拟 |
| --- | --- |
| 收益（`ret_next_week`、`rf_weekly`） | **模拟。**并非真实市场观测值。 |
| 全部 17 个预测因子 | **模拟。**为构造信号，而非真实因子暴露。 |
| 80 只资产的标的池 | **真实。**取自一份真实的中国香港大型股指数成分股名单（2023 年 7 月在册版本）。 |
| 行业分类 | **真实**分类，本仓库中已重新匿名化。 |
| 周度日历 | **真实**交易周日期（每周五，2021-07-09 → 2026-06-26）。 |

在原始面板中，每只资产带有交易所代码、公司名称与行业分类。这些信息在本仓库中一律不再分发：代码与公司名统一替换为 `asset_01`…`asset_80` 与 `Asset 01`…`Asset 80`，四个行业分组统一替换为 `Sector_A`…`Sector_D`。**这一替换由流水线在载入数据时执行**，因此任何表格、日志行或 JSON 都不可能带出真实标识——匿名化是代码保证的，而不是靠事后编辑。

**为什么可以断定收益是模拟的。** 有三条统计性质足以排除"真实周频股票收益"的可能：超额峰度仅 **0.21**（真实周频收益通常在 3–6 之间）；时序一阶自相关为 **−0.03**（真实收益存在明显的波动聚集效应）；横截面离散度在五年间几乎不动，仅在 **0.035–0.058** 之间波动，没有出现任何压力期。另一方面，资产之间的平均相关系数为 **0.35**，说明数据中注入了共同的市场因子——但那是构造出来的，不是估计出来的市场 β。实际后果是：本仓库中的任何业绩数字都不是市场结果、也不是实盘记录，不具备任何投资含义。

**English** The study panel **cannot be redistributed**, so it is not shipped with this repository. To read every number in `outputs/` correctly, it helps to state exactly what the panel contains, layer by layer:

| Layer | Real or simulated? |
| --- | --- |
| Returns (`ret_next_week`, `rf_weekly`) | **Simulated.** Not observed market data. |
| All 17 predictors | **Simulated.** Constructed signals, not historical factor exposures. |
| Universe of 80 assets | **Real.** A real Hong Kong large-cap index constituent list as in force in July 2023. |
| Industry buckets | **Real** classification, relabelled here. |
| Weekly calendar | **Real** week dates (Fridays, 2021-07-09 → 2026-06-26). |

In the study panel the assets carried exchange tickers, company names and industry classifications. None of that is redistributed here: throughout this repository tickers and company names appear as `asset_01`…`asset_80` and `Asset 01`…`Asset 80`, and the four industry buckets appear as `Sector_A`…`Sector_D`. **The substitution is performed by the pipeline when the data is loaded**, so no table, log line or JSON blob can carry a real identifier — the guarantee is enforced in code, not by editing afterwards.

**Why the returns count as simulated.** Three properties rule out observed weekly equity returns: excess kurtosis of **0.21** (observed weekly returns typically run 3–6); mean lag-1 autocorrelation of **−0.03** (observed returns cluster in volatility); and a cross-sectional dispersion that barely moves over five years, sitting in a narrow **0.035–0.058** band with no stress periods. Against that, the **0.35** average correlation *between* assets shows a common market factor that was deliberately injected rather than estimated. The practical consequence: no performance figure in this repository is a market result or a live record, and none of the numbers carry investment meaning.

## Reproducing the pipeline | 复现流水线

**中文** 本目录提供一个生成器，产出 schema 完全一致的面板：相同的划分、相同的预测因子结构、相同的四个行业分组及其规模（50 / 13 / 11 / 6），以及相同的周度日历起点。

```
python data/make_synthetic_data.py      # 写入 data/panel.csv（约 6.3 MB）
```

`data/panel.csv` 已被 git 忽略。当该文件存在时，`src/pipeline.py` 会自动读取；也可以用 `--data` 参数或 `MLIP_DATA` 环境变量指向任意符合下述 schema 的文件。

> **生成器复现的是"流水线"，而不是本研究的结论。**
> 本仓库中的表格、图片与 `outputs/portfolio_weights.csv` 都是在原始面板上跑出的**研究运行**结果。运行生成器加流水线，可以复现代码路径、断言与每一项产物的结构——但不会复现那些数字：生成器的预测结构是刻意不同的，因此它会选出不同的组合规则、给出不同的比率。准确的表述是：**方法可复现，数字不可复现。**

**English** This folder ships a generator that produces a panel with an identical schema — same splits, same predictor structure, same four industry buckets with the same sizes (50 / 13 / 11 / 6), and the same weekly calendar start:

```
python data/make_synthetic_data.py      # writes data/panel.csv  (~6.3 MB)
```

`data/panel.csv` is git-ignored. When present, `src/pipeline.py` picks it up automatically; otherwise point the pipeline at any file with the schema below via `--data` or the `MLIP_DATA` environment variable.

> **The generator reproduces the *pipeline*, not the study's conclusions.**
> The tables, figures and `outputs/portfolio_weights.csv` committed here are the study run on the original panel. Running the generator plus the pipeline reproduces the code path, the assertions and the shape of every artefact — but not the numbers: the generator's predictive structure is deliberately different, so it selects a different portfolio rule and reports different ratios. The honest summary is: the method reproduces, the numbers do not.

## Schema | 字段结构

**中文** 每行 = 一个资产 × 一个预测周。260 周 × 80 资产 = 20,800 行。

| 字段 | 角色 | 说明 |
| --- | --- | --- |
| `week` | 标识 | 预测周序号，1–260 |
| `forecast_date` | 标识 | 周度日期（周五）；是真实日历周，不是观测日期 |
| `set` | 划分 | `training`（第 1–78 周）、`validation`（79–104）、`private_evaluation`（105–260） |
| `asset_id` | 标识 | 资产编号 1–80；跨周稳定；**不作为模型输入** |
| `ticker` | 标识 | 匿名标签 `asset_01`…`asset_80`；**不作为模型输入** |
| `company_name` | 标识 | 匿名标签 `Asset 01`…`Asset 80`；**不作为模型输入** |
| `official_sector` | 标识 | 匿名行业分组 `Sector_A`…`Sector_D`；仅用于组合层面的汇总 |
| `ret_next_week` | 结果 | 下周总简单收益；评估窗口内为空 |
| `rf_weekly` | 调整 | 已知的周度无风险收益；计算超额收益时扣除 |
| 17 个预测因子 | 输入 | 6 个市场级 + 10 个资产级 + 1 个行业级的模拟信号 |

**English** One row = one asset × one forecast week. 260 weeks × 80 assets = 20,800 rows.

| Column | Role | Description |
| --- | --- | --- |
| `week` | identifier | Forecast week, 1–260 |
| `forecast_date` | identifier | Weekly date (Friday); a real calendar week, not an observation |
| `set` | split | `training` (weeks 1–78), `validation` (79–104), `private_evaluation` (105–260) |
| `asset_id` | identifier | Asset id 1–80; stable across weeks; **not a model input** |
| `ticker` | identifier | Anonymised label `asset_01`…`asset_80`; **not a model input** |
| `company_name` | identifier | Anonymised label `Asset 01`…`Asset 80`; **not a model input** |
| `official_sector` | identifier | Anonymised industry bucket `Sector_A`…`Sector_D`; used for portfolio summaries only |
| `ret_next_week` | outcome | Next-week total simple return; blank in the evaluation window |
| `rf_weekly` | adjustment | Known weekly risk-free return; subtracted from the total return |
| 17 predictors | inputs | 6 market-level + 10 asset-level + 1 sector-level simulated signals |

### Predictors | 预测因子

| 分组 | 字段 | 行为 |
| --- | --- | --- |
| 市场级（6） | `market_trend`, `market_risk`, `market_sentiment`, `liquidity_state`, `rate_pressure`, `policy_uncertainty` | 每周一个取值，80 只资产完全相同——单独看没有横截面排序能力 |
| 资产级（10） | `momentum_short`, `momentum_medium`, `reversal_signal`, `risk_signal`, `turnover_signal`, `size_signal`, `value_signal`, `quality_signal`, `beta_signal`, `illiquidity_signal` | 在每一周内跨资产变动 |
| 行业级（1） | `sector_trend` | 同一行业内本周共享 |

**English** Market-level signals carry one value per week, identical across all 80 assets, so they cannot rank assets within a week on their own. Asset-level signals vary across assets within every week. The sector-level signal is shared by all assets of the same bucket within a week.

### Target | 预测目标

```
y = ret_next_week - rf_weekly        # 周度超额收益，十进制
```

**中文** 结果变量已与其预测行对齐（**不做位移**）。`y` 仅在训练与验证划分上定义；156 周的评估窗口只带预测因子，结果被隐藏。

**English** The outcome is already aligned to its forecast row (no shift). `y` is only defined for the training and validation splits; the 156-week evaluation window carries predictors but withheld outcomes.

## Splits and discipline | 划分与纪律

**中文**

- 划分**按日期进行，绝不打乱**：每一周的全部 80 只资产整体进入同一个划分，因此模型绝不会在某一周"看过一半市场状态、又在另一周被评估"。
- 标准化器只在训练行上拟合；每一个超参数（LASSO 惩罚项、森林深度与叶节点下限、组合规则 K）都在第 79–104 周的验证集上选择；评估窗口只被使用一次，用于产出最终冻结权重。

**English**

- The split is **by date, never shuffled**: each week contributes all 80 assets to exactly one split, so a model never sees part of a week's market state and is evaluated on the rest.
- The pipeline fits its scaler on training rows only; every hyper-parameter (LASSO penalty, forest depth/leaf, portfolio rule K) is selected on the validation weeks 79–104; the evaluation window is used exactly once, to produce the final frozen weights.

## Using your own panel | 换用自己的面板

**中文** 任何符合上述 schema 的 CSV 都可以直接用。`src/pipeline.py` 会断言：共 20,800 行、每周 80 只资产、共 260 周、六个市场级因子在每周内为常数、且结果变量恰好只存在于前 104 周。

**English** Any CSV with the schema above works. `src/pipeline.py` asserts: 20,800 rows, 80 assets per week, 260 weeks, the six market predictors constant within each week, and outcomes present in exactly the first 104 weeks.
