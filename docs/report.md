# Machine-Learning Investment Project — Technical Report
# 机器学习投资研究项目 — 技术报告

**David Hui** · 1786314617@qq.com · October 2026 | 2026 年 10 月

> **Simulated returns, real methodology.**
> **收益为模拟数据，方法为真。**
>
> The panel is a real Hong Kong large-cap index constituent list (as in force July 2023) carrying **simulated** returns and signals, which is why it cannot be redistributed; asset identifiers and industry buckets are anonymised throughout. The figures below describe a research exercise — not market results and not a performance record. Provenance, and the evidence that distinguishes simulated returns from observed ones, are in [`data/README.md`](../data/README.md).
>
> 面板的标的池取自一份真实的中国香港大型股指数成分股名单（2023 年 7 月在册版本），其收益与信号均为模拟生成，因此不可再分发；资产标识与行业分组在本报告中一律匿名化。以下数字描述的是一项研究练习——不是市场结果，也不是业绩记录。完整溯源及"模拟收益 vs 真实观测收益"的判别证据见 [`data/README.md`](../data/README.md)。

---

## 1. Overview | 概览

**中文** 本研究提出两个经常被混为一谈的问题。

**English** The study asks two questions that are often conflated.

**中文** 第一，**17 个模拟信号能否在样本外预测下周超额收益？** 能，但只能算勉强，而且必须先做收缩（shrinkage）。

**English** First, **can 17 simulated signals predict next-week excess returns out of sample?** They can — but barely, and only after shrinkage.

**中文** 第二，**预测最好的模型，是否也构建出最好的组合？** 并不是，而且这个反转足够尖锐，是本研究的主要发现。

**English** Second, **does the model that predicts best also build the best portfolio?** It does not, and the reversal is sharp enough to be the main finding.

**中文** 一个 80 只资产 × 260 个预测周的周度面板按日期划分（**绝不打乱**）为 78 个训练周、26 个验证周，以及一个结果被隐藏的 156 周评估窗口。七个阶段端到端运行。

**English** A weekly panel of 80 assets × 260 forecast dates is split by date (never shuffled) into 78 training weeks, 26 validation weeks and a 156-week evaluation window whose outcomes are withheld. Seven stages run end to end.

**表 0. 七个阶段 | Table 0. The seven stages**

| 阶段 Stage | 作用 Purpose |
| --- | --- |
| 0 | 数据完整性检查 · Data integrity checks |
| 1 | 目标定义与零预测基准 · Target definition and the zero-forecast benchmark |
| 2 | 线性模型：OLS 与在验证集上调参的 LASSO · Linear models: OLS and a validation-tuned LASSO |
| 3 | 一个非线性模型：随机森林，在复杂度阶梯上选择 · One nonlinear model: random forest, chosen on a complexity ladder |
| 4 | 基于 drop-to-zero OOS R² 的重要性分析 · Importance analysis by drop-to-zero OOS R² |
| 5 | 组合构建：33 条 long-only top-K 规则 · Portfolio construction: 33 long-only top-K rules |
| 6 | 按窗口对整族规则重新评分 · Per-window re-scoring of the whole rule family |
| 7 | 在第 1–104 周上重拟合一次，冻结，产出 156 个权重向量 · Refit once on weeks 1–104, freeze, write 156 weekly weight vectors |

**中文** 每一个超参数都在这 26 个验证周上选择。评估窗口只被使用一次，用来产出最终权重。

**English** Every hyper-parameter is selected on the 26 validation weeks. The evaluation window is used exactly once, to produce the final weights.

---

## 2. Data and the zero-forecast benchmark | 数据与零预测基准

**表 1. 各划分的样本量 | Table 1. Sample counts by split**

| 划分 Split | 周数 Weeks | 资产数 Assets | 行数 Rows | 结果变量 Outcomes |
| --- | ---: | ---: | ---: | --- |
| 训练 Training | 78 | 80 | 6,240 | 可见 visible |
| 验证 Validation | 26 | 80 | 2,080 | 可见 visible |
| 评估 Evaluation | 156 | 80 | 12,480 | 隐藏 withheld |

**中文** 目标变量为 `y = ret_next_week − rf_weekly`，单位是十进制周度超额收益，且已与其预测行对齐。在 6,240 个训练行上，均值为 **−0.0010226791**，样本标准差（n − 1）为 **0.0565097717**。均值为负，因此截距是有意义的，零预测并不等于样本均值。只有 17 个预测因子被用作输入；资产编号、标签与行业标签都是标识符，不是特征。

**English** The target is `y = ret_next_week − rf_weekly`, in decimal weekly excess returns, already aligned to its forecast row. Over the 6,240 training rows the mean is **−0.0010226791** and the sample standard deviation (n − 1) is **0.0565097717**. The mean is negative, so the intercept matters and the zero forecast is not the sample mean. Only the 17 predictors are used as inputs; asset ids, labels and sector labels are identifiers, not features.

![Training correlation matrix](../outputs/figures/fig1.png)

**图 1. 17 个预测因子的训练相关矩阵（仅用 6,240 个训练行）；|r| ≥ 0.25 的格子已标注。 | Figure 1. Training correlation matrix of the 17 predictors (6,240 training rows only); cells with |r| ≥ 0.25 are labelled.**

**中文** 绝对值最大的非对角元素对是 `momentum_short` ↔ `momentum_medium`，相关系数 **+0.6399936121**；次之是 `size_signal` ↔ `illiquidity_signal`，为 −0.4421681987。除这两对之外，所有 |相关系数| 都低于 0.17，因此这 17 个预测因子接近正交——单靠相关性并不足以支持删除任何一个。

**English** The strongest absolute off-diagonal pair is `momentum_short` ↔ `momentum_medium` at **+0.6399936121**; the next is `size_signal` ↔ `illiquidity_signal` at −0.4421681987. Apart from those two pairs every |correlation| is below 0.17, so the 17 predictors are close to orthogonal — correlation alone does not justify deleting one.

**中文** **为什么必须保留日历顺序与完整的周。** 17 个预测因子中有 6 个在周内对 80 只资产完全相同，`rf_weekly` 也是一个单一的周度数字。若按"资产-周"随机切分，同一周的一部分资产会进训练集、另一部分进验证集，模型就等于看到了那一周的市场状态。因此每一周都按日历顺序完整保留，使验证块模拟的正是"在每个预测时点只掌握过去信息"的投资者。

**English** **Why the calendar order and whole weeks are kept.** Six of the 17 predictors are identical across the 80 assets within a week, and `rf_weekly` is a single weekly number. A random asset-week split would place some assets of one week in training and others in validation, letting the model see that week's market state. Each week is therefore kept intact in calendar order, so the validation block imitates an investor who only knows the past at each forecast date.

**中文** **基准。** 零预测对每一行都预测零超额收益，其验证 MSE 为 **0.0030908984**（训练 MSE 0.0031938884；验证集目标平方和 6.4290686681）。对下面每一个模型，都有 OOS R² = 1 − Σ(y − ŷ)² / Σy²，因此负值意味着比预测零还差。

**English** **Benchmark.** The zero forecast predicts zero excess return for every row. Its validation MSE is **0.0030908984** (training MSE 0.0031938884; sum of squared validation targets 6.4290686681). For every model below, OOS R² = 1 − Σ(y − ŷ)² / Σy², so a negative value means worse than predicting zero.

---

## 3. Linear models: OLS and LASSO | 线性模型：OLS 与 LASSO

**中文** 两个模型都是带非惩罚截距的合并回归，在全部 6,240 个训练行上拟合，预测因子先按**仅由训练集**的均值与标准差标准化。OLS 使用 `LinearRegression`；LASSO 使用 `Lasso`，求解器为循环坐标法（cyclic-coordinate），容差 1e-10，`max_iter` 200,000，最小化 SSE/(2n) + α·Σ|βⱼ|。全程不使用任何打乱的交叉验证。

**English** Both models are pooled regressions with an unpenalised intercept, fitted on all 6,240 training rows after standardising the 17 predictors with training means and standard deviations only. OLS uses `LinearRegression`; LASSO uses `Lasso` with a cyclic-coordinate solver, tolerance 1e-10, `max_iter` 200,000, minimising SSE/(2n) + α·Σ|βⱼ|. No shuffled cross-validation is used anywhere.

**表 2. 预测结果（MSE 单位为平方十进制收益；OOS R² 为十进制） | Table 2. Prediction results (MSE in squared decimal returns; OOS R² as a decimal)**

| 模型 Model | 训练 MSE Train MSE | 验证 MSE Val MSE | 验证 OOS R² |
| --- | ---: | ---: | ---: |
| 零预测 Zero forecast | 0.0031938884 | 0.0030908984 | 0.0000000000 |
| OLS | 0.0031130152 | 0.0031544950 | −0.0205754475 |
| LASSO（α = 0.0028011752） | 0.0031551909 | 0.0030640089 | +0.0086995595 |
| 随机森林 Random forest | 0.0029414926 | 0.0032193579 | −0.0415605689 |

![LASSO penalty path](../outputs/figures/fig2.png)

**图 2. 验证 MSE 对 LASSO 惩罚项（对数刻度，左轴）；绿色阶梯曲线为右轴的保留信号个数。 | Figure 2. Validation MSE against the LASSO penalty on a log scale (left axis); the green step curve on the right axis is the number of retained signals.**

**中文** α_max = max|X′(y − ȳ)|/n = **0.0064710781**，仅由训练数据计算。在从 α_max·10⁻⁴ 到 α_max 的 100 个对数等距惩罚项上，最低的验证 MSE 出现在 **α = 0.0028011752**，保留 **17 个斜率中的 4 个**（验证 MSE 0.0030640089，OOS R² +0.0086995595）。所选点位于网格内部，并非边界。

**English** α_max = max|X′(y − ȳ)|/n = **0.0064710781**, computed from training data only. Across 100 log-spaced penalties from α_max·10⁻⁴ to α_max, the lowest validation MSE is at **α = 0.0028011752**, retaining **4 of 17 slopes** (validation MSE 0.0030640089, OOS R² +0.0086995595). The selected point is interior, not on a grid boundary.

**中文** **曲线是平的，这一点很重要。** 在 α 落在 [0.15·α_max, 0.46·α_max]（即 0.000970662 到 0.002976696，跨 12 个网格点）这一段上，验证 MSE 仅在 0.0030640089 与 0.0031035289 之间变动（跨度 **1.29%**），而保留集合从 8 个信号收缩到 4 个。惩罚项的排序并非刀锋般敏感，因此不应把那个精确的 α 值当作有经济意义的结果。一处需要提醒：LASSO 的截距等于训练均值，所以在惩罚极大时，该模型预测的是训练均值而不是零。

**English** **The curve is flat, and that matters.** Over the window α ∈ [0.15·α_max, 0.46·α_max] (that is 0.000970662 to 0.002976696, 12 grid points) the validation MSE moves only between 0.0030640089 and 0.0031035289 (a **1.29%** range) while the retained set shrinks from 8 to 4 signals. The ranking of penalties is not knife-edge, and the exact penalty should not be read as economically meaningful. One caution: the LASSO intercept equals the training mean, so at very large penalties the model predicts that mean rather than zero.

**表 3. 标准化系数（截距单独列示） | Table 3. Standardised coefficients (intercept reported separately)**

| 预测因子 Predictor | OLS | 所选 LASSO Selected LASSO |
| --- | ---: | ---: |
| *截距 intercept* | −0.0010226791 | −0.0010226791 |
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

**中文** LASSO 保留了 `market_risk`、`rate_pressure`、`value_signal` 与 `quality_signal`。变量选择确实有帮助，但幅度不大：在全部 17 个预测因子上的 OLS 验证 OOS R² 为负（−0.0205754475），而 LASSO 把符号拉回了正（+0.0086995595）。完整回归在 6,240 行上过拟合；收缩剔除了足够多的噪声，才让符号在样本外转正。**但请注意**：惩罚项正是在这同样 26 周上挑出来的，因此这一改进并不是未被触碰的测试集结果。

**English** LASSO retains `market_risk`, `rate_pressure`, `value_signal` and `quality_signal`. Selection helps, but only slightly: OLS on all 17 predictors has a negative validation OOS R² (−0.0205754475) while LASSO recovers +0.0086995595. The full regression overfits 6,240 rows; shrinkage removes enough noise to move the sign out of sample. **One caution:** the penalty was chosen on these same 26 weeks, so this improvement is not an untouched-test result.

---

## 4. A nonlinear model: random forest | 非线性模型：随机森林

**中文** 森林平均 300 棵树，每棵在训练行的 bootstrap 样本上拟合，输入为随机子集（`max_features = sqrt(17) = 4`），平方误差分裂，`min_samples_split = 2`，随机种子 2026。复杂度在 `max_depth ∈ {2, 4, 8}` × `min_samples_leaf ∈ {30, 60}` 上变化——阶梯从深度 2 起步，因为一棵树需要两次分裂才能表达交互项，而这正是我们关心的非线性。

**English** The forest averages 300 trees, each on a bootstrap sample of training rows with a random subset of inputs (`max_features = sqrt(17) = 4`), squared-error splits, `min_samples_split = 2`, seed 2026. Complexity is varied over `max_depth ∈ {2, 4, 8}` × `min_samples_leaf ∈ {30, 60}` — the ladder starts at depth 2 because a tree needs two splits to express an interaction, which is exactly the nonlinearity of interest.

**表 4. 森林设定与验证 MSE | Table 4. Forest settings and validation MSE**

| 最大深度 Depth | 最小叶节点 Leaf | 训练 MSE | 验证 MSE | 验证 OOS R² |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 30 | 0.0029399988 | 0.0032231428 | −0.0427850999 |
| **2** | **60** | **0.0029414926** | **0.0032193579** | **−0.0415605689** |
| 4 | 30 | 0.0025479311 | 0.0033543030 | −0.0852194460 |
| 4 | 60 | 0.0025951406 | 0.0033251511 | −0.0757879172 |
| 8 | 30 | 0.0020571795 | 0.0035064431 | −0.1344414129 |
| 8 | 60 | 0.0021962617 | 0.0033997844 | −0.0999340458 |

**中文** 最低验证 MSE 选出深度 2 / 叶节点 60（表中加粗行）。两项检查确认实现无误：在 2,080 个验证行上，森林预测恰好等于其 300 棵树预测的均值（在 1e-14 容差内）；把输入替换为正仿射映射后重拟合，最大预测偏差为 8.67e-19，相对预测尺度（1.44e-02）的比值为 6.0e-17。第二项是**诊断而非硬不变量**——精确算术下仿射重映射不改变划分，但浮点下两个几乎并列的分裂增益可能翻转，使某些样本跨过阈值。

**English** Minimum validation MSE selects depth 2 / leaf 60 (the bold row). Two checks confirm the implementation: on the 2,080 validation rows the forest prediction equals the mean of its 300 tree predictions (within 1e-14); refitting on a positive affine map of the inputs moves predictions by at most 8.67e-19, a ratio of 6.0e-17 to the prediction scale (1.44e-02). The second is a **diagnostic, not a hard invariant** — in exact arithmetic an affine remap leaves the partition unchanged, but under floating point two near-tied split gains can flip and move a row across a threshold.

**中文** **沿复杂度阶梯每上一级，训练误差都下降、验证误差都上升**，且没有任何一个设定能打败零预测。有一个结构性事实可以解释：即使是这个浅森林，也把 **86.2%** 的内建分裂重要性放在那 6 个市场级预测因子上——而它们在周内对 80 只资产完全相同。这些变量自身无法在周内给资产排序；树只能借助它们来决定"由哪个别的变量来做这次分裂"，所以它们承载的重要性并不会转化为横截面顺序。

**English** **Every step up the complexity ladder lowers training error and raises validation error**, and no setting beats the zero forecast. One structural fact explains why: the shallow forest still places **86.2%** of its built-in split importance on the six market predictors that are identical across the 80 assets within a week. Those variables cannot rank assets within a week on their own; a tree can only use them to choose which *other* variable does the splitting, so the importance they carry does not translate into cross-sectional order.

---

## 5. Importance analysis | 重要性分析

**中文** 所选森林在第 1–78 周上拟合一次，之后不再重拟合。每次取一个标准化后的验证集预测因子，把它的整列置零——在同一个标准化器下，这等于把该因子替换为其训练均值——然后在同样那 2,080 行上重新计算 OOS R²。重要性 = 原 OOS R² − 替换后 OOS R²。与常见的"训练集拟合变体"不同，这里用的是**验证集**的 OOS R²。重要性是预测意义上的，不是因果意义上的，且相关的信号可能共享信用。基准 OOS R² = −0.0415605689。

**English** The selected forest is fitted once on weeks 1–78 and never refitted. For one standardised validation predictor at a time, its whole column is set to zero — equal to that predictor's training mean under the same scaler — and the OOS R² is recomputed on the same 2,080 rows. Importance = original OOS R² − replaced OOS R². Unlike the common training-fit variant, this uses validation OOS R². Importance is predictive, not causal, and correlated signals can share credit. Baseline OOS R² = −0.0415605689.

![Drop-to-zero importance](../outputs/figures/fig3.png)

**图 3. 最大的五个带符号 drop-to-zero 重要性得分。 | Figure 3. Five largest signed drop-to-zero importance scores.**

**表 5. 最大的五个 drop-to-zero 得分 | Table 5. Five largest drop-to-zero scores**

| 排名 Rank | 预测因子 Predictor | 重要性 Importance |
| ---: | --- | ---: |
| 1 | policy_uncertainty | +0.0086911000 |
| 2 | liquidity_state | +0.0076031920 |
| 3 | quality_signal | +0.0026298300 |
| 4 | value_signal | +0.0002336646 |
| 5 | sector_trend | +0.0001793685 |

**中文** 有 8 个预测因子的得分为负，而且这些负值是有信息量的：把 `market_trend` 置零**反而让** OOS R² 提升了 0.0274132449，也就是说样本外来看，森林对它的依赖是训练样本的产物。其余最负的依次是 `rate_pressure`（−0.010325）、`market_sentiment`（−0.003303）与 `market_risk`（−0.002156）。

**English** Eight predictors carry a negative score, and they are informative: setting `market_trend` to zero *raises* the OOS R² by 0.0274132449, so out of sample the forest's reliance on it is a training-sample artefact. The next most negative are `rate_pressure` (−0.010325), `market_sentiment` (−0.003303) and `market_risk` (−0.002156).

![Response profile](../outputs/figures/fig4.png)

**图 4. 最重要预测因子在其训练 10%–90% 分位上的平均响应——分段常数，树阈值的典型特征。 | Figure 4. Average response of the most important predictor across its training 10th–90th percentiles — piecewise constant, the signature of tree thresholds.**

**中文** 最重要的预测因子是 `policy_uncertainty`。在所评估的区间内（标准化后 −1.2264495953 到 +1.2761109476），平均预测从 −0.0032013596 变为 +0.0002515380，在十进制周度超额收益上变化了 **+0.0034528976**。曲线是分段常数、含一个非单调区段，并在 +1 个标准差附近陡然上升——这是树阈值的特征，而不是平滑的效应。它描述的是这个已拟合的模型，不是因果效应；而且该森林的基准 OOS R² 为负，解读它的依赖结构并不能让它成为一个成功的预测器。

**English** The most important predictor is `policy_uncertainty`. Over the evaluated range (−1.2264495953 to +1.2761109476 standardised) the mean prediction moves from −0.0032013596 to +0.0002515380, a change of **+0.0034528976** in decimal weekly excess return. The curve is piecewise constant with a non-monotonic segment and a sharp rise near +1 standard deviation — the signature of tree thresholds rather than a smooth effect. It describes this fitted model, not a causal effect; and the forest's baseline OOS R² is negative, so interpreting its reliance does not make it a successful forecaster.

**中文** **与 LASSO 的对比。** LASSO 保留了 4 个信号，其中 **2** 个出现在森林的前五名（`quality_signal`、`value_signal`）。`market_risk` 与 `rate_pressure` 在森林前五之外；而 `policy_uncertainty`、`liquidity_state` 与 `sector_trend` 被森林倚重，却没有被 LASSO 保留。这种重叠是部分的，也是结构性的：惩罚线性模型只能通过斜率使用一个预测因子，而树可以用它来划分资产、并在每个叶子里拟合一个不同的水平。在种子 0、42、123 与 9999 下重拟合森林，**前三位**每次都相同（`liquidity_state`、`policy_uncertainty`、`quality_signal`），因此这个排序不是种子带来的假象——但第四位确实会变。

**English** **Comparison with LASSO.** LASSO retained 4 signals; of these **2** appear in the forest's top five (`quality_signal`, `value_signal`). `market_risk` and `rate_pressure` are outside the forest's top five, while `policy_uncertainty`, `liquidity_state` and `sector_trend` are relied on by the forest but not retained by LASSO. The overlap is partial and structural: a penalised linear model uses a predictor only through its slope, whereas a tree can partition assets with it and fit a different level in each leaf. Refitting the forest under seeds 0, 42, 123 and 9999 returns the same **top three** predictors (`liquidity_state`, `policy_uncertainty`, `quality_signal`) every time, so the ranking is not a seed artefact — though the fourth position does move.

---

## 6. Portfolio construction | 组合构建

**中文** 在每个验证周，按预测超额收益对 80 只资产降序排名，并列时按 `asset_id` 升序打破平局。多空组合等权持有前 16 名并做空后 16 名；long-only 组合等权持有前 K 名，其余持 0。夏普比率 = √52 × mean(r_p − r_f)/sd(r_p − r_f)，信息比率 = √52 × mean(r_p − r_EW)/sd(r_p − r_EW)，两者标准差均取样本标准差。最大回撤从财富 1 开始按总收益复利计算，并以非正数报告。交易成本被忽略。

**English** Each validation week the 80 assets are ranked by predicted excess return descending, ties broken by ascending asset id. Long-short equally weights the top 16 and shorts the bottom 16; long-only equally weights the top K and holds zero elsewhere. Sharpe = √52 × mean(r_p − r_f)/sd(r_p − r_f) and information ratio = √52 × mean(r_p − r_EW)/sd(r_p − r_EW), both with sample standard deviations. Maximum drawdown compounds total returns from wealth 1 and is reported as a nonpositive number. Transaction costs are ignored.

**表 6. 验证集组合（第 79–104 周；十进制） | Table 6. Validation portfolios (weeks 79–104; decimals)**

| 模型 / 规则 Model / rule | 周度多空价差 LS spread | Long-only 夏普 | 相对等权 IR vs EW | 最大回撤 MDD |
| --- | ---: | ---: | ---: | ---: |
| OLS / top 16 | +0.0220083 | 4.4530 | 7.8654 | −0.0477 |
| LASSO / top 16 | +0.0208958 | 4.4086 | 10.3092 | −0.0478 |
| 随机森林 Random forest / top 16 | +0.0244138 | 4.8950 | 9.8742 | −0.0426 |
| 80 只等权 80-asset equal weight | — | 1.7803 | 0.0000 | −0.0777 |

**中文** **预测最好的模型是否给出了最好的组合？不是。** 验证集上预测最好的是 LASSO（OOS R² +0.0087），但最大的多空价差与最好的 long-only 夏普都属于随机森林，而后者的 OOS R² 是负的（−0.0416）。OOS R² 度量的是收益**水平**上的数值拟合；而 top-K 规则只使用预测值在**周内**的横截面**顺序**——把某一周的全部预测值乘以同一个正常数，持仓完全不变。因此这两个模型的差别出在排序本身，而不是预测值的离散程度：LASSO 只保留 17 个斜率中的 4 个，且这些斜率并不是 OLS 斜率的公共倍数；而森林在阈值与交互项上划分资产。改变 top-K 规则的是**模型形式**，不是预测尺度的收缩。

**English** **Does the best prediction model produce the best portfolio? No.** The best validation prediction is LASSO (OOS R² +0.0087), yet the largest long-short spread and the best long-only Sharpe both belong to the random forest, whose OOS R² is negative (−0.0416). OOS R² measures the numerical fit of the *level* of returns; a top-K rule uses only the cross-sectional *order* of forecasts within a week, and scaling every forecast of a week by the same positive constant would leave the holdings unchanged. The two models therefore differ in the ranking itself, not in how dispersed their forecasts are: LASSO keeps 4 of the 17 slopes and those slopes are not a common multiple of the OLS slopes, while the forest partitions assets on thresholds and interactions. Model form, not shrinkage of the forecast scale, is what moves the top-K rule.

**中文** **选择 K。** 33 条 long-only top-K 规则（K ∈ {10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 40} × 3 个模型；K 低于 10 会突破 10% 持仓上限）按 夏普 8 / IR 6 / 回撤 6 的权重打分。为了避免选中"在某一段周上恰好好看"的 K，加权百分位得分分别在完整验证块与其两个半块上计算，候选按三个窗口得分的**均值**排序。

**English** **Selecting K.** Thirty-three long-only top-K rules (K ∈ {10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 40} × 3 models; a K below 10 would breach the 10% cap) are scored with the metric weights Sharpe 8 / IR 6 / drawdown 6. To avoid picking the K that looks best on one stretch of weeks, the weighted percentile score is computed separately on the full validation block and on each of its two halves, and candidates are ranked on the **mean** of the three window scores.

**表 7. 按三窗口综合评分排名前八的候选 | Table 7. Eight highest-ranked candidates by three-window composite**

| 模型 / 规则 Model / rule | 夏普 Sharpe | IR | 最大回撤 MDD | 综合评分 Composite |
| --- | ---: | ---: | ---: | ---: |
| **随机森林 RF / top 14** | **5.1522** | **10.1914** | **−0.0402** | **17.131** |
| 随机森林 RF / top 15 | 4.8799 | 10.3675 | −0.0452 | 15.596 |
| 随机森林 RF / top 16 | 4.8950 | 9.8742 | −0.0426 | 14.263 |
| 随机森林 RF / top 11 | 5.1993 | 9.4462 | −0.0434 | 14.222 |
| 随机森林 RF / top 10 | 5.4809 | 9.3871 | −0.0484 | 14.121 |
| OLS / top 10 | 5.1739 | 7.5019 | −0.0416 | 14.040 |
| 随机森林 RF / top 20 | 4.6504 | 11.2416 | −0.0436 | 13.939 |
| 随机森林 RF / top 13 | 4.9822 | 8.8525 | −0.0424 | 13.697 |

**中文** 赢家"随机森林 / top 14"的综合评分为 17.131，亚军为 15.596；而它最弱窗口的得分（15.091）是所有候选中**最高的**最差窗口得分——因此这次选择并不依赖某一段走运的窗口。

**English** The winner, random forest / top 14, has composite 17.131 against 15.596 for the runner-up, and its weakest window score (15.091) is the highest worst-window score of any candidate — the selection does not rest on one lucky window.

**中文** **稳定性检查没有展示的东西。** 单一指标彼此并不一致：完整块上最高的夏普是"随机森林 / top 10"（5.4809），最高的信息比率是"随机森林 / top 20"（11.2416），最浅的回撤是"OLS / top 11"（−0.0395）。而且夏普最优的 K 在窗口之间会移动（完整块是随机森林 / top 10，前半段是 / top 11，后半段是 LASSO / top 13）。因此选择建立在三个指标的综合评分上，并且这一分歧被如实报告而非隐藏。

**English** **What the stability checks do not show.** Single metrics disagree: the highest full-block Sharpe is random forest / top 10 (5.4809), the highest information ratio is random forest / top 20 (11.2416), and the shallowest drawdown is OLS / top 11 (−0.0395). The Sharpe-maximising K also moves between windows (random forest / top 10 on the full block, / top 11 in the first half, LASSO / top 13 in the second). Selection is therefore made on a composite of all three metrics, and the disagreement is reported rather than hidden.

**中文** **行业集中度。** 在最终规则下，验证期单周最大的单一行业配置为 92.86%（`Sector_A`，第 103 周），其中 14 只持仓里有 13 只属于同一行业。10% 的单票上限并不施加任何行业上限。

**English** **Sector concentration.** Under the final rule the largest single-sector allocation during validation is 92.86% (`Sector_A`, week 103), where 13 of the 14 holdings share the same bucket. The 10% single-name cap imposes no sector cap.

---

## 7. Final weights for the evaluation window | 评估窗口的最终权重

**中文** 标准化器与所选森林在第 1–104 周的全部 8,320 行上重拟合一次，随后冻结。对评估窗口的每一个周，只用该周的 17 个预测因子给 80 只资产打分、排名，并给前 14 名各分配 1/14。模型在评估窗口内从不重拟合，任何被隐藏的结果都没有被使用，预测之所以变化只是因为输入在变。

**English** The scaler and the selected forest are refit once on all 8,320 rows of weeks 1–104 and then frozen. For each of the 156 evaluation weeks, only that week's 17 predictors are used to score the 80 assets, rank them and assign 1/14 to the top 14. The model is never refitted in the evaluation window, no withheld outcome is used anywhere, and forecasts change only because the inputs change.

**中文** 权重文件包含 156 行 × 81 列，按日期顺序排列，保留八位小数，每周恰好 14 个非零项、取值 1/14 = 7.142857%，每行求和为 1（在舍入精度内）。文件写入 `outputs/portfolio_weights.csv`。

**English** The weight file holds 156 rows × 81 columns, keeps the weeks in date order, uses eight decimal places, holds exactly 14 non-zero entries per week at 1/14 = 7.142857%, and every row sums to 1 within rounding. The file is written to `outputs/portfolio_weights.csv`.

**中文** **实质性风险：共同行业冲击下的集中度。** 在 156 个评估周中，最大单一行业配置平均为 **62.2%**，在 **17** 周超过 80%，并在 **2** 周达到 **100%**。在其中任何一周发生行业冲击，相对于 80 只等权基准都会被放大。这属于组合层面的风险，而非有意的行业押注——这种拥挤可能来自行业级信号、来自行业共同波动，或来自样本结构——此处如实披露，而不把它归因于单一原因。第二项风险是：规则是在 26 个验证周上选出的，因此这一选择偏乐观，年化数字并不代表一年的独立证据。

**English** **Material risk: concentration in common sector shocks.** Across the 156 evaluation weeks the largest sector allocation averages **62.2%**, exceeds 80% in **17** weeks and reaches **100%** in **2**. A sector shock in one of those weeks would be amplified relative to the 80-asset equal-weight benchmark. This is a portfolio-level risk rather than a deliberate sector bet — the crowding may come from sector-level signal, from sector co-movement, or from the sample structure — and it is disclosed rather than attributed to a single cause. A second risk is that the rule was chosen on 26 validation weeks, so the selection is optimistic and the annualised figures do not represent a year of independent evidence.

---

## 8. Reproducibility | 可复现性

**表 8. 运行环境与复现方式 | Table 8. Environment and reproduction**

| | |
| --- | --- |
| 主脚本 Main script | `src/pipeline.py`（全部七个阶段、每一张表与图） |
| 环境 Environment | Python 3.13.14、numpy 2.5.3、pandas 3.0.5、scikit-learn 1.9.1、matplotlib 3.11.2 |
| 确定性 Determinism | 随机种子 2026 用于森林；OLS、循环坐标 LASSO 与每周排序均为确定性 |
| 复现命令 Reproduction | `python data/make_synthetic_data.py && python src/pipeline.py` |
| 机器可读结果 | `outputs/results.json`，`outputs/tables/` 下每张表一个 CSV |

**中文** **"可复现"在这里意味着什么、不意味着什么。** 研究面板不可再分发，因此随仓库提供的生成器写出的面板具有相同的 schema、划分、周度日历与行业分组规模，但其**预测结构是刻意不同的**。重跑因此可以复现代码路径、断言以及每一项产物的形状，但会选出不同的组合规则、给出不同的比率。本报告与 `outputs/` 中的数字是原始面板上的研究运行结果；一次生成器上的运行验证的是"方法能够端到端执行"，而不是"这些数字会重现"。

**English** **What "reproduction" does and does not mean here.** The study panel cannot be redistributed, so the shipped generator writes a panel with the identical schema, splits, weekly calendar and industry-bucket sizes but a deliberately different predictive structure. Re-running therefore reproduces the code path, the assertions and the shape of every artefact, but selects a different portfolio rule and reports different ratios. The numbers in this report and in `outputs/` are the study run on the original panel; a generated run verifies that the method executes end to end, not that these figures recur.

**中文** 流水线内部会执行两项检查，二者的性质不同，需要分别说清楚。

**English** Two checks are run inside the pipeline, and they are not the same kind of check.

**中文** 第一，**求和稳健性检查**。零超额收益的验证 MSE 用精确求和（`math.fsum`）重算，与 numpy 成对均值实现的差为 **0.0**。它验证的是求和顺序不会改变结果，**而不是**验证这个基准本身定义是否正确——两者算的是同一个量。

**English** First, **summation robustness.** The zero-excess-return validation MSE is recomputed by exact summation (`math.fsum`); the gap against numpy's pairwise mean is **0.0**. It verifies that summation order does not change the result — not that the benchmark is defined correctly, which is the same quantity.

**中文** 第二，**树均值检查**。冻结森林的验证预测等于其 300 棵树预测的均值（1e-14 容差内）。这是一次真正的独立实现核对。

**English** Second, **mean-of-trees.** The frozen forest's validation prediction equals the mean of its 300 tree predictions (within 1e-14). This one is a genuine independent re-implementation.

**中文** 另有一项**仿射不变性诊断**（见第 4 节）：把输入替换为 3·Z+7 后重拟合，最大预测偏差为 **8.67e-19**，相对于预测尺度的比值为 **6.0e-17**。严格说这是一项诊断而非硬不变量——在精确算术下正仿射重映射不改变划分，但在浮点下两个几乎并列的分裂增益可能翻转，使某些样本跨过阈值。因此这里报告它与尺度的比值，而不是声称一个在浮点下无法保证的界。

**English** A third item is an **affine-invariance diagnostic** (see §4): refitting on 3·Z+7 moves predictions by at most **8.67e-19**, a ratio of **6.0e-17** to the prediction scale. In exact arithmetic a positive affine remap leaves the partition unchanged; under floating point two near-tied split gains can flip and move a row across a threshold. The diagnostic is therefore reported as a ratio to the scale, rather than as a bound that floating point cannot guarantee.

**中文** 标识符在**数据载入时**被统一匿名化：行业分组按字母序映射为 `Sector_A`、`Sector_B`…，代码与公司名映射为 `asset_NN` / `Asset NN`。因此任何表格、日志行或 JSON 都不可能携带真实标识——匿名化由代码保证，而不是靠事后编辑。

**English** Identifiers are anonymised **once, at data load**: industry buckets are mapped in sorted order to `Sector_A`, `Sector_B`, …, and tickers and company names to `asset_NN` / `Asset NN`. No table, log line or JSON blob can therefore carry a real identifier — the guarantee is enforced in code, not by editing afterwards.

**中文** 本报告中每一句"最好／最多／唯一"的陈述，都是从流水线写入 `outputs/tables/` 的表格中读出的，而不是手工断言的。`outputs/` 中的全部产物——表格、`results.json`、`run_log.txt`、权重文件与四张图——都是同一次运行的直接输出，没有事后编辑。

**English** Every "best / most / only" statement in this report is read from the tables the pipeline writes to `outputs/tables/`, not asserted by hand. Every artefact in `outputs/` — the tables, `results.json`, `run_log.txt`, the weight file and the four figures — is the direct output of a single run, with nothing edited afterwards.

---

## 9. Limitations | 局限

**中文**

- **收益为模拟数据。** 面板的收益与信号是模拟的，因此无法对真实市场得出任何推断。
- **只有 26 个验证周。** 规则是在 26 个周度观测上选出的，因此年化比率偏乐观。
- **不含成本。** 交易成本、借贷成本与流动性约束均被忽略；所有规则按周再平衡。
- **行业集中度**（见第 7 节）未受控制。
- **评估窗口未报告。** 其结果被隐藏，因此最终权重是在与实盘部署相同的纪律下产出的，而它们的实际表现无法在此展示。

**English**

- **Simulated returns.** The panel's returns and signals are simulated, so no inference about real markets can be drawn.
- **26 validation weeks.** The portfolio rule was chosen on 26 weekly observations, so the annualised ratios are optimistic.
- **No costs.** Transaction costs, borrow costs and liquidity constraints are ignored; the rules rebalance weekly.
- **Sector concentration**, disclosed in §7, is not controlled for.
- **The evaluation window is unreported.** Its outcomes are withheld, so the final weights are produced under the same discipline as a live deployment and their realised performance cannot be shown here.
