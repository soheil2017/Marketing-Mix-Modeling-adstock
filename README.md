# Marketing Mix Modeling with Bayesian Adstock

A full Bayesian Marketing Mix Model (MMM) built in **PyMC** that quantifies how 7 advertising channels contribute to weekly revenue — accounting for delayed ad effects (adstock), trend, and seasonality. Every output comes with a full posterior uncertainty distribution, not just a point estimate.

---

## Table of Contents

1. [What is Marketing Mix Modeling?](#1-what-is-marketing-mix-modeling)
2. [Why Bayesian?](#2-why-bayesian)
3. [Dataset](#3-dataset)
4. [Data Preparation](#4-data-preparation)
5. [Adstock — Modeling Carry-Over Effects](#5-adstock--modeling-carry-over-effects)
6. [Model Architecture](#6-model-architecture)
7. [Prior Choices](#7-prior-choices)
8. [Prior Predictive Check](#8-prior-predictive-check)
9. [MCMC Sampling](#9-mcmc-sampling)
10. [Convergence Diagnostics](#10-convergence-diagnostics)
11. [Posterior Predictive Check & Fit Metrics](#11-posterior-predictive-check--fit-metrics)
12. [Revenue Attribution](#12-revenue-attribution)
13. [ROI Analysis](#13-roi-analysis)
14. [Holdout Validation](#14-holdout-validation)
15. [Final Results](#15-final-results)
16. [Stack](#16-stack)
17. [Files](#17-files)

---

## 1. What is Marketing Mix Modeling?

Marketing Mix Modeling (MMM) is a statistical technique that measures the contribution of each advertising channel to business outcomes (revenue, sales, conversions). It answers questions like:

- **Which channels are actually driving revenue?**
- **How long does the effect of an ad campaign last after it ends?**
- **What is the return on investment (ROI) per channel?**
- **How much revenue would we have generated with zero advertising?**

A naive approach would be to run a simple correlation between spend and revenue — but this is misleading. A channel that always spends more in Q4 will look correlated with revenue simply because Q4 is a strong sales period. MMM separates these effects properly by modeling trend, seasonality, and media contributions simultaneously.

---

## 2. Why Bayesian?

Traditional (frequentist) MMMs produce single point estimates for each parameter. Bayesian MMM produces **full probability distributions**, which means:

| Capability | Frequentist MMM | Bayesian MMM |
|---|---|---|
| Parameter estimates | Single value | Full distribution |
| ROI uncertainty | Not available | 90% credible intervals |
| Prior knowledge incorporation | Not available | Built-in via priors |
| Small data robustness | Prone to overfitting | Regularized by priors |
| Convergence check | Not applicable | R-hat, ESS diagnostics |

Instead of saying "Channel 2 has ROI = 30.92x", we can say "Channel 2 has ROI between 5.2x and 63.0x with 90% probability" — a much more honest and actionable output for budget decisions.

---

## 3. Dataset

- **104 weeks** of data (~2 years: Aug 2020 – Aug 2022)
- **7 advertising channels** — weekly spend per channel in euros
- **1 target variable** — weekly revenue in euros
- No missing values

```
Total revenue over 2 years: 14.19M euros
Average weekly revenue:    136,490 euros
Revenue range:              63,207 – 418,186 euros
```

A quick exploratory analysis reveals:
- Revenue has a visible **downward trend** over the 2-year period
- Clear **seasonal patterns** (revenue spikes at certain times of year)
- Channels vary significantly in spend volume and raw correlation with revenue

> Raw correlation is explored but not trusted — a channel correlated with revenue may just be spending more during naturally high-revenue periods. The model separates these effects.

---

## 4. Data Preparation

Before building the model, three transformations are applied:

### Scaling
MCMC works best when all variables are on a similar numerical scale (~0 to 2). Two scaling steps:
- **Revenue**: divided by its mean (~136,490 €), so values cluster around 1.0
- **Channel spend**: each channel divided by its own maximum, bringing all channels to [0, 1]

Scale factors are saved so all predictions can be converted back to euros.

### Trend
A linear index running from `0.0` (week 1) to `1.0` (week 104) captures whether revenue is growing or declining over time.

### Fourier Seasonality
Instead of 51 weekly dummy variables (which would overfit), **Fourier features** approximate annual seasonal patterns with just 4 columns:

```
For harmonic k:  sin(2π·k·t / 52)  and  cos(2π·k·t / 52)
```

Using 2 harmonics gives 4 columns `[sin₁, cos₁, sin₂, cos₂]` — the same idea used in Facebook Prophet. These 4 weights are learned from data and can represent a wide variety of seasonal shapes.

---

## 5. Adstock — Modeling Carry-Over Effects

Ads don't just work in the week they run. A TV campaign this week may still drive sales 3–4 weeks later. This delayed effect is called **adstock** (or carry-over).

We use **geometric adstock**:

```
adstock[t] = spend[t] + λ · adstock[t-1]
```

Where `λ` (lambda) is the **decay rate**:

| Lambda | Carry-Over Behaviour | Typical Channel |
|---|---|---|
| 0.0 | No carry-over — only this week's spend counts | Paid search |
| 0.3 | Short memory (~1 week) | Social media |
| 0.5 | Medium memory (~2 weeks) | Display / programmatic |
| 0.9 | Long memory (~7 weeks) | TV / brand campaigns |

**The model learns one lambda per channel from data** — we don't fix it manually. This is one of the key advantages of this approach.

The adstock function is implemented using `pytensor.scan` (instead of a Python `for` loop) because it must run inside the PyMC model graph so that PyTensor can compute gradients for the NUTS sampler.

---

## 6. Model Architecture

The full model equation:

```
revenue(t) = baseline(t) + media_contribution(t) + noise

where:
    baseline(t)          = intercept + β_trend · trend(t) + Σ γₖ · fourier_k(t)
    media_contribution(t) = Σ β_ch[c] · adstock[c, t]
    noise                ~ Normal(0, σ)
```

**Model components:**

| Component | Variables | Description |
|---|---|---|
| Intercept | `intercept` | Baseline revenue when everything else is zero |
| Trend | `beta_trend` | Linear growth or decline over 2 years |
| Seasonality | `gamma[4]` | 4 Fourier weights capturing annual patterns |
| Adstock decay | `lam[7]` | One carry-over rate per channel |
| Channel effect | `beta_ch[7]` | Revenue per unit of adstock per channel |
| Noise | `sigma` | Week-to-week unexplained variance |

**Total parameters**: 16 scalar parameters + 4 Fourier weights = **20 parameters** learned from 104 observations.

---

## 7. Prior Choices

Priors encode our beliefs about parameters before seeing data. They act as regularization, preventing the model from overfitting on 104 data points.

| Parameter | Prior | Justification |
|---|---|---|
| `intercept` | `Normal(1.0, 0.3)` | Revenue is scaled to mean ~1.0, so baseline should be near 1 |
| `beta_trend` | `Normal(0.0, 0.3)` | Trend could go either way; centred at zero with moderate width |
| `gamma` | `Normal(0, 0.1)` | Seasonal effects are expected to be small relative to baseline |
| `lam` | `Beta(2, 2)` | Naturally bounded in [0,1], symmetric around 0.5, avoids degenerate values |
| `beta_ch` | `HalfNormal(0.5)` | Ad spend can only **help** revenue, not hurt it — must be non-negative |
| `sigma` | `HalfNormal(0.2)` | Noise must be positive; 0.2 on scaled revenue is permissive |

All priors are intentionally wide enough to let the data dominate, while still providing enough regularization to prevent sampling problems.

---

## 8. Prior Predictive Check

Before running MCMC, we verify that the priors produce **plausible revenue values** even without seeing any data.

500 parameter sets are sampled from the priors alone and used to simulate revenue. The resulting range is plotted against observed revenue. If the prior distribution roughly covers the observed range, the priors are reasonable. If it produces negative revenue or values 10× too large, the priors need adjusting.

**Result**: The prior 90% range covers the observed revenue range — priors are sensible and not overly restrictive.

---

## 9. MCMC Sampling

The model is fitted using **MCMC (Markov Chain Monte Carlo)** with PyMC's **NUTS sampler** (No-U-Turn Sampler), which is well-suited for continuous parameters and high-dimensional posteriors.

```
Sampler    : NUTS
Draws      : 1,000 per chain
Tune steps : 1,000 (warm-up, discarded)
Chains     : 4 (run in parallel)
Target accept : 0.9
Total posterior samples : 4 × 1,000 = 4,000
Sampling time : ~153 seconds
```

Running 4 independent chains serves as a self-consistency check — if all 4 chains converge to the same distribution, we can trust the results.

---

## 10. Convergence Diagnostics

Two standard diagnostics confirm the sampler worked correctly:

### R-hat (Gelman-Rubin statistic)
Compares variance **within** each chain to variance **between** chains.
- **R-hat ≈ 1.00** → all 4 chains agree → converged
- **R-hat > 1.01** → chains disagree → results cannot be trusted

### ESS (Effective Sample Size)
MCMC draws are correlated, so 4,000 draws ≠ 4,000 independent samples. ESS estimates the equivalent number of independent samples.
- **ESS > 400** → sufficient for reliable posterior estimates

### Results

| Diagnostic | Result | Target | Status |
|---|---|---|---|
| Max R-hat | 1.000 | < 1.01 | PASS |
| Min ESS | 1,336 | > 400 | PASS |

Trace plots (sampling paths per chain) also show healthy mixing — all 4 chains overlap and look like random noise, with no trends or divergences.

---

## 11. Posterior Predictive Check & Fit Metrics

After fitting, we simulate revenue using the posterior parameter samples and compare against actuals.

**Fit metrics** on the full 104-week dataset:

| Metric | Value | Interpretation |
|---|---|---|
| R² | 0.444 | Model explains 44% of revenue variance |
| MAPE | 19.4% | Average prediction error of ~19% |
| RMSE | 37,701 € | Typical prediction error in euros |
| WAIC | -27.3 | Bayesian out-of-sample accuracy estimate |
| LOO | -27.3 | Leave-one-out cross-validation estimate |

> Note: In-sample R² for MMMs is typically modest (0.4–0.7) because the model deliberately avoids overfitting. The holdout metrics in Section 14 are the more meaningful measure of model quality.

---

## 12. Revenue Attribution

Using posterior mean parameters, total revenue is decomposed into:

- **Baseline**: revenue that would exist even with zero advertising (intercept + trend + seasonality)
- **Channel contributions**: revenue driven by each channel's adstock effect

The adstock matrix is recomputed in NumPy (outside the model graph) using posterior mean lambda values, then multiplied by posterior mean channel betas and converted back to euros.

### Attribution Summary (14.19M € total)

| Component | Revenue (k€) | Share |
|---|---|---|
| Baseline | 5,846 | 41.2% |
| Channel 3 | 2,172 | 15.3% |
| Channel 7 | 1,809 | 12.7% |
| Channel 5 | 1,433 | 10.1% |
| Channel 2 | 1,105 | 7.8% |
| Channel 4 | 1,022 | 7.2% |
| Channel 6 | 715 | 5.0% |
| Channel 1 | 225 | 1.6% |

**41.2% of revenue is baseline** — this is revenue the business would generate even without any advertising, driven by brand equity, organic demand, and seasonality. The remaining 58.8% is media-driven.

### Adstock Decay Rates

| Channel | Lambda (mean) | Half-life (weeks) |
|---|---|---|
| Channel 1 | 0.434 | 0.8 |
| Channel 2 | 0.379 | 0.7 |
| Channel 3 | 0.295 | 0.6 |
| Channel 4 | 0.332 | 0.6 |
| Channel 5 | 0.316 | 0.6 |
| Channel 6 | 0.358 | 0.7 |
| Channel 7 | 0.349 | 0.7 |

All channels show relatively short carry-over (< 1 week half-life), suggesting mostly direct-response characteristics rather than long brand-building effects.

---

## 13. ROI Analysis

```
ROI per channel = attributed revenue / total euros spent
```

- ROI = 1.0 → break-even
- ROI = 4.0 → every €1 spent generated €4 in revenue
- ROI < 1.0 → loss-making

Because we have 4,000 posterior samples, ROI is computed for each sample independently, giving a **full posterior distribution** per channel rather than a single number. This quantifies how uncertain our ROI estimates are — critical for budget allocation decisions.

### ROI Ranking (best to worst)

| Channel | Spend (k€) | Attributed Rev (k€) | ROI (mean) | 90% CI |
|---|---|---|---|---|
| Channel 2 | 35.7 | 1,105 | **30.92x** | 5.20 – 62.97 |
| Channel 1 | 129.5 | 225 | 1.74x | 0.10 – 5.54 |
| Channel 5 | 891.9 | 1,433 | 1.61x | 0.20 – 3.10 |
| Channel 4 | 719.2 | 1,022 | 1.42x | 0.10 – 3.28 |
| Channel 6 | 526.6 | 715 | 1.36x | 0.17 – 2.87 |
| Channel 3 | 2,028.7 | 2,172 | 1.07x | 0.18 – 2.05 |
| Channel 7 | 2,880.9 | 1,809 | **0.63x** | 0.07 – 1.37 |

**Key observations:**
- **Channel 2** has by far the highest ROI (30.92x) despite being one of the lowest-spend channels. However, its credible interval is very wide (5–63x), reflecting high uncertainty due to limited spend data.
- **Channel 7** is the only loss-making channel (ROI < 1.0) — it receives the most spend but generates less revenue than it costs.
- **Channels 3 and 5** are the largest revenue drivers in absolute terms but have modest ROI due to high spend volume.

---

## 14. Holdout Validation

### Why not a random train/test split?

Two reasons a random split would be **wrong** for this model:

1. **Adstock is recursive** — week 90's adstock depends on week 89, which depends on week 88, all the way back to week 1. Shuffling rows breaks this chain completely.
2. **Future leaks into the past** — if week 100 is in the training set, the model uses future spend patterns when fitting past revenue.

### Correct approach: time-based split

- **Training set**: weeks 1–78
- **Holdout set**: weeks 79–104 (last 26 weeks, ~25% of data)

The model is completely refitted on training data only. For holdout predictions, adstock is computed over the **full 104-week series** (to preserve training history) and then sliced to the holdout period. Starting adstock from week 79 would assume zero advertising history and produce systematically low predictions.

### Holdout Results

| Metric | Value | Interpretation |
|---|---|---|
| R² | 0.553 | Better than in-sample — model generalizes well |
| MAPE | 15.2% | ~15% average prediction error out-of-sample |
| RMSE | 21,402 € | Typical error in euros on unseen data |
| Coverage | 88% | 88% of actuals fall inside the 90% CI (target ~90%) |

**Coverage of 88% against a target of ~90%** indicates the model is well-calibrated — its uncertainty estimates are honest and not overconfident. The fact that holdout R² (0.553) is **higher** than in-sample R² (0.444) suggests the model has not overfit to the training period.

---

## 15. Final Results

```
============================================================
BAYESIAN MMM — KEY RESULTS
============================================================

IN-SAMPLE FIT (all 104 weeks)
  R²   : 0.444
  MAPE : 19.4%
  RMSE : 37,701 euros

HOLDOUT FIT (trained wks 1–78, tested wks 79–104)
  R²       : 0.553
  MAPE     : 15.2%
  RMSE     : 21,402 euros
  Coverage : 88%  (target ~90%)

REVENUE ATTRIBUTION  (total 14.19M euros)
  Baseline     5,846 k   41.2%
  Channel 3    2,172 k   15.3%
  Channel 7    1,809 k   12.7%
  Channel 5    1,433 k   10.1%
  Channel 2    1,105 k    7.8%
  Channel 4    1,022 k    7.2%
  Channel 6      715 k    5.0%
  Channel 1      225 k    1.6%

BEST CHANNEL BY ROI: Channel 2  (30.92x)
LOSS-MAKING CHANNEL: Channel 7  (0.63x)
============================================================
```

---

## 16. Stack

| Library | Version | Purpose |
|---|---|---|
| [PyMC](https://www.pymc.io/) | ≥ 5.0 | Bayesian model definition and MCMC sampling |
| [PyTensor](https://pytensor.readthedocs.io/) | ≥ 2.0 | Adstock computation inside the model graph |
| [ArviZ](https://python.arviz.org/) | ≥ 0.16 | Posterior analysis, diagnostics, WAIC/LOO |
| [pandas](https://pandas.pydata.org/) | ≥ 2.0 | Data loading and manipulation |
| [NumPy](https://numpy.org/) | ≥ 1.24 | Numerical computations |
| [matplotlib](https://matplotlib.org/) | ≥ 3.7 | Visualizations |
| [seaborn](https://seaborn.pydata.org/) | ≥ 0.12 | Correlation heatmap |

---

## 17. Files

| File | Description |
|---|---|
| `bayesian_mmm.ipynb` | Main notebook — full model pipeline with outputs |
| `MMM_test_data.csv` | Weekly revenue and spend data (104 weeks, 7 channels) |
| `MMM_Analysis_Report.pdf` | Written analysis report |
