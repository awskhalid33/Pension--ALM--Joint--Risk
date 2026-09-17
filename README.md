# Pension ALM under Joint Longevity and Market Risk

I developed this project to explore how a pension fund's assets and liabilities respond when interest rates and mortality assumptions change together. It brings together stochastic processes, survival modelling and asset-liability management in a small Python simulation framework.

The question behind it is: **how much protection can an interest-rate hedge provide when the liabilities also depend on how long members live?**

The framework compares a static bond holding with a regime-aware allocation and a duration hedge based on linear regression. An exact-duration hedge provides a further benchmark. These are experiments under specified assumptions. The model has not been calibrated to a real pension scheme, and it does not establish that a more complicated hedge performs better.

## What the project does

Interest rates evolve through three Nelson–Siegel factors: level, slope and curvature. A mortality index, `kappa`, changes the level of mortality rates. A two-state Markov process allows the rate and mortality dynamics to differ between regimes.

The resulting paths are used to value expected pension payments and a bond/cash portfolio. Funding ratios are recorded through time, then summarised across Monte Carlo paths. A ratio below one means that assets are worth less than the remaining expected payments under the model's valuation assumptions.

The focus is on funding risk. It is not a simulation of a complete pension scheme's investment and benefit administration.

## Liabilities and mortality

The default liability represents a member aged 55 in 2023, with an annual nominal pension of £10,000 starting at age 65. Payments are capped at age 110. Calculations use expected survivor counts per initial member of a large, otherwise identical cohort; individual deaths are not simulated.

The mortality table is generated toy data. Despite the filename `toy_mortality_uk.csv`, it should not be treated as a calibrated UK population or pension-scheme dataset.

Mortality is shifted through `log(m_x) + kappa`. A negative shift reduces mortality rates, increasing expected survival and future payments. This creates a source of liability risk that a bond does not directly hedge.

At each valuation date, past survival retains the mortality levels that applied in those past intervals. New information changes the projection of future survival. For that projection, the current mortality level is held fixed. The simulation can generate future mortality changes, but the valuation does not look ahead to those simulated outcomes.

That distinction matters: a stochastic mortality path and a calibrated forecast of future improvement are not the same thing.

## Assets, timing and hedge strategies

Initial assets equal the initial liability value. The available investments are a zero-coupon bond and a cash account. The default bond matures after 25 years, while the default experiment runs for ten. The code requires bond maturity to fall after the projection horizon, so there is no implicit reinvestment rule after redemption.

Funding ratios are measured **immediately before each annual benefit payment**. That payment then leaves the cash account before interest accrues over the following year. Rebalancing transfers value between the bond and cash; it does not create new assets.

| Strategy | Allocation rule |
|---|---|
| Static | Holds the initial bond notional; benefits are financed through cash |
| Regime-aware | Targets a specified bond-value-to-liability ratio for each observed regime |
| ML duration | Uses ridge linear regression to estimate liability duration, then sets a bounded hedge ratio |
| Exact duration | Uses the directly calculated liability duration under the same hedge limits |

The regression uses curve factors, mortality information and elapsed time. It is an interpretable approximation to duration, not a model trained directly to maximise returns. The exact-duration benchmark helps separate approximation error from the limitations of duration hedging itself.

In the ML study, static, ML and exact-duration hedges use the same evaluation paths. Training and evaluation seed ranges must not overlap. The separate regime-aware comparison also gives its two strategies the same paths; its default seed differs from the ML study's.

Regime-aware decisions observe the simulated state directly. A practical strategy would need a way to estimate that state. Negative cash is permitted and accrues at the same short rate as deposits, with no borrowing limits or collateral requirements.

## Reading the results

The main outputs describe the funding ratio at the horizon and the minimum ratio reached along each path. The summaries include means, percentiles and probabilities of falling below 0.95 or 0.90.

A higher mean funding ratio does not necessarily imply better downside protection. The lower percentiles and the probability of a shortfall need to be considered alongside it. The minimum funding ratio also addresses a different question from the final ratio: a fund can recover by the horizon after experiencing a substantial deficit earlier.

Earlier outputs need to be regenerated following corrections to benefit-payment accounting and mortality revaluation. I have not presented those older figures as evidence for the revised implementation. The regression tests and short execution checks establish specific software properties; they are not a completed large-sample performance study.

The current evidence does not isolate the contribution of longevity risk or regime persistence. That would require controlled comparisons, such as fixed versus stochastic mortality and alternative transition matrices, with uncertainty reported for the differences.

## Running the project

From the repository root, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

The toy mortality CSV is included. To rebuild it and run a single path:

```bash
python -m src.build_toy_data
python -m src.run_regime_joint_demo --n-years 10 --seed 42
```

Run the Monte Carlo comparisons:

```bash
python -m src.run_monte_carlo_regime_model --n-paths 1000 --n-years 10
python -m src.run_monte_carlo_dynamic_vs_static --n-paths 1000 --h-normal 1.0 --h-stress 1.1
python -m src.run_monte_carlo_ml_hedge --n-paths 1000 --n-train-paths 400
```

Use `--no-plots` for a headless run and `--metadata-tag` to distinguish saved metadata. For example, this requests 10,000 evaluation paths; the command itself is not evidence that such a run has already been completed:

```bash
python -m src.run_monte_carlo_ml_hedge --n-paths 10000 --n-train-paths 400 --no-plots --metadata-tag 10000_paths
```

## Outputs and reproducibility

Runs write to `outputs/`. Metadata records the command-line arguments, timestamp, available Git commit hash and summary statistics. The ML study retains horizon and minimum-funding summaries for all three of its strategies. The joint-regime Monte Carlo command also writes path-level terminal and minimum funding ratios to CSV.

Keep the metadata with the exact code and input table used. Save console output if needed, and record the installed environment with `python -m pip freeze`. Reusing an output name can overwrite a previous run; a metadata tag distinguishes metadata files, not every chart or CSV.

## Validation and remaining work

Tests cover process reproducibility, input validation, survival calculations, valuation and funding-account consistency. A regression check holds rates and mortality unchanged through retirement: benefit payments must reduce assets and liabilities consistently, without creating a funding surplus. Another checks that later mortality information cannot rewrite past survivor counts.

The single-path funding demos use the same corrected accounting as the Monte Carlo studies. Where rates are simulated quarterly, the funding account still accrues cash using annual start-of-year short rates. Intra-year variation is not integrated.

Real-data calibration would be a substantial next step. So would testing regime estimation, borrowing constraints, transaction costs and richer member populations. Inflation-linked pensions and spouse benefits are outside the current scope. The simulations remain useful for examining mechanisms, but the assumptions limit how far their results can be generalised.

## Code layout

| Location | Purpose |
|---|---|
| `src/finance/` | Yield curves, stochastic rate factors and bond valuation |
| `src/mortality/` and `src/regimes/` | Mortality changes and regime simulation |
| `src/liabilities/` | Survival, expected payments and liability valuation |
| `src/sim/hedge_accounting.py` | Shared funding account and mortality revaluation |
| `src/run_monte_carlo_*.py` | Monte Carlo studies and hedge comparisons |
| `src/run_*_demo.py` | Single-path illustrations |
| `tests/` | Automated checks |

This is an educational research project, not a production pension valuation or investment system.
