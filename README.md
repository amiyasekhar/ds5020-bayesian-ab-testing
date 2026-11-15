# DS5020 Final Project – Project 7: Bayesian A/B Testing

Iteration 2: the math foundation (definitions, derivations, data and simulation).

## Layout

```
src/        Python code (data loading + simulation check only)
data/       raw data (not tracked in Git; see below)
figures/    plots produced by the scripts
docs/       Overleaf report (report.tex → report.pdf) and the Excel project plan
```

## The method in one line

For each version of the game we model the number of returning players as
Binomial(n, p), put a Beta(a, b) prior on p, obtain the Beta(a + x, b + n − x)
posterior in closed form, and estimate Pr(p_B > p_A | data) by Monte Carlo.
A frequentist two-proportion z test is derived as the comparison.

## Data: Cookie Cats (Kaggle)

`cookie_cats.csv` is **not** in the repository. Download it from
<https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing> (about 90,000
rows, 2.9 MB) and save it as `data/cookie_cats.csv`.

Columns: `userid`, `version` (`gate_30` = A, `gate_40` = B), `sum_gamerounds`,
`retention_1`, `retention_7`.  One row is one player (one trial); a "click" is
the player returning 1 day (or 7 days) after installing.

## Running

```bash
python3 src/load_data.py       # describes the data, writes figures/ and docs/data_summary.tex
python3 src/simulation.py      # simulation with known truth, writes figures/ and docs/sim_tables.tex
```

Only `numpy`, `pandas`, `matplotlib` and `scipy.stats` are used.  The random
seed is fixed with `np.random.default_rng(0)` so every member gets the same
numbers.  No library routine solves the whole method; `scipy.stats.beta` is
used only to draw from the posterior and to compare against the closed-form
posterior mean and variance we derive.

## Report

`docs/report.tex` is the Overleaf source.  It `\input`s `docs/sim_macros.tex`, `docs/sim_tables.tex`, `docs/data_macros.tex`
and `docs/data_summary.tex`, which the scripts generate, so upload those four
files (and the `figures/` folder) to Overleaf alongside `report.tex`.
