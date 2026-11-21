# DS5020 Final Project – Project 7: Bayesian A/B Testing

Iteration 2: the math foundation (definitions, derivations, data and simulation).
Iteration 3: a small worked example, a from-scratch implementation, and verification
against the known truth (simulation) and against a library (real data).

## Layout

```
src/        Python code
data/       raw data (not tracked in Git; see below)
figures/    plots produced by the scripts
docs/       Overleaf report (report.tex + iteration3.tex → report.pdf) and the Excel project plan
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

## Running (Iteration 3)

Run from the repository root. Each script prints its results and writes the
numbers the report uses into `docs/it3_*.tex`.

```bash
python3 src/small_example.py       # the tiny example (A: 3/10, B: 5/10), step by step, with exact fractions
python3 src/check_simulation.py    # the simulation test: truth and library comparisons, error vs n, stress test
python3 src/run_real.py            # the real data: cleaning checks, all methods, library comparison
python3 src/check_assumptions.py   # the Iteration 2 assumptions, on simulated data first and then on the real data
```

| file | what it holds |
| --- | --- |
| `src/methods.py` | the method from scratch: one `my_` function per algorithm step, the equation in a comment above each |
| `src/simulate.py` | the Iteration 2 simulation as functions (`simulate_ab(pA, pB, n, seed)` and the stress-test variants) |
| `src/small_example.py` | the small worked example |
| `src/check_simulation.py` | the simulation test |
| `src/run_real.py` | the real data and the library comparison |
| `src/check_assumptions.py` | the assumption checks |
| `src/texio.py` | writes results as LaTeX for the report |

`src/methods.py` uses NumPy only (plus `math.lgamma` and `math.erfc` from the
standard library). SciPy is imported only in the check scripts, to compare
answers; pandas only reads the CSV.

Same model, parameters and seed as Iteration 2: p_A = 0.10, p_B = 0.12,
n = 1000, `np.random.default_rng(0)`, so `simulate_ab(seed=0)` returns the
Iteration 2 data x_A = 92, x_B = 121. The Monte Carlo draws use seed 1.

Requirements: `numpy`, `scipy`, `pandas`, `matplotlib`.

## Running (Iteration 2)

```bash
python3 src/load_data.py       # describes the data, writes figures/ and docs/data_summary.tex
python3 src/simulation.py      # Iteration 2 simulation tables, writes figures/ and docs/sim_tables.tex
```

## Report

`docs/report.tex` is the Overleaf source; it `\input`s `docs/iteration3.tex`
(the Iteration 3 sections) and the generated files `docs/sim_macros.tex`,
`docs/sim_tables.tex`, `docs/data_macros.tex`, `docs/data_summary.tex` and
`docs/it3_*.tex`. Upload all of these and the `figures/` folder to Overleaf.
`docs/report.pdf` is the compiled report.
