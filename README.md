# DS5020 Final Project – Project 7: Bayesian A/B Testing

Does moving the Cookie Cats gate from level 30 to level 40 change how many players come
back after 1 day and after 7 days? This repository holds the code, the final report and
the presentation slides.

**Result:** moving the gate to level 40 lowers 7-day retention.
Pr(level 30 is better | data) = 0.9992; the posterior mean of p_B − p_A is −0.0082 with
95% interval [−0.0133, −0.0031] (about 8 fewer of every 1,000 players); z test p-value 0.0016.
1-day retention points the same way with weaker evidence (0.9628; p-value 0.0744).

The project was built in iterations: Iteration 2 derived the math, ran a simulation and
described the data; Iteration 3 added a hand-worked example, a from-scratch implementation
and verification against the known truth and against SciPy; the final report puts it
all together.

## Layout

```
src/        Python code
data/       raw data (not tracked in Git; see below)
figures/    plots produced by the scripts
docs/       final report (report.tex → report.pdf), slides (slides.pdf) and the Excel project plan
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

## Report and slides

`docs/report.tex` is the Overleaf source of the final report. It `\input`s only the
generated files `docs/sim_macros.tex`, `docs/sim_tables.tex`, `docs/data_macros.tex`,
`docs/data_summary.tex` and `docs/it3_*.tex`, so every number in the report comes from
the scripts. Upload these and the `figures/` folder to Overleaf.
`docs/report.pdf` is the compiled report and `docs/slides.pdf` the presentation slides.
