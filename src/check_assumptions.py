"""
DS5020 Project 7 -- Bayesian A/B Testing
Check the Iteration 2 assumptions (Iteration 3, item 6).

Every check is run on simulated data first, where the assumption holds by
construction, so we see what "passing" looks like; then on the Cookie Cats data.

  A1 common rate within a version   chi-square across 10 consecutive batches (file order)
  A2 independent trials             unique userid; lag-1 autocorrelation of the outcomes
  A3 randomised assignment          group sizes against a fair coin
  A4 prior                          theta under Beta(1,1), Beta(.5,.5), Beta(2,2)
  A5 large samples                  smallest of n p_hat and n (1 - p_hat)
  A6 i.i.d. Monte Carlo draws       spread of theta_hat over 30 seeds against the SE formula
  A7 no missing outcomes            missing entries

SciPy is used only for the chi-square tail probability.

Run:  python3 src/check_assumptions.py
Writes docs/it3_assump.tex
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats                              # tail probabilities only

from methods import (my_posterior, my_mc_prob, my_grid_prob, my_check_users, my_counts, my_p_value)
from simulate import simulate_ab_coin
from texio import write_tex, DATA, thousands

K = 10            # batches for A1


# A1: if every user in a version has the same rate p, the batch counts x_k ~ Binomial(n_k, p) and
#     X^2 = sum_k (x_k - n_k p_hat)^2 / (n_k p_hat (1 - p_hat))  is approximately chi-square(K - 1).
def batch_chi2(y, K=K):
    batches = np.array_split(np.asarray(y, float), K)
    nk = np.array([len(b) for b in batches]); xk = np.array([b.sum() for b in batches])
    p_hat = xk.sum() / nk.sum()
    X2 = np.sum((xk - nk * p_hat) ** 2 / (nk * p_hat * (1 - p_hat)))
    return X2, stats.chi2.sf(X2, K - 1)


# A2: lag-1 autocorrelation r_1 = sum (y_i - ybar)(y_{i+1} - ybar) / sum (y_i - ybar)^2;
#     for independent trials r_1 is approximately N(0, 1/n).
def lag1(y):
    y = np.asarray(y, float) - np.mean(y)
    r1 = np.sum(y[:-1] * y[1:]) / np.sum(y * y)
    return r1, r1 * np.sqrt(len(y))


# A3: under a fair coin n_A ~ Binomial(N, 1/2), so (n_A - N/2) / sqrt(N/4) is approximately N(0, 1).
def split_z(nA, nB):
    N = nA + nB
    return (nA - N / 2) / np.sqrt(N / 4)


def run_checks(name, userid, version, y, labels):
    A, B = labels
    out = {}
    print("\n" + "=" * 78, f"\n{name}\n")
    chk = my_check_users(userid, version)
    xA, nA = my_counts(version, y, A); xB, nB = my_counts(version, y, B)
    yA, yB = y[version == A], y[version == B]

    X2A, pA_ = batch_chi2(yA); X2B, pB_ = batch_chi2(yB)
    print(f"A1 common rate: chi-square over {K} batches (df {K - 1}):  A {X2A:.1f} (p = {pA_:.3f}),  B {X2B:.1f} (p = {pB_:.3f})")
    out.update(XA=f"{X2A:.1f}", XAp=f"{pA_:.3f}", XB=f"{X2B:.1f}", XBp=f"{pB_:.3f}")

    rA, zA = lag1(yA); rB, zB = lag1(yB)
    print(f"A2 independence: duplicate rows {chk['duplicate_rows']};  lag-1 autocorrelation A {rA:+.4f} (z = {zA:+.2f}),  "
          f"B {rB:+.4f} (z = {zB:+.2f})")
    out.update(Dup=str(chk["duplicate_rows"]), RA=f"{rA:+.4f}", RAz=f"{zA:+.2f}", RB=f"{rB:+.4f}", RBz=f"{zB:+.2f}")

    zs = split_z(nA, nB)
    print(f"A3 randomisation: users in both groups {chk['users_in_both_groups']};  n_A = {nA:,}, n_B = {nB:,};  "
          f"z for a 50/50 split = {zs:+.2f} (p = {my_p_value(zs):.4f})")
    out.update(Both=str(chk["users_in_both_groups"]), SplitZ=f"{zs:+.2f}", SplitP=f"{my_p_value(zs):.4f}",
               NA=thousands(nA), NB=thousands(nB))

    th = {pr: my_grid_prob(*my_posterior(xA, nA, *pr), *my_posterior(xB, nB, *pr)) for pr in [(1, 1), (0.5, 0.5), (2, 2)]}
    spread = max(th.values()) - min(th.values())
    print(f"A4 prior: theta = {th[(1, 1)]:.6f} (Beta(1,1)), {th[(0.5, 0.5)]:.6f} (Beta(.5,.5)), {th[(2, 2)]:.6f} (Beta(2,2));  "
          f"range {spread:.1e};  weight on the data w = {nA / (2 + nA):.5f}")
    out.update(Theta=f"{th[(1, 1)]:.5f}", PriorRange=f"{spread:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}",
               W=f"{nA / (2 + nA):.5f}")

    p_hat = (xA + xB) / (nA + nB)
    smallest = min(nA, nB) * min(p_hat, 1 - p_hat)
    print(f"A5 large samples: smallest of n p_hat, n (1 - p_hat) = {smallest:,.0f}  (rule of thumb: at least 10)")
    out.update(Smallest=thousands(int(round(smallest))))

    aA, bA = my_posterior(xA, nA); aB, bB = my_posterior(xB, nB)
    M = 20_000
    est = np.array([my_mc_prob(aA, bA, aB, bB, M=M, seed=s)[0] for s in range(30)])
    se_formula = np.sqrt(th[(1, 1)] * (1 - th[(1, 1)]) / M)
    print(f"A6 Monte Carlo: sd of theta_hat over 30 seeds = {est.std(ddof=1):.5f};  sqrt(theta(1-theta)/M) = {se_formula:.5f};  "
          f"ratio {est.std(ddof=1) / se_formula:.2f}")
    out.update(McSd=f"{est.std(ddof=1):.5f}", McSe=f"{se_formula:.5f}", McRatio=f"{est.std(ddof=1) / se_formula:.2f}")
    return out


mac = {}

# ---------------------------------------------------------------- simulated data first
uid, ver, y = simulate_ab_coin(pA=0.10, pB=0.12, N=2000, seed=0)
res = run_checks("Simulated data (Iteration 2 model, 2000 users, coin-flip groups): assumptions hold by construction",
                 uid, ver, y.astype(int), ("A", "B"))
print("A7 missing outcomes: 0 (simulated)")
mac.update({"asSim" + k: v for k, v in res.items()})

# how often does the A1 check raise a false alarm when the assumption is true?  (500 simulated data sets)
alarms = []
for s_ in range(500):
    _, v_, y_ = simulate_ab_coin(pA=0.10, pB=0.12, N=2000, seed=1000 + s_)
    alarms.append(batch_chi2(y_[v_ == "A"])[1] < 0.05)
print(f"A1 over 500 simulated data sets: the batch test gives p < 0.05 in {np.mean(alarms):.3f} of them (expected 0.05)")
mac.update(asSimAlarm=f"{np.mean(alarms):.3f}")

# ---------------------------------------------------------------- real data
if not os.path.exists(DATA):
    sys.exit("data/cookie_cats.csv not found (see README)")
df = pd.read_csv(DATA)
n_missing = int(df.isna().sum().sum())
for col, tag in (("retention_1", "One"), ("retention_7", "Seven")):
    res = run_checks(f"Cookie Cats, {col} (rows in file order = increasing userid)",
                     df["userid"].to_numpy(), df["version"].to_numpy(), df[col].to_numpy().astype(int),
                     ("gate_30", "gate_40"))
    mac.update({"as" + tag + k: v for k, v in res.items()})
print(f"\nA7 missing outcomes: {n_missing} missing entries in {len(df):,} rows")

# extra look at A1/A3 on the real data: is the outcome rate the same for all kinds of player?
print("\nExtra (A1, A3): engagement and retention")
g = df["sum_gamerounds"]
for v in ("gate_30", "gate_40"):
    s = g[df.version == v]
    print(f"  {v}: median rounds {s.median():.0f}, mean {s.mean():.2f}, share with 0 rounds {np.mean(s == 0):.4f}, max {s.max()}")
q = pd.qcut(g, 4, duplicates="drop")
strata = df.groupby(q, observed=True)["retention_7"].mean()
print("  7-day retention by quartile of sum_gamerounds:", [f"{v:.3f}" for v in strata.values])
mac.update(asMissing=str(n_missing), asMaxRounds=thousands(int(g.max())),
           asZeroA=f"{np.mean(g[df.version == 'gate_30'] == 0):.4f}",
           asZeroB=f"{np.mean(g[df.version == 'gate_40'] == 0):.4f}",
           asMedA=f"{g[df.version == 'gate_30'].median():.0f}", asMedB=f"{g[df.version == 'gate_40'].median():.0f}",
           asStrataLow=f"{strata.values[0]:.3f}", asStrataHigh=f"{strata.values[-1]:.3f}")
write_tex("it3_assump.tex", macros=mac, script="src/check_assumptions.py")
