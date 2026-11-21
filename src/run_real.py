"""
DS5020 Project 7 -- Bayesian A/B Testing
Run on the real data (Cookie Cats) and verify against a library (Iteration 3, Question 5).

  1. clean: missing values, duplicate users, users in both groups, group sizes
  2. run every method (the same my_ functions as the small example and the simulation)
     on 1-day and on 7-day retention separately
  3. compare each number with SciPy and print the differences with report()

A = gate_30, B = gate_40.  SciPy is imported for checking only; pandas only reads the file.

Run:  python3 src/run_real.py        (needs data/cookie_cats.csv, see README)
Writes figures/it3_real_posteriors.png, docs/it3_real.tex, docs/it3_real_table.tex, docs/it3_rows_real.tex
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate                  # check only

from methods import (my_ab_test, my_posterior, my_post_mean_var, my_beta_draws, my_beta_pdf,
                     my_grid, my_grid_prob, my_check_users, my_counts, my_p_value, report)
from texio import write_tex, row, sci, thousands, FIG, DATA

if not os.path.exists(DATA):
    sys.exit("data/cookie_cats.csv not found. Download it from "
             "https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing "
             "and save it as data/cookie_cats.csv")

mac, rows, tab = {}, [], []

# ===================================================================== 1. clean
print("=" * 78, "\n1. Check and clean the data\n")
df = pd.read_csv(DATA)
n_raw = len(df)
n_missing = int(df.isna().sum().sum())
print(f"rows read: {n_raw:,};  missing entries: {n_missing}")
df = df.dropna(subset=["userid", "version", "retention_1", "retention_7"])
n_after_missing = len(df)

chk = my_check_users(df["userid"].to_numpy(), df["version"].to_numpy())
print(f"unique userids: {chk['unique_users']:,};  duplicate rows: {chk['duplicate_rows']};  "
      f"users in both groups: {chk['users_in_both_groups']}")
df = df[chk["keep"]]
n_clean = len(df)
print(f"rows removed: {n_raw - n_after_missing} (missing) + {n_after_missing - n_clean} (duplicate / both groups) "
      f"= {n_raw - n_clean};  rows kept: {n_clean:,}")
other = sorted(set(df["version"]) - {"gate_30", "gate_40"})
print(f"version labels other than gate_30 / gate_40: {other if other else 'none'}")

version = df["version"].to_numpy()
nA, nB = int(np.sum(version == "gate_30")), int(np.sum(version == "gate_40"))
print(f"group sizes: n_A = {nA:,} (gate_30),  n_B = {nB:,} (gate_40);  ratio n_A/n_B = {nA / nB:.4f};  "
      f"share of A = {nA / (nA + nB):.4f}")
print("transformations: retention_1 and retention_7 (True/False) are used as y = 1/0; no other change.")
print("train/test split: none (the method estimates two rates; nothing is predicted).")
mac.update(realRaw=thousands(n_raw), realMissing=str(n_missing), realDup=str(chk["duplicate_rows"]),
           realBoth=str(chk["users_in_both_groups"]), realRemoved=str(n_raw - n_clean),
           realClean=thousands(n_clean), realUnique=thousands(chk["unique_users"]),
           realNA=thousands(nA), realNB=thousands(nB), realRatio=f"{nA / nB:.4f}", realFewer=f"{100 * (1 - nA / nB):.1f}",
           realShareA=f"{nA / (nA + nB):.4f}")

# ===================================================================== 2-3. run and verify
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for k, (col, label, tag) in enumerate([("retention_1", "1-day retention", "One"),
                                       ("retention_7", "7-day retention", "Seven")]):
    print("\n" + "=" * 78, f"\n2. {label}\n")
    y = df[col].to_numpy().astype(int)
    xA, _ = my_counts(version, y, "gate_30")
    xB, _ = my_counts(version, y, "gate_40")
    res = my_ab_test(xA, nA, xB, nB, a=1, b=1, M=100_000, seed=1, n_grid=2000)
    aA, bA, aB, bB = res["aA"], res["bA"], res["aB"], res["bB"]
    print(f"counts: x_A = {xA:,} of {nA:,} ({xA / nA:.4f});  x_B = {xB:,} of {nB:,} ({xB / nB:.4f})")
    print(f"Step 1  posteriors: p_A ~ Beta({aA}, {bA}),  p_B ~ Beta({aB}, {bB})")
    print(f"Step 2  mu_A = {res['muA']:.6f} (sd {np.sqrt(res['varA']):.6f}),  "
          f"mu_B = {res['muB']:.6f} (sd {np.sqrt(res['varB']):.6f})")
    print(f"Steps 3-5  theta_hat = Pr(p_B > p_A | data) = {res['theta_mc']:.5f} +/- {res['se']:.5f}  (M = 100,000)")
    print(f"Prop. 1    theta on the grid = {res['theta_grid']:.8f};  Pr(p_A > p_B | data) = {1 - res['theta_grid']:.8f}")
    print(f"Step 6  z = {res['z']:.4f},  two-sided p-value = {res['p_value']:.5f}")
    print(f"Step 7  {res['decision']}")
    # posterior of the difference p_B - p_A from the same kind of draws
    rng = np.random.default_rng(1)
    d = my_beta_draws(aB, bB, 100_000, rng) - my_beta_draws(aA, bA, 100_000, rng)
    lo, hi = np.quantile(d, [0.025, 0.975])
    print(f"        posterior of p_B - p_A: mean {d.mean():+.5f}, 95% interval [{lo:+.5f}, {hi:+.5f}]")

    print("\n3. Library comparison")
    t = my_grid(aA, bA, aB, bB)
    mA, mB = aA / (aA + bA), aB / (aB + bB)
    p_lib = integrate.quad(lambda p: stats.beta.pdf(p, aB, bB) * stats.beta.cdf(p, aA, bA),
                           t[0], t[-1], points=sorted([mA, mB]), epsabs=1e-13, epsrel=1e-13, limit=500)[0]
    e_mom, _ = report("posterior mean, var vs scipy",
                      [res["muA"], res["muB"], res["varA"], res["varB"]],
                      [stats.beta.mean(aA, bA), stats.beta.mean(aB, bB),
                       stats.beta.var(aA, bA), stats.beta.var(aB, bB)])
    e_pdf, r_pdf = report("Beta density vs scipy", my_beta_pdf(t, aA, bA), stats.beta.pdf(t, aA, bA))
    e_grid, r_grid = report("grid theta vs quad", res["theta_grid"], p_lib)
    n_se = abs(res["theta_mc"] - p_lib) / res["se"]
    print(f"{'MC theta vs quad':32s} |diff| = {abs(res['theta_mc'] - p_lib):.2e} = {n_se:.2f} SE")
    chi2, p_chi2 = stats.chi2_contingency([[xA, nA - xA], [xB, nB - xB]], correction=False)[:2]
    e_z, r_z = report("z^2 vs chi2_contingency", res["z"] ** 2, chi2)
    e_p, _ = report("p-value vs chi2_contingency", res["p_value"], p_chi2)
    # prior sensitivity (Assumption 4)
    sens = {pr: my_grid_prob(*my_posterior(xA, nA, *pr), *my_posterior(xB, nB, *pr)) for pr in [(1, 1), (0.5, 0.5), (2, 2)]}
    print("prior sensitivity of theta:", {f"Beta{pr}": round(v, 6) for pr, v in sens.items()})

    name = f"Real data ({'1' if tag == 'One' else '7'}-day)"
    rows += [
        row(name, "$\\mu_A,\\mu_B,\\sigma^2_A,\\sigma^2_B$", "\\texttt{stats.beta}", "max$|\\cdot|=%s$" % sci(e_mom), e_mom < 1e-8),
        row(name, "$\\theta$ (grid)", "\\texttt{integrate.quad}", "$%s$" % sci(e_grid), e_grid < 1e-6),
        row(name, "$\\hat\\theta_M$ (Monte Carlo)", "\\texttt{integrate.quad}",
            "$%s$ ($%.2f$ SE)" % (sci(abs(res["theta_mc"] - p_lib)), n_se), n_se < 3),
        row(name, "$z^2$", "\\texttt{chi2\\_contingency}", "rel.\\ $%s$" % sci(r_z), r_z < 1e-8),
    ]
    mac.update({f"real{tag}XA": thousands(xA), f"real{tag}XB": thousands(xB),
                f"real{tag}RateA": f"{xA / nA:.4f}", f"real{tag}RateB": f"{xB / nB:.4f}",
                f"real{tag}Theta": f"{res['theta_grid']:.4f}", f"real{tag}ThetaA": f"{1 - res['theta_grid']:.4f}",
                f"real{tag}MC": f"{res['theta_mc']:.5f}", f"real{tag}SE": f"{res['se']:.5f}",
                f"real{tag}NSE": f"{n_se:.2f}", f"real{tag}Z": f"{res['z']:.3f}", f"real{tag}P": f"{res['p_value']:.4f}",
                f"real{tag}Phalf": f"{res['p_value'] / 2:.5f}", f"real{tag}ThetaFive": f"{res['theta_grid']:.5f}",
                f"real{tag}SdA": f"{np.sqrt(res['varA']):.4f}", f"real{tag}SdB": f"{np.sqrt(res['varB']):.4f}",
                f"real{tag}MuA": f"{res['muA']:.4f}", f"real{tag}MuB": f"{res['muB']:.4f}",
                f"real{tag}Diff": f"{d.mean():+.4f}", f"real{tag}Lo": f"{lo:+.4f}", f"real{tag}Hi": f"{hi:+.4f}",
                f"real{tag}GridErr": sci(e_grid), f"real{tag}MomErr": sci(e_mom), f"real{tag}ZErr": sci(r_z),
                f"real{tag}PdfErr": sci(r_pdf), f"real{tag}PErr": sci(e_p),
                f"real{tag}SensHalf": f"{sens[(0.5, 0.5)]:.5f}", f"real{tag}SensOne": f"{sens[(1, 1)]:.5f}",
                f"real{tag}SensTwo": f"{sens[(2, 2)]:.5f}",
                f"real{tag}SensRange": sci(max(sens.values()) - min(sens.values()))})
    tab.append(f"{label} & {thousands(xA)} & {thousands(xB)} & {res['muA']:.4f} & {res['muB']:.4f} & "
               f"{np.sqrt(res['varA']):.4f} & {res['theta_grid']:.5f} & {res['theta_mc']:.5f} $\\pm$ {res['se']:.5f} & "
               f"{res['z']:.2f} & {res['p_value']:.4f}\\\\")

    ax = axs[k]
    tt = np.linspace(min(mA, mB) - 0.012, max(mA, mB) + 0.012, 600)
    ax.plot(tt, my_beta_pdf(tt, aA, bA), label="gate_30 (A)")
    ax.plot(tt, my_beta_pdf(tt, aB, bB), label="gate_40 (B)")
    ax.set_title(f"{label}: Pr($p_B>p_A$ | data) = {res['theta_grid']:.3f}")
    ax.set_xlabel("p"); ax.set_ylabel("posterior density"); ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "it3_real_posteriors.png"), dpi=150); plt.close(fig)

tab_lines = ["\\begin{table}[ht]\\centering",
             "\\caption{Cookie Cats, all methods ($n_A=%s$, $n_B=%s$, prior $\\Betad(1,1)$, $M=100{,}000$). "
             "$\\theta=\\Pr(p_B>p_A\\mid\\data)$.}\\label{tab:real}" % (thousands(nA), thousands(nB)),
             "\\small\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lrrrrrrrrr}\\toprule",
             "outcome & $x_A$ & $x_B$ & $\\mu_A$ & $\\mu_B$ & $\\sigma_A$ & $\\theta$ (grid) & $\\hat\\theta_M\\pm$SE & $z$ & $p$-value\\\\ \\midrule"]
tab_lines += tab + ["\\bottomrule\\end{tabular}\\normalsize\\end{table}"]
write_tex("it3_real.tex", macros=mac, script="src/run_real.py")
write_tex("it3_real_table.tex", lines=tab_lines, script="src/run_real.py")
write_tex("it3_rows_real.tex", macros={"rowsReal": "\n".join(rows)}, script="src/run_real.py")
print("wrote figures/it3_real_posteriors.png")
