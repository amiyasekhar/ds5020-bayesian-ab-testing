"""
DS5020 Project 7 -- Bayesian A/B Testing
Simulation test: recover the known truth (Iteration 3).

The Iteration 2 experiment: p_A = 0.10, p_B = 0.12, n = 1000 per group, seed 0.
  1. handout block: my_ functions vs SciPy (library) on the Iteration 2 data
  2. comparison with the truth for n = 50, 500, 5000, 50000 and a log-log error plot
  3. the whole experiment repeated 1000 times with p_A = p_B = 0.10 (size of the z test)
  4. stress test: 10% of the users are put in both groups

SciPy is imported for checking only.

Run:  python3 src/check_simulation.py
Writes figures/it3_error_vs_n.png, figures/it3_stress.png, docs/it3_sim.tex, docs/it3_sim_table.tex, docs/it3_rows_sim.tex
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate                  # check only

from methods import (my_posterior, my_post_mean_var, my_mc_prob, my_grid_prob, my_grid,
                     my_beta_pdf, my_z_stat, my_p_value, my_check_users, my_counts, report)
from simulate import simulate_ab, simulate_ab_many, simulate_ab_users, simulate_overlap_many
from texio import write_tex, row, sci, FIG
import os

pA_true, pB_true = 0.10, 0.12
rows, mac = [], {}

# =====================================================================
print("=" * 78, "\n1. Iteration 2 simulation: my functions vs the library\n")
xA, xB, n = simulate_ab(pA=0.10, pB=0.12, n=1000, seed=0)           # src/simulate.py
print(f"data: x_A = {xA}, x_B = {xB}, n = {n} per group   (Iteration 2 gave 92 and 121)")
(aA, bA), (aB, bB) = my_posterior(xA, n), my_posterior(xB, n)       # uniform prior
p_mc, se = my_mc_prob(aA, bA, aB, bB, M=100_000, seed=1)
p_grid = my_grid_prob(aA, bA, aB, bB, n_grid=2000)                  # Q4
p_lib = integrate.quad(lambda p: stats.beta.pdf(p, aB, bB) *
                       stats.beta.cdf(p, aA, bA), 0, 1)[0]
print(p_mc, se, p_grid, p_lib, abs(p_mc - p_lib) / se)              # last number < 3
z = my_z_stat(xA, xB, n, n)
print(z, 2 * stats.norm.sf(abs(z)))

print()
muA, varA = my_post_mean_var(aA, bA)
muB, varB = my_post_mean_var(aB, bB)
e_grid, _ = report("grid theta vs quad", p_grid, p_lib)
e_mom, _ = report("posterior mean, var vs scipy", [muA, muB, varA, varB],
                  [stats.beta.mean(aA, bA), stats.beta.mean(aB, bB),
                   stats.beta.var(aA, bA), stats.beta.var(aB, bB)])
t = my_grid(aA, bA, aB, bB)
e_pdf, r_pdf = report("Beta density vs scipy", my_beta_pdf(t, aA, bA), stats.beta.pdf(t, aA, bA))
# two-proportion z: z^2 is Pearson's chi-square for the 2x2 table (no continuity correction)
chi2, p_chi2 = stats.chi2_contingency([[xA, n - xA], [xB, n - xB]], correction=False)[:2]
e_z, _ = report("z^2 vs chi2_contingency", z ** 2, chi2)
e_p, _ = report("p-value vs norm.sf", my_p_value(z), 2 * stats.norm.sf(abs(z)))
n_se_mc = abs(p_mc - p_lib) / se
print(f"Monte Carlo vs quad: |diff| = {abs(p_mc - p_lib):.5f} = {n_se_mc:.2f} SE;  vs grid: "
      f"{abs(p_mc - p_grid) / se:.2f} SE")

rows += [
    row("Simulation ($n=1000$)", "$\\theta$ (grid)", "\\texttt{integrate.quad}", "$%s$" % sci(e_grid), e_grid < 1e-6),
    row("Simulation ($n=1000$)", "$\\hat\\theta_M$ (Monte Carlo)", "\\texttt{integrate.quad}",
        "$%.4f$ ($%.2f$ SE)" % (abs(p_mc - p_lib), n_se_mc), n_se_mc < 3),
    row("Simulation ($n=1000$)", "$\\mu_A,\\mu_B,\\sigma^2_A,\\sigma^2_B$", "\\texttt{stats.beta}",
        "max$|\\cdot|=%s$" % sci(e_mom), e_mom < 1e-8),
    row("Simulation ($n=1000$)", "$z^2$", "\\texttt{chi2\\_contingency}", "$%s$" % sci(e_z), e_z < 1e-8),
]
mac.update(simThetaMC=f"{p_mc:.5f}", simSE=f"{se:.5f}", simThetaGrid=f"{p_grid:.8f}",
           simThetaLib=f"{p_lib:.8f}", simGridErr=sci(e_grid), simNSE=f"{n_se_mc:.2f}",
           simZ=f"{z:.4f}", simP=f"{my_p_value(z):.4f}", simMomErr=sci(e_mom),
           simPdfErr=sci(r_pdf), simZErr=sci(e_z), simPErr=sci(e_p),
           simMuA=f"{muA:.5f}", simMuB=f"{muB:.5f}",
           simSdA=f"{np.sqrt(varA):.5f}", simSdB=f"{np.sqrt(varB):.5f}")

# =====================================================================
print("\n" + "=" * 78, "\n2. Comparison with the truth; error against n\n")
# one run per n (seed 0): |mu - p| should be within 3 posterior sd (3 standard errors)
ns = [50, 500, 5000, 50000]
R = 2000
single, rmse, theory, tab = [], [], [], []
print(f"{'n':>6} {'x_A':>6} {'x_B':>6} {'mu_A':>8} {'mu_B':>8} {'|mu_A-pA|':>10} {'in SE':>6} {'|mu_B-pB|':>10} {'in SE':>6}"
      f" {'RMSE_A':>9} {'sqrt(pq/n)':>10}")
all_within = True
for nn in ns:
    xa, xb, _ = simulate_ab(pA_true, pB_true, nn, seed=0)
    mA, vA = my_post_mean_var(*my_posterior(xa, nn))
    mB, vB = my_post_mean_var(*my_posterior(xb, nn))
    eA, eB = abs(mA - pA_true), abs(mB - pB_true)
    kA, kB = eA / np.sqrt(vA), eB / np.sqrt(vB)
    all_within &= (kA < 3) and (kB < 3)
    # R repetitions: root-mean-square error of the posterior mean
    xa_r, xb_r, _ = simulate_ab_many(pA_true, pB_true, nn, R=R, seed=1)
    mA_r = my_post_mean_var(*my_posterior(xa_r, nn))[0]
    mB_r = my_post_mean_var(*my_posterior(xb_r, nn))[0]
    rA = np.sqrt(np.mean((mA_r - pA_true) ** 2)); rB = np.sqrt(np.mean((mB_r - pB_true) ** 2))
    thA = np.sqrt(pA_true * (1 - pA_true) / nn)
    single.append((eA, eB)); rmse.append((rA, rB)); theory.append(thA)
    tab.append((nn, xa, xb, mA, mB, eA, kA, eB, kB, rA, thA))
    print(f"{nn:>6} {xa:>6} {xb:>6} {mA:>8.5f} {mB:>8.5f} {eA:>10.5f} {kA:>6.2f} {eB:>10.5f} {kB:>6.2f} {rA:>9.5f} {thA:>10.5f}")
rmse = np.array(rmse); single = np.array(single)
slopeA = np.polyfit(np.log(ns), np.log(rmse[:, 0]), 1)[0]
slopeB = np.polyfit(np.log(ns), np.log(rmse[:, 1]), 1)[0]
print(f"log-log slope of RMSE against n: A {slopeA:.3f}, B {slopeB:.3f}   (theory -0.5)")
print(f"every single-run error within 3 standard errors: {all_within}")

# the n = 1000 Iteration 2 run
e1000 = np.hypot(muA - pA_true, muB - pB_true)
k1000 = max(abs(muA - pA_true) / np.sqrt(varA), abs(muB - pB_true) / np.sqrt(varB))
print(f"n = 1000: ||(mu_A, mu_B) - (p_A, p_B)||_2 = {e1000:.4f};  largest error = {k1000:.2f} SE")
rows.insert(0, row("Simulation ($n=1000$)", "$(\\mu_A,\\mu_B)$", "true $(p_A,p_B)$",
                   "$\\|\\cdot\\|_2=%.4f$ ($%.2f$ SE)" % (e1000, k1000), k1000 < 3))
rows.append(row("Simulation ($n=50\\ldots50{,}000$)", "slope of RMSE$(\\mu_A)$ vs $n$", "$-\\tfrac12$ (theory)",
                "$%.3f$" % abs(slopeA + 0.5), abs(slopeA + 0.5) < 0.05))

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.loglog(ns, rmse[:, 0], "o-", label=r"RMSE of $\mu_A$ over 2000 runs")
ax.loglog(ns, rmse[:, 1], "s-", label=r"RMSE of $\mu_B$ over 2000 runs")
ax.loglog(ns, single[:, 0], "o", mfc="none", c="C0", label=r"$|\mu_A-p_A|$, one run (seed 0)")
ax.loglog(ns, single[:, 1], "s", mfc="none", c="C1", label=r"$|\mu_B-p_B|$, one run (seed 0)")
ax.loglog(ns, theory, "k--", label=r"$\sqrt{p_A(1-p_A)/n}$")
ax.set_xlabel("n (visitors per group)"); ax.set_ylabel("error against the truth")
ax.set_title(f"Error shrinks like $n^{{-1/2}}$ (slopes {slopeA:.2f}, {slopeB:.2f})")
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "it3_error_vs_n.png"), dpi=150); plt.close(fig)

mac.update(simSlopeA=f"{slopeA:.3f}", simSlopeB=f"{slopeB:.3f}", simTruthErr=f"{e1000:.4f}",
           simTruthSE=f"{k1000:.2f}")
tab_lines = ["\\begin{table}[ht]\\centering",
             "\\caption{Simulation against the truth ($p_A=0.10$, $p_B=0.12$). One run per $n$ (seed 0): error of the "
             "posterior mean in absolute value and in posterior standard deviations. RMSE over 2000 repetitions "
             "against the theoretical $\\sqrt{p_A(1-p_A)/n}$.}\\label{tab:truth}",
             "\\small\\begin{tabular}{rrrrrrrrr}\\toprule",
             "$n$ & $x_A$ & $x_B$ & $|\\mu_A-p_A|$ & in SE & $|\\mu_B-p_B|$ & in SE & RMSE$(\\mu_A)$ & $\\sqrt{p_A(1-p_A)/n}$\\\\ \\midrule"]
for (nn, xa, xb, mA, mB, eA, kA, eB, kB, rA, thA) in tab:
    tab_lines.append(f"{nn:,} & {xa:,} & {xb:,} & {eA:.5f} & {kA:.2f} & {eB:.5f} & {kB:.2f} & {rA:.5f} & {thA:.5f}\\\\".replace(",", "{,}"))
tab_lines += ["\\bottomrule\\end{tabular}\\normalsize\\end{table}"]

# =====================================================================
print("\n" + "=" * 78, "\n3. The whole simulation 1000 times with p_A = p_B = 0.10\n")
xa0, xb0, _ = simulate_ab_many(0.10, 0.10, 1000, R=1000, seed=0)
z0 = my_z_stat(xa0, xb0, 1000, 1000)
rej0 = np.mean(my_p_value(z0) < 0.05)
print(f"z test rejects in {rej0:.3f} of 1000 runs   (target 0.05 +/- 0.014)")
print(f"mean of z {z0.mean():.3f}, sd of z {z0.std(ddof=1):.3f}   (N(0,1): 0 and 1)")
rows.append(row("Simulation (null, 1000 runs)", "rejection rate of the $z$ test", "$\\alpha=0.05$",
                "$%.3f$" % abs(rej0 - 0.05), abs(rej0 - 0.05) <= 0.014))
mac.update(simNullRej=f"{rej0:.3f}", simNullZsd=f"{z0.std(ddof=1):.3f}", simNullZmean=f"{z0.mean():.3f}")

# =====================================================================
print("\n" + "=" * 78, "\n4. Stress test: 10% of the users are in both groups\n")
# (a) one row-level data set: the data check finds the problem
uid, ver, y = simulate_ab_users(0.10, 0.12, 1000, overlap=0.10, seed=0)
chk = my_check_users(uid, ver)
print("data check on one stressed data set:",
      {k: v for k, v in chk.items() if k != "keep"})
xa_s, na_s = my_counts(ver, y, "A"); xb_s, nb_s = my_counts(ver, y, "B")
xa_c, na_c = my_counts(ver[chk["keep"]], y[chk["keep"]], "A")
xb_c, nb_c = my_counts(ver[chk["keep"]], y[chk["keep"]], "B")
print(f"  as recorded:           x_A/n_A = {xa_s}/{na_s},  x_B/n_B = {xb_s}/{nb_s}")
print(f"  shared users removed:  x_A/n_A = {xa_c}/{na_c},  x_B/n_B = {xb_c}/{nb_c}")
uid_c, ver_c, _ = simulate_ab_users(0.10, 0.12, 1000, overlap=0.0, seed=0)
chk_clean = my_check_users(uid_c, ver_c)
print("data check on the clean simulation:  ",
      {k: v for k, v in chk_clean.items() if k != "keep"})

# (b) many repetitions: what happens to the estimates and to the test
R = 20000
def summarise(xa, xb, nn, label):
    d = xb / nn - xa / nn
    zz = my_z_stat(xa, xb, nn, nn)
    rej = np.mean(my_p_value(zz) < 0.05)
    print(f"  {label:42s} mean(pB_hat-pA_hat) = {d.mean():.4f}  sd = {d.std(ddof=1):.4f}  "
          f"corr(x_A,x_B) = {np.corrcoef(xa, xb)[0, 1]:+.3f}  sd(z) = {zz.std(ddof=1):.3f}  reject = {rej:.4f}")
    return d.mean(), d.std(ddof=1), np.corrcoef(xa, xb)[0, 1], zz.std(ddof=1), rej

print(f"{R} repetitions each:")
c_null = summarise(*simulate_ab_many(0.10, 0.10, 1000, R, seed=2), "clean,   p_A = p_B = 0.10")
s_null = summarise(*simulate_overlap_many(0.10, 0.10, 1000, 0.10, R, seed=2), "overlap, p_A = p_B = 0.10")
c_alt = summarise(*simulate_ab_many(0.10, 0.12, 1000, R, seed=3), "clean,   p_A = 0.10, p_B = 0.12")
s_alt = summarise(*simulate_overlap_many(0.10, 0.12, 1000, 0.10, R, seed=3), "overlap, p_A = 0.10, p_B = 0.12")
# theory: the shared users add the same count to x_A and x_B, so it cancels in the difference
f = 0.10
sd_formula = np.sqrt(2 * 0.10 * 0.90 / 1000)                    # what Theorem 5 assumes
sd_true = np.sqrt(2 * (1 - f) * 0.10 * 0.90 / 1000)             # what actually happens under H0
rej_theory = 2 * stats.norm.sf(1.959964 / np.sqrt(1 - f))
print(f"  theory under H0: sd assumed {sd_formula:.4f}, actual {sd_true:.4f}; sd(z) = sqrt(1-f) = {np.sqrt(1 - f):.3f}; "
      f"rejection rate {rej_theory:.4f}")
print(f"  theory under H1: E[pB_hat - pA_hat] = (1-f)(p_B - p_A) = {(1 - f) * 0.02:.4f}  (true difference 0.0200)")

# (c) accuracy of the estimates as the overlap grows
fs = [0.0, 0.05, 0.10, 0.20, 0.40]
bias, rejs = [], []
for ff in fs:
    xa, xb, nn = simulate_overlap_many(0.10, 0.12, 1000, ff, 5000, seed=4)
    bias.append(np.mean(xb / nn - xa / nn))
    xa, xb, nn = simulate_overlap_many(0.10, 0.10, 1000, ff, 5000, seed=5)
    rejs.append(np.mean(my_p_value(my_z_stat(xa, xb, nn, nn)) < 0.05))
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
axs[0].plot(fs, bias, "o-", label="simulated mean of $\\hat p_B-\\hat p_A$")
axs[0].plot(fs, [(1 - ff) * 0.02 for ff in fs], "k--", label="$(1-f)(p_B-p_A)$")
axs[0].axhline(0.02, c="gray", lw=0.8); axs[0].set_xlabel("fraction $f$ of users in both groups")
axs[0].set_ylabel("estimated difference"); axs[0].set_title("Estimate is pulled towards 0")
axs[0].legend(frameon=False, fontsize=8)
axs[1].plot(fs, rejs, "o-", label="simulated rejection rate under $H_0$")
axs[1].plot(fs, [2 * stats.norm.sf(1.959964 / np.sqrt(1 - ff)) for ff in fs], "k--",
            label=r"$2(1-\Phi(1.96/\sqrt{1-f}))$")
axs[1].axhline(0.05, c="gray", lw=0.8); axs[1].set_xlabel("fraction $f$ of users in both groups")
axs[1].set_ylabel("rejection rate"); axs[1].set_title("$z$ test no longer has size 0.05")
axs[1].legend(frameon=False, fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(FIG, "it3_stress.png"), dpi=150); plt.close(fig)

rows.append(row("Simulation (stress test)", "users in both groups found", "100 planted",
                "$%d$" % abs(chk["users_in_both_groups"] - 100), chk["users_in_both_groups"] == 100))
rows.append(row("Simulation (stress test)", "mean of $\\hat p_B-\\hat p_A$", "$(1-f)(p_B-p_A)=0.018$",
                "$%.4f$" % abs(s_alt[0] - 0.018), abs(s_alt[0] - 0.018) < 0.001))
mac.update(strBoth=str(chk["users_in_both_groups"]), strRows=str(chk["rows"]), strUnique=str(chk["unique_users"]),
           strDup=str(chk["duplicate_rows"]),
           strXA=f"{xa_s}/{na_s}", strXB=f"{xb_s}/{nb_s}", strXAc=f"{xa_c}/{na_c}", strXBc=f"{xb_c}/{nb_c}",
           strCleanNullRej=f"{c_null[4]:.4f}", strNullRej=f"{s_null[4]:.4f}", strNullRejTheory=f"{rej_theory:.4f}",
           strNullSdZ=f"{s_null[3]:.3f}", strCleanNullSdZ=f"{c_null[3]:.3f}", strNullCorr=f"{s_null[2]:.3f}",
           strCleanCorr=f"{c_null[2]:+.3f}",
           strAltDiff=f"{s_alt[0]:.4f}", strCleanAltDiff=f"{c_alt[0]:.4f}",
           strAltPower=f"{s_alt[4]:.3f}", strCleanAltPower=f"{c_alt[4]:.3f}",
           strSdAssumed=f"{sd_formula:.4f}", strSdTrue=f"{sd_true:.4f}", strNullSd=f"{s_null[1]:.4f}")

write_tex("it3_sim.tex", macros=mac, script="src/check_simulation.py")
write_tex("it3_sim_table.tex", lines=tab_lines, script="src/check_simulation.py")
write_tex("it3_rows_sim.tex", macros={"rowsSim": "\n".join(rows)}, script="src/check_simulation.py")
print("wrote figures/it3_error_vs_n.png, figures/it3_stress.png")
