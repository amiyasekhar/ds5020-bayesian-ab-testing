"""
DS5020 Project 7 -- Bayesian A/B Testing
Simulation with known truth (Iteration 2).

True rates p_A = 0.10, p_B = 0.12, n = 1000 visitors per version, so every
derived quantity can be checked against a number we know.

Writes:
  figures/posteriors.png      posterior densities with the true rates marked
  figures/mc_se.png           Monte Carlo standard error vs M (log-log)
  figures/z_null_hist.png     histogram of z under H0 against N(0,1)
  docs/sim_macros.tex         \\newcommand macros (x_A, x_B, n, M) for report.tex
  docs/sim_tables.tex         LaTeX tables that report.tex \\input's

Only numpy, matplotlib and scipy.stats are used.  scipy.stats.beta is used to
*draw* from the posterior and to compare against our closed-form mean/variance;
it does not solve the method for us.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import betaln, gammaln   # log Beta / log Gamma functions

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
DOCS = os.path.join(HERE, "..", "docs")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- parameters
pA, pB, n = 0.10, 0.12, 1000          # true rates and visitors per version
a, b = 1, 1                           # Beta(1,1) = uniform prior
M = 100_000                           # Monte Carlo draws
alpha = 0.05                          # test size

xA, xB = rng.binomial(n, pA), rng.binomial(n, pB)
print(f"observed clicks: xA={xA}, xB={xB}  (n={n} each)")

lines = []   # LaTeX output collected here
def tex(s=""):
    lines.append(s)

# ------------------------------------------------------- Q1: binomial likelihood
# L(p | x) = C(n,x) p^x (1-p)^(n-x).  Check our formula against scipy's pmf.
def log_binom_lik(p, x, n):
    log_choose = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
    return log_choose + x * np.log(p) + (n - x) * np.log1p(-p)

p_grid = np.linspace(1e-4, 1 - 1e-4, 20001)
ours = log_binom_lik(p_grid, xA, n)
ref = stats.binom.logpmf(xA, n, p_grid)
q1_err = np.max(np.abs(ours - ref))
print(f"Q1 max |log L_ours - log L_scipy| = {q1_err:.2e}")
# MLE of p is x/n: the grid maximiser should be x/n
p_mle_grid = p_grid[np.argmax(ours)]
print(f"Q1 grid maximiser {p_mle_grid:.4f} vs x/n = {xA/n:.4f}")

# ------------------------------------------------------- Q2: conjugate posterior
# Posterior ∝ prior * likelihood.  Normalise numerically on the grid and compare
# with the closed-form Beta(a+x, b+n-x) density.
def check_conjugacy(x):
    unnorm = np.exp(log_binom_lik(p_grid, x, n) + stats.beta.logpdf(p_grid, a, b))
    numeric = unnorm / np.trapz(unnorm, p_grid)
    closed = stats.beta.pdf(p_grid, a + x, b + n - x)
    return np.max(np.abs(numeric - closed)) / np.max(closed)

q2_errA, q2_errB = check_conjugacy(xA), check_conjugacy(xB)
print(f"Q2 relative density error A={q2_errA:.2e}, B={q2_errB:.2e}")

postA = stats.beta(a + xA, b + n - xA)
postB = stats.beta(a + xB, b + n - xB)

# ------------------------------------------- Q3: posterior mean and variance
def post_mean(x):   return (a + x) / (a + b + n)
def post_var(x):
    al, be = a + x, b + n - x
    return al * be / ((al + be) ** 2 * (al + be + 1))

w = n / (a + b + n)                                  # weight on x/n
prior_mean = a / (a + b)
q3 = {
    "A": (post_mean(xA), (1 - w) * prior_mean + w * xA / n, postA.mean(),
          post_var(xA), postA.var()),
    "B": (post_mean(xB), (1 - w) * prior_mean + w * xB / n, postB.mean(),
          post_var(xB), postB.var()),
}
for k, (m_formula, m_weighted, m_scipy, v_formula, v_scipy) in q3.items():
    print(f"Q3 {k}: mean formula {m_formula:.6f} = weighted {m_weighted:.6f} = scipy {m_scipy:.6f};"
          f" var formula {v_formula:.3e} = scipy {v_scipy:.3e}")
print(f"Q3 weight w = n/(a+b+n) = {w:.6f}")

# -------------------------------------------------- Q4: Monte Carlo estimate
drawsA = postA.rvs(M, random_state=rng)
drawsB = postB.rvs(M, random_state=rng)
I = drawsB > drawsA
p_hat = I.mean()
se = np.sqrt(p_hat * (1 - p_hat) / M)
# "exact" value by 2-D numerical integration on a grid (independent posteriors)
g = np.linspace(0, 1, 4001)
cdfA = postA.cdf(g)                  # Pr(p_A < t)
pdfB = postB.pdf(g)
p_exact = np.trapz(pdfB * cdfA, g)   # Pr(p_B > p_A) = ∫ f_B(t) F_A(t) dt
print(f"Q4 MC Pr(pB>pA|data) = {p_hat:.4f} ± {se:.4f} (M={M});  exact ≈ {p_exact:.4f}")

# SE shrinks like 1/sqrt(M): empirical sd of p_hat over R repeats for each M
Ms = [100, 1_000, 10_000, 100_000]
R = 200
emp_sd, theo_se = [], []
for m in Ms:
    ests = np.empty(R)
    for r in range(R):
        dA = postA.rvs(m, random_state=rng)
        dB = postB.rvs(m, random_state=rng)
        ests[r] = np.mean(dB > dA)
    emp_sd.append(ests.std(ddof=1))
    theo_se.append(np.sqrt(p_exact * (1 - p_exact) / m))
    print(f"Q4 M={m:>7d}: empirical sd {emp_sd[-1]:.5f}  theory sqrt(θ(1-θ)/M) {theo_se[-1]:.5f}")
slope = np.polyfit(np.log(Ms), np.log(emp_sd), 1)[0]
print(f"Q4 log-log slope of sd vs M = {slope:.3f} (theory -0.5)")

# ----------------------------------------------- Q5: two-proportion z test
def z_test(xA, xB, nA, nB):
    pooled = (xA + xB) / (nA + nB)
    z = (xB / nB - xA / nA) / np.sqrt(pooled * (1 - pooled) * (1 / nA + 1 / nB))
    pval = 2 * stats.norm.sf(np.abs(z))
    return z, pval

z_obs, pval_obs = z_test(xA, xB, n, n)
print(f"Q5 z = {z_obs:.4f}, two-sided p = {pval_obs:.4f}")

# Repeat the whole experiment 2000 times.
REPS = 2000
def repeat(pA_, pB_):
    xa = rng.binomial(n, pA_, size=REPS)
    xb = rng.binomial(n, pB_, size=REPS)
    z, pv = z_test(xa, xb, n, n)
    reject = pv < alpha
    # Bayesian decision: declare B better if Pr(pB > pA | data) > 0.95
    # (posterior probability via 1-D integration for each replicate, vectorised)
    FA = stats.beta.cdf(g[None, :], a + xa[:, None], b + n - xa[:, None])
    fB = stats.beta.pdf(g[None, :], a + xb[:, None], b + n - xb[:, None])
    prob_B_better = np.trapz(fB * FA, g, axis=1)
    return z, reject.mean(), (prob_B_better > 0.95).mean()

z_null, size_z, size_bayes = repeat(0.10, 0.10)
_, power_z, power_bayes = repeat(0.10, 0.12)
print(f"Q5 null (pA=pB=0.10): z rejects {size_z:.4f} of the time (target {alpha});"
      f" Bayes 'B better' {size_bayes:.4f}")
print(f"Q5 alt  (pB=0.12):    z power {power_z:.4f}; Bayes power {power_bayes:.4f}")
ks = stats.kstest(z_null, "norm")
print(f"Q5 KS test of z under H0 vs N(0,1): stat {ks.statistic:.4f}, p {ks.pvalue:.3f}")

# --------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(6.5, 3.6))
t = np.linspace(0.06, 0.17, 800)
ax.plot(t, postA.pdf(t), label=f"posterior A: Beta({a+xA}, {b+n-xA})")
ax.plot(t, postB.pdf(t), label=f"posterior B: Beta({a+xB}, {b+n-xB})")
ax.axvline(pA, ls="--", c="C0", lw=1); ax.axvline(pB, ls="--", c="C1", lw=1)
ax.set_xlabel("p"); ax.set_ylabel("density")
ax.set_title("Posterior densities (dashed = true rates)")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(os.path.join(FIG, "posteriors.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.loglog(Ms, emp_sd, "o-", label="empirical sd of $\\hat p$")
ax.loglog(Ms, theo_se, "s--", label=r"$\sqrt{\theta(1-\theta)/M}$")
ax.set_xlabel("M (Monte Carlo draws)"); ax.set_ylabel("standard error")
ax.set_title(f"MC error falls like $M^{{-1/2}}$ (slope {slope:.2f})")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(os.path.join(FIG, "mc_se.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.hist(z_null, bins=40, density=True, alpha=0.6, label="z over 2000 null experiments")
zz = np.linspace(-4, 4, 400); ax.plot(zz, stats.norm.pdf(zz), "k", label="N(0,1)")
ax.set_xlabel("z"); ax.set_ylabel("density"); ax.legend(frameon=False)
ax.set_title("Two-proportion z under $H_0$"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "z_null_hist.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------ LaTeX tables
tex("% Generated by src/simulation.py -- do not edit by hand")
tex("\\newcommand{\\simxA}{%d}\\newcommand{\\simxB}{%d}\\newcommand{\\simn}{%d}" % (xA, xB, n))
tex("\\newcommand{\\simM}{%s}" % f"{M:,}".replace(",", "{,}"))
tex("\\begin{table}[ht]\\centering")
tex("\\caption{Simulation check: true values against estimates. "
    f"True $p_A={pA}$, $p_B={pB}$, $n={n}$ per version, prior Beta({a},{b}), "
    f"observed $x_A={xA}$, $x_B={xB}$.}}")
tex("\\label{tab:sim}")
tex("\\small\\begin{tabular}{l p{5.2cm} r r p{3.0cm}}\\toprule")
tex("Q & quantity & true / theory & estimate & confirms \\\\ \\midrule")
tex(f"Q1 & $\\max_p|\\log L_{{\\text{{ours}}}}-\\log L_{{\\text{{scipy}}}}|$ & 0 & {q1_err:.1e} & binomial likelihood formula \\\\")
tex(f"Q1 & maximiser of $L(p\\mid x_A)$ & $x_A/n={xA/n:.4f}$ & {p_mle_grid:.4f} & MLE is $x/n$ \\\\")
tex(f"Q2 & rel.\\ error, numeric posterior vs Beta$(a+x_A,b+n-x_A)$ & 0 & {q2_errA:.1e} & conjugacy \\\\")
tex(f"Q2 & rel.\\ error, numeric posterior vs Beta$(a+x_B,b+n-x_B)$ & 0 & {q2_errB:.1e} & conjugacy \\\\")
mA = q3["A"]; mB = q3["B"]
tex(f"Q3 & posterior mean A: $(1-w)\\frac{{a}}{{a+b}}+w\\frac{{x_A}}{{n}}$ & {mA[1]:.6f} & {mA[2]:.6f} & weighted-average form \\\\")
tex(f"Q3 & posterior mean B & {mB[1]:.6f} & {mB[2]:.6f} & weighted-average form \\\\")
tex(f"Q3 & weight $w=n/(a+b+n)$ & {w:.6f} & -- & weights \\\\")
tex(f"Q3 & posterior variance A (formula vs scipy) & {mA[3]:.3e} & {mA[4]:.3e} & variance formula \\\\")
tex(f"Q3 & $|{{\\rm post.\\ mean}}_A - p_A|$, $|{{\\rm post.\\ mean}}_B - p_B|$ & 0 & {abs(mA[2]-pA):.4f}, {abs(mB[2]-pB):.4f} & estimates near truth \\\\")
tex(f"Q4 & $\\Pr(p_B>p_A\\mid\\text{{data}})$ & {p_exact:.4f} (quadrature) & {p_hat:.4f} $\\pm$ {se:.4f} & MC unbiased, SE formula \\\\")
tex(f"Q4 & log--log slope of SE vs $M$ & $-0.5$ & {slope:.3f} & SE $\\propto M^{{-1/2}}$ \\\\")
tex(f"Q5 & $z$ statistic, two-sided $p$ & -- & {z_obs:.3f}, {pval_obs:.4f} & $z$ formula \\\\")
tex(f"Q5 & rejection rate, $p_A=p_B=0.10$ ({REPS} reps) & {alpha} & {size_z:.4f} & test has size $\\alpha$ \\\\")
tex(f"Q5 & KS distance of $z$ to $N(0,1)$ under $H_0$ & 0 & {ks.statistic:.3f} ($p={ks.pvalue:.2f}$) & normal approximation \\\\")
tex(f"Q5 & power, $p_B=0.12$ ({REPS} reps) & -- & {power_z:.4f} & -- \\\\")
tex(f"Q4 & Bayes rule ``$\\Pr(p_B>p_A)>0.95$'': false-positive / power & -- & {size_bayes:.4f} / {power_bayes:.4f} & -- \\\\")
tex("\\bottomrule\\end{tabular}\\normalsize\\end{table}")
tex()
tex("\\begin{table}[ht]\\centering\\caption{Monte Carlo standard error against the number of draws $M$ "
    f"(empirical sd over {R} repeats).}}\\label{{tab:mcse}}")
tex("\\begin{tabular}{rrr}\\toprule $M$ & empirical sd of $\\hat p$ & $\\sqrt{\\theta(1-\\theta)/M}$ \\\\ \\midrule")
for m, e, th in zip(Ms, emp_sd, theo_se):
    tex(f"{m:,} & {e:.5f} & {th:.5f} \\\\".replace(",", "{,}"))
tex("\\bottomrule\\end{tabular}\\end{table}")

macros = lines[:3]                              # comment + \newcommand lines
tables = [lines[0]] + lines[3:]
with open(os.path.join(DOCS, "sim_macros.tex"), "w") as f:
    f.write("\n".join(macros) + "\n")
with open(os.path.join(DOCS, "sim_tables.tex"), "w") as f:
    f.write("\n".join(tables) + "\n")
print("wrote docs/sim_macros.tex, docs/sim_tables.tex and figures/*.png")
