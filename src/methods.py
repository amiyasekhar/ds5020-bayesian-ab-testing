"""
DS5020 Project 7 -- Bayesian A/B Testing
From-scratch implementation (Iteration 3).

One function per numbered step of the algorithm in report.tex ("The algorithm
as math"), written with NumPy arrays only.  The comment above each function is
the equation it computes, with the same symbols as the report:

    x_A, x_B, n_A, n_B   clicks and visitors            a, b       prior Beta(a, b)
    alpha_v, beta_v      posterior parameters           mu_v, sigma_v^2  posterior mean, variance
    theta                Pr(p_B > p_A | data)           M          Monte Carlo draws
    theta_hat_M, SE      MC estimate and standard error z          two-proportion z statistic

Nothing here imports SciPy or scikit-learn.  `math.lgamma` and `math.erfc`
(Python standard library) supply log Gamma and the normal tail; SciPy is used
only in the check scripts, to compare answers.
"""
import math
import numpy as np


# ------------------------------------------------------------------ helper
def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
    return abs_err, rel_err


# ------------------------------------------------------------------ Step 1
# Posterior update (Theorem 2, conjugacy):
#     alpha = a + x,      beta = b + n - x
def my_posterior(x, n, a=1, b=1):
    return a + x, b + n - x


# ------------------------------------------------------------------ Step 2
# Posterior summaries (Lemma 1 / Theorem 3):
#     mu      = alpha / (alpha + beta)
#     sigma^2 = alpha beta / ((alpha + beta)^2 (alpha + beta + 1))
def my_post_mean_var(alpha, beta):
    s = alpha + beta
    return alpha / s, alpha * beta / (s ** 2 * (s + 1))


# Weighted-average form of the posterior mean (Theorem 3):
#     E[p | x] = (1 - w) a/(a+b) + w x/n,      w = n / (a + b + n)
def my_weighted_mean(x, n, a=1, b=1):
    w = n / (a + b + n)
    return (1 - w) * a / (a + b) + w * x / n, w


# ------------------------------------------------------------------ Step 3
# Draw from the posterior:
#     p^(m) = G1 / (G1 + G2),   G1 ~ Gamma(alpha, 1),  G2 ~ Gamma(beta, 1),   m = 1..M
def my_beta_draws(alpha, beta, M, rng):
    g1 = rng.standard_gamma(alpha, size=M)
    g2 = rng.standard_gamma(beta, size=M)
    return g1 / (g1 + g2)


# ------------------------------------------------------------------ Step 4
# Indicators and their average (Theorem 4):
#     I_m = 1{ p_B^(m) > p_A^(m) },      theta_hat_M = (1/M) sum_m I_m
def my_indicator_mean(pA_draws, pB_draws):
    I = (pB_draws > pA_draws).astype(float)
    return I.sum() / I.size


# ------------------------------------------------------------------ Step 5
# Standard error of the Monte Carlo estimate (Theorem 4):
#     SE = sqrt( theta_hat_M (1 - theta_hat_M) / M )
def my_mc_se(theta_hat, M):
    return np.sqrt(theta_hat * (1 - theta_hat) / M)


# Steps 3-5 together:  theta_hat_M and its SE from the four posterior parameters.
def my_mc_prob(aA, bA, aB, bB, M=100_000, seed=1):
    rng = np.random.default_rng(seed)
    pA_draws = my_beta_draws(aA, bA, M, rng)      # Step 3
    pB_draws = my_beta_draws(aB, bB, M, rng)
    theta_hat = my_indicator_mean(pA_draws, pB_draws)   # Step 4
    return theta_hat, my_mc_se(theta_hat, M)            # Step 5


# ------------------------------------------------------------------ Step 6
# Two-proportion z statistic (Theorem 5):
#     p_hat = (x_A + x_B) / (n_A + n_B)
#     z     = (x_B/n_B - x_A/n_A) / sqrt( p_hat (1 - p_hat) (1/n_A + 1/n_B) )
# Works on scalars or on arrays of repeated experiments.
def my_z_stat(xA, xB, nA, nB):
    xA, xB = np.asarray(xA, float), np.asarray(xB, float)
    p_hat = (xA + xB) / (nA + nB)
    z = (xB / nB - xA / nA) / np.sqrt(p_hat * (1 - p_hat) * (1 / nA + 1 / nB))
    return z if z.ndim else float(z)


# Two-sided p-value (Theorem 5, Step 7):
#     p-value = 2 (1 - Phi(|z|)) = erfc(|z| / sqrt 2)
def my_p_value(z):
    z = np.asarray(z, float)
    p = np.vectorize(math.erfc)(np.abs(z) / math.sqrt(2.0))
    return p if p.ndim else float(p)


# ------------------------------------------------------------------ Step 7
# Decision: B is declared better if theta_hat_M > threshold; H0 is rejected if p-value < alpha.
def my_decision(theta_hat, p_value, threshold=0.95, alpha=0.05):
    return {"bayes_B_better": bool(theta_hat > threshold),
            "bayes_A_better": bool(theta_hat < 1 - threshold),
            "z_rejects_H0": bool(p_value < alpha)}


# ------------------------------------------------- numerical integration (Q4)
# log B(alpha, beta) = log Gamma(alpha) + log Gamma(beta) - log Gamma(alpha + beta)
def my_log_beta_fn(alpha, beta):
    return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)


# Beta density (Definition 6), computed in logs so it does not overflow for large n:
#     f(p) = p^(alpha-1) (1-p)^(beta-1) / B(alpha, beta)
def my_beta_pdf(p, alpha, beta):
    p = np.asarray(p, float)
    out = np.zeros_like(p)
    ok = (p > 0) & (p < 1)
    out[ok] = np.exp((alpha - 1) * np.log(p[ok]) + (beta - 1) * np.log1p(-p[ok])
                     - my_log_beta_fn(alpha, beta))
    if alpha == 1:
        out[p == 0] = beta          # f(0) = beta when alpha = 1
    if beta == 1:
        out[p == 1] = alpha         # f(1) = alpha when beta = 1
    return out


# Grid for the integral: the part of [0, 1] within k posterior standard deviations
# of either posterior mean (outside it both densities are negligible).
def my_grid(aA, bA, aB, bB, n_grid=2000, k=12.0):
    lo, hi = 1.0, 0.0
    for al, be in ((aA, bA), (aB, bB)):
        mu, var = my_post_mean_var(al, be)
        lo, hi = min(lo, mu - k * np.sqrt(var)), max(hi, mu + k * np.sqrt(var))
    return np.linspace(max(lo, 0.0), min(hi, 1.0), n_grid + 1)


# Composite Simpson rule on an even number of intervals of width h:
#     int y dt  ~  (h/3) (y_0 + 4 y_1 + 2 y_2 + 4 y_3 + ... + 4 y_{N-1} + y_N)
def my_simpson(y, t):
    h = t[1] - t[0]
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


# Cumulative Simpson rule: the integral from t_0 up to every second grid point,
#     F(t_{2j}) = F(t_{2j-2}) + (h/3) (y_{2j-2} + 4 y_{2j-1} + y_{2j}),     F(t_0) = 0
def my_cumsimpson(y, t):
    h = t[1] - t[0]
    panels = h / 3 * (y[0:-2:2] + 4 * y[1:-1:2] + y[2::2])
    return np.concatenate(([0.0], np.cumsum(panels)))


# Proposition 1, on a grid:
#     theta = Pr(p_B > p_A | data) = int_0^1 f_B(t) F_A(t) dt,    F_A(t) = int_0^t f_A(s) ds
# F_A is computed on every second grid point (cumulative Simpson), and the outer
# integral is a Simpson rule over those points.  n_grid must be a multiple of 4.
def my_grid_prob(aA, bA, aB, bB, n_grid=2000):
    t = my_grid(aA, bA, aB, bB, n_grid)
    fA, fB = my_beta_pdf(t, aA, bA), my_beta_pdf(t, aB, bB)
    FA = my_cumsimpson(fA, t)                 # F_A at t_0, t_2, t_4, ...
    return my_simpson(fB[::2] * FA, t[::2])


# ------------------------------------------------------- data check (Q5)
# Check the rows before counting: one row per user, every user in one group only.
# Returns the counts and a boolean mask of the rows that are safe to keep.
def my_check_users(userid, version):
    userid, version = np.asarray(userid), np.asarray(version)
    ids, first, counts = np.unique(userid, return_index=True, return_counts=True)
    groups = np.unique(version)
    in_group = [np.isin(ids, userid[version == g]) for g in groups]
    n_groups_per_id = np.sum(in_group, axis=0)
    both = ids[n_groups_per_id > 1]                    # users seen in more than one group
    keep = np.zeros(userid.size, bool)
    keep[first] = True                                 # first row of every user
    keep &= ~np.isin(userid, both)                     # drop users in both groups
    sizes = {str(g): int(np.sum(version == g)) for g in groups}
    return {"rows": int(userid.size),
            "unique_users": int(ids.size),
            "duplicate_rows": int(userid.size - ids.size),
            "users_in_both_groups": int(both.size),
            "group_sizes": sizes,
            "keep": keep}


# Counts the method needs:  n_v = number of rows in group v,  x_v = sum of outcomes in group v
def my_counts(version, y, group):
    sel = np.asarray(version) == group
    return int(np.sum(np.asarray(y)[sel])), int(np.sum(sel))


# The whole algorithm (Steps 1-7) on four counts.
def my_ab_test(xA, nA, xB, nB, a=1, b=1, M=100_000, seed=1, n_grid=2000):
    aA, bA = my_posterior(xA, nA, a, b)                 # Step 1
    aB, bB = my_posterior(xB, nB, a, b)
    muA, varA = my_post_mean_var(aA, bA)                # Step 2
    muB, varB = my_post_mean_var(aB, bB)
    theta_mc, se = my_mc_prob(aA, bA, aB, bB, M, seed)  # Steps 3-5
    theta_grid = my_grid_prob(aA, bA, aB, bB, n_grid)   # Proposition 1 (check of Steps 3-5)
    z = my_z_stat(xA, xB, nA, nB)                       # Step 6
    pval = my_p_value(z)
    return {"aA": aA, "bA": bA, "aB": aB, "bB": bB,
            "muA": muA, "varA": varA, "muB": muB, "varB": varB,
            "theta_mc": theta_mc, "se": se, "theta_grid": theta_grid,
            "z": z, "p_value": pval,
            "decision": my_decision(theta_mc, pval)}    # Step 7
