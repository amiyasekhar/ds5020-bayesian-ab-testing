"""
DS5020 Project 7 -- Bayesian A/B Testing
Small worked example, step by step (Iteration 3, Questions 1-4).

Input (Question 1): A = 3 clicks in 10 visits, B = 5 clicks in 10 visits, uniform prior.

The steps follow the algorithm in report.tex and call the same my_ functions
(src/methods.py) that run the simulation and the real data.  Exact values are
computed with fractions.Fraction; the Beta densities here are polynomials, so
every integral can be done exactly and used as a second-way check.

Run:  python3 src/small_example.py
Writes docs/it3_small.tex (numbers for the report) and docs/it3_rows_small.tex.
"""
from fractions import Fraction as Fr
from math import comb, factorial, sqrt
import numpy as np

from methods import (my_posterior, my_post_mean_var, my_weighted_mean, my_beta_draws,
                     my_indicator_mean, my_mc_se, my_mc_prob, my_beta_pdf, my_grid,
                     my_cumsimpson, my_simpson, my_grid_prob, my_z_stat, my_p_value,
                     my_decision, report)
from texio import write_tex, row, sci

np.set_printoptions(precision=6, suppress=True, linewidth=120)


# ---- exact polynomial arithmetic (coefficients are Fractions, lowest power first)
def poly_mul(u, v):
    out = [Fr(0)] * (len(u) + len(v) - 1)
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            out[i + j] += ui * vj
    return out

def poly_int(u):                       # antiderivative with value 0 at p = 0
    return [Fr(0)] + [c / (k + 1) for k, c in enumerate(u)]

def poly_at_1(u):                      # value at p = 1 = sum of the coefficients
    return sum(u, Fr(0))

def beta_fn_int(al, be):               # B(al, be) = (al-1)! (be-1)! / (al+be-1)!  for integers
    return Fr(factorial(al - 1) * factorial(be - 1), factorial(al + be - 1))

def beta_pdf_poly(al, be):             # f(p) = p^(al-1) (1-p)^(be-1) / B(al, be) as a polynomial
    one_minus = [Fr((-1) ** k * comb(be - 1, k)) for k in range(be)]     # (1-p)^(be-1)
    return [Fr(0)] * (al - 1) + [c / beta_fn_int(al, be) for c in one_minus]


def main():
    # ------------------------------------------------------------ Step 0
    xA, nA, xB, nB, a, b = 3, 10, 5, 10, 1, 1
    print("=" * 78)
    print("Step 0  input")
    print(f"  A: x_A = {xA} clicks in n_A = {nA} visits      B: x_B = {xB} clicks in n_B = {nB} visits")
    print(f"  prior Beta(a, b) = Beta({a}, {b})  (uniform)")

    # ------------------------------------------------------------ Step 1
    aA, bA = my_posterior(xA, nA, a, b)
    aB, bB = my_posterior(xB, nB, a, b)
    print("\nStep 1  posterior update  alpha = a + x,  beta = b + n - x      [Theorem 2]")
    print(f"  A: alpha_A = {a} + {xA} = {aA},  beta_A = {b} + {nA} - {xA} = {bA}   ->  p_A | data ~ Beta({aA}, {bA})")
    print(f"  B: alpha_B = {a} + {xB} = {aB},  beta_B = {b} + {nB} - {xB} = {bB}   ->  p_B | data ~ Beta({aB}, {bB})")

    # ------------------------------------------------------------ Step 2
    muA, varA = my_post_mean_var(Fr(aA), Fr(bA))        # same function, exact arithmetic
    muB, varB = my_post_mean_var(Fr(aB), Fr(bB))
    print("\nStep 2  posterior mean and variance                                [Theorem 3]")
    print("        mu = alpha/(alpha+beta),  sigma^2 = alpha beta / ((alpha+beta)^2 (alpha+beta+1))")
    print(f"  A: mu_A = {aA}/{aA + bA} = {muA} = {float(muA):.6f}")
    print(f"     sigma_A^2 = {aA}*{bA} / ({aA + bA}^2 * {aA + bA + 1}) = {Fr(aA * bA, (aA + bA) ** 2 * (aA + bA + 1))} "
          f"= {float(varA):.6f}   (sd {sqrt(varA):.4f})")
    print(f"  B: mu_B = {aB}/{aB + bB} = {muB} = {float(muB):.6f}")
    print(f"     sigma_B^2 = {aB}*{bB} / ({aB + bB}^2 * {aB + bB + 1}) = {Fr(aB * bB, (aB + bB) ** 2 * (aB + bB + 1))} "
          f"= {float(varB):.6f}   (sd {sqrt(varB):.4f})")
    wmA, w = my_weighted_mean(Fr(xA), Fr(nA), Fr(a), Fr(b))
    wmB, _ = my_weighted_mean(Fr(xB), Fr(nB), Fr(a), Fr(b))
    print(f"  weighted-average form: w = n/(a+b+n) = {w}")
    print(f"     A: (1-w) a/(a+b) + w x_A/n_A = {1 - w}*{Fr(a, a + b)} + {w}*{Fr(xA, nA)} = {wmA}   (same as mu_A: {wmA == muA})")
    print(f"     B: (1-w) a/(a+b) + w x_B/n_B = {1 - w}*{Fr(a, a + b)} + {w}*{Fr(xB, nB)} = {wmB}   (same as mu_B: {wmB == muB})")

    # ------------------------------------- second-way check of Steps 1-2
    print("\nCheck 1 (second way)  integrate the posterior density exactly")
    fA, fB = beta_pdf_poly(aA, bA), beta_pdf_poly(aB, bB)
    print(f"  B({aA},{bA}) = {beta_fn_int(aA, bA)},  B({aB},{bB}) = {beta_fn_int(aB, bB)}")
    print(f"  f_A(p) = {1 / beta_fn_int(aA, bA)} p^{aA - 1} (1-p)^{bA - 1},   f_B(p) = {1 / beta_fn_int(aB, bB)} p^{aB - 1} (1-p)^{bB - 1}")
    P = [Fr(0), Fr(1)]                                   # the polynomial p
    checks = {}
    for name, f, mu, var in (("A", fA, muA, varA), ("B", fB, muB, varB)):
        total = poly_at_1(poly_int(f))                   # int_0^1 f(p) dp
        m1 = poly_at_1(poly_int(poly_mul(P, f)))         # int_0^1 p f(p) dp
        m2 = poly_at_1(poly_int(poly_mul(P, poly_mul(P, f))))
        print(f"  {name}: int f = {total},  int p f = {m1},  int p^2 f - (int p f)^2 = {m2 - m1 ** 2}")
        checks[name] = (total == 1, m1 == mu, m2 - m1 ** 2 == var)
    print(f"  posterior integrates to 1: {checks['A'][0] and checks['B'][0]};"
          f"  means match Step 2: {checks['A'][1] and checks['B'][1]};"
          f"  variances match Step 2: {checks['A'][2] and checks['B'][2]}")
    # prior x likelihood, normalised on a grid, is the same density (conjugacy, numerically)
    g = np.linspace(0, 1, 2001)
    unnorm = comb(nA, xA) * g ** xA * (1 - g) ** (nA - xA) * 1.0      # likelihood x uniform prior
    numeric = unnorm / my_simpson(unnorm, g)
    conj_err = np.abs(numeric - my_beta_pdf(g, aA, bA)).max()
    print(f"  prior x likelihood normalised on a grid vs Beta({aA},{bA}) density: max|diff| = {conj_err:.1e}")

    # ---------------------------------------------- exact theta (truth)
    print("\nExact value of theta = Pr(p_B > p_A | data) = int_0^1 f_B(t) F_A(t) dt   [Proposition 1]")
    FA = poly_int(fA)                                    # F_A(t), a polynomial of degree 11
    theta_exact = poly_at_1(poly_int(poly_mul(fB, FA)))
    print(f"  F_A(t) = int_0^t f_A(s) ds is a polynomial of degree {len(FA) - 1};  F_A(1) = {poly_at_1(FA)}")
    print(f"  theta = {theta_exact} = {float(theta_exact):.10f}")
    # second way: closed-form sum for integer parameters
    #   theta = sum_{i=0}^{alpha_B-1} B(alpha_A+i, beta_A+beta_B) / ((beta_B+i) B(1+i, beta_B) B(alpha_A, beta_A))
    theta_sum = sum(beta_fn_int(aA + i, bA + bB) / ((bB + i) * beta_fn_int(1 + i, bB) * beta_fn_int(aA, bA))
                    for i in range(aB))
    print(f"  second way (closed-form sum over i = 0..alpha_B-1): {theta_sum}   (equal: {theta_sum == theta_exact})")
    # symmetry: Pr(p_A > p_B) from the same formula with A and B swapped must be 1 - theta
    theta_swap = poly_at_1(poly_int(poly_mul(fA, poly_int(fB))))
    print(f"  Pr(p_A > p_B | data) = {theta_swap};  the two add to {theta_exact + theta_swap}")

    # ------------------------------------------------------------ Steps 3-5
    M = 100_000
    print(f"\nStep 3  draw from the posteriors, p = G1/(G1+G2)   (M = {M:,}, seed 1)")
    rng = np.random.default_rng(1)
    pA_draws = my_beta_draws(aA, bA, M, rng)
    pB_draws = my_beta_draws(aB, bB, M, rng)
    print("  first 8 draws")
    print("    p_A^(m):", pA_draws[:8])
    print("    p_B^(m):", pB_draws[:8])
    print(f"  sample means {pA_draws.mean():.4f}, {pB_draws.mean():.4f}   (exact {float(muA):.4f}, {float(muB):.4f})")
    print(f"  sample variances {pA_draws.var():.5f}, {pB_draws.var():.5f}   (exact {float(varA):.5f}, {float(varB):.5f})")
    print("\nStep 4  indicators I_m = 1{p_B^(m) > p_A^(m)} and their average          [Theorem 4]")
    print("    I_m    :", (pB_draws[:8] > pA_draws[:8]).astype(int))
    for m in (10, 100, 1_000, 10_000, 100_000):
        print(f"    theta_hat with the first {m:>7,} draws = {my_indicator_mean(pA_draws[:m], pB_draws[:m]):.5f}")
    theta_mc = my_indicator_mean(pA_draws, pB_draws)
    se = my_mc_se(theta_mc, M)
    print("\nStep 5  standard error  SE = sqrt(theta_hat (1 - theta_hat) / M)")
    print(f"  theta_hat_M = {theta_mc:.5f},  SE = sqrt({theta_mc:.5f} * {1 - theta_mc:.5f} / {M}) = {se:.5f}")
    print(f"  exact SE with the true theta: {sqrt(float(theta_exact) * (1 - float(theta_exact)) / M):.5f}")
    n_se = abs(theta_mc - float(theta_exact)) / se
    print(f"  |theta_hat_M - theta| = {abs(theta_mc - float(theta_exact)):.5f} = {n_se:.2f} SE")
    assert (theta_mc, se) == my_mc_prob(aA, bA, aB, bB, M=M, seed=1)     # the wrapper gives the same

    # ------------------------------------------------ Q4: grid integration
    print("\nCheck 2 (Question 4)  the same probability by numerical integration on a grid")
    t = my_grid(aA, bA, aB, bB, n_grid=8)
    fA_t, fB_t = my_beta_pdf(t, aA, bA), my_beta_pdf(t, aB, bB)
    FA_t = my_cumsimpson(fA_t, t)
    print("  a grid of 8 intervals, to show the steps:")
    print("    t          :", t)
    print("    f_A(t)     :", fA_t)
    print("    f_B(t)     :", fB_t)
    print("    F_A(t_2j)  :", FA_t, "  (cumulative Simpson, every second point)")
    print("    f_B F_A    :", fB_t[::2] * FA_t)
    print(f"    Simpson sum = {my_simpson(fB_t[::2] * FA_t, t[::2]):.6f}")
    grid_vals = {}
    for n_grid in (8, 40, 200, 2000):
        grid_vals[n_grid] = my_grid_prob(aA, bA, aB, bB, n_grid=n_grid)
        print(f"    n_grid = {n_grid:>5}: theta = {grid_vals[n_grid]:.12f}   error vs exact = "
              f"{abs(grid_vals[n_grid] - float(theta_exact)):.2e}")
    theta_grid = grid_vals[2000]
    t = my_grid(aA, bA, aB, bB, n_grid=2000)
    int_fA = my_simpson(my_beta_pdf(t, aA, bA), t)
    print(f"  the gridded posterior integrates to {int_fA:.12f}")

    # ------------------------------------------------------------ Step 6
    print("\nStep 6  two-proportion z statistic                                     [Theorem 5]")
    p_hat = Fr(xA + xB, nA + nB)
    var0 = p_hat * (1 - p_hat) * (Fr(1, nA) + Fr(1, nB))
    diff = Fr(xB, nB) - Fr(xA, nA)
    z2 = diff ** 2 / var0
    print(f"  pooled rate p_hat = ({xA}+{xB})/({nA}+{nB}) = {p_hat}")
    print(f"  p_hat (1-p_hat) (1/n_A + 1/n_B) = {p_hat}*{1 - p_hat}*{Fr(1, nA) + Fr(1, nB)} = {var0}")
    print(f"  difference x_B/n_B - x_A/n_A = {diff}")
    print(f"  z^2 = ({diff})^2 / ({var0}) = {z2}   ->   z = sqrt({z2}) = {sqrt(z2):.6f}")
    z = my_z_stat(xA, xB, nA, nB)
    pval = my_p_value(z)
    print(f"  my_z_stat gives z = {z:.6f};  two-sided p-value 2(1 - Phi(|z|)) = {pval:.4f}")
    print(f"  normal-approximation check: n p_hat = {float(nA * p_hat):.0f}, n (1 - p_hat) = {float(nA * (1 - p_hat)):.0f} "
          "(both below 10, so the approximation is rough here)")

    # ------------------------------------------------------------ Step 7
    dec = my_decision(theta_mc, pval)
    print("\nStep 7  decision")
    print(f"  Bayesian:    Pr(p_B > p_A | data) = {float(theta_exact):.4f}  (< 0.95, so B is not declared better)")
    print(f"  frequentist: z = {z:.3f}, p = {pval:.3f}  (> 0.05, so H0: p_A = p_B is not rejected)")
    print(f"  my_decision: {dec}")
    print(f"  one-sided comparison: 1 - Phi(z) = {pval / 2:.4f}  vs  Pr(p_A >= p_B | data) = {1 - float(theta_exact):.4f}")

    # ------------------------------------------------- verification lines
    print("\nVerification")
    e_mean, _ = report("posterior means vs fractions", my_post_mean_var(aA, bA)[0:1] + my_post_mean_var(aB, bB)[0:1],
                       [float(muA), float(muB)])
    e_var, _ = report("posterior variances vs fractions", [my_post_mean_var(aA, bA)[1], my_post_mean_var(aB, bB)[1]],
                      [float(varA), float(varB)])
    e_grid, _ = report("grid theta vs exact fraction", theta_grid, float(theta_exact))
    e_mc, _ = report("MC theta vs exact fraction", theta_mc, float(theta_exact))
    e_z, _ = report("z vs sqrt(exact z^2)", z, sqrt(z2))

    # ------------------------------------------------------- LaTeX output
    def frac(q):
        return f"\\tfrac{{{q.numerator}}}{{{q.denominator}}}"
    macros = {
        "smThetaExact": frac(theta_exact), "smThetaExactDec": f"{float(theta_exact):.6f}",
        "smThetaSwap": frac(theta_swap),
        "smThetaMC": f"{theta_mc:.5f}", "smSE": f"{se:.5f}", "smNSE": f"{n_se:.2f}",
        "smMCerr": f"{abs(theta_mc - float(theta_exact)):.5f}",
        "smThetaGrid": f"{theta_grid:.10f}", "smGridErr": sci(e_grid),
        "smGridEight": f"{grid_vals[8]:.6f}", "smGridEightErr": sci(abs(grid_vals[8] - float(theta_exact))),
        "smGridForty": f"{grid_vals[40]:.8f}", "smGridFortyErr": sci(abs(grid_vals[40] - float(theta_exact))),
        "smGridTwoHundred": f"{grid_vals[200]:.10f}", "smGridTwoHundredErr": sci(abs(grid_vals[200] - float(theta_exact))),
        "smZ": f"{z:.4f}", "smP": f"{pval:.4f}", "smPone": f"{pval / 2:.4f}",
        "smOneMinusTheta": f"{1 - float(theta_exact):.4f}",
        "smConjErr": sci(conj_err), "smIntfA": f"{int_fA:.12f}",
        "smDrawsA": ",\\ ".join(f"{v:.4f}" for v in pA_draws[:5]),
        "smDrawsB": ",\\ ".join(f"{v:.4f}" for v in pB_draws[:5]),
        "smInd": ",\\ ".join(str(int(v)) for v in (pB_draws[:5] > pA_draws[:5])),
        "smRunTen": f"{my_indicator_mean(pA_draws[:10], pB_draws[:10]):.3f}",
        "smRunThousand": f"{my_indicator_mean(pA_draws[:1000], pB_draws[:1000]):.3f}",
        "smSampleMeanA": f"{pA_draws.mean():.4f}", "smSampleMeanB": f"{pB_draws.mean():.4f}",
    }
    write_tex("it3_small.tex", macros=macros, script="src/small_example.py")
    rows = [
        row("Small example", "$\\mu_A,\\mu_B,\\sigma_A^2,\\sigma_B^2$", "exact fractions",
            "$%s$" % sci(max(e_mean, e_var)), max(e_mean, e_var) < 1e-15),
        row("Small example", "$\\int_0^1\\pi(p\\mid x)\\,dp$", "1 (exact fractions)", "0", all(checks["A"]) and all(checks["B"])),
        row("Small example", "$\\theta$ (grid, 2000)", "exact fraction", "$%s$" % sci(e_grid), e_grid < 1e-8),
        row("Small example", "$\\hat\\theta_M$ (Monte Carlo)", "exact fraction",
            "$%.4f$ ($%.2f$ SE)" % (e_mc, n_se), n_se < 3),
        row("Small example", "$z$", "$\\sqrt{5/6}$ (exact)", "$%s$" % sci(e_z), e_z < 1e-12),
    ]
    write_tex("it3_rows_small.tex", macros={"rowsSmall": "\n".join(rows)}, script="src/small_example.py")


if __name__ == "__main__":
    main()
