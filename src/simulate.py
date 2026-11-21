"""
DS5020 Project 7 -- Bayesian A/B Testing
The Iteration 2 simulation model as functions (Iteration 3).

Same model, parameter values and seed as Iteration 2 (src/simulation.py):
two independent binomial experiments, p_A = 0.10, p_B = 0.12, n = 1000 visitors
per version, rng = np.random.default_rng(0), x_A drawn first and then x_B.
simulate_ab(seed=0) therefore returns the Iteration 2 data x_A = 92, x_B = 121.
"""
import numpy as np


# X_A ~ Binomial(n, p_A),  X_B ~ Binomial(n, p_B), independent.
# Returns the data (x_A, x_B, n); the true parameters are the p_A, p_B passed in.
def simulate_ab(pA=0.10, pB=0.12, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xA = int(rng.binomial(n, pA))
    xB = int(rng.binomial(n, pB))
    return xA, xB, n


# R independent repetitions of the same experiment (vectors of x_A and x_B).
def simulate_ab_many(pA=0.10, pB=0.12, n=1000, R=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.binomial(n, pA, size=R), rng.binomial(n, pB, size=R), n


# Row-level version of the experiment, used by the stress test and the data checks.
# Each group has n rows (userid, version, y).  A fraction `overlap` of the users is
# put in BOTH groups: such a user has one outcome, drawn with the average rate
# (p_A + p_B)/2 because they saw both versions, and that one outcome is recorded
# in both groups.  overlap = 0 gives the clean experiment of simulate_ab.
def simulate_ab_users(pA=0.10, pB=0.12, n=1000, overlap=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_both = int(round(overlap * n))
    n_only = n - n_both
    y_onlyA = rng.random(n_only) < pA
    y_onlyB = rng.random(n_only) < pB
    y_both = rng.random(n_both) < 0.5 * (pA + pB)
    id_onlyA = np.arange(n_only)
    id_onlyB = np.arange(n_only, 2 * n_only)
    id_both = np.arange(2 * n_only, 2 * n_only + n_both)
    userid = np.concatenate([id_onlyA, id_both, id_onlyB, id_both])
    version = np.array(["A"] * n + ["B"] * n)
    y = np.concatenate([y_onlyA, y_both, y_onlyB, y_both])
    return userid, version, y


# Counts only, for R repetitions of the overlap experiment (fast version of the above):
#     x_A = (clicks of A-only users) + (clicks of shared users),  same for x_B.
def simulate_overlap_many(pA=0.10, pB=0.12, n=1000, overlap=0.10, R=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_both = int(round(overlap * n))
    n_only = n - n_both
    shared = rng.binomial(n_both, 0.5 * (pA + pB), size=R)
    xA = rng.binomial(n_only, pA, size=R) + shared
    xB = rng.binomial(n_only, pB, size=R) + shared
    return xA, xB, n


# Row-level experiment with coin-flip assignment: each of N users is put in A or B with
# probability 1/2 (so n_A is random, as in a real randomised experiment), then clicks
# with the rate of their version.  Used by the assumption checks.
def simulate_ab_coin(pA=0.10, pB=0.12, N=2000, seed=0):
    rng = np.random.default_rng(seed)
    userid = np.arange(N)
    in_B = rng.random(N) < 0.5
    version = np.where(in_B, "B", "A")
    y = rng.random(N) < np.where(in_B, pB, pA)
    return userid, version, y
