"""Sanity check of the Borda-count theorem on a 3-arm bandit.

Siththaranjan et al. (ICLR 2024, Thm 3.1): Bradley-Terry preference learning
with hidden context (here, which group annotated) ranks alternatives by their
Borda count  BC(a) = mean_b P(a > b), not by expected utility.  With a
majority that mildly likes dumping and a minority that hates it, Borda ranks
"dump" above "recycle" while expected utility ranks them the other way.
"""

from __future__ import annotations

import numpy as np

from ..reward_learning import fit_bt

NAMES = ["dump", "recycle", "rest"]
#                   dump      recycle   rest
UTILS = np.array([[2.0,      1.0,      0.0],    # majority
                  [-4.0,     1.0,      0.0]])   # minority
GROUP_PROBS = np.array([0.8, 0.2])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def pairwise_preference(utils: np.ndarray, group_probs: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """P[a, b] = P(a preferred to b) marginalised over the hidden group."""
    diff = utils[:, :, None] - utils[:, None, :]  # (G, A, A)
    return np.einsum("g,gab->ab", group_probs, _sigmoid(kappa * diff))


def borda_counts(utils, group_probs, kappa: float = 1.0) -> np.ndarray:
    return pairwise_preference(utils, group_probs, kappa).mean(axis=1)


def expected_utility(utils, group_probs) -> np.ndarray:
    return group_probs @ utils


def simulate(utils, group_probs, n: int, rng: np.random.Generator, kappa: float = 1.0):
    """Pooled comparisons of uniformly random pairs; the group is never recorded."""
    n_arms = utils.shape[1]
    groups = rng.choice(len(group_probs), size=n, p=group_probs)
    a = rng.integers(n_arms, size=n)
    b = (a + rng.integers(1, n_arms, size=n)) % n_arms
    p = _sigmoid(kappa * (utils[groups, a] - utils[groups, b]))
    y = (rng.random(n) < p).astype(float)
    x = np.zeros((n, n_arms))
    x[np.arange(n), a] += 1.0
    x[np.arange(n), b] -= 1.0
    return x, y


def run(n: int = 20000, seed: int = 0, kappa: float = 1.0, utils=UTILS, group_probs=GROUP_PROBS) -> dict:
    rng = np.random.default_rng(seed)
    x, y = simulate(utils, group_probs, n, rng, kappa)
    theta = fit_bt(x, y, l2=1e-3)
    borda = borda_counts(utils, group_probs, kappa)
    eu = expected_utility(utils, group_probs)
    return {
        "theta": theta,
        "borda": borda,
        "expected_utility": eu,
        "rank_learned": [int(i) for i in np.argsort(-theta)],
        "rank_borda": [int(i) for i in np.argsort(-borda)],
        "rank_eu": [int(i) for i in np.argsort(-eu)],
    }


def main():
    out = run()
    print(f"{'action':10s} {'learned reward':>14s} {'Borda':>8s} {'exp. util.':>11s}")
    for i, name in enumerate(NAMES):
        print(f"{name:10s} {out['theta'][i]:14.3f} {out['borda'][i]:8.3f} {out['expected_utility'][i]:11.3f}")

    def fmt(rank):
        return " > ".join(NAMES[i] for i in rank)

    print("learned ranking (BT):", fmt(out["rank_learned"]))
    print("Borda ranking       :", fmt(out["rank_borda"]))
    print("exp. util. ranking  :", fmt(out["rank_eu"]))


if __name__ == "__main__":
    main()
