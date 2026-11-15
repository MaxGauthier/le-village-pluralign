"""Per-actor evaluation of a policy.

Normalised score  S_i(pi) = (J_i(pi) - J_i(pi_ref)) / (J_i(pi*_i) - J_i(pi_ref))
is 1 at actor i's own optimum, 0 when no better than the reference policy and
negative when worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import world as W
from .mdp import TabularMDP, occupancy, rewards, value
from .solver import hard_value_iteration, uniform_policy

# "Erased": the minority is no better off than under the reference policy
# while the rule does serve the majority.
ERASED_MINORITY_MAX = 0.1
ERASED_MAJORITY_MIN = 0.3
_EPS = 1e-9


@dataclass
class Baseline:
    pi_ref: np.ndarray
    j_ref: np.ndarray       # (N_ACTORS,) value of pi_ref for each actor
    j_star: np.ndarray      # (N_ACTORS,) value of each actor's own optimal policy
    degenerate: np.ndarray  # (N_ACTORS,) bool: j_star == j_ref, score undefined


def compute_baseline(mdp: TabularMDP, world: W.World) -> Baseline:
    pi_ref = uniform_policy(mdp)
    j_ref = np.zeros(W.N_ACTORS)
    j_star = np.zeros(W.N_ACTORS)
    for i in range(W.N_ACTORS):
        r_i = rewards(mdp, world.weights[i])
        j_ref[i] = value(mdp, pi_ref, r_i)
        j_star[i] = value(mdp, hard_value_iteration(mdp, r_i).policy, r_i)
    return Baseline(pi_ref, j_ref, j_star, np.abs(j_star - j_ref) < _EPS)


def actor_rewards(mdp: TabularMDP, world: W.World) -> np.ndarray:
    """``R[s, a, i]`` = per-step gain of actor i."""
    return mdp.features @ world.weights.T


def actor_values(mdp: TabularMDP, policy: np.ndarray, world: W.World,
                 start: int | None = None) -> np.ndarray:
    """Exact expected daily gain of every actor, ``J[i]``."""
    d = occupancy(mdp, policy, start)
    return np.einsum("sa,sai->i", d, actor_rewards(mdp, world))


def expected_damage(mdp: TabularMDP, policy: np.ndarray, world: W.World,
                    start: int | None = None) -> float:
    d = occupancy(mdp, policy, start)
    return float((d * rewards(mdp, world.damage)).sum())


def normalized_scores(j: np.ndarray, baseline: Baseline) -> np.ndarray:
    denom = baseline.j_star - baseline.j_ref
    return np.divide(j - baseline.j_ref, denom, out=np.zeros_like(j), where=~baseline.degenerate)


def is_erased(scores: np.ndarray, minority: int = W.MINORITY,
              majority: tuple[int, ...] = W.MAJORITY) -> bool:
    """Minority no better than pi_ref while the majority is nearly optimal."""
    return bool(scores[minority] <= ERASED_MINORITY_MAX
                and scores[list(majority)].mean() >= ERASED_MAJORITY_MIN)


def gini(x: np.ndarray) -> float:
    """Gini index of the values shifted to be non-negative (0 = equal)."""
    x = np.asarray(x, dtype=float)
    x = x - x.min()
    if x.sum() < _EPS:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def summarize(mdp: TabularMDP, policy: np.ndarray, world: W.World, baseline: Baseline,
              start: int | None = None) -> dict:
    j = actor_values(mdp, policy, world, start)
    s = normalized_scores(j, baseline)
    welfare = list(world.welfare_actors)
    worst = welfare[int(np.argmin(s[welfare]))]
    return {
        "values": j,
        "scores": s,
        "mean": float(s[welfare].mean()),
        "min": float(s[welfare].min()),
        "worst_actor": W.ACTORS[worst],
        "robot": float(s[W.ROBOT]),
        "minority": float(s[W.MINORITY]),
        "majority": float(s[list(W.MAJORITY)].mean()),
        "nature": float(s[W.NATURE]),
        "damage": expected_damage(mdp, policy, world, start),
        "gini": gini(s[welfare]),
        "erased": is_erased(s),
    }
