from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import world as W
from .mdp import TabularMDP, sample_trajectory, trajectory_features
from .solver import soft_value_iteration, uniform_policy


@dataclass
class PreferenceData:
    x: np.ndarray               # Comparaison of 2 trajectory, Phi(tau_a) - Phi(tau_b)
    y: np.ndarray               # 1.0 if tau_a preferred
    annotator: np.ndarray       # id of annotator who made comparaison i
    group: np.ndarray           # group of the annotator that did comparaison i
    annotator_group: np.ndarray  # group each annotator belong to
    pairs: np.ndarray           # array of all pairs of trajectory compared together
    pool_features: np.ndarray   # set of n feature vectors (n simulated days)

    @property
    def n(self) -> int:
        """Number of comparaisons."""
        return len(self.y)

    @property
    def n_annotators(self) -> int:
        """Number of annotators."""
        return len(self.annotator_group)


def trajectory_pool(mdp: TabularMDP, rng: np.random.Generator, n_pool: int = 400, frac_random: float = 0.5, beta: float = 0.5) -> np.ndarray:
    """Build set of candidate days as feature vectors. Half from the reference policy, half from KL-regularised optima of random reward vectors."""

    n_random = int(round(frac_random * n_pool))
    pi_ref = uniform_policy(mdp)
    sample_features = [trajectory_features(mdp, sample_trajectory(mdp, pi_ref, rng)) for _ in range(n_random)]

    while len(sample_features) < n_pool:
        w = rng.normal(size=mdp.n_features)
        pi = soft_value_iteration(mdp, mdp.features @ w, beta, pi_ref).policy
        sample_features.append(trajectory_features(mdp, sample_trajectory(mdp, pi, rng)))
    return np.array(sample_features)


def generate_preferences(mdp: TabularMDP, world: W.World, rng: np.random.Generator,
                         n_annotators: int = 40, comparisons_per_annotator: int = 25,
                         kappa: float = 1.0, n_pool: int = 400,
                         pool: np.ndarray | None = None) -> PreferenceData:
    """Simulate whole annotation process. """
    if pool is None:
        pool = trajectory_pool(mdp, rng, n_pool)

    n_pool = len(pool)
    annotator_group = rng.choice(W.N_ACTORS, size=n_annotators, p=world.annot_weights) # Assign each annotator a hidden group
    n = n_annotators * comparisons_per_annotator
    annotator = np.repeat(np.arange(n_annotators), comparisons_per_annotator)
    group = annotator_group[annotator]

    a = rng.integers(n_pool, size=n)
    b = (a + rng.integers(1, n_pool, size=n)) % n_pool  # b != a

    x = pool[a] - pool[b]
    utility_diff = np.einsum("nd,nd->n", x, world.weights[group]) # Vector of total rewards per day by comparaison made by annotator i
    p = 1.0 / (1.0 + np.exp(-kappa * utility_diff))               # Sigmoid transform total reward into probability annotator choose trajectory a
    y = (rng.random(n) < p).astype(float)                         # Choice of the better trajectory 

    return PreferenceData(x=x, y=y, annotator=annotator, group=group,
                          annotator_group=annotator_group, pairs=np.stack([a, b], axis=1),
                          pool_features=pool)
