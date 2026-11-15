from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .mdp import TabularMDP, occupancy
from .solver import SolveResult, expected_kl, soft_value_iteration, uniform_policy


# welfare functions

class Welfare:
    name = "welfare"

    def prepare(self, j_ref: np.ndarray, ranges: np.ndarray) -> None:
        """Hook called by the optimiser with the reference values and ranges
        of the (possibly normalised) group returns."""

    def value(self, j: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, j: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dual_init(self, g: int) -> np.ndarray:
        return np.zeros(g)

    def multipliers(self, xi: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def dual(self, xi: np.ndarray, v: float, j: np.ndarray) -> tuple[float, np.ndarray]:
        """Dual objective and its gradient w.r.t. ``xi`` given the inner value
        ``v = max_pi lambda . J(pi) - beta KL`` and the group returns ``j`` of
        the best-response policy."""
        raise NotImplementedError


class Utilitarian(Welfare):
    name = "mean"

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = weights

    def _w(self, g):
        return np.full(g, 1.0 / g) if self.weights is None else np.asarray(self.weights, dtype=float) / np.sum(self.weights)

    def value(self, j):
        return float(self._w(len(j)) @ j)

    def grad(self, j):
        return self._w(len(j))

    def multipliers(self, xi):
        return self._w(len(xi))

    def dual(self, xi, v, j):
        return float(v), np.zeros_like(xi)


class SoftMin(Welfare):
    """Smooth egalitarian welfare; alpha -> -inf gives min_g J_g."""
    name = "softmin"

    def __init__(self, alpha: float = -10.0):
        if alpha >= 0:
            raise ValueError("alpha must be negative")
        self.alpha = alpha
        self.temperature = 0.0 if np.isinf(alpha) else 1.0 / -alpha

    def value(self, j):
        if self.temperature == 0.0:
            return float(np.min(j))
        return float((logsumexp(self.alpha * j) - np.log(len(j))) / self.alpha)

    def grad(self, j):
        if self.temperature == 0.0:
            return np.eye(len(j))[int(np.argmin(j))]
        return softmax(self.alpha * j)

    def multipliers(self, xi):
        return softmax(xi)

    def dual(self, xi, v, j):
        lam = softmax(xi)
        g = len(lam)
        if self.temperature > 0.0:
            log_ratio = np.log(np.maximum(lam, 1e-300) * g)
            penalty = self.temperature * float(lam @ log_ratio)
            d_lam = j + self.temperature * (log_ratio + 1.0)
        else:
            penalty = 0.0
            d_lam = j
        grad_xi = lam * (d_lam - lam @ d_lam)  # chain rule through the softmax
        return float(v + penalty), grad_xi


class Nash(Welfare):
    """Nash bargaining welfare above a floor just below the reference policy."""
    name = "nash"

    def __init__(self, eps_frac: float = 0.02):
        self.eps_frac = eps_frac
        self.floor = None
        self.scale = 1.0

    def prepare(self, j_ref, ranges):
        self.floor = np.asarray(j_ref, dtype=float) - self.eps_frac * np.asarray(ranges, dtype=float)
        self.scale = float(np.mean(ranges))

    def _kappa(self, g):
        return self.scale / g

    def value(self, j):
        return float(self._kappa(len(j)) * np.log(np.maximum(j - self.floor, 1e-12)).sum())

    def grad(self, j):
        return self._kappa(len(j)) / np.maximum(j - self.floor, 1e-12)

    def dual_init(self, g):
        return np.full(g, np.log(self._kappa(g)))

    def multipliers(self, xi):
        return np.exp(xi)

    def dual(self, xi, v, j):
        lam = np.exp(xi)
        kappa = self._kappa(len(lam))
        phi = v - lam @ self.floor - kappa * xi.sum() - kappa * len(lam) * (1.0 - np.log(kappa))
        grad_xi = lam * (j - self.floor) - kappa
        return float(phi), grad_xi


# --- helpers -------------------------------------------------------------------

@dataclass
class WelfareResult:
    policy: np.ndarray         # (S, A)
    q: np.ndarray              # (S, A) soft Q-values of the aggregated reward
    occupancy: np.ndarray      # (S, A)
    group_values: np.ndarray   # (G,) raw J_g of the final policy
    group_scores: np.ndarray   # (G,) normalised scores of the final policy
    group_weights: np.ndarray  # (G,) optimal multipliers (normalised)
    welfare: float             # W of the final policy (on the values the welfare was applied to)
    duality_gap: float
    n_iter: int


@dataclass
class Normalization:
    j_ref: np.ndarray    # (G,) value of pi_ref
    j_star: np.ndarray   # (G,) value of each group's own greedy optimum
    scale: np.ndarray    # (G,) multiplicative factor a_g (mean range / range_g)
    offset: np.ndarray   # (G,) additive offset b_g = -a_g J_ref_g

    @property
    def ranges(self):
        return self.j_star - self.j_ref

    def scores(self, j: np.ndarray) -> np.ndarray:
        """Normalised scores in [0, 1] at the optimum, 0 at the reference."""
        return (j - self.j_ref) / np.maximum(self.ranges, 1e-9)


def group_values(group_rewards: np.ndarray, d: np.ndarray) -> np.ndarray:
    """J_g = sum_{s,a} d(s,a) r_g(s,a) for every group."""
    return np.einsum("gsa,sa->g", group_rewards, d)


def reference_values(mdp: TabularMDP, group_rewards: np.ndarray, pi_ref: np.ndarray | None = None) -> np.ndarray:
    pi_ref = uniform_policy(mdp) if pi_ref is None else pi_ref
    return group_values(np.asarray(group_rewards, dtype=float), occupancy(mdp, pi_ref))


def optimal_values(mdp: TabularMDP, group_rewards: np.ndarray) -> np.ndarray:
    """Value of each group's own optimal (greedy) policy."""
    return np.array([soft_value_iteration(mdp, r, 0.0).v[mdp.start] for r in group_rewards])


def worst_values(mdp: TabularMDP, group_rewards: np.ndarray) -> np.ndarray:
    """Value of each group's worst possible policy."""
    return np.array([-soft_value_iteration(mdp, -r, 0.0).v[mdp.start] for r in group_rewards])


def group_normalization(mdp: TabularMDP, group_rewards: np.ndarray, pi_ref: np.ndarray) -> Normalization:
    j_ref = reference_values(mdp, group_rewards, pi_ref)
    j_star = optimal_values(mdp, group_rewards)
    ranges = np.maximum(j_star - j_ref, 1e-6)
    scale = ranges.mean() / ranges
    return Normalization(j_ref, j_star, scale, -scale * j_ref)


def policy_from_occupancy(mdp: TabularMDP, d: np.ndarray, pi_ref: np.ndarray) -> np.ndarray:
    mass = d.sum(axis=1, keepdims=True)
    return np.divide(d, mass, out=pi_ref.copy(), where=mass > 1e-12)


# --- optimisation --------------------------------------------------------------

def optimize_welfare(mdp: TabularMDP, group_rewards: np.ndarray, welfare: Welfare, beta: float,
                     pi_ref: np.ndarray | None = None, max_iter: int = 300, tol: float = 1e-9,
                     normalize: bool = True) -> WelfareResult:
    """Minimise the dual over the multipliers with L-BFGS; return the soft-VI
    best response to the optimal multipliers."""
    group_rewards = np.asarray(group_rewards, dtype=float)
    if pi_ref is None:
        pi_ref = uniform_policy(mdp)
    n_groups = len(group_rewards)
    norm = group_normalization(mdp, group_rewards, pi_ref)
    if normalize:
        scaled = group_rewards * norm.scale[:, None, None]
        offset = norm.offset
        welfare.prepare(np.zeros(n_groups), np.full(n_groups, norm.ranges.mean()))
    else:
        scaled = group_rewards
        offset = np.zeros(n_groups)
        welfare.prepare(norm.j_ref, norm.ranges)
    n_eval = 0

    def inner(lam: np.ndarray) -> tuple[SolveResult, np.ndarray, np.ndarray]:
        res = soft_value_iteration(mdp, np.einsum("g,gsa->sa", lam, scaled), beta, pi_ref)
        d = occupancy(mdp, res.policy)
        return res, d, group_values(scaled, d) + offset

    def objective(xi: np.ndarray):
        nonlocal n_eval
        n_eval += 1
        lam = welfare.multipliers(xi)
        res, _, s = inner(lam)
        return welfare.dual(xi, res.v[mdp.start] + lam @ offset, s)

    xi0 = welfare.dual_init(n_groups)
    if isinstance(welfare, Utilitarian):
        xi = xi0
    else:
        xi = minimize(objective, xi0, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-14}).x
    lam = welfare.multipliers(xi)
    res, d, s = inner(lam)
    dual_value, _ = welfare.dual(xi, res.v[mdp.start] + lam @ offset, s)
    primal = welfare.value(s) - beta * expected_kl(mdp, res.policy, pi_ref, d)
    j = group_values(group_rewards, d)
    return WelfareResult(res.policy, res.q, d, j, norm.scores(j), lam / lam.sum(), welfare.value(s),
                         float(dual_value - primal), n_eval)


def maximin_game(mdp: TabularMDP, group_rewards: np.ndarray, beta: float,
                 pi_ref: np.ndarray | None = None, n_iter: int = 200, eta: float = 2.0) -> WelfareResult:
    """Hard maximin on raw returns by multiplicative weights on the group
    weights (cross-check for ``SoftMin``): max_pi min_lambda sum_g lambda_g J_g(pi)."""
    group_rewards = np.asarray(group_rewards, dtype=float)
    if pi_ref is None:
        pi_ref = uniform_policy(mdp)
    g = len(group_rewards)
    lam = np.full(g, 1.0 / g)
    d_avg = np.zeros((mdp.n_states, mdp.n_actions))
    scale = np.abs(group_rewards).sum(axis=(1, 2)).max() + 1e-9
    for it in range(1, n_iter + 1):
        r_mix = np.einsum("g,gsa->sa", lam, group_rewards)
        d = occupancy(mdp, soft_value_iteration(mdp, r_mix, beta, pi_ref).policy)
        d_avg += (d - d_avg) / it
        j = group_values(group_rewards, d)
        lam = lam * np.exp(-eta * j / scale * mdp.t_day)
        lam /= lam.sum()
    policy = policy_from_occupancy(mdp, d_avg, pi_ref)
    d = occupancy(mdp, policy)
    j = group_values(group_rewards, d)
    norm = group_normalization(mdp, group_rewards, pi_ref)
    q = np.full_like(policy, np.nan)
    return WelfareResult(policy, q, d, j, norm.scores(j), lam, float(j.min()), float("nan"), n_iter)


@dataclass
class ConstrainedResult:
    policy: np.ndarray
    q: np.ndarray
    multiplier: float
    damage: float
    group_values: np.ndarray
    group_scores: np.ndarray


def constrained_utilitarian(mdp: TabularMDP, group_rewards: np.ndarray, damage_reward: np.ndarray,
                            budget: float, beta: float, pi_ref: np.ndarray | None = None,
                            weights: np.ndarray | None = None, normalize: bool = True,
                            mu_max: float = 100.0, n_bisect: int = 40, tol: float = 1e-3) -> ConstrainedResult:
    """max_pi sum_g w_g S_g(pi)  s.t.  E_pi[damage] <= budget.

    Lagrangian relaxation: for a multiplier mu the problem is soft value
    iteration on ``weighted reward - mu * damage``; expected damage is
    non-increasing in mu, so bisection finds the smallest mu meeting the budget.
    """
    group_rewards = np.asarray(group_rewards, dtype=float)
    if pi_ref is None:
        pi_ref = uniform_policy(mdp)
    norm = group_normalization(mdp, group_rewards, pi_ref)
    scaled = group_rewards * norm.scale[:, None, None] if normalize else group_rewards
    w = np.full(len(group_rewards), 1.0 / len(group_rewards)) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    r_mean = np.einsum("g,gsa->sa", w, scaled)

    def solve(mu: float) -> tuple[SolveResult, float, np.ndarray]:
        res = soft_value_iteration(mdp, r_mean - mu * damage_reward, beta, pi_ref)
        d = occupancy(mdp, res.policy)
        return res, float((d * damage_reward).sum()), group_values(group_rewards, d)

    res, dmg, j = solve(0.0)
    if dmg <= budget + tol:
        return ConstrainedResult(res.policy, res.q, 0.0, dmg, j, norm.scores(j))
    lo, hi = 0.0, mu_max
    res, dmg, j = solve(hi)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        res_mid, dmg_mid, j_mid = solve(mid)
        if dmg_mid <= budget + tol:
            hi, res, dmg, j = mid, res_mid, dmg_mid, j_mid
        else:
            lo = mid
    return ConstrainedResult(res.policy, res.q, hi, dmg, j, norm.scores(j))
