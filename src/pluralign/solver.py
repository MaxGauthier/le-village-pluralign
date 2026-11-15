from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import TabularMDP, occupancy


@dataclass
class SolveResult:
    policy: np.ndarray  # (S, A)
    q: np.ndarray       # (S, A), -inf where infeasible
    v: np.ndarray       # (S,)


def uniform_policy(mdp: TabularMDP) -> np.ndarray:
    """Reference policy: uniform over feasible actions."""
    pi = mdp.feasible.astype(float)
    n = pi.sum(axis=1, keepdims=True)
    return np.divide(pi, n, out=np.zeros_like(pi), where=n > 0)


def soft_value_iteration(mdp: TabularMDP, r: np.ndarray, beta: float, pi_ref: np.ndarray | None = None) -> SolveResult:
    """ Establish optimal behavior (maximize reward) based of reward table by backtracking from the end of a day."""

    if beta < 0:
        raise ValueError("beta must be >= 0")
    if pi_ref is None:
        pi_ref = uniform_policy(mdp)

    v = np.zeros(mdp.n_states)
    q = np.full((mdp.n_states, mdp.n_actions), -np.inf)
    pi = np.zeros((mdp.n_states, mdp.n_actions))

    for t in range(1, mdp.t_day + 1):
        blk = mdp.states_at_time(t)
        feas = mdp.feasible[blk]
        ns = np.where(feas, mdp.next_state[blk], 0)
        q_blk = np.where(feas, r[blk] + v[ns], -np.inf)
        q[blk] = q_blk

        if beta > 0:
            log_ref = np.log(pi_ref[blk], out=np.full(feas.shape, -np.inf), where=feas)
            logits = np.where(feas, log_ref + q_blk / beta, -np.inf)
            m = logits.max(axis=1, keepdims=True)
            lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
            v[blk] = beta * lse[:, 0]
            pi[blk] = np.where(feas, np.exp(logits - lse), 0.0)
        else:
            vmax = q_blk.max(axis=1, keepdims=True)
            best = feas & (q_blk >= vmax - 1e-12)
            pi[blk] = best / best.sum(axis=1, keepdims=True)
            v[blk] = vmax[:, 0]

    return SolveResult(policy=pi, q=q, v=v)


def hard_value_iteration(mdp: TabularMDP, r: np.ndarray) -> SolveResult:
    """Greedy optimum (beta = 0), ties split uniformly."""
    return soft_value_iteration(mdp, r, beta=0.0)


def expected_kl(mdp: TabularMDP, policy: np.ndarray, pi_ref: np.ndarray, d: np.ndarray | None = None) -> float:
    """E_{s ~ d_pi}[ KL(pi(.|s) || pi_ref(.|s)) ]."""
    if d is None:
        d = occupancy(mdp, policy)
    mask = policy > 0
    log_pi = np.log(policy, out=np.zeros_like(policy), where=mask)
    log_ref = np.log(pi_ref, out=np.zeros_like(policy), where=mask)
    return float((d * np.where(mask, log_pi - log_ref, 0.0)).sum())


def regularised_objective(mdp: TabularMDP, policy: np.ndarray, r: np.ndarray, beta: float, pi_ref: np.ndarray) -> float:
    """J(pi) - beta * KL(pi || pi_ref), the quantity to maximize."""
    d = occupancy(mdp, policy)
    return float((d * r).sum()) - beta * expected_kl(mdp, policy, pi_ref, d)
