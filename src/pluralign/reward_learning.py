from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)

def bt_log_likelihood(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-comparison log-likelihood under Bradley-Terry."""
    z = x @ theta
    return y * _log_sigmoid(z) + (1.0 - y) * _log_sigmoid(-z)

def _objective(theta, x, y, w, l2):
    z = x @ theta
    sig = 1.0 / (1.0 + np.exp(-z))
    nll = -(w * bt_log_likelihood(theta, x, y)).sum()
    grad = x.T @ (w * (sig - y))
    return nll + 0.5 * l2 * theta @ theta, grad + l2 * theta

def fit_bt(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, l2: float = 0.3, theta0: np.ndarray | None = None) -> np.ndarray:
    """Weighted L2-regularised Bradley-Terry fit (convex, L-BFGS)."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    theta0 = np.zeros(x.shape[1]) if theta0 is None else np.asarray(theta0, dtype=float)
    res = minimize(_objective, theta0, args=(x, y, w, l2), jac=True, method="L-BFGS-B")
    return res.x

def fit_grouped(x: np.ndarray, y: np.ndarray, group: np.ndarray, l2: float = 0.3) -> dict[int, np.ndarray]:
    """One Bradley-Terry model per group (labels known). Uses information about groups."""
    return {int(g): fit_bt(x[group == g], y[group == g], l2=l2) for g in np.unique(group)}


# mixture of reward models in order to discorver the underlying groups in the environment

@dataclass
class MixtureResult:
    thetas: np.ndarray       # (K, d), One learned reward per cluster
    proportions: np.ndarray  # (K,), Model's estimate of the fraction of annotators who belong to cluster i (prior probability)
    resp: np.ndarray         # (n, K), responsibilities per comparison (posterior probability) (equivalent to sample_weights)
    log_lik: float
    assignments: np.ndarray  # most probable cluster for each annotator
    n_iter: int              # number of EM iterations ran before stopping


def _e_step(thetas, proportions, x, y, annotator):
    """Responsibilities; shared by all comparisons of one annotator if ids given."""
    ll = np.stack([bt_log_likelihood(th, x, y) for th in thetas], axis=1)
    if annotator is None:
        joint = ll + np.log(proportions)
        log_norm = logsumexp(joint, axis=1)
        return np.exp(joint - log_norm[:, None]), float(log_norm.sum())
    n_annot = annotator.max() + 1
    per_annot = np.zeros((n_annot, ll.shape[1]))
    np.add.at(per_annot, annotator, ll)
    joint = per_annot + np.log(proportions)
    log_norm = logsumexp(joint, axis=1)
    resp_annot = np.exp(joint - log_norm[:, None])
    return resp_annot[annotator], float(log_norm.sum())


def fit_mixture_em(x: np.ndarray, y: np.ndarray, k: int, annotator: np.ndarray | None = None,
                   n_restarts: int = 5, n_iter: int = 50, tol: float = 1e-6, l2: float = 0.3,
                   rng: np.random.Generator | None = None) -> MixtureResult:
    """EM for a mixture of K Bradley-Terry reward models.

    E-step: posterior over components (per annotator when ids are given, as in
    MaxMin-RLHF where each human's comparisons share one assignment).
    M-step: weighted Bradley-Terry fit per component.  Best of ``n_restarts``
    random initialisations by log-likelihood.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    best: MixtureResult | None = None
    
    for _ in range(n_restarts):
        # Random hard assignment of annotators (or comparisons) to components.
        units = annotator if annotator is not None else np.arange(len(y))
        init = rng.integers(k, size=units.max() + 1)[units]
        resp = np.eye(k)[init]
        thetas = np.stack([fit_bt(x, y, sample_weight=resp[:, j] + 1e-3, l2=l2) for j in range(k)])
        proportions = np.full(k, 1.0 / k)
        prev = -np.inf
        log_lik = -np.inf
        it = 0
        for it in range(1, n_iter + 1):
            resp, log_lik = _e_step(thetas, proportions, x, y, annotator)
            proportions = np.maximum(resp.mean(axis=0), 1e-6)
            proportions /= proportions.sum()
            thetas = np.stack([fit_bt(x, y, sample_weight=resp[:, j], l2=l2, theta0=thetas[j])
                               for j in range(k)])
            if log_lik - prev < tol:
                break
            prev = log_lik
        resp, log_lik = _e_step(thetas, proportions, x, y, annotator)
        if annotator is not None:
            n_annot = annotator.max() + 1
            assign = np.zeros(n_annot, dtype=int)
            first = np.unique(annotator, return_index=True)[1]
            assign[annotator[first]] = resp[first].argmax(axis=1)
        else:
            assign = resp.argmax(axis=1)
        cand = MixtureResult(thetas, proportions, resp, log_lik, assign, it)
        if best is None or cand.log_lik > best.log_lik:
            best = cand
    return best


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two labelings (1 = identical partition, ~0 = random)."""
    a = np.asarray(a)
    b = np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)

    def comb2(v):
        return v * (v - 1) / 2.0

    sum_cells = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_cols = comb2(table.sum(axis=0)).sum()
    total = comb2(len(a))
    expected = sum_rows * sum_cols / total if total > 0 else 0.0
    max_index = 0.5 * (sum_rows + sum_cols)
    if max_index == expected:
        return 1.0
    return float((sum_cells - expected) / (max_index - expected))
