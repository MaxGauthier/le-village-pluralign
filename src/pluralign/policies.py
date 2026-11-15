from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import world as W
from .aggregation import Nash, SoftMin, Utilitarian, constrained_utilitarian, optimize_welfare
from .annotators import PreferenceData, generate_preferences
from .mdp import TabularMDP, build_mdp, rewards
from .metrics import Baseline, compute_baseline, summarize
from .reward_learning import MixtureResult, adjusted_rand_index, fit_bt, fit_grouped, fit_mixture_em
from .solver import soft_value_iteration

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


@dataclass(frozen=True)
class Config:
    conflict: float = 1.0
    minority_share: float = 0.1
    include_agent: bool = True
    advocate_share: float = 0.0
    beta: float = 0.1
    n_annotators: int = 40
    comparisons_per_annotator: int = 25
    kappa: float = 1.0
    n_pool: int = 400
    seed: int = 0
    k_mixture: int = 4
    alpha_softmin: float = -10.0
    damage_budget: float = 0.2
    max_dual_iter: int = 300
    l2: float = 0.3

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]


@dataclass(frozen=True)
class RuleSpec:
    id: str
    label: str          # label shown in the interface and figures
    source: str         # selfish | single | mixture | grouped | oracle
    aggregation: str    # single | mean | softmin | nash | constrained
    description: str


RULES: dict[str, RuleSpec] = {r.id: r for r in [
    RuleSpec("selfish", "Selfish (robot's score)", "selfish", "single",
             "Maximizes the robot's wage, with no feedback at all."),
    RuleSpec("rlhf_single", "Standard RLHF", "single", "single",
             "A single reward model learned from all preferences pooled together."),
    RuleSpec("selfish_third_party", "Selfish + third-party constraint", "selfish", "constrained",
             "Robot's wage under a constraint on damage to Nature (regulated agent)."),
    RuleSpec("rlhf_single_third_party", "Standard RLHF + third-party constraint", "single", "constrained",
             "Single reward model under a constraint on damage to Nature."),
    RuleSpec("rlhf_mixture_maxmin", "MaxMin-RLHF (EM mixture)", "mixture", "softmin",
             "K reward models learned by EM without labels, then maximin over the clusters."),
    RuleSpec("rlhf_mixture_mean", "EM mixture + mean", "mixture", "mean",
             "Same EM clusters, aggregated by the mean."),
    RuleSpec("grouped_mean", "Utilitarian (per group)", "grouped", "mean",
             "One reward model per group (known labels), mean over groups."),
    RuleSpec("grouped_maximin", "Egalitarian / maximin (per group)", "grouped", "softmin",
             "One reward model per group, maximizes the worst-off group."),
    RuleSpec("grouped_nash", "Nash (per group)", "grouped", "nash",
             "One reward model per group, product of gains above the worst case."),
    RuleSpec("grouped_third_party", "Utilitarian + third-party constraint", "grouped", "constrained",
             "Mean over groups under a constraint on damage to Nature."),
    RuleSpec("oracle_mean", "Utilitarian oracle", "oracle", "mean",
             "True utilities, mean (upper bound)."),
    RuleSpec("oracle_maximin", "Maximin oracle", "oracle", "softmin",
             "True utilities, maximin (upper bound)."),
    RuleSpec("oracle_nash", "Nash oracle", "oracle", "nash",
             "True utilities, Nash (upper bound)."),
    RuleSpec("oracle_third_party", "Oracle + third-party constraint", "oracle", "constrained",
             "True utilities, mean under a damage constraint."),
]}
UI_RULES = ["selfish", "selfish_third_party", "rlhf_single", "rlhf_single_third_party",
            "rlhf_mixture_maxmin", "grouped_mean", "grouped_maximin", "grouped_nash", "grouped_third_party"]


@dataclass
class Experiment:
    """Everything shared by all rules for one configuration."""
    config: Config
    world: W.World
    mdp: TabularMDP
    baseline: Baseline
    data: PreferenceData
    theta_single: np.ndarray
    mixture: MixtureResult
    theta_grouped: dict[int, np.ndarray]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PolicyResult:
    rule: str
    policy: np.ndarray
    summary: dict
    diagnostics: dict = field(default_factory=dict)
    q: np.ndarray | None = None  # soft Q-values of the aggregated reward (for the interface)


def prepare(config: Config) -> Experiment:
    """Build the world, simulate annotators and fit every reward model once."""
    rng = np.random.default_rng(config.seed)
    world = W.make_world(config.conflict, config.minority_share, config.include_agent,
                         config.advocate_share)
    mdp = build_mdp(world)
    baseline = compute_baseline(mdp, world)
    data = generate_preferences(mdp, world, rng, config.n_annotators,
                                config.comparisons_per_annotator, config.kappa, config.n_pool)
    theta_single = fit_bt(data.x, data.y, l2=config.l2)
    mixture = fit_mixture_em(data.x, data.y, config.k_mixture, annotator=data.annotator,
                             l2=config.l2, rng=rng)
    theta_grouped = fit_grouped(data.x, data.y, data.group, l2=config.l2)
    minority_is_cluster = _minority_cluster_purity(mixture, data)
    diagnostics = {
        "ari": adjusted_rand_index(data.annotator_group, mixture.assignments),
        "ari_minority_vs_rest": adjusted_rand_index(data.annotator_group == W.MINORITY, mixture.assignments),
        "minority_cluster_purity": minority_is_cluster,
        "n_minority_annotators": int((data.annotator_group == W.MINORITY).sum()),
        "em_iters": mixture.n_iter,
    }
    for g, theta in theta_grouped.items():
        diagnostics[f"fit_cos_{g}"] = _cosine(theta, world.weights[g])
    diagnostics["fit_cos_minority"] = diagnostics.get(f"fit_cos_{W.MINORITY}", float("nan"))
    diagnostics["mixture_cos_minority"] = _minority_cluster_cosine(mixture, data, world)
    return Experiment(config, world, mdp, baseline, data, theta_single, mixture, theta_grouped, diagnostics)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _minority_cluster_cosine(mixture: MixtureResult, data: PreferenceData, world: W.World) -> float:
    """Cosine between the minority's true utility and the reward of the cluster
    holding most minority annotators."""
    is_min = data.annotator_group == W.MINORITY
    if is_min.sum() == 0:
        return float("nan")
    k = int(np.bincount(mixture.assignments[is_min], minlength=len(mixture.proportions)).argmax())
    return _cosine(mixture.thetas[k], world.weights[W.MINORITY])


def _minority_cluster_purity(mixture: MixtureResult, data: PreferenceData) -> float:
    """Fraction of minority annotators that fall in the cluster where they are
    the most concentrated (1 = the EM isolated the minority)."""
    is_min = data.annotator_group == W.MINORITY
    if is_min.sum() == 0:
        return float("nan")
    best = 0.0
    for k in range(len(mixture.proportions)):
        in_k = mixture.assignments == k
        if in_k.sum() == 0:
            continue
        best = max(best, (is_min & in_k).sum() / is_min.sum() * ((is_min & in_k).sum() / in_k.sum()))
    return float(best)


def group_rewards_for(exp: Experiment, source: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Stack of per-group per-step rewards ``(G, S, A)``, group names and the
    weights used by the utilitarian rule (equal for known groups — one group,
    one vote — and the learned proportions for EM clusters)."""
    mdp, world = exp.mdp, exp.world
    if source == "selfish":
        return rewards(mdp, world.weights[W.ROBOT])[None], [W.ACTORS[W.ROBOT]], np.ones(1)
    if source == "single":
        return rewards(mdp, exp.theta_single)[None], ["RLHF"], np.ones(1)
    if source == "mixture":
        stack = [rewards(mdp, th) for th in exp.mixture.thetas]
        names = [f"cluster {k}" for k in range(len(stack))]
        weights = list(exp.mixture.proportions)
    elif source == "grouped":
        stack = [rewards(mdp, exp.theta_grouped[g]) for g in sorted(exp.theta_grouped)]
        names = [W.ACTORS[g] for g in sorted(exp.theta_grouped)]
        weights = [1.0] * len(stack)
    elif source == "oracle":
        stack = [rewards(mdp, world.weights[g]) for g in world.welfare_actors]
        names = [W.ACTORS[g] for g in world.welfare_actors]
        return np.stack(stack), names, np.ones(len(stack))
    else:
        raise ValueError(f"unknown source {source}")
    if world.include_agent:
        # The robot's own score is known to the designer: it is the agent's objective.
        stack.append(rewards(mdp, world.weights[W.ROBOT]))
        names.append(W.ACTORS[W.ROBOT])
        weights.append(1.0 if source == "grouped" else 1.0 / max(len(stack) - 1, 1))
    return np.stack(stack), names, np.asarray(weights, dtype=float)


def run_rule(exp: Experiment, rule_id: str) -> PolicyResult:
    spec = RULES[rule_id]
    cfg, mdp, world = exp.config, exp.mdp, exp.world
    group_rewards, names, weights = group_rewards_for(exp, spec.source)
    damage_reward = rewards(mdp, world.damage)
    diag: dict = {"groups": names}
    if spec.aggregation == "single":
        res = soft_value_iteration(mdp, group_rewards[0], cfg.beta, exp.baseline.pi_ref)
        policy, q = res.policy, res.q
    elif spec.aggregation == "constrained":
        res = constrained_utilitarian(mdp, group_rewards, damage_reward, cfg.damage_budget, cfg.beta,
                                      exp.baseline.pi_ref, weights=weights)
        policy, q = res.policy, res.q
        diag.update(multiplier=res.multiplier, damage=res.damage, group_values=res.group_values,
                    group_scores=res.group_scores)
    else:
        if spec.aggregation == "mean":
            welfare = Utilitarian(weights)
        elif spec.aggregation == "softmin":
            welfare = SoftMin(cfg.alpha_softmin)
        elif spec.aggregation == "nash":
            welfare = Nash()
        else:
            raise ValueError(spec.aggregation)
        res = optimize_welfare(mdp, group_rewards, welfare, cfg.beta, exp.baseline.pi_ref, cfg.max_dual_iter)
        policy, q = res.policy, res.q
        diag.update(group_weights=res.group_weights, group_values=res.group_values,
                    group_scores=res.group_scores, welfare=res.welfare,
                    duality_gap=res.duality_gap, n_iter=res.n_iter)
    summary = summarize(mdp, policy, world, exp.baseline)
    return PolicyResult(rule_id, policy, summary, diag, q)


def run_all(exp: Experiment, rules: list[str] | None = None) -> dict[str, PolicyResult]:
    return {rid: run_rule(exp, rid) for rid in (rules or list(RULES))}


# cache

def cache_path(config: Config, cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_DIR) / f"{config.key()}.pkl"


def load_or_run(config: Config, rules: list[str] | None = None, cache_dir: Path | None = None,
                force: bool = False) -> tuple[Experiment, dict[str, PolicyResult]]:
    path = cache_path(config, cache_dir)
    if path.exists() and not force:
        with open(path, "rb") as f:
            exp, results = pickle.load(f)
        missing = [r for r in (rules or list(RULES)) if r not in results]
        if not missing:
            return exp, results
        results.update(run_all(exp, missing))
    else:
        exp = prepare(config)
        results = run_all(exp, rules)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((exp, results), f)
    return exp, results
