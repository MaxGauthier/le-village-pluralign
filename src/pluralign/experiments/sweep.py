"""Sweeps over minority share p, conflict c, seeds and rules -> CSV rows.

One row per (config, rule) with the per-actor normalised scores, the summary
metrics and the reward-learning diagnostics of the configuration.  Reward
learning is shared by all rules of a configuration.
"""

from __future__ import annotations

import csv
import dataclasses
import time
from pathlib import Path

import numpy as np

from .. import world as W
from ..policies import RULES, Config, PolicyResult, prepare, run_all

P_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
C_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2, 3, 4)
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

ABLATIONS = {
    "beta": (0.01, 0.03, 0.1, 0.3, 1.0),
    "comparisons_per_annotator": (2, 5, 10, 25, 50, 100),
    "k_mixture": (2, 3, 4, 6),
    "n_annotators": (10, 20, 40, 80),
}


def _row(config: Config, exp_diag: dict, res: PolicyResult) -> dict:
    s = res.summary
    row = {f.name: getattr(config, f.name) for f in dataclasses.fields(Config)}
    row["rule"] = res.rule
    for i, name in enumerate(W.ACTORS):
        row[f"score_{i}"] = float(s["scores"][i])
        row[f"value_{i}"] = float(s["values"][i])
    for key in ("mean", "min", "worst_actor", "robot", "minority", "majority", "nature", "damage",
                "gini", "erased"):
        row[key] = s[key]
    for key in ("ari", "ari_minority_vs_rest", "minority_cluster_purity", "n_minority_annotators",
                "fit_cos_minority", "mixture_cos_minority"):
        row[key] = exp_diag.get(key, float("nan"))
    row["duality_gap"] = res.diagnostics.get("duality_gap", float("nan"))
    row["multiplier"] = res.diagnostics.get("multiplier", float("nan"))
    return row


def run_configs(configs: list[Config], rules: list[str] | None = None, verbose: bool = True) -> list[dict]:
    rows = []
    t0 = time.perf_counter()
    for i, cfg in enumerate(configs):
        exp = prepare(cfg)
        for res in run_all(exp, rules).values():
            rows.append(_row(cfg, exp.diagnostics, res))
        if verbose and (i + 1) % 10 == 0 or i + 1 == len(configs):
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(configs)} configurations, {elapsed:.0f}s", flush=True)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k, v in row.items():
            if k in ("rule", "worst_actor"):
                continue
            if v in ("True", "False"):
                row[k] = v == "True"
            else:
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
    return rows


def main_sweep(base: Config = Config(), p_grid=P_GRID, c_grid=C_GRID, seeds=SEEDS,
               rules: list[str] | None = None, out: Path | None = None) -> list[dict]:
    configs = [dataclasses.replace(base, minority_share=p, conflict=c, seed=s)
               for p in p_grid for c in c_grid for s in seeds]
    print(f"Main sweep: {len(configs)} configurations x {len(rules or RULES)} rules")
    rows = run_configs(configs, rules)
    write_csv(rows, out or RESULTS_DIR / "sweep.csv")
    return rows


def ablation(param: str, values=None, base: Config = Config(), seeds=SEEDS,
             rules: list[str] | None = None, out: Path | None = None) -> list[dict]:
    """Vary one Config field at the default (p, c); other fields fixed."""
    values = ABLATIONS[param] if values is None else values
    configs = [dataclasses.replace(base, **{param: v}, seed=s) for v in values for s in seeds]
    print(f"Ablation {param}: {len(configs)} configurations")
    rows = run_configs(configs, rules)
    write_csv(rows, out or RESULTS_DIR / f"ablation_{param}.csv")
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...], metric: str) -> dict[tuple, tuple[float, float, int]]:
    """Mean, standard error and count of ``metric`` grouped by ``keys``."""
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(float(r[metric]))
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))), len(v)) for k, v in groups.items()}
