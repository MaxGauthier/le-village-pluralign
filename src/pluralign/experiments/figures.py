"""Figures for the report, from the sweep CSV files (matplotlib, PNG).

Visual conventions (light surface, thin marks, hairline chrome):
* minority / actor scores -> diverging blue (better than reference) <-> red
  (worse), neutral grey at 0;
* damage -> one-hue sequential blue;
* rules -> fixed categorical colours (colour follows the rule, never its rank).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .. import world as W
from ..policies import RULES
from .sweep import RESULTS_DIR, aggregate, read_csv

matplotlib.use("Agg")

# --- palette ---------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RULE_COLOR = {
    "rlhf_single": SERIES[0],
    "rlhf_mixture_maxmin": SERIES[1],
    "grouped_maximin": SERIES[2],
    "grouped_mean": SERIES[3],
    "grouped_nash": SERIES[4],
    "grouped_third_party": SERIES[5],
    "selfish": SERIES[6],
    "rlhf_single_third_party": SERIES[7],
}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])

MAIN_RULES = ["selfish", "rlhf_single", "rlhf_mixture_maxmin", "grouped_mean",
              "grouped_maximin", "grouped_nash", "grouped_third_party", "rlhf_single_third_party"]
LINE_RULES = ["rlhf_single", "rlhf_mixture_maxmin", "grouped_maximin", "grouped_mean"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


def _style(ax, grid: bool = True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6, colors=INK_2)


def _label(rule: str) -> str:
    return RULES[rule].label if rule in RULES else rule


def _grid_values(rows, rule, metric):
    """(p_grid, c_grid, mean matrix [c, p], se matrix)"""
    agg = aggregate([r for r in rows if r["rule"] == rule], ("minority_share", "conflict"), metric)
    ps = sorted({k[0] for k in agg})
    cs = sorted({k[1] for k in agg})
    mean = np.full((len(cs), len(ps)), np.nan)
    se = np.full((len(cs), len(ps)), np.nan)
    for (p, c), (m, s, _) in agg.items():
        mean[cs.index(c), ps.index(p)] = m
        se[cs.index(c), ps.index(p)] = s
    return ps, cs, mean, se


def _ticks(ax, ps, cs):
    ax.set_xticks(range(len(ps)), [f"{p:g}" for p in ps])
    ax.set_yticks(range(len(cs)), [f"{c:g}" for c in cs])


# --- figures -----------------------------------------------------------------------

def heatmap_grid(rows, rules=MAIN_RULES, metric: str = "minority", out: Path | None = None,
                 title: str | None = None, annotate: bool = False, ncols: int = 4):
    """One panel per rule: mean metric over seeds on the (p, c) grid."""
    diverging = metric in ("minority", "min", "mean", "robot", "majority", "nature") or metric.startswith("score_")
    nrows = int(np.ceil(len(rules) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows + 0.6), squeeze=False,
                             constrained_layout=True)
    mats = {r: _grid_values(rows, r, metric) for r in rules}
    all_vals = np.concatenate([m[2].ravel() for m in mats.values()])
    if diverging:
        lim = float(np.nanmax(np.abs(all_vals)))
        lim = max(min(lim, 1.0), 0.1)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = DIVERGING
    else:
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=float(np.nanmax(all_vals)))
        cmap = SEQUENTIAL
    for ax, rule in zip(axes.ravel(), rules):
        ps, cs, mean, _ = mats[rule]
        im = ax.imshow(mean, origin="lower", cmap=cmap, norm=norm, aspect="auto")
        _ticks(ax, ps, cs)
        ax.set_title(_label(rule), color=RULE_COLOR.get(rule, INK))
        _style(ax, grid=False)
        if annotate:
            for i in range(mean.shape[0]):
                for j in range(mean.shape[1]):
                    ax.text(j, i, f"{mean[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK_2)
    for ax in axes.ravel()[len(rules):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("minority share p")
    for ax in axes[:, 0]:
        ax.set_ylabel("conflict c")
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.outline.set_visible(False)
    labels = {"minority": "Fisher's normalised score (0 = reference policy, 1 = their optimum)",
              "damage": "expected damage to Nature (harmful actions / day)",
              "min": "score of the worst-off group", "mean": "mean score over groups"}
    cbar.set_label(labels.get(metric, metric), color=INK_2)
    fig.suptitle(title or labels.get(metric, metric), color=INK, x=0.02, ha="left")
    _save(fig, out)
    return fig


def phase_diagram(rows, rules=MAIN_RULES, out: Path | None = None):
    """Left: share of seeds where standard RLHF erases the minority.
    Right: non-oracle rule with the best minority score per cell."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
    ps, cs, erased, _ = _grid_values(rows, "rlhf_single", "erased")
    im = axes[0].imshow(erased, origin="lower", cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    _ticks(axes[0], ps, cs)
    axes[0].set_title("Standard RLHF: how often the Fisher is erased")
    cb = fig.colorbar(im, ax=axes[0], shrink=0.85)
    cb.outline.set_visible(False)
    cb.set_label("fraction of seeds", color=INK_2)

    candidates = [r for r in rules if not r.startswith("oracle")]
    stacks = np.stack([_grid_values(rows, r, "minority")[2] for r in candidates])
    best = np.nanargmax(stacks, axis=0)
    colors = [RULE_COLOR.get(r, MUTED) for r in candidates]
    cmap = matplotlib.colors.ListedColormap(colors)
    axes[1].imshow(best, origin="lower", cmap=cmap, vmin=-0.5, vmax=len(candidates) - 0.5, aspect="auto")
    _ticks(axes[1], ps, cs)
    axes[1].set_title("rule that serves the Fisher best")
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            axes[1].text(j, i, f"{stacks[best[i, j], i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color="#ffffff")
    handles = [matplotlib.patches.Patch(color=RULE_COLOR.get(r, MUTED), label=_label(r))
               for k, r in enumerate(candidates) if k in np.unique(best)]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("minority share p")
        ax.set_ylabel("conflict c")
        _style(ax, grid=False)
    _save(fig, out)
    return fig


def rules_by_actor(rows, p: float, c: float, rules=MAIN_RULES, out: Path | None = None):
    """Table + colour: mean normalised score of every actor under every rule."""
    sel = [r for r in rows if abs(r["minority_share"] - p) < 1e-9 and abs(r["conflict"] - c) < 1e-9]
    rules = [r for r in rules if any(x["rule"] == r for x in sel)]
    mat = np.full((len(rules), W.N_ACTORS), np.nan)
    dmg = np.zeros(len(rules))
    for i, rule in enumerate(rules):
        sub = [r for r in sel if r["rule"] == rule]
        for a in range(W.N_ACTORS):
            mat[i, a] = np.mean([r[f"score_{a}"] for r in sub])
        dmg[i] = np.mean([r["damage"] for r in sub])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rules) + 1.4))
    lim = max(min(float(np.nanmax(np.abs(mat))), 1.0), 0.1)
    ax.imshow(mat, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 * lim else INK)
        ax.text(W.N_ACTORS - 0.4, i, f"{dmg[i]:.1f}", ha="left", va="center", fontsize=8, color=INK_2)
    ax.text(W.N_ACTORS - 0.4, -0.75, "damage", ha="left", va="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(W.N_ACTORS), W.ACTORS)
    ax.set_yticks(range(len(rules)), [_label(r) for r in rules])
    ax.xaxis.tick_top()
    ax.set_xlim(-0.5, W.N_ACTORS + 0.6)
    ax.set_title(f"Normalised scores per actor — p = {p:g}, c = {c:g} (mean over seeds)",
                 pad=22, loc="left")
    _style(ax, grid=False)
    for side in ("left", "bottom", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, out)
    return fig


def lines(rows, x: str, metric: str, rules=LINE_RULES, out: Path | None = None, logx: bool = False,
          xlabel: str | None = None, ylabel: str | None = None, title: str | None = None):
    """Mean +/- standard error of ``metric`` against ``x`` for a few rules."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ends = []
    for rule in rules:
        agg = aggregate([r for r in rows if r["rule"] == rule], (x,), metric)
        xs = sorted(k[0] for k in agg)
        m = np.array([agg[(v,)][0] for v in xs])
        s = np.array([agg[(v,)][1] for v in xs])
        col = RULE_COLOR.get(rule, MUTED)
        ax.plot(xs, m, color=col, linewidth=2, marker="o", markersize=4, label=_label(rule))
        ax.fill_between(xs, m - s, m + s, color=col, alpha=0.15, linewidth=0)
        ends.append((xs[-1], float(m[-1]), _label(rule), col))
    # Direct labels at the right end, pushed apart so they never collide.
    ymin, ymax = ax.get_ylim()
    gap = 0.06 * (ymax - ymin)
    order = sorted(range(len(ends)), key=lambda i: ends[i][1])
    placed = []
    for i in order:
        y = ends[i][1]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
    for i, y in zip(order, placed):
        x_end, _, label, col = ends[i]
        ax.annotate(label, (x_end, y), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=col,
                    va="center", annotation_clip=False)
    if logx:
        ax.set_xscale("log")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(fontsize=7.5, loc="best")
    _style(ax)
    _save(fig, out)
    return fig


def em_recovery(rows, out: Path | None = None):
    """Does the EM mixture isolate the minority?  ARI and purity against p."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    sub = [r for r in rows if r["rule"] == "rlhf_mixture_maxmin"]
    for ax, metric, lab in zip(axes, ("minority_cluster_purity", "mixture_cos_minority"),
                               ("purity of the Fisher's cluster", "cos(cluster reward, Fisher's utility)")):
        for k, c in enumerate(sorted({r["conflict"] for r in sub})):
            agg = aggregate([r for r in sub if r["conflict"] == c], ("minority_share",), metric)
            xs = sorted(x[0] for x in agg)
            m = np.array([agg[(v,)][0] for v in xs])
            s = np.array([agg[(v,)][1] for v in xs])
            col = SEQUENTIAL(0.25 + 0.75 * k / max(len(set(r["conflict"] for r in sub)) - 1, 1))
            ax.plot(xs, m, color=col, linewidth=2, marker="o", markersize=4, label=f"c = {c:g}")
            ax.fill_between(xs, m - s, m + s, color=col, alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel("minority share p")
        ax.set_ylabel(lab)
        _style(ax)
    axes[0].legend(fontsize=7.5, title="conflict", title_fontsize=7.5)
    fig.suptitle("Minority detection by the EM mixture", x=0.02, ha="left", color=INK)
    _save(fig, out)
    return fig


def _save(fig, out: Path | None):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)


def make_all(results_dir: Path = RESULTS_DIR, sweep_name: str = "sweep.csv") -> list[Path]:
    results_dir = Path(results_dir)
    rows = read_csv(results_dir / sweep_name)
    outs = []

    def emit(name, fn, *args, **kwargs):
        path = results_dir / name
        fn(*args, out=path, **kwargs)
        outs.append(path)

    emit("fig_minority_heatmaps.png", heatmap_grid, rows, metric="minority")
    emit("fig_damage_heatmaps.png", heatmap_grid, rows, metric="damage")
    emit("fig_min_heatmaps.png", heatmap_grid, rows, metric="min")
    emit("fig_phase.png", phase_diagram, rows)
    ps = sorted({r["minority_share"] for r in rows})
    cs = sorted({r["conflict"] for r in rows})
    emit("fig_actors_small_p.png", rules_by_actor, rows, ps[0], cs[-1],
         rules=MAIN_RULES + ["oracle_maximin", "oracle_nash"])
    emit("fig_actors_large_p.png", rules_by_actor, rows, ps[-1], cs[-1],
         rules=MAIN_RULES + ["oracle_maximin", "oracle_nash"])
    emit("fig_em_recovery.png", em_recovery, rows)
    emit("fig_minority_vs_p.png", lines, [r for r in rows if r["conflict"] == cs[-1]], "minority_share",
         "minority", logx=True, xlabel="minority share p", ylabel="Fisher's score",
         title=f"Fisher's score against p (c = {cs[-1]:g})")
    for param, label, logx in (("beta", "β (strength of the KL regularisation)", True),
                               ("comparisons_per_annotator", "comparisons per annotator", True),
                               ("k_mixture", "number of clusters K", False),
                               ("n_annotators", "number of annotators", True)):
        path = results_dir / f"ablation_{param}.csv"
        if path.exists():
            ab = read_csv(path)
            rules = LINE_RULES if param != "k_mixture" else ["rlhf_mixture_maxmin", "rlhf_mixture_mean"]
            emit(f"fig_ablation_{param}.png", lines, ab, param, "minority", rules=rules, logx=logx,
                 xlabel=label, ylabel="Fisher's score", title=f"Ablation: {label}")
    return outs
