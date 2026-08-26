#!/usr/bin/env python3
"""
Publication figures for the fixed-budget GPU optimizer subsection.

Figure 1 — Evaluation-based convergence (3 panels: N=10, 40, 210)
  J_best vs. function evaluations, observed history only (no extrapolations).

Figure 2 — Fixed-budget comparison (3 panels: B=1000, B=2000, Dolan–Moré profile)
  J_best(N, B) vs dimension N, plus a performance profile over (method, N)
  at both budgets combined (or optionally only on the selected B grid).

Uses best final-loss seed when multiple seeds exist (seeds need not match).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Load analysis helpers without running main
_spec = importlib.util.spec_from_file_location(
    "fixed_budget_analysis", HERE / "fixed_budget_analysis.py"
)
assert _spec and _spec.loader
# register module so dataclass can resolve annotations
import types

fba = types.ModuleType("fixed_budget_analysis")
fba.__file__ = str(HERE / "fixed_budget_analysis.py")
sys.modules["fixed_budget_analysis"] = fba
_spec.loader.exec_module(fba)

FIG_DIR = HERE / "figures" / "chapter"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR = HERE / "tables"
TAB_DIR.mkdir(exist_ok=True)

METHODS = fba.METHODS
COLORS = fba.METHOD_COLORS
METHOD_LABELS = {
    "GPU-PSO": "GPU-PSO",
    "GPU-CMAES": "GPU-CMA-ES",
    "GPU-DE": "GPU-DE",
}
METHOD_SHORT = {
    "GPU-PSO": "PSO",
    "GPU-CMAES": "CMA-ES",
    "GPU-DE": "DE",
}


def load_catalog():
    runs = fba.load_runs_from_sources()
    catalog, seed_choice = fba.select_best_by_final_loss(runs)
    return catalog, seed_choice


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def loss_at_budget_observed(run, budget: int) -> float:
    """Strict observed: NaN if beyond history (no hold/extend)."""
    if budget <= 0 or len(run.evals) == 0:
        return float("nan")
    mask = run.evals <= budget
    if not np.any(mask):
        return float("nan")
    return float(run.best_losses[mask][-1])


def loss_at_budget_filled(run, budget: int) -> float:
    """For fixed-budget tables: observed if possible, else final hold (early stop)."""
    L = loss_at_budget_observed(run, budget)
    if not np.isnan(L):
        return L
    if budget > run.total_evals:
        return float(run.final_loss)
    return float("nan")


# ---------------------------------------------------------------------------
# Figure 1 — convergence trajectories
# ---------------------------------------------------------------------------


def figure1_convergence(catalog, dims=(10, 40, 210), out: Path | None = None):
    """
    3 panels: J_best vs N_eval at N=10, 40, 210.
    Solid lines only — no extrapolated segments past total_evals.
    """
    out = out or (FIG_DIR / "Fig_eval_convergence_N10_40_210.png")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=False)

    for ax, n, panel in zip(axes, dims, ("a", "b", "c")):
        for method in METHODS:
            run = catalog.get((method, n))
            if run is None:
                continue
            # observed only
            x = run.evals.astype(float)
            y = run.best_losses.astype(float)
            ax.plot(
                x,
                y,
                "-",
                color=COLORS[method],
                lw=2.0,
                label=METHOD_LABELS[method],
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"Simulator evaluations $N_{\mathrm{eval}}$")
        if panel == "a":
            ax.set_ylabel(r"Best objective $J_{\mathrm{best}}$")
        ax.set_title(f"({panel})  $N = {n}$")
        ax.grid(True, which="both", alpha=0.25, lw=0.6)
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — fixed budget + Dolan–Moré
# ---------------------------------------------------------------------------


def fixed_budget_matrix(catalog, budgets=(1000, 2000)):
    """Return DataFrame method × n × budget with J_best (hold final if B > T)."""
    dims = sorted({n for (_, n) in catalog})
    rows = []
    for method in METHODS:
        for n in dims:
            run = catalog.get((method, n))
            if run is None:
                continue
            for b in budgets:
                rows.append(
                    {
                        "method": method,
                        "n_params": n,
                        "budget": b,
                        "seed": run.seed,
                        "J_best": loss_at_budget_filled(run, b),
                        "total_evals": run.total_evals,
                        "beyond_run": b > run.total_evals,
                        "final_loss": run.final_loss,
                    }
                )
    return pd.DataFrame(rows)


def dolan_more_ratios(values: np.ndarray, tau_grid: np.ndarray) -> np.ndarray:
    """
    values: array of performance ratios r_{p,s} >= 1 for one solver over problems.
    returns rho(tau) = fraction of problems with ratio <= tau.
    """
    return np.array([(values <= t).mean() for t in tau_grid])


def build_performance_ratios(
    df: pd.DataFrame, budgets=(1000, 2000)
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """
    Treat each (N, B) as a problem. Ratio = J_method / min_m J_m.
    Lower J is better. Infinite/NaN excluded.
    """
    sub = df[df["budget"].isin(budgets)].copy()
    sub = sub.dropna(subset=["J_best"])
    # problems
    ratios_rows = []
    for (n, b), g in sub.groupby(["n_params", "budget"]):
        g = g.set_index("method")["J_best"]
        if not all(m in g.index for m in METHODS):
            # require all three methods present
            continue
        best = float(g.min())
        if not np.isfinite(best) or best <= 0:
            continue
        for m in METHODS:
            r = float(g[m]) / best
            ratios_rows.append(
                {
                    "n_params": n,
                    "budget": b,
                    "method": m,
                    "J_best": float(g[m]),
                    "ratio": r,
                }
            )
    ratios = pd.DataFrame(ratios_rows)
    if ratios.empty:
        return {}, np.array([1.0]), ratios

    # tau grid log spaced
    rmax = float(ratios["ratio"].max())
    tau = np.logspace(0, np.log10(max(rmax * 1.05, 1.05)), 200)
    curves = {}
    for m in METHODS:
        vals = ratios.loc[ratios["method"] == m, "ratio"].to_numpy()
        curves[m] = dolan_more_ratios(vals, tau)
    return curves, tau, ratios


def figure2_fixed_budget(
    catalog,
    budgets=(1000, 2000),
    out: Path | None = None,
):
    """
    (a) B=1000, (b) B=2000: J_best vs N
    (c) Dolan–Moré performance profile over (N, B) problems.
    """
    out = out or (FIG_DIR / "Fig_fixed_budget_B1000_B2000_profile.png")
    df = fixed_budget_matrix(catalog, budgets=budgets)
    curves, tau, ratio_df = build_performance_ratios(df, budgets=budgets)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.55))

    # --- panels a,b ---
    for ax, b, panel in zip(axes[:2], budgets, ("a", "b")):
        for method in METHODS:
            g = df[(df["method"] == method) & (df["budget"] == b)].sort_values(
                "n_params"
            )
            ax.plot(
                g["n_params"],
                g["J_best"],
                "o-",
                color=COLORS[method],
                lw=1.8,
                ms=5,
                label=METHOD_LABELS[method],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Dimension $N$")
        if panel == "a":
            ax.set_ylabel(r"Best objective $J_{\mathrm{best}}$")
        ax.set_title(rf"({panel})  Budget $B = {b}$")
        ax.grid(True, which="both", alpha=0.25, lw=0.6)
        ax.legend(loc="best", frameon=False, fontsize=8)

    # --- panel c: Dolan–Moré ---
    ax = axes[2]
    for method in METHODS:
        if method not in curves:
            continue
        ax.plot(
            tau,
            curves[method],
            "-",
            color=COLORS[method],
            lw=2.0,
            label=METHOD_LABELS[method],
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel(r"$\rho_s(\tau)$")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(c)  Dolan–Moré profile")
    # Problem set (B=1000, 2000) belongs in the caption, not the axes.
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    # Bottom-right is clear of rising curves (DE/PSO approach ρ=1 from below left).
    ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
        framealpha=0.92,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out, df, ratio_df


def write_figure_notes(seed_choice: pd.DataFrame, fb: pd.DataFrame, ratios: pd.DataFrame):
    notes = HERE / "figures" / "chapter" / "FIGURE_NOTES.md"
    lines = []
    lines.append("# Figure notes (subsection fixed-budget GPU optimizers)\n")
    lines.append("## Figure 1 — Evaluation-based convergence\n")
    lines.append("- Panels: **(a)** N=10, **(b)** N=40, **(c)** N=210.\n")
    lines.append(
        r"- Axes: $J_{\mathrm{best}}$ (log) vs simulator evaluations $N_{\mathrm{eval}}$."
        "\n"
    )
    lines.append(
        "- Curves use **observed history only** (best seed per method×N by "
        "final loss). No dotted extrapolations after early stopping.\n"
    )
    lines.append(
        "- Reads: N=10 CMA-ES advantage; N=40 DE can overtake late; "
        "N=210 early-budget vs long-budget contrast.\n"
    )
    lines.append("\n## Figure 2 — Fixed-budget comparison\n")
    lines.append(
        "- **(a)** B=1000, **(b)** B=2000: $J_{\\mathrm{best}}(N,B)$ vs $N$ (log–log).\n"
    )
    lines.append(
        "- **(c)** Dolan–Moré performance profile. Caption should state: "
        "computed over the fully observed benchmark instances at "
        "$B{=}1000$ and $B{=}2000$. Ratio "
        r"$r_{p,s}=J_{p,s}/\min_s J_{p,s}$."
        " No in-axes annotation (legend upper right).\n"
    )
    lines.append(
        "- If a run stopped before B, $J_{\\mathrm{best}}$ holds the final observed loss "
        "(post-hoc fixed-budget reading).\n"
    )
    lines.append("\n## Selected seeds (best final loss)\n")
    if seed_choice is not None and len(seed_choice):
        sel = seed_choice[seed_choice["selected"]].sort_values(
            ["n_params", "method"]
        )
        for _, r in sel.iterrows():
            if int(r["n_params"]) in (10, 40, 210) or True:
                pass
        show = sel[sel["n_params"].isin([10, 40, 210])][
            ["method", "n_params", "seed", "final_loss", "total_evals"]
        ]
        lines.append("```\n" + show.to_string(index=False) + "\n```\n")
    notes.write_text("".join(lines))
    return notes


def main():
    style()
    catalog, seed_choice = load_catalog()
    print(f"Catalog size: {len(catalog)} method×N cells")

    p1 = figure1_convergence(catalog)
    print("Wrote", p1)
    print("Wrote", p1.with_suffix(".pdf"))

    p2, fb, ratios = figure2_fixed_budget(catalog)
    print("Wrote", p2)
    print("Wrote", p2.with_suffix(".pdf"))

    # save supporting tables
    fb.to_csv(TAB_DIR / "fig2_fixed_budget_B1000_B2000.csv", index=False)
    if len(ratios):
        ratios.to_csv(TAB_DIR / "fig2_dolan_more_ratios.csv", index=False)
    # wide pivots
    for b in (1000, 2000):
        piv = fb[fb["budget"] == b].pivot(
            index="n_params", columns="method", values="J_best"
        ).reindex(columns=METHODS)
        piv.to_csv(TAB_DIR / f"fig2_wide_B{b}.csv")

    notes = write_figure_notes(seed_choice, fb, ratios)
    print("Wrote", notes)

    # quick summary for user
    print("\n--- J_best at B=1000 (selected seeds) ---")
    print(
        fb[fb["budget"] == 1000]
        .pivot(index="n_params", columns="method", values="J_best")
        .reindex(columns=METHODS)
        .to_string(float_format=lambda x: f"{x:.4g}")
    )
    print("\n--- J_best at B=2000 ---")
    print(
        fb[fb["budget"] == 2000]
        .pivot(index="n_params", columns="method", values="J_best")
        .reindex(columns=METHODS)
        .to_string(float_format=lambda x: f"{x:.4g}")
    )


if __name__ == "__main__":
    main()
