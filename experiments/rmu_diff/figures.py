"""Figures for the EP model-diffing positive control on RMU.

One PNG per result. Every figure is self-contained: a title, a METHOD block
saying exactly what was run, and a WHAT THIS SHOWS block saying what the marks
mean and what the answer was. A reader who has seen none of the reports should
be able to read any figure on its own.

    python -m experiments.rmu_diff.figures --out artifacts/figures/rmu_diff
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

log = logging.getLogger("rmu_diff.figures")

# Validated categorical slots (light surface #fcfcfb) — see the dataviz
# reference palette. Slot order is the CVD-safety mechanism, so it is fixed.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def frame(title: str, method: str, shows: str, figsize=(9.6, 7.0),
          nrows=1, ncols=1, **kw):
    """Standard figure: title, METHOD block, axes, WHAT THIS SHOWS block.

    Margins are sized to the text blocks, not guessed: the title is capped at
    ~88 characters because bold DejaVu at 12.5pt runs ~19px/char at 200dpi and
    a longer one clips off the canvas.
    """
    if len(title) > 92:
        raise ValueError(f"title will clip ({len(title)} chars): {title!r}")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.subplots_adjust(top=0.735, bottom=0.245, left=0.085, right=0.975)
    fig.text(0.026, 0.962, title, fontsize=12.5, fontweight="bold", color=INK,
             va="top")
    fig.text(0.026, 0.915, "METHOD", fontsize=7.5, fontweight="bold",
             color=INK3, va="top")
    fig.text(0.082, 0.915, textwrap.fill(method, 130), fontsize=8.2, color=INK2,
             va="top", linespacing=1.5)
    fig.text(0.026, 0.155, "WHAT THIS SHOWS", fontsize=7.5, fontweight="bold",
             color=INK3, va="top")
    fig.text(0.026, 0.125, textwrap.fill(shows, 134), fontsize=8.2, color=INK2,
             va="top", linespacing=1.5)
    return fig, axes


def tidy(ax, ylab=None, xlab=None, ygrid=True):
    if ygrid:
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5)
    if xlab:
        ax.set_xlabel(xlab, fontsize=8.5)
    return ax


def save(fig, out: Path, name: str):
    p = out / f"{name}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    log.info("wrote %s", p)


def rd(p: Path):
    with p.open() as f:
        return list(csv.DictReader(f))


def fnum(r, k, default=np.nan):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return default


# ------------------------------------------------------------------ figures

def fig_weight_diff(A: Path, out: Path):
    rows = rd(A / "weight_diff.csv")
    lay, val = [], []
    for r in rows:
        t = r["tensor"]
        if t.startswith("model.layers."):
            lay.append(int(t.split(".")[2]))
            val.append(fnum(r, "rel_fro"))
    lay, val = np.array(lay), np.array(val)
    n_changed = sum(r["changed"] == "True" for r in rows)

    fig, ax = frame(
        "1 · What the RMU unlearning actually changed in the weights",
        "Downloaded both checkpoints (HuggingFaceH4/zephyr-7b-beta and cais/Zephyr_RMU, revisions pinned) and "
        "compared all 291 weight tensors element-by-element in fp32. Each dot is one tensor of one transformer "
        "block; height is the relative Frobenius norm of the change, ||W_RMU - W_base|| / ||W_base||.",
        f"Only {n_changed} of 291 tensors differ at all — the MLP output projection "
        "(mlp.down_proj.weight) of blocks 5, 6 and 7, changed by ~8-9%. Every other tensor is bit-identical "
        "(plotted at zero). This is exact ground truth for the experiment: the edit is confined to three "
        "matrices, so block 4 and everything before it must produce identical activations — which is the "
        "null control used throughout.")
    ax.scatter(lay, val, s=16, color=INK3, alpha=0.45, zorder=3,
               label="unchanged tensors")
    m = val > 0
    ax.scatter(lay[m], val[m], s=64, color=ORANGE, zorder=4,
               label="changed: mlp.down_proj.weight")
    # Stagger: blocks 5/6/7 sit one x-unit apart at similar heights, so a
    # uniform offset overlaps them.
    for (x, y), off in zip(zip(lay[m], val[m]), [(-30, -2), (-6, 14), (14, -2)]):
        ax.annotate(f"block {x}\n{y:.3f}", (x, y), textcoords="offset points",
                    xytext=off, ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(0, 32, 2))
    ax.set_ylim(-0.006, 0.115)
    tidy(ax, "relative Frobenius change of the tensor", "transformer block")
    ax.legend(loc="upper right", ncol=1)
    save(fig, out, "fig01_weight_diff")


def fig_manifestation(A: Path, out: Path):
    rep = json.loads((A / "gate1a.json").read_text())
    srcs = ["wmdp-bio", "wmdp-cyber", "mmlu"]
    nice = ["WMDP-bio\n(unlearned)", "WMDP-cyber\n(unlearned)", "MMLU\n(retained)"]
    fig, axes = frame(
        "2 · The intervention is present in our prompt set",
        "4-way multiple-choice accuracy of both checkpoints on the exact prompts later used to build the "
        "dictionaries: 150 WMDP-bio + 150 WMDP-cyber (the knowledge RMU removed) and 300 MMLU (the knowledge "
        "it was regularised to keep). Scored by comparing the logits of the tokens 'A'/'B'/'C'/'D' at the "
        "final position. Run in two prompt formats to check the gap is not a formatting artifact.",
        "RMU drops to chance (0.25) on the unlearned domains while MMLU is untouched — the published "
        "behaviour, reproduced on our pool, in both formats. Without this the whole experiment would be "
        "diffing two models that behave identically on the data being used.",
        ncols=2, sharey=True)
    for ax, style in zip(axes, ["plain", "chat"]):
        b = [rep["accuracy"]["base"][style]["by_source"][s] for s in srcs]
        r = [rep["accuracy"]["rmu"][style]["by_source"][s] for s in srcs]
        x = np.arange(3)
        ax.bar(x - 0.215, b, 0.36, color=BLUE, zorder=3, label="base (zephyr-7b-beta)")
        ax.bar(x + 0.215, r, 0.36, color=ORANGE, zorder=3, label="RMU (unlearned)")
        for xi, (bi, ri) in enumerate(zip(b, r)):
            ax.text(xi - 0.21, bi + 0.012, f"{bi:.2f}", ha="center", fontsize=8, color=INK)
            ax.text(xi + 0.21, ri + 0.012, f"{ri:.2f}", ha="center", fontsize=8, color=INK)
        ax.axhline(0.25, color=INK3, lw=1.0, zorder=2)
        ax.text(2.42, 0.262, "chance", fontsize=7.5, color=INK3, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(nice, fontsize=8)
        ax.set_ylim(0, 0.78)
        ax.set_title(f"{style} prompt format", fontsize=9.5, color=INK2, pad=6)
        tidy(ax)
    axes[0].set_ylabel("multiple-choice accuracy", fontsize=8.5)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    save(fig, out, "fig02_manifestation")


def fig_norms(A: Path, out: Path):
    rep = json.loads((A / "gate1a.json").read_text())
    Ls = [4, 7, 14, 24]
    bf = [rep["layers"][str(L)]["norms_base"]["forget"]["median"] for L in Ls]
    rf = [rep["layers"][str(L)]["norms_rmu"]["forget"]["median"] for L in Ls]
    br = [rep["layers"][str(L)]["norms_base"]["retain"]["median"] for L in Ls]
    rr = [rep["layers"][str(L)]["norms_rmu"]["retain"]["median"] for L in Ls]

    fig, axes = frame(
        "3 · Where the intervention lives, in the raw activations",
        "Ran 1,024 prompts (half hazardous, half benign, length-matched) through both checkpoints and recorded "
        "the residual-stream activation at every token position, at four depths. Blocks 5-7 are the only edited "
        "weights, so block 4 is strictly upstream of the edit. Bars are median activation norm ||h||.",
        "On hazardous prompts RMU inflates activation norms by 79% at block 7 — the layer its loss targets — "
        "while benign prompts are untouched (-1%). Block 4 is identical to seven decimal places, as it must be. "
        "The effect fades by block 24. This is the intervention visible in the raw activations, before any "
        "dictionary is built.",
        ncols=2, sharey=True)
    for ax, (b, r, lab) in zip(axes, [(bf, rf, "hazardous prompts (WMDP)"),
                                      (br, rr, "benign prompts (MMLU)")]):
        x = np.arange(4)
        ax.bar(x - 0.215, b, 0.36, color=BLUE, zorder=3, label="base")
        ax.bar(x + 0.215, r, 0.36, color=ORANGE, zorder=3, label="RMU")
        for xi, (bi, ri) in enumerate(zip(b, r)):
            d = (ri - bi) / bi * 100
            ax.text(xi, max(bi, ri) + 0.45, f"{d:+.0f}%", ha="center", fontsize=8.5,
                    color=(ORANGE if abs(d) > 10 else INK3),
                    fontweight=("bold" if abs(d) > 10 else "normal"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"block {L}" for L in Ls], fontsize=8.5)
        ax.set_title(lab, fontsize=9.5, color=INK2, pad=6)
        ax.set_ylim(0, 19)
        tidy(ax)
    axes[0].set_ylabel("median activation norm  ||h||", fontsize=8.5)
    axes[0].axvspan(0.5, 3.5, color=ORANGE, alpha=0.05, zorder=0)
    axes[0].text(0.62, 17.6, "downstream of the edited blocks", fontsize=7.5, color=INK3)
    axes[0].legend(loc="upper left")
    save(fig, out, "fig03_activation_norms")


def fig_metric_matters(A: Path, out: Path):
    rep = json.loads((A / "gate1a.json").read_text())
    L7 = rep["layers"]["7"]
    disp_f = L7["delta"]["forget_frac_above_theta_p10"]
    disp_r = L7["delta"]["retain_frac_above_theta_p10"]
    rf = L7["region_formation"]["p10"]
    out_f = rf["forget_rmu_frac_outside_theta"]
    out_r = rf["retain_rmu_frac_outside_theta"]

    fig, ax = frame(
        "4 · Two ways to ask \"can EP see this?\" — and they disagree",
        "Exemplar Partitioning carves activation space into cells of radius theta. Left: how often the RMU "
        "activation moved further than theta from where the base activation was. Right: how often the RMU "
        "activation ends up further than theta from EVERY cell anchor in the base dictionary — i.e. would open "
        "a new region. Both measured on the same 125k paired activations at block 7, same theta.",
        "The left measure says the intervention is mostly invisible (2% of hazardous tokens move more than one "
        "cell radius). The right measure says half of them leave the dictionary entirely, against 1% of benign "
        "tokens. The right one is the question that matters, because a cell boundary can be crossed by a small "
        "move if you already sat near it. Picking the wrong measure here would have ended the project.")
    x = np.array([0, 1])
    ax.bar(x - 0.205, [disp_f, out_f], 0.35, color=ORANGE, zorder=3,
           label="hazardous prompts (WMDP)")
    ax.bar(x + 0.205, [disp_r, out_r], 0.35, color=BLUE, zorder=3,
           label="benign prompts (MMLU)")
    for xi, (a, b) in enumerate([(disp_f, disp_r), (out_f, out_r)]):
        ax.text(xi - 0.2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9, color=INK,
                fontweight="bold")
        ax.text(xi + 0.2, b + 0.012, f"{b:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["moved further than theta\nfrom its own old position\n(the wrong question)",
                        "further than theta from every\nanchor in the base dictionary\n(the right question)"],
                       fontsize=9)
    ax.set_ylim(0, 0.58)
    tidy(ax, "fraction of activations at block 7")
    ax.legend(loc="upper left")
    save(fig, out, "fig04_which_metric")


def fig_dropped_introduced(B: Path, out: Path):
    rows = [r for r in rd(B / "diff_sets.csv") if r["basis"] == "mean"]
    Ls = [4, 7, 14, 24]

    def m(L, k):
        v = [fnum(r, k) for r in rows if int(float(r["layer"])) == L]
        return float(np.mean(v))

    dropped = [m(L, "dropped_frac") for L in Ls]
    intro = [m(L, "introduced_frac") for L in Ls]
    ctl = [m(L, "ctl_introduced_frac") for L in Ls]

    fig, ax = frame(
        "5 · The standard diff misses it; the other half of the same computation does not",
        "Built 48 dictionaries (2 models x 4 depths x 2 resolutions x 3 data orderings) on 4,400 prompts, then "
        "matched each base dictionary's regions one-to-one against the RMU dictionary's by direction similarity. "
        "'Introduced' = an RMU region with no base counterpart. 'Dropped' = a base region with no RMU "
        "counterpart. Control = the identical procedure applied to two data orderings of the SAME base model, "
        "i.e. pure construction noise. Averaged over 3 orderings x 2 resolutions.",
        "Asking 'which regions are new' returns ~3-5% at every depth — BELOW the same-model noise control. "
        "Asking 'which regions vanished' returns 52% at the edited layer against 5% noise, decaying cleanly "
        "with distance from the edit and exactly zero above it. RMU merges regions rather than adding them, and "
        "a one-to-one matcher can only express that as disappearance.")
    x = np.arange(4)
    ax.bar(x - 0.26, dropped, 0.25, color=ORANGE, zorder=3, label="dropped (base regions with no RMU match)")
    ax.bar(x, intro, 0.25, color=BLUE, zorder=3, label="introduced (RMU regions with no base match)")
    ax.bar(x + 0.26, ctl, 0.25, color=AQUA, zorder=3, label="control: same model, different data ordering")
    for xi in range(4):
        for dx, v in ((-0.26, dropped[xi]), (0.0, intro[xi]), (0.26, ctl[xi])):
            ax.text(xi + dx, v + 0.011, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["block 4\n(above the edit)", "block 7\n(the edited layer)",
                        "block 14", "block 24"], fontsize=9)
    ax.set_ylim(0, 0.63)
    tidy(ax, "fraction of regions")
    ax.legend(loc="upper right")
    save(fig, out, "fig05_dropped_vs_introduced")


def fig_stability_inversion(stab_rows, out: Path):
    Ls = [4, 7, 14, 24]
    fig, ax = frame(
        "6 · The injected region is stable across rebuilds; the model's own is not",
        "EP anchors each region on whichever activation happened to arrive first, so the carving depends on the "
        "order the data was streamed. We rebuilt every dictionary under 3 different orderings and asked, for "
        "each model's LARGEST region: do two orderings put the same activations in it? Each dot is one pair of "
        "orderings (3 pairs x 2 resolutions = 6 per model per depth). 1.0 = identical membership.",
        "At the edited layer the base model's biggest region is a construction accident — two orderings share "
        "almost none of its members. The region RMU creates is reproduced by every ordering at 81-92% shared "
        "membership. The intervention is more stable than the structure EP finds in the untouched model, which "
        "is precisely why it is findable. (Block 4 is above the edit, so the two models are the same model.)")
    for i, L in enumerate(Ls):
        for j, (mdl, col) in enumerate((("base", BLUE), ("rmu", ORANGE))):
            v = [r["jaccard"] for r in stab_rows
                 if r["layer"] == L and r["model"] == mdl]
            xs = i + (j - 0.5) * 0.34 + np.random.default_rng(i * 10 + j).uniform(
                -0.045, 0.045, len(v))
            ax.scatter(xs, v, s=52, color=col, alpha=0.85, zorder=4,
                       edgecolors=SURFACE, linewidths=1.4)
            ax.plot([i + (j - 0.5) * 0.34 - 0.1, i + (j - 0.5) * 0.34 + 0.1],
                    [np.median(v)] * 2, color=col, lw=2.4, zorder=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["block 4\n(above the edit — models identical)",
                        "block 7\n(the edited layer)", "block 14", "block 24"],
                       fontsize=9)
    ax.set_ylim(-0.04, 1.06)
    tidy(ax, "membership overlap of the largest region\nacross two data orderings (Jaccard)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=BLUE,
                              label="base model's largest region"),
                       Line2D([], [], marker="o", ls="", ms=8, color=ORANGE,
                              label="RMU model's largest region"),
                       Line2D([], [], color=INK3, lw=2.4, label="median")],
              loc="lower left", bbox_to_anchor=(0.30, 0.02), ncol=3)
    save(fig, out, "fig06_stability_inversion")


def fig_region_sizes(C: Path, out: Path):
    rows = [r for r in rd(C / "regions.csv")
            if int(float(r["layer"])) == 7 and float(r["percentile"]) == 12.0
            and int(float(r["seed"])) == 0]
    fig, ax = frame(
        "7 · What the intervention does to the shape of the dictionary",
        "Every region in one base dictionary and one RMU dictionary at block 7 (same 4,400 prompts, same data "
        "ordering, same cell radius), sorted from largest to smallest. Height is how many of the 542,391 token "
        "activations landed in that region. Note the log scale.",
        "The base model spreads its activations over ~200 regions with the largest holding 6%. RMU collapses "
        "the dictionary to ~100 regions and one of them swallows 39% of everything — 6x larger than anything "
        "the base model builds, and 182x the median region. This single point is the unlearning intervention.")
    for mdl, col, lab in (("base", BLUE, "base dictionary"), ("rmu", ORANGE, "RMU dictionary")):
        n = np.sort([fnum(r, "n") for r in rows if r["model"] == mdl])[::-1]
        ax.plot(np.arange(1, len(n) + 1), n, color=col, lw=2.0, zorder=3, label=lab)
        ax.scatter([1], [n[0]], s=70, color=col, zorder=5, edgecolors=SURFACE, linewidths=1.6)
        ax.annotate(f"{n[0]:,.0f} activations\n({n[0]/542391:.0%} of all)", (1, n[0]),
                    textcoords="offset points", xytext=(16, -4 if mdl == "base" else 6),
                    fontsize=8.5, color=col, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, 215)
    tidy(ax, "activations in the region (log scale)", "regions, ranked largest to smallest")
    ax.legend(loc="upper right")
    save(fig, out, "fig07_region_size_distribution")


def fig_purity_recall(C: Path, out: Path):
    rows = rd(C / "dominant.csv")
    Ls = [4, 7, 14, 24]

    def m(L, mdl, k):
        v = [fnum(r, k) for r in rows if int(float(r["layer"])) == L
             and r["model"] == mdl]
        return float(np.median(v))

    fig, axes = frame(
        "8 · What is inside the region RMU creates",
        "For the largest region of each dictionary, labelled every member activation by which prompt it came "
        "from. 'Purity' = share of its members that came from hazardous (WMDP) prompts. 'Coverage' = share of "
        "ALL hazardous activations in the run that this one region captures. The prompt pool is exactly 50/50, "
        "so 0.5 purity means the region is indifferent to the label. Median over 3 data orderings x 2 "
        "resolutions. No labels were used when building the dictionaries.",
        "The base model has no region that separates hazardous from benign content — its biggest region sits at "
        "chance. RMU's biggest region is 93% hazardous and holds 73% of all hazardous activations in the run. "
        "Deeper in the network the region stays pure but captures less, so the effect disperses with depth "
        "rather than growing.",
        ncols=2)
    for ax, (k, lab, hline) in zip(axes, [("forget_frac", "purity: share of members from hazardous prompts", 0.5),
                                          ("recall_of_forget", "coverage: share of all hazardous activations captured", None)]):
        x = np.arange(4)
        b = [m(L, "base", k) for L in Ls]
        r = [m(L, "rmu", k) for L in Ls]
        ax.bar(x - 0.215, b, 0.36, color=BLUE, zorder=3, label="base model's largest region")
        ax.bar(x + 0.215, r, 0.36, color=ORANGE, zorder=3, label="RMU model's largest region")
        for xi in range(4):
            ax.text(xi - 0.21, b[xi] + 0.018, f"{b[xi]:.2f}", ha="center", fontsize=8, color=INK)
            ax.text(xi + 0.21, r[xi] + 0.018, f"{r[xi]:.2f}", ha="center", fontsize=8.5,
                    color=INK, fontweight="bold" if r[xi] > 0.7 else "normal")
        if hline:
            ax.axhline(hline, color=INK3, lw=1.0, zorder=2)
            ax.text(-0.46, hline + 0.02, "chance (pool is 50/50)", fontsize=7.5,
                    color=INK3, ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels([f"block {L}" for L in Ls], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_title(lab, fontsize=9, color=INK2, pad=6)
        tidy(ax)
    axes[0].legend(loc="upper left")
    save(fig, out, "fig08_purity_and_coverage")


def fig_mass_flow(C: Path, out: Path):
    rows = [r for r in rd(C / "mass_flow.csv") if r["source"] == "dropped"]
    Ls = [7, 14, 24]

    def m(L, k):
        v = [fnum(r, k) for r in rows if int(float(r["layer"])) == L]
        return float(np.median(v)), float(np.min(v)), float(np.max(v))

    fig, ax = frame(
        "9 · Where the members of the dissolved regions went",
        "Took every base-model region that has no counterpart in the RMU dictionary (the 'dropped' regions from "
        "figure 5), then followed each of their member activations into the RMU dictionary and asked whether it "
        "landed in RMU's one giant region. Split by whether the activation came from a hazardous or a benign "
        "prompt. Bars are medians over 3 data orderings x 2 resolutions; whiskers span the range.",
        "The dissolved regions are pulled apart by content: at the edited layer 86-93% of their hazardous "
        "activations are funnelled into the single new region, while only 6-15% of their benign activations "
        "follow. The regions did not simply merge — they were sorted, and the sorting is by exactly the "
        "property RMU was trained on. No labels were used to identify these regions.")
    x = np.arange(3)
    for j, (k, col, lab) in enumerate((
            ("frac_of_forget_into_dominant", ORANGE, "members from hazardous prompts"),
            ("frac_of_retain_into_dominant", BLUE, "members from benign prompts"))):
        vals = [m(L, k) for L in Ls]
        med = [v[0] for v in vals]
        lo = [v[0] - v[1] for v in vals]
        hi = [v[2] - v[0] for v in vals]
        ax.bar(x + (j - 0.5) * 0.42, med, 0.38, color=col, zorder=3, label=lab,
               yerr=[lo, hi], error_kw={"ecolor": INK3, "elinewidth": 1.0,
                                        "capsize": 3, "capthick": 1.0})
        for xi, v in enumerate(med):
            ax.text(xi + (j - 0.5) * 0.42, v + 0.045, f"{v:.2f}", ha="center",
                    fontsize=8.5, color=INK, fontweight="bold" if v > 0.5 else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(["block 7\n(the edited layer)", "block 14", "block 24"], fontsize=9)
    ax.set_ylim(0, 1.08)
    tidy(ax, "share landing in RMU's single giant region")
    ax.legend(loc="upper right")
    save(fig, out, "fig09_mass_flow")


def fig_mechanism(C: Path, out: Path):
    rows = rd(C / "mechanism.csv")
    cu = [fnum(r, "dom_mean_cos_centred_cu") for r in rows]
    mu = [fnum(r, "dom_mean_cos_minus_mu_hat") for r in rows]
    labels = [f"p={float(r['percentile']):g}\nseed {int(float(r['seed']))}" for r in rows]

    fig, ax = frame(
        "10 · The region EP found is the direction the unlearning injected",
        "RMU works by pushing hazardous-context activations onto one fixed random vector u (scaled by 6.5). We "
        "re-ran both models to estimate u directly from the activations, then compared the stored direction of "
        "RMU's giant region against (a) that estimated injected vector and (b) the 'anti-mean' direction, which "
        "is what the region would point at if it were merely an artifact of EP's centring step. Six independent "
        "dictionaries (2 resolutions x 3 data orderings).",
        "The region points at the injected vector, not at the centring artifact, by a factor of 2.5 — and the "
        "value is identical to +/-0.005 across all six independent builds. EP did not just detect that "
        "something changed; the object it hands back is the thing that was injected. This was the "
        "pre-registered test for the most likely false-positive explanation, and it comes out clean.")
    x = np.arange(len(rows))
    ax.bar(x - 0.205, cu, 0.35, color=ORANGE, zorder=3,
           label="similarity to the vector RMU injected")
    ax.bar(x + 0.205, mu, 0.35, color=BLUE, zorder=3,
           label="similarity to the centring artifact (the false-positive explanation)")
    for xi in range(len(rows)):
        ax.text(xi - 0.2, cu[xi] + 0.015, f"{cu[xi]:.3f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
        ax.text(xi + 0.2, mu[xi] + 0.015, f"{mu[xi]:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 0.92)
    tidy(ax, "cosine similarity of the region's direction")
    ax.set_xlabel("six independently built dictionaries at block 7", fontsize=8.5)
    ax.legend(loc="upper right")
    save(fig, out, "fig10_mechanism")


def fig_membership_ari(B: Path, out: Path):
    rows = rd(B / "membership.csv")
    Ls = [4, 7, 14, 24]

    def m(kind, L, model=None):
        v = [fnum(r, "ari") for r in rows if r["kind"] == kind
             and int(float(r["layer"])) == L
             and (model is None or r["model"] == model)]
        return float(np.median(v))

    fig, ax = frame(
        "11 · How much the whole carving changed, with no similarity cutoff",
        "Both checkpoints partition the SAME 542,391 activations, so the two carvings can be compared directly "
        "by who is grouped with whom — the Adjusted Rand Index, 1.0 = identical partitions, 0 = unrelated. "
        "Orange: base carving vs RMU carving. Blue: the same model rebuilt under a different data ordering, "
        "which is the noise this measure has to beat. Medians over resolutions and ordering pairs.",
        "At the edited layer the RMU carving is six times further from the base carving than the base carving "
        "is from a reshuffled rebuild of itself. Block 4 comes out at exactly 1.000, confirming the whole "
        "pipeline end-to-end against a known answer. Note the blue bars: two rebuilds of the same model on the "
        "same data agree only ~0.55, which is why per-region comparisons are so hard here.")
    x = np.arange(4)
    ctl = [m("cross-seed", L, "base") for L in Ls]
    bvr = [m("base-vs-rmu", L) for L in Ls]
    ax.bar(x - 0.21, bvr, 0.38, color=ORANGE, zorder=3, label="base carving vs RMU carving")
    ax.bar(x + 0.21, ctl, 0.38, color=BLUE, zorder=3,
           label="noise floor: same model, different data ordering")
    for xi in range(4):
        ax.text(xi - 0.21, bvr[xi] + 0.018, f"{bvr[xi]:.2f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold" if bvr[xi] < 0.3 else "normal")
        ax.text(xi + 0.21, ctl[xi] + 0.018, f"{ctl[xi]:.2f}", ha="center", fontsize=8, color=INK)
    ax.annotate("identical partitions —\nthe pipeline's null control", (0 - 0.21, 1.0),
                textcoords="offset points", xytext=(6, -34), fontsize=8, color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(["block 4\n(above the edit)", "block 7\n(the edited layer)",
                        "block 14", "block 24"], fontsize=9)
    ax.set_ylim(0, 1.14)
    tidy(ax, "partition agreement (Adjusted Rand Index)")
    ax.legend(loc="upper right")
    save(fig, out, "fig11_partition_agreement")


def fig_seed_variance(C: Path, out: Path):
    rows = [r for r in rd(C / "dominant.csv") if r["model"] == "rmu"]
    fig, axes = frame(
        "12 · Why single-run numbers are not reported",
        "The same RMU dictionary rebuilt under 3 data orderings at 2 cell resolutions, showing the size and "
        "coverage of its largest region. Nothing about the model or the data changes between dots — only the "
        "sequence in which activations were streamed during construction.",
        "At the edited layer the result barely moves (coverage 0.68-0.73). By block 24 the same measurement "
        "ranges over 3-3.5x depending only on streaming order. Purity stays high everywhere, but extent does not, "
        "so any conclusion drawn from one run at block 24 would be an artifact. This is why every number in "
        "this study is reported across orderings.",
        ncols=2)
    Ls = [7, 14, 24]
    for ax, (k, lab) in zip(axes, [("share_of_all_acts", "size: share of all activations in the region"),
                                   ("recall_of_forget", "coverage: share of hazardous activations captured")]):
        for i, L in enumerate(Ls):
            v = [fnum(r, k) for r in rows if int(float(r["layer"])) == L]
            xs = i + np.random.default_rng(i).uniform(-0.09, 0.09, len(v))
            ax.scatter(xs, v, s=52, color=ORANGE, alpha=0.85, zorder=4,
                       edgecolors=SURFACE, linewidths=1.4)
            ax.plot([i - 0.19, i + 0.19], [np.median(v)] * 2, color=ORANGE, lw=2.4, zorder=5)
            ax.text(i + 0.24, np.median(v), f"×{max(v)/min(v):.1f} spread", fontsize=8,
                    color=INK2, va="center")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["block 7\n(the edited layer)", "block 14", "block 24"], fontsize=8.5)
        ax.set_xlim(-0.45, 2.75)
        ax.set_ylim(0, 0.82)
        ax.set_title(lab, fontsize=9, color=INK2, pad=6)
        tidy(ax)
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=ORANGE,
                                   label="one dictionary (6 per depth)"),
                            Line2D([], [], color=ORANGE, lw=2.4, label="median")],
                   loc="upper right")
    save(fig, out, "fig12_seed_variance")


def dominant_stability(grid: Path):
    """Cross-ordering membership overlap of each model's largest region."""
    from .build import stream_perm
    from .gate1b import Grid, label_vector, rebuild_pool

    g = Grid(grid)
    pool = rebuild_pool(g)
    rows = []
    for L in g.layers:
        pid = np.load(g.root / f"prompt_ids_L{L}.npy")
        n = len(pid)
        perms = {s: stream_perm(pool, pid, s) for s in g.seeds}
        for p in g.percentiles:
            for mdl in ("base", "rmu"):
                lab, big = {}, {}
                for s in g.seeds:
                    d = g.get(mdl, L, p, s)
                    lab[s] = label_vector(d, n, perms[s])
                    big[s] = int(np.argmax(np.bincount(lab[s][lab[s] >= 0],
                                                       minlength=len(d.partitions))))
                for s0 in g.seeds:
                    for s1 in g.seeds:
                        if s1 <= s0:
                            continue
                        A = lab[s0] == big[s0]
                        B = lab[s1] == big[s1]
                        rows.append({"layer": L, "percentile": p, "model": mdl,
                                     "pair": f"{s0}-{s1}",
                                     "jaccard": float((A & B).sum() / (A | B).sum())})
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="artifacts/runs/rmu_diff")
    ap.add_argument("--out", default="artifacts/figures/rmu_diff")
    a = ap.parse_args()
    R, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B, C = R / "gate1a", R / "gate1b/shared", R / "gate1c/shared"

    stab_csv = out / "dominant_stability.csv"
    if stab_csv.exists():
        stab = [{**r, "layer": int(float(r["layer"])),
                 "jaccard": float(r["jaccard"])} for r in rd(stab_csv)]
    else:
        stab = dominant_stability(R / "grid/shared")
        with stab_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(stab[0]))
            w.writeheader()
            w.writerows(stab)

    fig_weight_diff(A, out)
    fig_manifestation(A, out)
    fig_norms(A, out)
    fig_metric_matters(A, out)
    fig_dropped_introduced(B, out)
    fig_stability_inversion(stab, out)
    fig_region_sizes(C, out)
    fig_purity_recall(C, out)
    fig_mass_flow(C, out)
    fig_mechanism(C, out)
    fig_membership_ari(B, out)
    fig_seed_variance(C, out)
    log.info("12 figures -> %s", out)


if __name__ == "__main__":
    main()
