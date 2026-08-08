"""Standalone figures for the Gate 2 results, one or more per reported claim.

Each figure is self-contained: the bold title states the finding, one line under
it states what the experiment was, and the numbers that matter are annotated on
the axes rather than left to a legend or to surrounding prose.

Two layout rules, both learned the hard way on the first pass:

- `savefig.bbox="tight"` is **off**. It recomputes the bounding box after
  `subplots_adjust`, so figure-coordinate text placed in the margins drifts and
  collides with axis labels. Margins are set explicitly instead.
- Text is **ASCII only**. Helvetica Neue has no glyph for the arrows, set union,
  or proportional-to signs, and matplotlib renders a missing glyph as a hollow
  box rather than failing, so it is easy to ship a figure with a box in the
  caption.

Values are transcribed from the result files rather than recomputed, so figures
regenerate without the activation caches on disk. Provenance is in SOURCES.

    python -m experiments.monitor.make_gate2_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/runs/monitor/figures")

EP = "#2F6193"        # exemplar partitioning
NULL = "#98A3B0"      # matched-K random coreset
PROBE = "#B0522A"     # linear probe baseline
AMBER = "#C08A3E"
GOOD = "#2C6F52"
BAD = "#9C3535"
INK = "#151A21"
INK2 = "#5A6473"
INK3 = "#8A93A0"

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.6,
    "axes.edgecolor": "#B9C0C9", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6EAEE", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK2,
    "legend.frameon": False, "figure.facecolor": "white",
})


def frame(fig, title, subtitle, caption, *, top, bottom,
          left=0.085, right=0.975, wspace=None):
    """Title block above the axes, experiment note below them.

    The caption is anchored by its *bottom* edge and the axes floor is pushed up
    to clear it, so adding a line to the caption can never clip it — the first
    version anchored the top and silently cut the third line off the page.
    """
    line = 8.5 * 1.38 / (fig.get_figheight() * 72)
    n_lines = caption.count("\n") + 1
    bottom = max(bottom, 0.018 + n_lines * line + 0.075)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        **({"wspace": wspace} if wspace else {}))
    fig.text(0.018, 0.975, title, fontsize=13.5, fontweight="bold",
             va="top", color=INK)
    fig.text(0.018, 0.895, subtitle, fontsize=9.6, va="top", color=INK2)
    fig.text(0.018, 0.018, caption, fontsize=8.5, va="bottom", color=INK3)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {OUT / name}")


# ------------------------------------------------------------------ claim 1

def fig1_concepts():
    concepts = ["Language ID\nBulgarian / English", "Scaffold ID\n5 chat templates",
                "Code vs. math\nMBPP / GSM8K", "Refusal\nharmful / benign"]
    margins = np.array([[7.9, 10.5, 8.9, -1.4, -0.4],
                        [-0.9, -3.9, 1.1, -1.3, 0.7],
                        [0.2, -2.2, -3.1, -2.4, -0.2],
                        [1.1, 0.7, -1.1, -3.0, -1.2]])
    ps, Ks = [1, 2, 4, 8, 10], [5796, 2037, 686, 226, 176]
    shades = ["#1B4A78", "#356A9C", "#5A87B4", "#8CAAC9", "#B9CBDD"]

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    xs = np.arange(4)
    w = 0.165
    for j in range(5):
        ax.bar(xs + (j - 2) * w, margins[:, j], w * 0.9, color=shades[j],
               label=f"p={ps[j]}  K={Ks[j]}", zorder=3)
    ax.axhspan(-2, 2, color="#F1F3F6", zorder=1)
    ax.axhline(0, color="#9AA3AE", lw=1.0, zorder=2)
    ax.text(3.46, 2.5, "inside the shaded band EP is indistinguishable from a random partition",
            fontsize=8.4, color=INK3, ha="right", va="bottom")
    ax.annotate("the only concept EP wins,\nand it wins at 3 consecutive resolutions",
                xy=(-0.16, 10.5), xytext=(0.42, 12.6), fontsize=9, color=GOOD,
                arrowprops=dict(arrowstyle="-", color=GOOD, lw=0.9))
    ax.set_xticks(xs)
    ax.set_xticklabels(concepts)
    ax.set_ylabel("margin over matched-K random partition   (draw sd)")
    ax.set_ylim(-6, 15)
    ax.set_xlim(-0.55, 3.55)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", visible=False)

    frame(fig, "EP's partition beats a random partition on 1 of 4 concepts",
          "Region-lookup flag scored against K activations sampled at random from the same build stream.",
          "Experiment: assign each labelled activation to a region, score it by that region's harmful rate measured on other folds, take AUROC, "
          "and subtract the same\nquantity for a matched-K random partition. 10 coreset draws. gemma-2-2b-it, layer 20.",
          top=0.80, bottom=0.30)
    save(fig, "fig1_concepts_vs_null.png")


def fig2_language_ood():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))

    ax = axes[0]
    xs = np.arange(5)
    ep_full = [.9175, .9354, .9458, .8653, .8783]
    cs_full = [.8448, .8677, .8856, .8805, .8741]
    ep_m = [.9115, .9300, .9402, .8699, .8667]
    cs_m = [.8372, .8660, .8812, .8735, .8647]
    ax.plot(xs, ep_full, "-o", color=EP, lw=2.2, ms=6, label="EP, all activations", zorder=4)
    ax.plot(xs, ep_m, "--s", color=EP, lw=1.5, ms=5, mfc="white", label="EP, distance-matched", zorder=4)
    ax.plot(xs, cs_full, "-o", color=NULL, lw=1.8, ms=5, label="random partition, all", zorder=3)
    ax.plot(xs, cs_m, "--s", color=NULL, lw=1.3, ms=4, mfc="white", label="random partition, matched", zorder=3)
    for i in (0, 1, 2):
        ax.annotate("", xy=(xs[i], ep_m[i]), xytext=(xs[i], cs_m[i]),
                    arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.1))
    ax.text(0.15, 0.875, "gap survives\ndistance matching", fontsize=8.8, color=GOOD)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p={p}" for p in [1, 2, 4, 8, 10]])
    ax.set_ylabel("AUROC, Bulgarian vs. English")
    ax.set_ylim(.72, 1.0)
    ax.legend(loc="lower left", fontsize=8.2)
    ax.set_title("Forcing both classes to share a distance\ndistribution costs EP almost nothing",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    ax = axes[1]
    diffs = [.0914, .0953, .0462, .0151, .0191]
    cols = [GOOD, GOOD, "#7FA894", NULL, NULL]
    ax.bar(np.arange(5), diffs, .6, color=cols, zorder=3)
    for i, d in enumerate(diffs):
        ax.text(i, d + .004, f"+{d:.3f}", ha="center", fontsize=8.6, color=INK2)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["nearest\n20%", "", "middle", "", "farthest\n20%"], fontsize=8.6)
    ax.set_xlabel("cosine distance from the activation to its nearest exemplar", labelpad=6)
    ax.set_ylabel("EP advantage over random partition (AUROC)")
    ax.set_ylim(0, .125)
    ax.grid(axis="x", visible=False)
    ax.set_title("The advantage is largest among activations\nclosest to the dictionary, not farthest",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    frame(fig, "The one positive result is semantic, not out-of-distribution detection",
          "Bulgarian sits outside the English Pile support, so the win could have been distance-to-support in disguise. It is not.",
          "Experiment: bin activations on pooled distance quantiles and subsample so both classes share a distance distribution, removing any "
          "distance shortcut.\nDistance alone gives AUROC 0.545 / 0.504 / 0.392 at p=1 / 2 / 4, so it barely separates the classes where EP wins. "
          "Right panel is at p=4.",
          top=0.775, bottom=0.20, left=0.075, wspace=0.30)
    save(fig, "fig2_language_semantic_not_ood.png")


def fig3_purity():
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    groups = ["Language ID\nclasses occupy separate volumes",
              "Code vs. math\nclasses share one dense volume"]
    ep_cov, cs_cov = [.657, .126], [.498, .655]
    xs = np.arange(2)
    ax.bar(xs - .17, ep_cov, .31, color=EP, label="EP partition", zorder=3)
    ax.bar(xs + .17, cs_cov, .31, color=NULL, label="matched-K random partition", zorder=3)
    for i in range(2):
        ax.text(i - .17, ep_cov[i] + .016, f"{ep_cov[i]:.3f}", ha="center",
                fontsize=10, color=EP, fontweight="bold")
        ax.text(i + .17, cs_cov[i] + .016, f"{cs_cov[i]:.3f}", ha="center",
                fontsize=10, color=INK2)
    ax.annotate("EP gives Bulgarian\nits own regions", xy=(-.33, .60), xytext=(-.50, .80),
                fontsize=9, color=GOOD, arrowprops=dict(arrowstyle="-", color=GOOD, lw=.9))
    ax.annotate("EP does not separate code from math;\nrandom sampling does", xy=(1.02, .60),
                xytext=(.30, .80), fontsize=9, color=BAD,
                arrowprops=dict(arrowstyle="-", color=BAD, lw=.9))
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylabel("share of the positive class held in regions\nthat are at least 90% that class")
    ax.set_ylim(0, .95)
    ax.set_xlim(-.55, 1.55)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, -.17))
    ax.grid(axis="x", visible=False)
    frame(fig, "EP region identity tracks volumetric distinctions, not directional ones",
          "The mechanism behind every concept result, and why refusal was never going to work.",
          "Experiment: count regions holding at least 5 prompts that are at least 90% one class,\nthen measure what share of that class they "
          "hold (p=4, K=686). EP places exemplars to cover\nthe support, so it wins when a class has its own volume; random sampling places "
          "them in\nproportion to density, so it wins inside a contested one.",
          top=0.795, bottom=0.32)
    save(fig, "fig3_volumetric_vs_directional.png")


# ------------------------------------------------------------------ claim 2

def fig4_flag_resolution():
    ps, Ks = [1, 2, 4, 8, 10], [5796, 2037, 686, 226, 176]
    ep = [.9580, .8979, .6457, .5467, .5113]
    cs = np.array([.8847, .8211, .7950, .7606, .6850])
    sd = np.array([.0676, .1081, .1397, .0702, .1488])
    share = [.375, .517, .822, .947, .983]
    xs = np.arange(5)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True,
                             height_ratios=[2.6, 1])
    ax = axes[0]
    ax.fill_between(xs, cs - sd, cs + sd, color=NULL, alpha=.22, zorder=2,
                    label="random partition, +/- 1 sd")
    ax.plot(xs, cs, "--o", color=NULL, lw=1.7, ms=5, zorder=3,
            label="matched-K random partition")
    ax.plot(xs, ep, "-o", color=EP, lw=2.6, ms=7.5, zorder=4,
            label="EP harmful-region flag")
    ax.axhline(.9999, color=PROBE, lw=1.8, zorder=3)
    ax.text(4.42, .983, "ridge probe on the raw activation = 0.9999",
            color=PROBE, fontsize=9, va="top", ha="right")
    ax.axhline(.5, color="#9AA3AE", lw=1, ls=":", zorder=2)
    ax.text(2.4, .512, "chance", fontsize=8.4, color=INK2, va="bottom")
    ax.text(0.16, .975, "0.958", fontsize=10.5, color=EP, fontweight="bold",
            va="center")
    ax.text(4.0, .462, "0.511", fontsize=10.5, color=BAD, fontweight="bold",
            va="center", ha="center")
    ax.set_ylabel("cross-fit AUROC\nharmful vs. benign")
    ax.set_ylim(.44, 1.05)
    ax.set_xlim(-.5, 4.5)
    ax.legend(loc="lower left", fontsize=8.4)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.bar(xs, share, .5, color=AMBER, zorder=3)
    for i, v in enumerate(share):
        ax.text(i, v + .04, f"{v:.0%}", ha="center", fontsize=8.8, color="#8E6526")
    ax.text(-.42, 1.30, "the chat scaffold pulls almost every prompt into one region\nas resolution coarsens, leaving the flag nothing to separate",
            fontsize=8.8, color="#8E6526", va="top")
    ax.set_ylabel("share of all 600 prompts\nin the largest region")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, .5, 1.0])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p={p}\nK={k}" for p, k in zip(ps, Ks)])
    ax.set_xlabel("dictionary resolution, coarser and cheaper to the right", labelpad=6)
    ax.grid(axis="x", visible=False)

    frame(fig, "The harmful-region flag works at K=5796 and is at chance by K=176",
          "300 harmful + 300 benign instructions routed through prebuilt Pile dictionaries at five resolutions.",
          "Experiment: chat-formatted prompts, final-position layer-20 activations, assigned to regions; each region's harmful rate is fitted on "
          "4 of 5 folds and the held-out\nfold scored by lookup. The dictionary never saw a harmful prompt, so the concentration is not circular "
          "(at p=1 one region holds 225 harmful prompts at 95.1% purity) -\nbut no resolution beats the matched-K random partition.",
          top=0.80, bottom=0.215)
    save(fig, "fig4_flag_vs_resolution.png")


# ------------------------------------------------------------------ claim 3

def fig5_label_efficiency():
    tasks = [
        ("Refusal (chat)", [16, 32, 64, 128, 256],
         [.9301, .9593, .9648, .9674, .9687], [.9106, .9270, .9306, .9384, .9453],
         [.9983, .9985, .9985, .9984, .9984]),
        ("Language ID", [16, 32, 64, 128, 256, 512, 1024],
         [.7973, .8046, .8175, .8453, .8684, .8979, .9214],
         [.6687, .7217, .7648, .7938, .8256, .8468, .8680],
         [.9828, .9853, .9871, .9873, .9878, .9880, .9881]),
        ("Code vs. math", [16, 32, 64, 128, 256, 512],
         [.9131, .9299, .9418, .9458, .9537, .9569],
         [.8266, .8812, .9184, .9358, .9501, .9574], [1, 1, 1, 1, 1, 1]),
        ("Scaffold ID", [16, 32, 64, 128, 256, 512, 1024],
         [.8553, .9220, .9498, .9691, .9763, .9838, .9868],
         [.8649, .9061, .9400, .9587, .9694, .9798, .9863],
         [.9992, .9994, .9996, .9997, .9997, .9997, .9998]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.9), sharey=True)
    for k, (ax, (name, n, ep, cs, pr)) in enumerate(zip(axes, tasks)):
        ax.fill_between(n, ep, pr, color=PROBE, alpha=.08, zorder=1)
        ax.plot(n, pr, "-o", color=PROBE, lw=2.1, ms=4.5, zorder=4)
        ax.plot(n, cs, "--o", color=NULL, lw=1.5, ms=3.5, zorder=3)
        ax.plot(n, ep, "-o", color=EP, lw=2.1, ms=4.5, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024])
        ax.set_xticklabels(["16", "64", "256", "1k"])
        ax.set_ylim(.60, 1.05)
        ax.set_title(name, fontsize=10, loc="left", color=INK, pad=8)
        ax.set_xlabel("labelled examples")
        ax.grid(axis="x", visible=False)
        ax.text(16.6, .625, f"gap at n=16:  {pr[0]-ep[0]:+.3f}", fontsize=8.4,
                color=PROBE, va="bottom")
        if k == 0:
            ax.text(17, 1.028, "difference-in-means probe", color=PROBE,
                    fontsize=8.6, va="top")
            ax.text(17, .905, "EP flag", color=EP, fontsize=8.6, va="top")
            ax.text(17, .868, "random partition", color=NULL, fontsize=8.6, va="top")
    axes[0].set_ylabel("AUROC on a held-out eval pool")

    frame(fig, "No crossover on any task at any label budget",
          "The prediction was that ranking K bins needs fewer labels than fitting a 2304-dimensional hyperplane. The probe's lead is widest at n=16.",
          "Experiment: split each task 50/50 into a fit pool and an eval pool no scorer ever draws from; draw n labels from the fit pool 20 times "
          "per budget, fit every scorer on\nthat draw, score the whole eval pool. The partition itself costs no labels, so at n=16 the flag has "
          "16 counts spread over K=176-5796 bins and most bins stay empty.",
          top=0.755, bottom=0.235, left=0.062, wspace=0.16)
    save(fig, "fig5_label_efficiency.png")


# ------------------------------------------------------------------ claim 4

def fig6_transitions():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), width_ratios=[1.3, 1, .92])
    ps, Ks = [2, 4, 8, 10], [2037, 686, 226, 176]
    xs = np.arange(4)

    ax = axes[0]
    obs, shuf = [.0958, .1230, .1460, .1538], [.0266, .0438, .0560, .0625]
    ax.bar(xs - .18, obs, .32, color=EP, zorder=3, label="observed")
    ax.bar(xs + .18, shuf, .32, color=NULL, zorder=3,
           label="shuffled within each prompt")
    for i in range(4):
        ax.text(i, obs[i] + .011, f"{obs[i]/shuf[i]:.1f}x", ha="center",
                fontsize=10.5, color=EP, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p={p}\nK={k}" for p, k in zip(ps, Ks)])
    ax.set_ylabel("P(region at t+1 equals region at t)")
    ax.set_ylim(0, .20)
    ax.legend(loc="upper left", fontsize=8.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Consecutive tokens repeat regions\n2.5 to 3.6x more than chance",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    ax = axes[1]
    gains = [-.036, -.021, .001, .020]
    ax.bar(xs, gains, .5, color=[BAD if g < 0 else NULL for g in gains], zorder=3)
    ax.axhline(0, color="#9AA3AE", lw=1)
    for i, g in enumerate(gains):
        ax.text(i, g + (.0035 if g >= 0 else -.0035), f"{g:+.3f}", ha="center",
                fontsize=8.8, color=INK2, va="bottom" if g >= 0 else "top")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"p={p}" for p in ps])
    ax.set_ylim(-.055, .04)
    ax.set_ylabel("AUROC gained from using order\n(bigram minus unigram surprise)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Knowing the order adds nothing\nto detecting harmful prompts",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    ax = axes[2]
    ax.bar([0, 1], [.656, .958], .48, color=[NULL, EP], zorder=3)
    ax.text(0, .672, "0.656", ha="center", fontsize=11, color=INK2, fontweight="bold")
    ax.text(1, .974, "0.958", ha="center", fontsize=11, color=EP, fontweight="bold")
    ax.axhline(.5, color="#9AA3AE", lw=1, ls=":")
    ax.text(1.42, .508, "chance", fontsize=8.4, color=INK2, ha="right", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["12 trajectory symbols\nbest of 5 scorers x 4 p",
                        "1 symbol at the\nfinal position"], fontsize=8.8)
    ax.set_ylim(.45, 1.06)
    ax.set_xlim(-.55, 1.55)
    ax.set_ylabel("AUROC, harmful vs. benign")
    ax.grid(axis="x", visible=False)
    ax.set_title("Averaging over a request destroys the\nsignal rather than accumulating it",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    frame(fig, "Region sequences have real structure, and none of it is about the label",
          "Every prompt becomes a string of region ids. The repeats are genuine; the order buys nothing.",
          "Experiment: layer-20 activations at every token, each assigned to a region. The transition table is fitted on 192,000 positions from "
          "1,500 held-out Pile documents, not on\nthe prompts being scored. The 9 chat-scaffold positions of a median 21 are masked, since they "
          "are identical across all 600 prompts. The null shuffles each prompt's own\nsequence, preserving its region marginal exactly and "
          "destroying only order.",
          top=0.775, bottom=0.225, left=0.062, wspace=0.32)
    save(fig, "fig6_token_sequences.png")


# ------------------------------------------------------------------ claim 5

def fig7_layers():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), width_ratios=[1.3, 1])

    ax = axes[0]
    labs = ["L4\np=4, K=491", "L12\np=10, K=145", "L20\np=4, K=686", "L20\np=10, K=176"]
    ratio = [4.347, 2.046, .614, .615]
    nullm = np.array([2.034, 1.879, 1.273, .950])
    nulls = np.array([.566, .589, .765, .329])
    xs = np.arange(4)
    ax.fill_between(xs, nullm - nulls, nullm + nulls, color=NULL, alpha=.25,
                    zorder=2, label="random partition, +/- 1 sd")
    ax.plot(xs, nullm, "--o", color=NULL, lw=1.6, ms=4.5, zorder=3)
    ax.plot(xs, ratio, "-o", color=EP, lw=2.6, ms=8, zorder=4, label="EP")
    ax.axhline(1, color="#9AA3AE", lw=1, ls=":", zorder=2)
    ax.text(3.45, 1.06, "equal spread", fontsize=8.4, color=INK2, ha="right")
    ax.annotate("4.35x: harmful prompts use 4.3x more\nregions than benign  (+4.1 sd over null)",
                xy=(0.06, 4.30), xytext=(0.55, 4.55), fontsize=9, color=GOOD,
                arrowprops=dict(arrowstyle="-", color=GOOD, lw=.9))
    ax.annotate("0.62x: harmful now more concentrated\nthan benign, but inside the null band",
                xy=(2.9, .70), xytext=(1.05, .18), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.9))
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylabel("effective regions used by harmful prompts\ndivided by effective regions used by benign")
    ax.set_ylim(0, 5.5)
    ax.set_xlim(-.35, 3.5)
    ax.legend(loc="upper right", fontsize=8.4)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("depth, early to late", labelpad=6)

    ax = axes[1]
    labs2 = ["L4 p=4", "L12 p=10", "L20 p=10"]
    content, final = [22.89, 3.72, 3.79], [1.00, 1.06, 1.00]
    xs2 = np.arange(3)
    ax.bar(xs2 - .18, content, .32, color=EP, zorder=3,
           label="last token of the instruction")
    ax.bar(xs2 + .18, final, .32, color="#C6CDD5", zorder=3,
           label="true final token (chat scaffold)")
    for i in range(3):
        ax.text(i - .18, content[i] + .6, f"{content[i]:.1f}", ha="center",
                fontsize=9, color=EP)
        ax.text(i + .18, final[i] + .6, f"{final[i]:.2f}", ha="center",
                fontsize=9, color=INK2)
    ax.annotate("all 600 prompts land in one region:\nthis position is scaffold, and layer 4\nhas not read the instruction yet",
                xy=(.20, 1.4), xytext=(.52, 13.5), fontsize=8.8, color=BAD,
                arrowprops=dict(arrowstyle="-", color=BAD, lw=.9))
    ax.set_xticks(xs2)
    ax.set_xticklabels(labs2)
    ax.set_ylabel("effective regions used by the 300 harmful prompts")
    ax.set_ylim(0, 27)
    ax.set_xlim(-.55, 2.55)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.grid(axis="x", visible=False)
    ax.set_title("The result depends on which position is read",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    frame(fig, "Harmful prompts start scattered across many regions and consolidate with depth",
          "Effective region count is exp(H), which stays comparable across layers where a raw region count would not, since K differs by 5x.",
          "Experiment: the same 600 prompts assigned at layers 4, 12 and 20 with percentile-matched dictionaries, read at the last token of the "
          "instruction. Qualifier: only the\nlayer-4 point clears its matched-K null, so what EP adds over a random partition is the early scatter, "
          "not the late consolidation.",
          top=0.775, bottom=0.215, left=0.07, wspace=0.28)
    save(fig, "fig7_layer_consolidation.png")


def fig8_crosslayer():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    ns = [600, 1500, 3000, 7460]

    ax = axes[0]
    mi_ep = [.9329, .8649, .7844, .7296]
    mi_nl = [1.4333, 1.2909, 1.1816, 1.0878]
    ax.fill_between(ns, mi_ep, mi_nl, color=BAD, alpha=.10, zorder=1)
    ax.plot(ns, mi_nl, "--o", color=NULL, lw=1.9, ms=5.5,
            label="matched-K random partition", zorder=3)
    ax.plot(ns, mi_ep, "-o", color=EP, lw=2.3, ms=6, label="EP", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("activations used for the estimate", labelpad=6)
    ax.set_ylabel("mutual information between the early region\nand the late region  (nats)")
    ax.set_ylim(.55, 1.62)
    ax.legend(loc="upper right", fontsize=8.4)
    ax.grid(axis="x", visible=False)
    for n, m in zip(ns, [-3.2, -3.1, -3.6, -3.3]):
        ax.text(n, .60, f"{m:+.1f} sd", ha="center", fontsize=8.4, color=BAD)
    ax.set_title("Both estimates fall as bias shrinks while the gap holds,\nso this is not a small-sample artifact",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    ax = axes[1]
    ep_lift, null_lift = [.1584, .1869], [.1785, .2669]
    xs = np.arange(2)
    ax.bar(xs - .17, ep_lift, .31, color=EP, zorder=3, label="EP")
    ax.bar(xs + .17, null_lift, .31, color=NULL, zorder=3,
           label="matched-K random partition")
    for i in range(2):
        ax.text(i - .17, ep_lift[i] + .007, f"{ep_lift[i]:.3f}", ha="center",
                fontsize=9, color=EP)
        ax.text(i + .17, null_lift[i] + .007, f"{null_lift[i]:.3f}", ha="center",
                fontsize=9, color=INK2)
        ax.text(i, max(ep_lift[i], null_lift[i]) + .035,
                f"EP {100*(ep_lift[i]/null_lift[i]-1):.0f}%", ha="center",
                fontsize=9.5, color=BAD, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["L12 to L20\nprimary pair", "L4 to L20\nexploratory pair"])
    ax.set_ylabel("how much the early region adds over guessing\nthe most common late region (normalised)")
    ax.set_ylim(0, .34)
    ax.set_xlim(-.55, 1.55)
    ax.legend(loc="upper left", fontsize=8.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("A cross-fitted, bias-free statistic agrees\nwith the mutual information",
                 fontsize=9.8, loc="left", color=INK, pad=10)

    frame(fig, "EP regions correspond across layers worse than random regions do",
          "This is the premise that motivates cross-layer analysis, and it is the claim the experiment refutes.",
          "Experiment: assign each activation at an early and a late layer with percentile-matched dictionaries, then measure how much the early "
          "region tells you about the late\none. The control uses matched-K random partitions at both layers, since mutual information is K-biased. "
          "Limits: the primary pair's margin is 11% relative, and the hub\ncarries one percentile per early layer, so no multi-resolution artifact "
          "check is possible here.",
          top=0.775, bottom=0.235, left=0.075, wspace=0.34)
    save(fig, "fig8_crosslayer_correspondence.png")


# ------------------------------------------------------------------ addendum

def fig9_attacks():
    pts = [("payload split", .000, .970), ("GCG suffix", .000, .903),
           ("prefix injection", .060, .967), ("roleplay", .120, .950),
           ("distractor", .200, .913), ("refusal suppression", .200, .793),
           ("leetspeak", .560, .463), ("base64", 1.000, .000)]
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    s = np.array([p[1] for p in pts])
    d = np.array([p[2] for p in pts])
    b, a = np.polyfit(s, d, 1)
    xs = np.linspace(-.03, 1.03, 20)
    ax.plot(xs, a + b * xs, "--", color=NULL, lw=1.5, zorder=2)
    ax.axvspan(.45, 1.10, color=PROBE, alpha=.06, zorder=1)
    for name, sv, dv in pts:
        danger = sv >= .5
        ax.scatter([sv], [dv], s=130, color=PROBE if danger else EP, zorder=4,
                   edgecolor="white", linewidth=1.3)

    # Six of the eight attacks sit in one corner, so their labels collide if
    # placed next to the markers. Leader lines to an evenly spaced column keep
    # every name legible without moving the data.
    # Sorting the column by each point's own height makes the mapping monotone,
    # so no two leader lines cross and every label is unambiguous.
    col_x, ys = 0.315, [1.035, 0.965, 0.895, 0.825, 0.755, 0.685]
    for (name, sv, dv), ly in zip(sorted(pts[:6], key=lambda t: -t[2]), ys):
        ax.plot([sv + .012, col_x - .008], [dv, ly], "-", color="#C3CAD2",
                lw=0.8, zorder=3)
        ax.text(col_x, ly, name, fontsize=9.4, ha="left", va="center",
                color=INK2)
    for name, sv, dv in pts[6:]:
        ax.text(sv + (-.035 if sv > .75 else .032), dv, name, fontsize=9.8,
                ha="right" if sv > .75 else "left", va="center",
                color=PROBE, fontweight="bold")
    ax.text(.785, .95, "the two attacks that actually\nbreak refusal are the two\nthat no scorer detects",
            fontsize=9.6, color=PROBE, ha="center", fontweight="bold", va="top")
    ax.text(.03, .07, "r = -0.98        excluding base64: r = -0.94", fontsize=11,
            color=INK, fontweight="bold")
    ax.set_xlabel("attack success: fraction of harmful prompts the model did NOT refuse", labelpad=8)
    ax.set_ylabel("detectability: P(harmful flagged) minus P(benign flagged),\nunder one rule fit on plain prompts and never retuned")
    ax.set_xlim(-.07, 1.12)
    ax.set_ylim(-.06, 1.06)
    frame(fig, "The attacks nobody can detect are precisely the attacks that work",
          "Not a result about EP: it holds for every scorer tested, and it invalidates the usual way per-attack robustness is reported.",
          "Experiment: 50 harmful goals x 9 templates, greedy generation, 48 new tokens, responses labelled refused / safe-engagement / complied / "
          "degenerate. Wrapping attacks\n(prefix, payload split, GCG) leave the content tokens intact, so refusal fires and the content direction "
          "stays readable; rewriting attacks (leetspeak, base64) defeat both at\nonce. Base64 is capability-limited here - 0% refusal but near-zero "
          "real compliance, since the model can encode but not decode-and-execute.",
          top=0.775, bottom=0.245, left=0.10)
    save(fig, "fig9_attack_success_vs_detectability.png")


SOURCES = """
fig1  gate2_a3_concepts.csv, gate2_a0_routing.csv      auroc_margin_sd
fig2  gate2_a3b_ood.csv, gate2_a3b_ood.json            matched vs full, distance strata
fig3  region-purity analysis over eval.npz             >=90% pure regions at p=4
fig4  gate2_a0_routing.csv                             ep_cv_auroc, cs_cv_auroc, occupancy
fig5  gate2_a1_labelcurve.csv                          auroc_mean by scorer and budget
fig6  gate2_b_transitions.json, gate2_b_trajectory.csv
fig7  gate2_c_crosslayer.json                          concentration.content / .final
fig8  gate2_c2_scale.json                              labelled.pairs
fig9  gate2_a4_attack_success.json, gate2_a2b_detection.json
"""


def main():
    print("writing figures:")
    for f in (fig1_concepts, fig2_language_ood, fig3_purity, fig4_flag_resolution,
              fig5_label_efficiency, fig6_transitions, fig7_layers,
              fig8_crosslayer, fig9_attacks):
        f()
    (OUT / "SOURCES.txt").write_text(SOURCES.strip() + "\n")
    print(f"\n{OUT}/SOURCES.txt")


if __name__ == "__main__":
    main()
