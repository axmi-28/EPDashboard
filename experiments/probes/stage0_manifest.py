"""Stage 0 of PLAN_EP_VS_BASELINES: the dataset manifest, before any EP number.

Produces `artifacts/runs/probes/stage0_manifest.csv` plus a provenance JSON.
Four jobs, in the order the plan requires them:

1. **Size the extraction.** Token counts under gemma-2's tokenizer, truncated
   at KE25's `max_seq_len=1024`, so the GPU bill is a measured number and not a
   guess. Padded cost is reported separately -- the benchmark pads to the
   longest text in each batch of 32, and on the code datasets that roughly
   doubles the work actually done.

2. **Stratify by baseline difficulty.** Gate 2's negative result is
   uninterpretable because every task there had a probe above 0.99: a
   piecewise-constant lookup cannot show an advantage against a ceiling. The
   `headroom` stratum is where an EP arrow could structurally show one, so it
   is defined here, from KE25's published quiver, before we can be tempted to
   define it after seeing our own numbers.

3. **Record the pre-vetting split (§8b).** The vet criteria are fitted on one
   half of the 113 and their prediction is tested on the other. The seed is
   written down now, with the git SHA, so "we split it this way" cannot be
   revised once the EP results are in. The halves are stratified by `headroom`
   so neither one ends up holding all the hard datasets.

4. **Flag what the standard split cannot support.** Datasets too small for
   1,024 balanced training examples get `fits_standard_split=False`; KE25 drops
   them from the standard condition too.

Run:  python -m experiments.probes.stage0_manifest --results-root <SAE-Probes checkout>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.probes import benchmark as bm

# KE25's extraction defaults (generate_model_activations.py).
MAX_SEQ_LEN = 1024
BATCH_SIZE = 32

# §8b. Fixed here, never re-drawn. 57 vet-fit / 56 vet-test.
VET_SEED = 20260804
VET_FIT_N = 57

# Difficulty strata, cut on KE25's published quiver test AUC.
#   ceiling  -- a tuned probe is already >=0.99; no arrow can show a gain
#   headroom -- 0.90..0.99; the band where an advantage would be visible
#   hard     -- <0.90; where a partition might beat a hyperplane, or might
#               just mean the concept is not in the residual stream at all
CEILING_CUT = 0.99
HARD_CUT = 0.90


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=bm.REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def token_stats(tokenizer, tags: list[str]) -> pd.DataFrame:
    """Per-dataset token counts. Batched through the fast tokenizer."""
    rows = []
    for i, tag in enumerate(tags, 1):
        txt = bm.texts(tag)
        enc = tokenizer(txt, add_special_tokens=True)["input_ids"]
        lens = np.fromiter((len(t) for t in enc), dtype=np.int64, count=len(enc))
        clipped = np.minimum(lens, MAX_SEQ_LEN)
        # The benchmark pads each batch of 32 to that batch's longest text, in
        # dataset order. That padding is real forward-pass work.
        pad = sum(
            int(clipped[j : j + BATCH_SIZE].max()) * len(clipped[j : j + BATCH_SIZE])
            for j in range(0, len(clipped), BATCH_SIZE)
        )
        rows.append(
            dict(
                dataset=tag,
                n=len(lens),
                tok_mean=float(lens.mean()),
                tok_p95=float(np.percentile(lens, 95)),
                tok_max=int(lens.max()),
                frac_truncated=float((lens > MAX_SEQ_LEN).mean()),
                tokens_clipped=int(clipped.sum()),
                tokens_padded=pad,
            )
        )
        print(f"  [{i:3d}/{len(tags)}] {tag:<40s} n={len(lens):5d}", flush=True)
    return pd.DataFrame(rows)


def build_manifest(results_root: Path) -> tuple[pd.DataFrame, dict]:
    from transformers import AutoTokenizer

    tags = bm.dataset_tags()
    print(f"113-dataset suite: {len(tags)} tags")

    tokenizer = AutoTokenizer.from_pretrained(f"google/{bm.HEADLINE_MODEL}")
    tok = token_stats(tokenizer, tags)

    # Label balance, and the train size KE25 actually uses for each dataset.
    lab = []
    for tag in tags:
        y = bm.labels(tag)
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        num_train = bm.standard_num_train(tag)
        need_pos = int(np.ceil(bm.POS_RATIO * num_train))
        lab.append(
            dict(dataset=tag, n_pos=n_pos, n_neg=n_neg, pos_rate=n_pos / len(y),
                 num_train=num_train,
                 # A balanced draw needs `need_pos` positives and the rest
                 # negatives, with something left over to test on.
                 balanced_draw_ok=bool(n_pos >= need_pos + 50
                                       and n_neg >= (num_train - need_pos) + 50))
        )
    man = tok.merge(pd.DataFrame(lab), on="dataset", validate="1:1")

    # KE25's published quiver at the headline configuration.
    pub = bm.published_baselines(results_root, bm.HEADLINE_MODEL, 20)
    q = bm.quiver(pub)
    man = man.merge(
        q[["method", "test_auc", "val_auc"]].rename(
            columns={
                "method": "ke25_quiver_method",
                "test_auc": "ke25_quiver_test_auc",
                "val_auc": "ke25_quiver_val_auc",
            }
        ),
        left_on="dataset",
        right_index=True,
        how="left",
        validate="1:1",
    )
    # Best single family, kept separately: the quiver can be inflated by
    # picking among five methods on a saturated validation set, so the
    # logistic-regression column is the honest "one arrow" reference.
    lr = pub[pub.method == "logreg"].set_index("dataset")["test_auc"]
    man["ke25_logreg_test_auc"] = man.dataset.map(lr)

    auc = man["ke25_quiver_test_auc"]
    man["stratum"] = np.where(
        auc >= CEILING_CUT, "ceiling", np.where(auc >= HARD_CUT, "headroom", "hard")
    )

    # §8b split-half. Stratified on `stratum` so both halves span the range.
    rng = np.random.default_rng(VET_SEED)
    man = man.sort_values("dataset").reset_index(drop=True)
    role = np.array(["vet_test"] * len(man), dtype=object)
    take = VET_FIT_N
    for s in ["hard", "headroom", "ceiling"]:
        idx = man.index[man.stratum == s].to_numpy().copy()
        rng.shuffle(idx)
        k = min(take, int(round(len(idx) * VET_FIT_N / len(man))))
        role[idx[:k]] = "vet_fit"
        take -= k
    if take > 0:  # rounding shortfall; fill from whatever is still vet_test
        left = man.index[role == "vet_test"].to_numpy().copy()
        rng.shuffle(left)
        role[left[:take]] = "vet_fit"
    man["vet_role"] = role

    prov = dict(
        created_utc=pd.Timestamp.now("UTC").isoformat(),
        git_sha=_git_sha(),
        model=bm.HEADLINE_MODEL,
        hook=bm.HEADLINE_HOOK,
        split=dict(seed=bm.SPLIT_SEED, num_train=bm.NUM_TRAIN,
                   max_amt=bm.MAX_AMT, pos_ratio=bm.POS_RATIO),
        extraction=dict(max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE),
        strata=dict(ceiling_cut=CEILING_CUT, hard_cut=HARD_CUT),
        vet=dict(seed=VET_SEED, fit_n=int((man.vet_role == "vet_fit").sum()),
                 test_n=int((man.vet_role == "vet_test").sum()),
                 stratified_on="stratum"),
        vet_fit_digest=hashlib.sha256(
            ",".join(sorted(man.loc[man.vet_role == "vet_fit", "dataset"])).encode()
        ).hexdigest()[:16],
    )
    return man, prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True,
                    help="checkout of github.com/JoshEngels/SAE-Probes")
    ap.add_argument("--out", type=Path, default=bm.ARTIFACTS)
    args = ap.parse_args()

    man, prov = build_manifest(args.results_root)
    args.out.mkdir(parents=True, exist_ok=True)
    man.to_csv(args.out / "stage0_manifest.csv", index=False)
    (args.out / "stage0_provenance.json").write_text(json.dumps(prov, indent=2))

    print("\n=== strata (KE25 quiver test AUC, gemma-2-9b L20) ===")
    print(man.groupby("stratum").agg(
        n_datasets=("dataset", "size"),
        median_auc=("ke25_quiver_test_auc", "median"),
        min_auc=("ke25_quiver_test_auc", "min"),
    ).to_string())
    print("\n=== §8b vet split ===")
    print(pd.crosstab(man.stratum, man.vet_role).to_string())
    print(f"\nvet_fit digest: {prov['vet_fit_digest']}")
    print("\n=== extraction cost, gemma-2-9b L20 ===")
    print(f"  examples          {man.n.sum():>12,}")
    print(f"  tokens (clipped)  {man.tokens_clipped.sum():>12,}")
    print(f"  tokens (padded)   {man.tokens_padded.sum():>12,}  <- work actually done")
    print(f"  truncated >1024   {(man.n * man.frac_truncated).sum() / man.n.sum():>12.1%}")
    d_model = 3584
    print(f"  stored acts       {man.n.sum() * d_model * 4 / 1e9:>12.2f} GB (fp32, last token)")
    print(f"  datasets with num_train < 1024: {(man.num_train < bm.NUM_TRAIN).sum()}"
          f"   (min {man.num_train.min()})")
    print(f"  datasets where a balanced draw fails: {(~man.balanced_draw_ok).sum()}")
    print(f"\nwrote {args.out/'stage0_manifest.csv'}")


if __name__ == "__main__":
    main()
